from fineTuningMethod import FineTuningMethod
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from privacyKnowledge import PrivacyKnowledge
from utilityKnowledge import UtilityKnowledge

from plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd

class MergeSynData(FineTuningMethod):
    def __init__(
            self,
            trained_generators: dict,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        super().__init__(trained_generators, privacyCalculator, utilityCalculator)
        self._n_histogram_bins = 10
        self.column_frequency_error = 0.2

    @staticmethod
    def name() -> str:
        return "mergesyndata"
    

    def fine_tune(self, current_generator: Plugin, real: DataLoader, syn: DataLoader, count: int, priv_metric_req: PrivacyKnowledge, priv_val_req: float, util_metric_req: UtilityKnowledge, util_val_req: float, error_range: float) -> DataLoader:
        self.count = count
        self.current_generator = current_generator
        self.level = current_generator.get_privacy_level()
        print("fine tune")
        
        ## Fine tune privacy
        if priv_metric_req is not None and util_metric_req is not None:
            new_syn = self.privacy_utility_tradeoff(real, syn, priv_metric_req, priv_val_req, util_metric_req, util_val_req, error_range)
        elif priv_metric_req is not None:
            new_syn = self.privacy_fine_tuning(real, syn, priv_metric_req, priv_val_req, error_range)
        elif util_metric_req is not None:
            new_syn = self.utility_fine_tuning(real, syn, util_metric_req, util_val_req, error_range)

        return new_syn

    
    def privacy_utility_tradeoff(self, real: DataLoader, syn: DataLoader, priv_metric_req: PrivacyKnowledge, priv_val_req: float, util_metric_req: UtilityKnowledge, util_val_req: float, error_range: float) -> DataLoader:
        counter = 0
        while counter < 10:

            satisfied = True

            ## Calculate the privacy
            priv_val = priv_metric_req.calculate(real, syn)
            if not priv_metric_req.satisfied(priv_val_req, priv_val, error_range):
                ## Check in what direction the privacy has to change
                result = priv_metric_req.change(priv_val_req, priv_val)
                if result['direction'] == 'up':
                    new = self.increase_privacy(real, syn, priv_metric_req, priv_val_req, result['amount'], error_range)
                elif result['direction'] == 'down':
                    new = self.decrease_privacy(real, syn, priv_metric_req, priv_val_req, result['amount'], error_range)

                if new is not None:
                    syn = new
                else:
                    satisfied = False

            print('priv increased: ' + str(priv_metric_req.calculate(real, syn)))
            
            ## Calculate the utility
            util_val = util_metric_req.calculate(real, syn)
            if not util_metric_req.satisfied(util_val_req, util_val, error_range):
                ## Check in what direction the utility has to change
                result = util_metric_req.change(util_val_req, util_val)
                if result['direction'] == 'up':
                    new = self.increase_utility(real, syn, util_metric_req, util_val_req, result['amount'], error_range)
                elif result['direction'] == 'down':
                    new = syn

                if new is not None:
                    syn = new
                else:
                    satisfied = False

            print('util increased: ' + str(util_metric_req.calculate(real, syn)))

            priv_val = priv_metric_req.calculate(real, syn)
            util_val = util_metric_req.calculate(real, syn)
            if satisfied and priv_metric_req.satisfied(priv_val_req, priv_val, error_range) and (util_metric_req.satisfied(util_val_req, util_val, error_range) or util_metric_req.change(util_val_req, util_val)['direction'] == 'down'):
                return syn
            
            # if satisfied and (util_metric_req.satisfied(util_val_req, util_val, error_range) or util_metric_req.change(util_val_req, util_val)['direction'] == 'down'):
            #     return syn

            counter = counter + 1

        return None
    

    def privacy_fine_tuning(self, real: DataLoader, syn: DataLoader, priv_metric_req: PrivacyKnowledge, priv_val_req: float, error_range: float) -> DataLoader:
        ## Calculate the privacy
        priv_val = priv_metric_req.calculate(real, syn)
        if priv_metric_req.satisfied(priv_val_req, priv_val, error_range):
            return syn
        
        counter = 0
        while counter < 10:
            ## Check in what direction the privacy has to change
            result = priv_metric_req.change(priv_val_req, priv_val)
            if result['direction'] == 'up':
                new = self.increase_privacy(real, syn, priv_metric_req, priv_val_req, result['amount'], error_range)
            elif result['direction'] == 'down':
                new = self.decrease_privacy(real, syn, priv_metric_req, priv_val_req, result['amount'], error_range)
        
            if new is not None:
                return new
            else:
                counter = counter + 1
        
        return None


    def utility_fine_tuning(self, real: DataLoader, syn: DataLoader, util_metric_req: UtilityKnowledge, util_val_req: float, error_range: float) -> DataLoader:
        ## Calculate the utility
        util_val = util_metric_req.calculate(real, syn)
        if util_metric_req.satisfied(util_val_req, util_val, error_range):
            return syn
        
        counter = 0
        while counter < 10:
            ## Check in what direction the utility has to change
            result = util_metric_req.change(util_val_req, util_val)
            if result['direction'] == 'up':
                new = self.increase_utility(real, syn, util_metric_req, util_val_req, result['amount'], error_range)

            ## Don't do anything if utility is above the requested value, higher utility than requested is ok
        
            if new is not None:
                return new
            else:
                counter = counter + 1
        
        return None
        
    
    def increase_privacy(self, real: DataLoader, syn: DataLoader, priv_metric_req: PrivacyKnowledge, priv_val_req: float, amount: float, error_range: float) -> DataLoader:
        print('Increase privacy')
        
        requested_level = self.current_generator.get_privacy_level().level

        ## Counter to avoid getting stuck when no improvements are made
        no_change = 0
        privacy_satisfied = False
        while not privacy_satisfied:
            rows_added = False
            ## Merge syn data from other privacy levels
            for level, new_generator in self.trained_generators.items():
                if level == requested_level:
                    continue

                ## Generate new synthetic data from other privacy level
                new_data = new_generator.generate(self.count)

                ## TODO encode new data with same encoder as real data
                # new_data = self.create_data_loader(self.encode(new_data.dataframe()))

                ## Current privacy of syn data
                cur_priv = self.privacy_calc.calculatePrivacy(real, syn)

                ## privacy of syn_data of different privacy level
                new_priv = self.privacy_calc.calculatePrivacy(real, new_data)
                
                ## Don't merge syn data with data from other level if the other data has lower privacy
                if new_priv < cur_priv:
                    continue

                ## Find rows of syn data with shortest distance from other rows in syn data
                distances = euclidean_distances(syn.dataframe(), syn.dataframe())
                ranked_rows = np.argsort(distances.mean(axis=1))

                ## Find rows of syn data from other level with largest distance from syn data
                distances2 = euclidean_distances(new_data.dataframe(), syn.dataframe())
                ranked_rows2 = np.flip(np.argsort(distances2.mean(axis=1)))
                selected_rows = new_data.dataframe().iloc[ranked_rows2[:int((self.count * amount))]]

                ## remove most similar rows from syn and add most dissimilar rows from new_data
                combined_data = syn.dataframe().drop(ranked_rows[:int((self.count * amount))])
                combined_data = pd.concat([combined_data, selected_rows], ignore_index=True)
                combined_data_loader = GenericDataLoader(combined_data)
                combined_priv = self.privacy_calc.calculatePrivacy(real, combined_data_loader)

                ## Check if the merge improved the privacy
                if combined_priv < cur_priv:
                    continue

                ## Set rows_added bool True
                rows_added = True

                ## Test if privacy function is satisfied
                priv_val = priv_metric_req.calculate(real, combined_data_loader)
                if priv_metric_req.satisfied(priv_val_req, priv_val, error_range):
                    return combined_data_loader
                
                ## Continue with next syn data from other privacy level
                syn = combined_data_loader
            
            ## Reset no_change counter
            if rows_added:
                no_change = 0
            else:
                no_change = no_change + 1
                if no_change >= 10:
                        return None
        
        ## Requested privacy can not be reached
        return None


    def decrease_privacy(self, real: DataLoader, syn: DataLoader, priv_metric_req: PrivacyKnowledge, priv_val_req: float, amount: float, error_range: float) -> DataLoader:
        print('Decrease privacy')
        
        requested_level = self.current_generator.get_privacy_level().level

        ## Counter to avoid getting stuck when no improvements are made
        no_change = 0
        privacy_satisfied = False
        while not privacy_satisfied:
            rows_added = False
            ## Merge syn data from other privacy levels
            for level, new_generator in self.trained_generators.items():
                if level == requested_level:
                    continue

                ## Generate new synthetic data from other privacy level
                new_data = new_generator.generate(self.count)

                ## TODO encode new data with same encoder as real data
                # new_data = self.create_data_loader(self.encode(new_data.dataframe()))

                ## Current privacy of syn data
                cur_priv = self.privacy_calc.calculatePrivacy(real, syn)

                ## privacy of syn_data of different privacy level
                new_priv = self.privacy_calc.calculatePrivacy(real, new_data)
                
                ## Don't merge syn data with data from other level if the other data has higher privacy
                if new_priv > cur_priv:
                    continue

                ## Find rows of syn data with furthest distance from other rows in syn data
                distances = euclidean_distances(syn.dataframe(), syn.dataframe())
                ranked_rows = np.flip(np.argsort(distances.mean(axis=1)))

                ## Find rows of syn data from other level with closest distance from syn data
                distances2 = euclidean_distances(new_data.dataframe(), syn.dataframe())
                ranked_rows2 = np.argsort(distances2.mean(axis=1))
                selected_rows = new_data.dataframe().iloc[ranked_rows2[:int((self.count * amount))]]

                ## remove most dissimilar rows from syn and add most similar rows from new_data
                combined_data = syn.dataframe().drop(ranked_rows[:int((self.count * amount))])
                combined_data = pd.concat([combined_data, selected_rows], ignore_index=True)
                combined_data_loader = GenericDataLoader(combined_data)
                combined_priv = self.privacy_calc.calculatePrivacy(real, combined_data_loader)

                ## Check if the merge improved the privacy
                if combined_priv > cur_priv:
                    continue

                ## Set rows_added bool True
                rows_added = True

                ## Test if privacy function is satisfied
                priv_val = priv_metric_req.calculate(real, syn)
                if priv_metric_req.satisfied(priv_val_req, priv_val, error_range):
                    return combined_data_loader
                
                ## Continue with next syn data from other privacy level
                syn_data_loader = combined_data_loader
            
            ## Reset no_change counter
            if rows_added:
                no_change = 0
            else:
                no_change = no_change + 1
                if no_change >= 10:
                        return None
        
        ## Requested privacy can not be reached
        return None

    def increase_utility(self, real: DataLoader, syn: DataLoader, util_metric_req: UtilityKnowledge, util_val_req: float, amount: float, error_range: float) -> DataLoader:
        print('Increase utility')
        
        requested_level = self.current_generator.get_privacy_level().level

        real_data = real.dataframe()
        syn_data = syn.dataframe()

        ## Counter to avoid getting stuck when no improvements are made
        no_change = 0

        freq_satisfied = False
        while not freq_satisfied:
            if no_change >= 10:
                break
            
            ## Test if utility function is satisfied before all frequencies are
            util_val = util_metric_req.calculate(real, syn)
            if util_metric_req.satisfied(util_val_req, util_val, error_range):
                return GenericDataLoader(syn_data)

            ## Set freq_satisfied True, if one column does not satisfy freq than it is set back to False
            freq_satisfied = True

            for column in real_data:
                ## merge datasets untill column frequencies are satisfied
                freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
                real_column_freqs = freqs[column][0]
                syn_column_freqs = freqs[column][1]
                
                ## Calculate bin size
                bin_size = list(real_column_freqs.keys())[1] - list(real_column_freqs.keys())[0]
                
                ## Check if freq is within the allowed error range
                if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
                    continue
                else:
                    freq_satisfied = False

                ## Use syn data of other privacy levels to merge with syn data of requested level
                for level, new_generator in self.trained_generators.items():
                    if level == requested_level:
                        continue

                    ## Continue with next column if this column's frequencies are satisfied
                    if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
                        break

                    ## Generate new synthetic data from other privacy level
                    new_data = new_generator.generate(count = 100).dataframe()

                    ## Encode new data with same encoder as real data
                    # new_data = self.create_data_loader(self.encode(new_data.dataframe()))

                    ## Merge syn data with new data until no suitable rows can be found in new data
                    row_found = True
                    while row_found:

                        ## Set row_found False, if one row is found for a bin than it is set back to True
                        row_found = False

                        for bin in real_column_freqs:
                            real_freq = real_column_freqs[bin]
                            syn_freq = syn_column_freqs[bin]

                            ## Values in current bin for column are underrepresented
                            if syn_freq < (real_freq - self.column_frequency_error):
                                ## Find row from syn data of other level with value for column in current bin
                                index = self.find_row(new_data, column, bin, bin + bin_size)
                                
                                ## No row with value in current bin for column
                                if index is None:
                                    continue

                                ## Remove row from syn to keep the requested amount of rows
                                remove_row_index = syn_data.sample().index
                                syn_data = syn_data.drop(remove_row_index)

                                ## Add new row
                                row = new_data.loc[index].to_dict()
                                new_data = new_data.drop(index)
                                syn_data = syn_data.append(row, ignore_index=True)
                                row_found = True

                                ## Reset no_change counter
                                no_change = 0

                            ## Values in current bin for column are overrepresented
                            elif syn_freq > (real_freq + self.column_frequency_error):
                                ## Find row with value for column in current bin
                                index = self.find_row(syn_data, column, bin, bin + bin_size)

                                ## No row with value in current bin for column
                                if index is None:
                                    continue

                                ## Remove row from syn data to keep the requested amount of rows
                                row = syn_data.loc[index].to_dict()
                                syn_data = syn_data.drop(index)

                                ## Add new row where value for column is not in current bin
                                row_added = False
                                while not row_added:
                                    row = self.current_generator.generate(count = 1).dataframe()
                                    if row.iloc[0][column] < bin or row.iloc[0][column] >= bin + bin_size:
                                        syn_data = syn_data.append(row, ignore_index=True)
                                        row_added = True
                                        row_found = True
                                
                                ## Reset no_change counter
                                no_change = 0

                        freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
                        real_column_freqs = freqs[column][0]
                        syn_column_freqs = freqs[column][1]
        
                ## Increase counter
                no_change = no_change +1

        new_data_loader = GenericDataLoader(syn_data)
        util_val = util_metric_req.calculate(real, new_data_loader)
        if util_metric_req.satisfied(util_val_req, util_val, error_range):
            return new_data_loader

        ## Utility can not be reached
        return None

    
    def column_frequencies(self, real: pd.DataFrame, syn: pd.DataFrame, n_histogram_bins: int = 10) -> dict:
        """Get percentual frequencies for each possible real categorical value.

        Returns:
            The observed and expected frequencies (as a percent).
        """
        res = {}
        for col in real.columns:
            local_bins = min(n_histogram_bins, len(real[col].unique()))

            if len(real[col].unique()) < 5:  # categorical
                gt = (real[col].value_counts() / len(real)).to_dict()
                synth = (syn[col].value_counts() / len(syn)).to_dict()
            else:
                gt_vals, bins = np.histogram(real[col], bins=local_bins)
                synth_vals, _ = np.histogram(syn[col], bins=bins)
                gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
                synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

            for val in gt:
                if val not in synth or synth[val] == 0:
                    synth[val] = 1e-11
            for val in synth:
                if val not in gt or gt[val] == 0:
                    gt[val] = 1e-11

            if gt.keys() != synth.keys():
                raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
            res[col] = (gt, synth)

        return res


    def validate_column_frequencies(self, real_column_freqs, syn_column_freqs):
        for bin in real_column_freqs:
            real_freq = real_column_freqs[bin]
            fake_freq = syn_column_freqs[bin]
            if fake_freq < (real_freq - self.column_frequency_error) or fake_freq > (real_freq + self.column_frequency_error):
                return False
        return True
            
                
    def find_row(self, data, column, bin_low, bin_high):
        for ind in data.index:
            if data[column][ind] >= bin_low and data[column][ind] < bin_high:
                return ind