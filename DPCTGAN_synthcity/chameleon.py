## Synthcity imports
# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics._utils import get_frequency

from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator,  InferenceEvaluator

from user import User
from privacyLevel import PrivacyLevels, RequirementLevels, EvaluationLevels
from knowledgeComponent import KnowledgeComponent
from dataRequest import DataRequest
from privacyLevelMetric import PrivacyLevelMetric, NearestNeighborDistance, CommonRows, kAnonymity, DataLeakage, DataMismatch, ReIdentification
from plugin import Plugin

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import copy
from sklearn.metrics.pairwise import euclidean_distances

import privacy_functions


class Chameleon():
    """
    Main class for the data chameleon.

    Constructor Args:
        
    """

    def __init__(self, privacy_level_metric = NearestNeighborDistance()):
        self.generators = {}
        self.syn_data = {}
        self.users = []
        self.synthetic_data_requests = []

        self.privacy_level_metric = privacy_level_metric
        self.knowledge_component = KnowledgeComponent(privacy_level_metric)

        self.syn_data_length = 1000
        self._n_histogram_bins = 10
        self.column_frequency_error = 0.2
        self.rows_add_priv_increase = 10
    
    def load_real_data(self, data, sensitive_features):
        self.original_data = data
        self.sensitive_features = sensitive_features
        self.aux_cols = list(data.sample(n=3,axis='columns').columns.values)
        self.train_data, self.control_data = train_test_split(data, test_size=0.2)
        self.loader = GenericDataLoader(self.train_data, sensitive_features=sensitive_features)
        return self.loader
    
    def create_data_loader(self, data):
        return GenericDataLoader(data, sensitive_features=self.sensitive_features)

    def update_real_data(self, data):
        # Get the new data
        new_data = data.merge(self.original_data.drop_duplicates(), how='left')
        # Perform Kolmogorov-Smirnov test per feature
        for feature in self.original_data:
            x = ks_2samp(self.original_data[feature], new_data[feature])
            # Check if pvalue is smaller than threshold (0.05 or 0.01), if so, the data is different and we should retrain our models
            threshold = 0.5
            if x.pvalue < threshold:
                print(feature)
                print(new_data[feature])
                print(x)
                self.original_data = data
                self.train_data, self.control_data = train_test_split(data, test_size=0.2)
                self.loader = GenericDataLoader(self.train_data, sensitive_features=self.sensitive_features)
                self.train_generators()
                return

    def add_user(self, name, owner, privacy_level = PrivacyLevels.SECRET):
        new_user = User(name, owner, privacy_level)
        self.users.append(new_user)
        
    def set_user_privacy_level(self, name, level):
        for user in self.users:
            if user.get_name() == name:
                user.set_privacy_level(level)
                return
        raise ValueError("There does not exist a user with this name")
    
    def get_user(self, name):
        for user in self.users:
            if user.get_name() == name:
                return user
    
    def get_privacy_level_metric(self):
        return self.privacy_level_metric
            
    def get_requirement_level(self, value):
        return self.knowledge_component.level_by_requirement(value)
    
    def get_evaluation_level(self, value):
        return self.knowledge_component.level_by_evaluation(value)

    def add_generator(self, generator):
        if isinstance(generator, Plugin):
            self.generators[generator.get_privacy_level().level] = generator
            self.syn_data[generator.get_privacy_level().level] = pd.DataFrame
        else:
            raise ValueError("generator must be an instance of the Plugin class")
        
    def train_generators(self):
        print(" ")
        ## Fit generators
        for level, generator in self.generators.items():
            generator.fit(self.loader)

        ## Generate synthetic data such that the data can be delivered faster on request
        # for level, generator in self.generators.items():
        #     self.syn_data[level] = self.generate(generator, self.syn_data_length)

    def update_utility(self, generator: Plugin):
        generator.update(self.loader)

    def generate_synthetic_data(self, user_name: str, requested_level: PrivacyLevels, count: int, privacy_func: any=None, utility_func: any=None):
        user = self.get_user(user_name)
        if user is None:
            raise RuntimeError("There is no user with this name.")
        

        ## Find the user's privacy level
        user_privacy_level = user.get_privacy_level()


        ## Check if user is allowed to request data from the specified privacy level (only allowed to request data more private than their own privacy level)
        if user_privacy_level.level > requested_level.level:
            raise RuntimeError("The privacy level of the user must be lower than the privacy level of the requested data.")


        ## Check if there is previously generated data for this privacy level if the user did not yet received data from this level
        # cwd = os.getcwd()
        # if not user.get_data_requested():
        #     if os.path.isdir(cwd + '/data'):
        #         # directory exists
        #         file = cwd + '/data/synthetic_' + requested_level.__str__() + '.csv'
        #         if os.path.exists(file):
        #             print('Using previously generated data.')
        #             synthetic_data = pd.read_csv(file)
        #             return synthetic_data
        #     else:
        #         # create directory
        #         os.makedirs(cwd + '/data')


        ## Find the appropriate generator based on the requested privacy level
        if requested_level.level in self.generators:
            suitable_generator = self.generators.get(requested_level.level)
        else:
            raise RuntimeError("No suitable generator found, first add generators for all privacy levels")
        

        print(" ")
        print("---Generating synthetic data for privacy level: " + str(requested_level) + "---")

        ## Generate synthetic data, fine tune generators untill required privacy level is reached
        # correct_privacy_level = False
        ## Keep track of used generators in the fine tuning process
        # used_generators = {}
        # while not correct_privacy_level:
            ## Add current generator to used_generators
            # used_generators[suitable_generator.get_privacy_level().level] = suitable_generator
            
            # Generate synthetic data
            # synthetic_data = self.generate(suitable_generator, count)
            
            # Check if the generated data fits the requested privacy level
            # syn_data_privacy_level = self.check_metrics_data(synthetic_data)
            # if  syn_data_privacy_level.level > requested_level.level:
            #     # Privacy level is too high
            #     suitable_generator = self.decrease_privacy(suitable_generator, used_generators, requested_level)
            # elif syn_data_privacy_level.level < requested_level.level:
            #     # Privacy level is too low
            #     suitable_generator = self.increase_privacy(suitable_generator, used_generators, requested_level)
            # else:
            #     correct_privacy_level = True
        
        ## Save synthetic dataset with privacy level
        # synthetic_data.dataframe().to_csv(cwd + '/data/synthetic_' + requested_level.__str__() + '.csv')
        # user.set_data_requested(True)

        ## Generate data
        synthetic_data = self.generate(suitable_generator, count)

        ## Fine tune the data if privacy and/or utility function are included
        if privacy_func is not None and utility_func is not None:
            synthetic_data = self.privacy_fine_tuning(synthetic_data, suitable_generator, count, privacy_func)
            synthetic_data = self.utility_fine_tuning(synthetic_data, suitable_generator, count, utility_func)
        elif privacy_func is not None:
            synthetic_data = self.privacy_fine_tuning(synthetic_data, suitable_generator, count, privacy_func)
        elif utility_func is not None:
            synthetic_data = self.utility_fine_tuning(synthetic_data, suitable_generator, count, utility_func)

        ## Generate synthetic data, fine tune generators untill required utility is reached if privacy and/or utility function are included
        # if privacy_func is not None and utility_func is not None:
        #     priv_satisfied = privacy_func(self.loader, synthetic_data)
        #     print(priv_satisfied)
        #     fine_tune = self.privacy_utility_tradeoff(synthetic_data, requested_level, count, privacy_func, utility_func)
        #     if fine_tune == None:
        #         raise RuntimeError("This combination of privacy and utility can not be achieved.")
        #     else:
        #         # TODO validated = self.validate_privacy_syn(fine_tune, requested_level)
        #         validated = True
        #     if validated:
        #         return fine_tune
        #     else:
        #         raise RuntimeError("This combination of privacy and utility can not be achieved for this privacy level.")

            # updated_generator = copy.deepcopy(suitable_generator)
            # privacy_satisfied = False
            # utility_satisfied = False
            # while not privacy_satisfied or not utility_satisfied:
            #     if not privacy_satisfied:
            #         print("---Testing privacy of synthetic data---")
            #         privacy = privacy_func(synthetic_data.dataframe())
            #         print("Privacy satisfied? " + str(privacy))
            #         if privacy:
            #             privacy_satisfied = True
            #         else:
            #             # TODO update privacy
            #             print("Current privacy budjet: " + str(updated_generator.get_privacy_budget()))
            #             # set utility satisfied false because we update the generator/synthetic data
            #             utility_satisfied = False
            #             synthetic_data = self.generate(updated_generator, count)
            #     elif not utility_satisfied:
            #         print("---Testing utility of synthetic data---")
            #         utility = utility_func(synthetic_data.dataframe())
            #         print("Utility satisfied? " + str(utility))
            #         if utility:
            #             utility_satisfied = True
            #         else:
            #             # TODO update utility
            #             self.update_utility(updated_generator)
            #             # TODO give warning if privacy level changed
            #             # privacy_budjet = updated_generator.get_privacy_budget()
            #             # if privacy_budjet >= requested_level.epsilon:
            #             #     raise RuntimeError("This utility can not be achieved for this privacy level.")

            #             # set privacy satisfied false because we update the generator/synthetic data
            #             privacy_satisfied = False
            #             synthetic_data = self.generate(updated_generator, count)

        # elif privacy_func is not None:
        #     fine_tune = self.privacy_fine_tuning(synthetic_data, requested_level, count, privacy_func, utility_func)
        #     if fine_tune == None:
        #         raise RuntimeError("This amount of privacy can not be achieved.")
        #     else:
        #         validated = self.validate_privacy_syn(fine_tune, requested_level)
        #     if validated:
        #         return fine_tune
        #     else:
        #         raise RuntimeError("This amount of privacy can not be achieved for this privacy level.")

            # Need to copy the baseline generator to avoid that we update it and lose the baseline model for future generations
            # updated_generator = copy.deepcopy(suitable_generator)
            # privacy_satisfied = False
            # while not privacy_satisfied:
            #     print("---Testing privacy of synthetic data---")
            #     privacy = privacy_func(synthetic_data.dataframe())
            #     print("Privacy satisfied? " + str(privacy))
            #     print("Current privacy budjet: " + str(updated_generator.get_privacy_budget()))
            #     if privacy:
            #         privacy_satisfied = True
            #     else:
            #         # TODO increase/update privacy
            #         synthetic_data = self.generate(updated_generator, count)

        # elif utility_func is not None:
        #     fine_tune = self.utility_fine_tuning(synthetic_data, requested_level, count, privacy_func, utility_func)
        #     if fine_tune == None:
        #         raise RuntimeError("This amount of utility can not be achieved.")
        #     else:
        #         validated = self.validate_privacy_syn(fine_tune, requested_level)
        #     if validated:
        #         return fine_tune
        #     else:
        #         raise RuntimeError("This amount of utility can not be achieved for this privacy level.")

            # Need to copy the baseline generator to avoid that we update it and lose the baseline model for future generations
            # updated_generator = copy.deepcopy(suitable_generator)
            # utility_satisfied = False
            # while not utility_satisfied:
            #     print("---Testing utility of synthetic data---")
            #     utility = utility_func(synthetic_data.dataframe())
            #     print("Utility satisfied? " + str(utility))
            #     print("Current privacy budjet: " + str(updated_generator.get_privacy_budget()))
            #     if utility:
            #         utility_satisfied = True
            #     else:
            #         # TODO increase/update utility
            #         self.update_utility(updated_generator)
            #         # TODO give warning if privacy level changed
            #         # privacy_budjet = updated_generator.get_privacy_budget()
            #         # if privacy_budjet >= requested_level.epsilon:
            #         #     raise RuntimeError("This utility can not be achieved for this privacy level.")
            #         synthetic_data = self.generate(updated_generator, count)
        
        # else:
            ## Ask user to confirm the synthetic data
            # final_data = self.confirm_synthetic_data(suitable_generator, requested_level, synthetic_data, count)
        
        print("---Releasing synthetic data---")

        # print(synthetic_data.dataframe())
        # print('encode')
        # syn = suitable_generator.get_encoder().transform(synthetic_data.dataframe())
        # print(pd.DataFrame(syn))
        # print('decode')
        # syn = suitable_generator.get_encoder().inverse_transform(synthetic_data.dataframe())
        # print(pd.DataFrame(syn))
        # syn, encoders = synthetic_data.encode()
        # print(syn)
        # print('decode')
        # syn = syn.decode(encoders)
        # print(syn)

        # new_gen_privacy_level = requested_level.level + 1
        # new_generator = self.generators.get(new_gen_privacy_level)
        # synthetic_data2 = self.generate(new_generator, count)

        # print(synthetic_data)
        # print(synthetic_data2)
        # syn, encoders = synthetic_data.encode()
        # syn2, encoders2 = synthetic_data2.encode()

        # syn3 = 0.5 * syn.dataframe() + 0.5 * syn2.dataframe()
        # syn = GenericDataLoader(syn3, sensitive_features=self.sensitive_features).decode(encoders)
        # print(syn)

        return synthetic_data.dataframe()
    
    def generate(self, generator, count):
        try:
            synthetic_data = generator.generate(count = count)
        except RuntimeError as e:
            if e.message == 'Fit the generator first':
                generator.fit(self.loader)
                synthetic_data = generator.generate(count = count)
            else:
                raise RuntimeError("Something went wrong, try adding the generators again")
        return synthetic_data
    
    def privacy_utility_tradeoff(self, synthetic_data, requested_level: PrivacyLevels, count: int, privacy_func: any=None, utility_func: any=None):
        ...

    def privacy_fine_tuning(self, synthetic_data, generator, count: int, privacy_func: any=None):
        ## Test if privacy function is satisfied
        if privacy_func(self.loader, synthetic_data):
            return synthetic_data
        
        fine_tuned = self.increase_privacy(synthetic_data, generator, privacy_func)
        
        if fine_tuned is not None:
            return fine_tuned
        else:
            return None

    def utility_fine_tuning(self, synthetic_data, generator, count: int, utility_func: any=None):
        ## Test if utility function is satisfied
        if utility_func(self.loader, synthetic_data):
            return synthetic_data
        
        fine_tuned = self.increase_utility(synthetic_data, generator, utility_func)

        if fine_tuned is not None:
            return fine_tuned
        else:
            return None
    
    def increase_privacy(self, syn_data_loader, generator, privacy_func):
        requested_level = generator.get_privacy_level().level
        print(requested_level)

        ## Merge syn data from other privacy levels
        for level, data in self.syn_data.items():
            if level == requested_level:
                continue

            ## Current nearest neighbor distance between rows in syn_data and real data
            mean_dist = self.privacy_level_metric.evaluate(self.loader, syn_data_loader)
            print(mean_dist)

            ## nearest neighbor distance between syn_data of different privacy level and real data
            mean_dist_2 = self.privacy_level_metric.evaluate(self.loader, data)
            print(mean_dist_2)
            
            ## Don't merge syn data with data from other level if the other data is closer to the real data than the current syn data itself
            if mean_dist_2 < mean_dist:
                continue

            ## Find rows of syn data with shortest distance from other rows in syn data
            distances = euclidean_distances(syn_data_loader.dataframe(), syn_data_loader.dataframe())
            ranked_rows = np.argsort(distances.mean(axis=1))

            ## Find rows of syn data from other level with largest distance from syn data
            distances2 = euclidean_distances(data.dataframe(), syn_data_loader.dataframe())
            ranked_rows2 = np.flip(np.argsort(distances2.mean(axis=1)))
            selected_rows = data.dataframe().iloc[ranked_rows2[:100]]

            ## remove most similar rows from syn_data and add most dissimilar rows from syn_data2
            combined_data = syn_data_loader.dataframe().drop(ranked_rows[:100])
            combined_data = pd.concat([combined_data, selected_rows], ignore_index=True)
            combined_data_loader = self.create_data_loader(combined_data)
            combined_dist = self.privacy_level_metric.evaluate(self.loader, combined_data_loader)

            ## Check if the merge improved the privacy
            if combined_dist < mean_dist:
                continue

            ## Test if privacy function is satisfied
            if privacy_func(self.loader, combined_data_loader):
                return combined_data_loader
            
            ## Continue with next syn data from other privacy level
            syn_data_loader = combined_data_loader
        
        ## Generate new rows to add to the syn data
        ## Counter to avoid getting stuck when no improvements are made
        no_change = 0
        privacy_satisfied = False
        while not privacy_satisfied:
            for level, new_generator in self.generators.items():
                if level < requested_level:
                    continue

                ## Current nearest neighbor distance between rows in syn_data and real data
                mean_dist = self.privacy_level_metric.evaluate(self.loader, syn_data_loader)
                print(mean_dist)

                new_data = syn_data_loader.dataframe()
                rows = self.generate(new_generator, 100).dataframe()
                dist = euclidean_distances(rows, syn_data_loader.dataframe()).mean(axis=1)
                indices = np.where(dist < euclidean_distances(syn_data_loader.dataframe(), syn_data_loader.dataframe()).mean(axis=1).mean())[0]
                selected_rows = rows.loc[indices]

                ## Remove row from syn to keep the requested amount of rows
                remove_rows_index = new_data.sample(selected_rows.shape[0]).index
                new_data = new_data.drop(remove_rows_index)

                ## Add selected rows to increase the average distance
                new_data = new_data.append(selected_rows, ignore_index=True)

                new_data_loader = self.create_data_loader(new_data)
                new_dist = self.privacy_level_metric.evaluate(self.loader, new_data_loader)
                print(new_dist)

                ## Check if the merge improved the privacy
                if new_dist <= mean_dist:
                    ## Privacy can not be reached
                    if no_change >= 10:
                        return None
                    no_change = no_change + 1
                    continue

                ## Test if privacy function is satisfied
                if privacy_func(self.loader, new_data_loader):
                    return new_data_loader
                
                ## Reset no_change counter
                no_change = 0
                
                ## Continue to add more rows
                syn_data_loader = new_data_loader
        
        ## Requested privacy can not be reached
        return None

    def decrease_privacy(self, generator, used_generators, requested_level):
        ...
    
    def increase_utility(self, syn_data_loader, generator, utility_func):
        requested_level = generator.get_privacy_level().level
        print(requested_level)

        real_data = self.loader.dataframe()
        syn_data = syn_data_loader.dataframe()

        ## Counter to avoid getting stuck when no improvements are made
        no_change = 0

        freq_satisfied = False
        while not freq_satisfied:
            if no_change >= 10:
                break
            
            ## Test if utility function is satisfied before all frequencies are
            if utility_func(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)):
                return GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)

            ## Set freq_satisfied True, if one column does not satisfy freq than it is set back to False
            freq_satisfied = True

            for column in real_data:
                print(column)

                ## merge datasets untill column frequencies are satisfied
                freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
                real_column_freqs = freqs[column][0]
                # print(real_column_freqs)
                syn_column_freqs = freqs[column][1]
                # print(syn_column_freqs)
                
                ## Calculate bin size
                bin_size = list(real_column_freqs.keys())[1] - list(real_column_freqs.keys())[0]
                
                ## Check if freq is within the allowed error range
                if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
                    continue
                else:
                    freq_satisfied = False

                ## Use syn data of other privacy levels to merge with syn data of requested level
                for level, new_generator in self.generators.items():
                    if level == requested_level:
                        continue

                    ## Continue with next column if this column's frequencies are satisfied
                    if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
                        break

                    ## Generate new synthetic data from other privacy level
                    new_data = self.generate(new_generator, 100).dataframe()

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
                                    row = self.generate(generator, 1).dataframe()
                                    if row.iloc[0][column] < bin or row.iloc[0][column] >= bin + bin_size:
                                        syn_data = syn_data.append(row, ignore_index=True)
                                        row_added = True
                                        row_found = True
                                
                                ## Reset no_change counter
                                no_change = 0

                        freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
                        real_column_freqs = freqs[column][0]
                        # print(real_column_freqs)
                        syn_column_freqs = freqs[column][1]
                        # print(syn_column_freqs)

                        ## Calculate mean difference
                        sum = 0
                        counter = 0
                        for bin in real_column_freqs:
                            real_freq = real_column_freqs[bin]
                            fake_freq = syn_column_freqs[bin]
                            sum = sum + abs(real_freq - syn_freq)
                            counter = counter + 1
                        avg = sum / counter
                        print('avg: ' + str(avg))
        
                ## Increase counter
                no_change = no_change +1

        new_data_loader = GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)
        if utility_func(self.loader, new_data_loader):
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
            
    
    def change_column_frequencies(self, real, syn, requested_level):
        real_data = real.dataframe()
        syn_data = syn.dataframe()
        generator = self.generators[requested_level.level]

        freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)

        for column in real_data:
            print(column)

            ## merge datasets untill column frequencies are satisfied
            freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
            real_column_freqs = freqs[column][0]
            # print(real_column_freqs)
            syn_column_freqs = freqs[column][1]
            # print(syn_column_freqs)

            # if not self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
                ## Change current column
            
            ## Calculate bin size
            bin_size = list(real_column_freqs.keys())[1] - list(real_column_freqs.keys())[0]
            
            ## Check if freq is within the allowed error range
            freq_satisfied = self.validate_column_frequencies(real_column_freqs, syn_column_freqs)

            ## Use syn data of other privacy levels to merge with syn data of requested level
            for level, data in self.syn_data.items():
                if level == requested_level:
                    continue
                
                ## Copy the data to local variable such that the original syn data is not changed
                data = copy.deepcopy(data)

                no_change = 0
                while not freq_satisfied:
                    if no_change >= 10:
                        break
                    for bin in real_column_freqs:
                        real_freq = real_column_freqs[bin]
                        syn_freq = syn_column_freqs[bin]
                        if syn_freq < (real_freq - self.column_frequency_error):
                            ## Find row from syn data of other level with value for column in current bin
                            index = self.find_row(data, column, bin, bin + bin_size)
                            if index is None:
                                no_change = no_change + 1
                                continue

                            ## Remove row from syn to keep the requested amount of rows
                            remove_row_index = syn_data.sample().index
                            syn_data = syn_data.drop(remove_row_index)

                            ## Add new row
                            row = data.loc[index].to_dict()
                            data = data.drop(index)
                            syn_data = syn_data.append(row, ignore_index=True)

                        if syn_freq > (real_freq + self.column_frequency_error):
                            # Find row with value for column in current bin
                            index = self.find_row(syn_data, column, bin, bin + bin_size)
                            if index is None:
                                no_change = no_change + 1
                                continue

                            ## Remove row
                            row = syn_data.loc[index].to_dict()
                            syn_data = syn_data.drop(index)

                            ## Add new row where value for column is not in current bin
                            row_added = False
                            while not row_added:
                                row = self.generate(generator, 1).dataframe()
                                if row.iloc[0][column] < bin or row.iloc[0][column] >= bin + bin_size:
                                    syn_data = syn_data.append(row, ignore_index=True)
                                    row_added = True

                    freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
                    real_column_freqs = freqs[column][0]
                    # print(real_column_freqs)
                    syn_column_freqs = freqs[column][1]
                    # print(syn_column_freqs)
                    freq_satisfied = self.validate_column_frequencies(real_column_freqs, syn_column_freqs)
        
        return GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)
                
    def find_row(self, data, column, bin_low, bin_high):
        for ind in data.index:
            if data[column][ind] >= bin_low and data[column][ind] < bin_high:
                return ind
           









    def confirm_synthetic_data(self, generator, level, syn_data, count):
        print(" ")
        print("This is the evaluation report of the current synthetic data:")
        confirmed = False
        # updated_generator = copy.deepcopy(generator)
        updated_generator = generator
        while not confirmed:
            ## TODO print privacy/utility summary of the synthetic data
            #self.evaluation_report(syn_data_gen.dataframe())
            print("---Privacy utility report---")
            print(self.privacy_level_metric.name() + ": " + str(self.privacy_level_metric.evaluate(self.loader, syn_data)))

            ## Ask user to confirm the synthetic data
            confirm_reply = input('Are you satisfied with the utility of the synthetic data (yes/no)? ')
            if confirm_reply == 'yes':
                return syn_data.dataframe()
            elif confirm_reply == 'no':
                ## Ask user if privacy needs to be increased or decreased
                # update = False
                # while not update:
                #     update_reply = input('Do you want to increase or decrease the privacy (up/down)? ')
                #     if update_reply == 'up':
                #         ## TODO increase privacy
                #         syn_data_gen = syn_data_gen
                #     elif update_reply == "down":
                #         ## TODO decrease privacy
                #         syn_data_gen = syn_data_gen
                #     else:
                #         print('Please reply with up (to increase the privacy and decrease the utility) or down (to decrease the privacy and increase the utility).')
                self.update_utility(updated_generator)
                syn_data = self.generate(updated_generator, count)
                privacy_budjet = updated_generator.get_privacy_budget()
                # TODO
                # if privacy_budjet >= level.epsilon:
                #     raise RuntimeError("This utility can not be achieved for this privacy level.")
            else:
                print('Please reply with yes or no.')

    def validate_privacy_syn(self, synthetic_data, requested_level: PrivacyLevels):
        syn_priv_level = self.check_metrics_data(synthetic_data)
        if syn_priv_level >= requested_level:
            return True
        else:
            return False
        
    def check_metrics_data(self, syn_data):
        syn_data_loader = GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)
        metric_value = self.privacy_level_metric.evaluate(self.loader, syn_data_loader)
        print(metric_value)
        return self.get_evaluation_level(metric_value)

    def evaluation_report(self, syn):
        ## Singling out attack
        # evaluator = SinglingOutEvaluator(ori=self.original_data, syn=syn, control=self.control_data)
        # evaluator.evaluate()
        # print('Singling out attack: ' + evaluator.risk())
        ## Linkability attack
        evaluator = LinkabilityEvaluator(ori=self.original_data, syn=syn, control=self.control_data, aux_cols=self.aux_cols)
        evaluator.evaluate()
        print('Linkability attack: ' + str(evaluator.risk()))
        # ## Inference attack
        # evaluator = InferenceEvaluator(ori=self.original_data, syn=syn, control=self.control_data, aux_cols=self.aux_cols, secret=self.sensitive_features)
        # evaluator.evaluate()
        # print('Inference attack: ' + evaluator.risk())
    
    def add_syn_data_request(self, request):
        if isinstance(request, DataRequest):
            self.synthetic_data_requests.append(request)
        else:
            raise ValueError("request must be an instance of the DataRequest class")