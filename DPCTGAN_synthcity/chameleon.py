## Synthcity imports
# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.metrics._utils import get_frequency
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.metrics.eval_statistical import InverseKLDivergence
from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance

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
from sklearn.metrics import jaccard_score
from importlib import import_module

import privacy_functions

from configurationParser import ConfigHandler
import privacyKnowledge
import utilityKnowledge
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
import fineTuningMethod


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

        self.configparser = ConfigHandler()
        self.privacyCalculator = PrivacyCalculator()
        self.utilityCalculator = UtilityCalculator()

        self.encoder = None

        self.syn_data_length = 1000
        self._n_histogram_bins = 10
        self.rows_add_priv_increase = 10

        self.column_frequency_error = 0.2

        self.handleConfigs()
        self.handleSynDataRequirements()
        self.handleMetrics()
        self.handleFineTuningMethod()

    def handleConfigs(self):
        self.configparser.parseRequirements()
        self.configparser.parseFineTuningMetrics()
        self.configparser.parseFineTuningMethod()

    def handleSynDataRequirements(self):
        ## Get privacy level
        level = int(self.configparser.getPrivacyLevel())
        self.privacyLevel = self.getPrivacyLevel(level)
        
        ## Get privacy metric used to define the requirements of syn data and instantiate the right class
        priv_req = self.configparser.getPrivacyMetricRequirement()
        if priv_req == 'none':
            self.privacyMetricRequirement = None
            self.privacyValueRequirement = None
        else:
            self.privacyMetricRequirement = getattr(privacyKnowledge, privacyKnowledge.CLASS_NAME_FILE[priv_req])()
            ## Get the required value of privacy metric of the syn data
            self.privacyValueRequirement = float(self.configparser.getPrivacyValueRequirement())
        
        ## Get utility metric used to define the requirements of syn data and instantiate the right class
        util_req = self.configparser.getUtilityMetricRequirement()
        if util_req == 'none':
            self.utilityMetricRequirement = None
            self.utilityValueRequirement = None
        else:
            self.utilityMetricRequirement = getattr(utilityKnowledge, utilityKnowledge.CLASS_NAME_FILE[util_req])()
            ## Get the required value of utility metric of the syn data
            self.utilityValueRequirement = float(self.configparser.getUtilityValueRequirement())
        
        ## Get the allow error range around the required privacy and utility values
        ange = self.configparser.getRange()
        if range == 'none':
            self.requirementRrange = None
        else:
            self.requirementRrange = float(range)
    
    def handleMetrics(self):
        ## Get privacy metrics with weights
        privacyMetrics = self.configparser.getPrivacyMetrics()
        self.privacyCalculator.setMetrics(privacyMetrics)
        ## Get utility metrics with weights
        utilityMetrics = self.configparser.getUtilityMetrics()
        self.utilityCalculator.setMetrics(utilityMetrics)
    
    def handleFineTuningMethod(self):
        ## Get the fine tuning class
        module = import_module(self.configparser.getFineTuningModule())
        className = self.configparser.getFineTuningClass()
        if module == 'none':
            self.fineTuningModule = None
        else:
            self.fineTuningModule = module
            self.fineTuningInstance = getattr(module, className)(self.generators, self.privacyCalculator, self.utilityCalculator)


    def getPrivacyLevel(self, level: int) -> PrivacyLevels:
        if level == 1:
            return PrivacyLevels.LOW
        elif level == 2:
            return PrivacyLevels.MEDIUM
        elif level == 3:
            return PrivacyLevels.HIGH
        else:
            return PrivacyLevels.SECRET
    
    def load_real_data(self, data, sensitive_features):
        self.original_data = data
        self.sensitive_features = sensitive_features
        self.aux_cols = list(data.sample(n=3,axis='columns').columns.values)
        self.train_data, self.control_data = train_test_split(data, test_size=0.2)

        ## Create dataloader
        self.loader = GenericDataLoader(self.train_data, sensitive_features=sensitive_features)

        ## Create data encoder and encode real data
        # self.encoder = TabularEncoder(categorical_limit=30).fit(self.train_data)
        # self.real_encoded = self.encode(self.loader.dataframe())
        # self.real_encoded_loader = GenericDataLoader(self.real_encoded, sensitive_features=sensitive_features)

        # print(self.loader.dataframe())
        # print(self.real_encoded)

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


    def generate_synthetic_data(self, user_name: str, count: int):
        user = self.get_user(user_name)
        if user is None:
            raise RuntimeError("There is no user with this name.")
        
        ## Find the user's privacy level
        user_privacy_level = user.get_privacy_level()

        ## Check if user is allowed to request data from the specified privacy level (only allowed to request data more private than their own privacy level)
        if user_privacy_level.level > self.privacyLevel.level:
            raise RuntimeError("The privacy level of the user must be lower than the privacy level of the requested data.")

        ## Find the appropriate generator based on the requested privacy level
        if self.privacyLevel.level in self.generators:
            suitable_generator = self.generators.get(self.privacyLevel.level)
        else:
            raise RuntimeError("No suitable generator found, first add generators for all privacy levels")
        
        print(" ")
        print("---Generating synthetic data for privacy level: " + str(self.privacyLevel) + "---")

        ## Generate data
        synthetic_data = self.generate(suitable_generator, count)
        
        ## Delete null rows
        # null_rows = synthetic_data.dataframe().isnull().any(axis=1)
        # synthetic_data = self.create_data_loader(synthetic_data.dataframe().drop(np.where(null_rows)[0]))


        ## Encode syn data with same encoder used for real data
        # syn_encoded = self.encode(synthetic_data.dataframe())
        # syn_encoded = self.create_data_loader(syn_encoded)


        ## If no specific privacy and/or utility requirements are included, release synthetic data after confirmation of user
        if self.privacyMetricRequirement is None and self.utilityMetricRequirement is None:
            synthetic_data = self.confirm_synthetic_data(synthetic_data, suitable_generator, self.privacyLevel, count)
        ## Fine tune the data if specific privacy and/or utility requirements are included
        else:
            if self.fineTuningModule is None:
                raise RuntimeError("No fine tuning method specified, please specify a method in the config file.")
            else:
                synthetic_data = self.fineTuningInstance.fine_tune(suitable_generator, self.loader, synthetic_data, count, self.privacyMetricRequirement, self.privacyValueRequirement, self.utilityMetricRequirement, self.utilityValueRequirement, self.requirementRrange)
        
        if synthetic_data == None:
            print('The privacy and/or utility can not be reached. Try retraining the generators, adding a new generator or adapting the privacy or utility requirements.')
            return None
        
        ## Decode syn data
        # synthetic_data = self.decode(syn_encoded.dataframe())
        
        print("---Releasing synthetic data---")
        return synthetic_data


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

    
    def confirm_synthetic_data(self, syn_data, generator, level, count):
        print(" ")
        print("This is the evaluation report of the current synthetic data:")

        confirmed = False
        while not confirmed:
            ## TODO print privacy/utility summary of the synthetic data
            #self.evaluation_report(syn_data_gen.dataframe())
            print("---Privacy utility report---")
            print(syn_data.dataframe())
            print(self.privacy_level_metric.name() + ": " + str(self.privacy_level_metric.evaluate(self.loader, syn_data)))

            ## Ask user to confirm privacy of the synthetic data
            confirm_privacy_reply = input('Are you satisfied with the privacy of the synthetic data (yes/no)? ')
            if confirm_privacy_reply == 'yes':
                ## Ask user to confirm privacy of the synthetic data
                confirm_utility_reply = input('Are you satisfied with the utility of the synthetic data (yes/no)? ')
                if confirm_utility_reply == 'yes':
                    confirmed = True
                elif confirm_utility_reply == 'no':
                    updated_data = self.increase_utility_stepwise(syn_data, generator)
                else:
                    print('Please reply with yes or no.')
            elif confirm_privacy_reply == 'no':
                updated_data = self.increase_privacy_stepwise(syn_data, generator, count)
            else:
                print('Please reply with yes or no.')
            
            if updated_data is not None:
                syn_data = updated_data
            else:
                print('This is the highest possible privacy and utility.')
                return syn_data

        return syn_data
    

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
    

    
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        df_encoded = pd.get_dummies(X, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'])

        print(df_encoded)
        return df_encoded
        # return self.encoder.transform(X)


    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)


    def get_encoder(self) -> TabularEncoder:
        return self.encoder
    
    

    # def increase_privacy_stepwise(self, syn_data_loader, generator, count):
    #     requested_level = generator.get_privacy_level().level

    #     ## Initial privacy metric value
    #     evaluator = NearestNeighborDistance()
    #     initial_privacy_score = evaluator.evaluate(self.loader, syn_data_loader)

    #     ## Counter to avoid getting stuck when no improvements are made
    #     no_change = 0
    #     privacy_satisfied = False
    #     while not privacy_satisfied:
    #         rows_added = False
    #         ## Merge syn data from other privacy levels
    #         for level, new_generator in self.generators.items():
    #             if level == requested_level:
    #                 continue

    #             ## Generate new synthetic data from other privacy level
    #             new_data = self.generate(new_generator, count)

    #             ## encode new data with same encoder as real data
    #             # new_data = self.create_data_loader(self.encode(new_data.dataframe()))

    #             ## Current nearest neighbor distance between rows in syn_data and real data
    #             mean_dist = self.privacy_level_metric.evaluate(self.loader, syn_data_loader)

    #             ## nearest neighbor distance between syn_data of different privacy level and real data
    #             mean_dist_2 = self.privacy_level_metric.evaluate(self.loader, new_data)
                
    #             ## Don't merge syn data with data from other level if the other data is closer to the real data than the current syn data itself
    #             if mean_dist_2 < mean_dist:
    #                 continue

    #             ## Find rows of syn data with shortest distance from other rows in syn data
    #             distances = euclidean_distances(syn_data_loader.dataframe(), syn_data_loader.dataframe())
    #             ranked_rows = np.argsort(distances.mean(axis=1))

    #             ## Find rows of syn data from other level with largest distance from syn data
    #             distances2 = euclidean_distances(new_data.dataframe(), syn_data_loader.dataframe())
    #             ranked_rows2 = np.flip(np.argsort(distances2.mean(axis=1)))
    #             selected_rows = new_data.dataframe().iloc[ranked_rows2[:int((count/10))]]

    #             ## remove most similar rows from syn_data and add most dissimilar rows from syn_data2
    #             combined_data = syn_data_loader.dataframe().drop(ranked_rows[:int((count/10))])
    #             combined_data = pd.concat([combined_data, selected_rows], ignore_index=True)
    #             combined_data_loader = self.create_data_loader(combined_data)
    #             combined_dist = self.privacy_level_metric.evaluate(self.loader, combined_data_loader)

    #             ## Check if the merge improved the privacy
    #             if combined_dist < mean_dist:
    #                 continue

    #             ## Set rows_added bool True
    #             rows_added = True

    #             ## Test if utility increased enough
    #             updated_privacy_score = evaluator.evaluate(self.loader, combined_data_loader)
    #             if updated_privacy_score >= (initial_privacy_score + 0.05):
    #                 return combined_data_loader
                
    #             ## Continue with next syn data from other privacy level
    #             syn_data_loader = combined_data_loader
            
    #         ## Reset no_change counter
    #         if rows_added:
    #             no_change = 0
    #         else:
    #             no_change = no_change + 1
    #             if no_change >= 10:
    #                     return None
        
    #     ## Requested privacy can not be reached
    #     return None
    

    # def increase_utility_stepwise(self, syn_data_loader, generator):
    #     requested_level = generator.get_privacy_level().level

    #     real_data = self.loader.dataframe()
    #     syn_data = syn_data_loader.dataframe()

    #     ## Initial utility metric value
    #     evaluator = InverseKLDivergence()
    #     initial_utility_score = evaluator.evaluate(self.loader, syn_data_loader).get('marginal')
    #     print(initial_utility_score)

    #     ## Counter to avoid getting stuck when no improvements are made
    #     no_change = 0

    #     freq_satisfied = False
    #     while not freq_satisfied:
    #         if no_change >= 10:
    #             break
            
    #         ## Test if utility increased enough
    #         updated_utility_score = evaluator.evaluate(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)).get('marginal')
    #         if updated_utility_score >= (initial_utility_score + 0.01):
    #             return GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)

    #         ## Set freq_satisfied True, if one column does not satisfy freq than it is set back to False
    #         freq_satisfied = True

    #         for column in real_data:
    #             ## merge datasets untill column frequencies are satisfied
    #             freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
    #             real_column_freqs = freqs[column][0]
    #             syn_column_freqs = freqs[column][1]
                
    #             ## Calculate bin size
    #             bin_size = list(real_column_freqs.keys())[1] - list(real_column_freqs.keys())[0]
                
    #             ## Check if freq is within the allowed error range
    #             if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
    #                 continue
    #             else:
    #                 freq_satisfied = False

    #             ## Use syn data of other privacy levels to merge with syn data of requested level
    #             for level, new_generator in self.generators.items():
    #                 if level == requested_level:
    #                     continue

    #                 ## Continue with next column if this column's frequencies are satisfied
    #                 if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
    #                     break

    #                 ## Generate new synthetic data from other privacy level
    #                 new_data = self.generate(new_generator, 100).dataframe()

    #                 ## Encode new data with same encoder as real data
    #                 # new_data = self.create_data_loader(self.encode(new_data.dataframe()))

    #                 ## Merge syn data with new data until no suitable rows can be found in new data
    #                 row_found = True
    #                 while row_found:

    #                     ## Set row_found False, if one row is found for a bin than it is set back to True
    #                     row_found = False

    #                     for bin in real_column_freqs:
    #                         real_freq = real_column_freqs[bin]
    #                         syn_freq = syn_column_freqs[bin]

    #                         ## Values in current bin for column are underrepresented
    #                         if syn_freq < (real_freq - self.column_frequency_error):
    #                             ## Find row from syn data of other level with value for column in current bin
    #                             index = self.find_row(new_data, column, bin, bin + bin_size)
                                
    #                             ## No row with value in current bin for column
    #                             if index is None:
    #                                 continue

    #                             ## Remove row from syn to keep the requested amount of rows
    #                             remove_row_index = syn_data.sample().index
    #                             syn_data = syn_data.drop(remove_row_index)

    #                             ## Add new row
    #                             row = new_data.loc[index].to_dict()
    #                             new_data = new_data.drop(index)
    #                             syn_data = syn_data.append(row, ignore_index=True)
    #                             row_found = True

    #                             ## Reset no_change counter
    #                             no_change = 0

    #                         ## Values in current bin for column are overrepresented
    #                         elif syn_freq > (real_freq + self.column_frequency_error):
    #                             ## Find row with value for column in current bin
    #                             index = self.find_row(syn_data, column, bin, bin + bin_size)

    #                             ## No row with value in current bin for column
    #                             if index is None:
    #                                 continue

    #                             ## Remove row from syn data to keep the requested amount of rows
    #                             row = syn_data.loc[index].to_dict()
    #                             syn_data = syn_data.drop(index)

    #                             ## Add new row where value for column is not in current bin
    #                             row_added = False
    #                             while not row_added:
    #                                 row = self.generate(generator, 1).dataframe()
    #                                 if row.iloc[0][column] < bin or row.iloc[0][column] >= bin + bin_size:
    #                                     syn_data = syn_data.append(row, ignore_index=True)
    #                                     row_added = True
    #                                     row_found = True
                                
    #                             ## Reset no_change counter
    #                             no_change = 0

    #                     freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
    #                     real_column_freqs = freqs[column][0]
    #                     syn_column_freqs = freqs[column][1]
        
    #             ## Increase counter
    #             no_change = no_change +1

    #     ## Test if utility increased enough
    #     updated_utility_score = evaluator.evaluate(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)).get('marginal')
    #     if updated_utility_score >= (initial_utility_score + 0.01):
    #         return GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)

    #     ## Utility can not be reached
    #     return None



    # def increase_privacy_utility(self, syn_data_loader, generator, count, privacy_func, utility_func):
    #     requested_level = generator.get_privacy_level().level
    #     print(requested_level)

    #     real_data = self.loader.dataframe()
    #     syn_data = syn_data_loader.dataframe()

    #     ## Counter to avoid getting stuck when no improvements are made
    #     no_change = 0

    #     freq_satisfied = False
    #     while not freq_satisfied:
    #         if no_change >= 10:
    #             break
            
    #         ## Test if privacy and utility function is satisfied before all frequencies are
    #         if privacy_func(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)) and utility_func(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)):
    #             return GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)

    #         ## Set freq_satisfied True, if one column does not satisfy freq than it is set back to False
    #         freq_satisfied = True

    #         ## Current nearest neighbor distance between rows in syn_data and real data
    #         mean_dist = euclidean_distances(syn_data, syn_data).mean(axis=1).mean()
    #         # mean_dist = self.privacy_level_metric.evaluate(self.loader, GenericDataLoader(syn_data, sensitive_features=self.sensitive_features))

    #         for column in real_data:
    #             print(column)

    #             ## merge datasets untill column frequencies are satisfied
    #             freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
    #             real_column_freqs = freqs[column][0]
    #             syn_column_freqs = freqs[column][1]
                
    #             ## Calculate bin size
    #             bin_size = list(real_column_freqs.keys())[1] - list(real_column_freqs.keys())[0]
                
    #             ## Check if freq is within the allowed error range
    #             if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
    #                 continue
    #             else:
    #                 freq_satisfied = False

    #             ## Use syn data of other privacy levels to merge with syn data of requested level
    #             for level, new_generator in self.generators.items():
    #                 if level == requested_level:
    #                     continue

    #                 ## Continue with next column if this column's frequencies are satisfied
    #                 if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
    #                     break

    #                 ## Generate new synthetic data from other privacy level
    #                 new_data = self.generate(new_generator, 100).dataframe()

    #                 ## Merge syn data with new data until no suitable rows can be found in new data
    #                 row_found = 0
    #                 while row_found < 10:

    #                     ## Continue with next column if this column's frequencies are satisfied
    #                     if self.validate_column_frequencies(real_column_freqs, syn_column_freqs):
    #                         break

    #                     for bin in real_column_freqs:
    #                         real_freq = real_column_freqs[bin]
    #                         syn_freq = syn_column_freqs[bin]

    #                         ## Values in current bin for column are underrepresented
    #                         if syn_freq < (real_freq - self.column_frequency_error):
    #                             suitable_row = False
                                
    #                             while not suitable_row:
    #                                 ## Find row from syn data of other level with value for column in current bin
    #                                 index = self.find_row(new_data, column, bin, bin + bin_size)

    #                                 ## No row with value in current bin for column
    #                                 if index is None:
    #                                     row_found = row_found + 1
    #                                     break
                                    
    #                                 suitable_row = True
    #                                 row_found = 0

    #                                 row = new_data.iloc[index]
    #                                 # row = syn_data.loc[index].to_dict()
    #                                 new_data = new_data.drop(index)

    #                                 ## nearest neighbor distance between syn_data of different privacy level and real data
    #                                 mean_dist_2 = euclidean_distances(row.to_frame().transpose(), syn_data).mean(axis=1)
    #                                 # mean_dist_2 = self.privacy_level_metric.evaluate(self.loader, GenericDataLoader(row.to_frame().transpose(), sensitive_features=self.sensitive_features))
                                    
    #                                 ## Don't merge syn data with data from other level if the other data is closer to the real data than the current syn data itself, avoid decreasing privacy
    #                                 if mean_dist_2 < mean_dist:
    #                                     continue

    #                                 ## Remove row from syn to keep the requested amount of rows
    #                                 remove_row_index = syn_data.sample().index
    #                                 syn_data = syn_data.drop(remove_row_index)

    #                                 ## Add new row
    #                                 syn_data = syn_data.append(row, ignore_index=True)
    #                                 row_found = True

    #                                 ## Reset no_change counter
    #                                 no_change = 0

    #                         ## Values in current bin for column are overrepresented
    #                         elif syn_freq > (real_freq + self.column_frequency_error):
    #                             suitable_row = False
    #                             while not suitable_row:
    #                                 ## Find row with value for column in current bin
    #                                 index = self.find_row(syn_data, column, bin, bin + bin_size)

    #                                 ## No row with value in current bin for column
    #                                 if index is None:
    #                                     row_found = row_found + 1
    #                                     continue

    #                                 row_found = 0

    #                                 ## Remove row from syn data to keep the requested amount of rows
    #                                 row = syn_data.loc[index].to_dict()
    #                                 syn_data = syn_data.drop(index)

    #                                 ## Add new row where value for column is not in current bin
    #                                 row_added = False
    #                                 while not row_added:
    #                                     row = self.generate(generator, 1).dataframe()
    #                                     if row.iloc[0][column] < bin or row.iloc[0][column] >= bin + bin_size:
    #                                         ## nearest neighbor distance between syn_data of different privacy level and real data
    #                                         mean_dist_2 = euclidean_distances(row, syn_data).mean(axis=1)
    #                                         # mean_dist_2 = self.privacy_level_metric.evaluate(self.loader, GenericDataLoader(row, sensitive_features=self.sensitive_features))
    #                                         print(mean_dist_2)
                                    
    #                                         ## Don't merge syn data with data from other level if the other data is closer to the real data than the current syn data itself, avoid decreasing privacy
    #                                         if mean_dist_2 < mean_dist:
    #                                             continue

    #                                         syn_data = syn_data.append(row, ignore_index=True)
    #                                         row_added = True
    #                                         row_found = True
                                    
    #                                 ## Reset no_change counter
    #                                 no_change = 0

    #                     freqs = self.column_frequencies(real_data, syn_data, self._n_histogram_bins)
    #                     real_column_freqs = freqs[column][0]
    #                     # print(real_column_freqs)
    #                     syn_column_freqs = freqs[column][1]
    #                     # print(syn_column_freqs)

    #                     ## Calculate mean difference
    #                     sum = 0
    #                     counter = 0
    #                     for bin in real_column_freqs:
    #                         real_freq = real_column_freqs[bin]
    #                         fake_freq = syn_column_freqs[bin]
    #                         sum = sum + abs(real_freq - syn_freq)
    #                         counter = counter + 1
    #                     avg = sum / counter
    #                     print('avg: ' + str(avg))
        
    #             ## Increase counter
    #             no_change = no_change +1

    #     new_data_loader = GenericDataLoader(syn_data, sensitive_features=self.sensitive_features)
    #     if utility_func(self.loader, new_data_loader):
    #         return new_data_loader

    #     ## Utility can not be reached
    #     return None     