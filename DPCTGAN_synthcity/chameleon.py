## Synthcity imports
# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder

from protectionLevel import ProtectionLevel
from plugin import Plugin
from generatorCreator import GeneratorCreator

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from importlib import import_module
import pickle as pickle
from typing import Optional, Union
import threading

from configurationParser import ConfigHandler
from privacyKnowledge import PrivacyMetric
from utilityKnowledge import UtilityMetric
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from evaluationInfo import EvaluationInfo

from dataEncoderDecoder import DataEncoderDecoder


class Chameleon():
    """
    Main class for the data chameleon.
    
    """

    def __init__(self):
        self.generators = {}
        self.protection_levels = {}
        self.syn_data = {}

        self.data_encoders = None

        self.configparser = ConfigHandler()

        self.privacyCalculator = PrivacyCalculator()
        self.utilityCalculator = UtilityCalculator()
        self.evaluationInfo = EvaluationInfo()

        self.handleConfigs()
        
    def handleConfigs(self):
        self.configparser.parseConfigs()
        self.handleGenerator()
        self.handleFineTuningMetrics()
        self.handleFineTuningMethod()
        self.handleEncoding()
        self.handleEvaluationMetrics()

    def handleGenerator(self):
        ## Get plugin class generator
        module = import_module(self.configparser.getPluginModule())
        className = self.configparser.getPluginClass()
        if module == 'none':
            raise ValueError("Specify the module name and class name of the plugin in the config file.")
        else:
            self.pluginModule = module
            self.pluginClass = getattr(module, className)
    
    def handleFineTuningMetrics(self):
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
            self.fineTuningInstance = getattr(module, className)(self.privacyCalculator, self.utilityCalculator)

    def handleEncoding(self):
        ## Get information if we need to encode the data or not
        self.encode = self.configparser.getEncoding()

    def handleEvaluationMetrics(self):
        ## Get privacy and utility metrics for evaluation report
        metrics = self.configparser.getEvaluationMetrics()
        self.evaluationInfo.setMetrics(metrics)
    
    def load_private_data(self, data, sensitive_features):
        self.original_data = data
        self.sensitive_features = sensitive_features
        self.aux_cols = list(data.sample(n=3,axis='columns').columns.values)
        self.train_data, self.control_data = train_test_split(data, test_size=0.2)

        ## Create dataloader
        self.private_data = GenericDataLoader(self.train_data, sensitive_features=sensitive_features)

        ## Encode real data
        if self.encode:
            self.encode_data(self.private_data)
    
    def create_data_loader(self, data):
        return GenericDataLoader(data, sensitive_features=self.sensitive_features)
    
    def encode_data(self, data: DataLoader):
        ## Use encoder of Dataloader class
        self.private_data, self.private_data_encoders = data.encode()

        print('### encoded')
        print(self.private_data)
    
    def decode_data(self, data: DataLoader):
        ## Use decoder of Dataloader class
        if self.private_data_encoders is not None:
            decoded = data.decode(self.private_data_encoders)

        print('### decoded')
        print(decoded)
        return decoded
    
    def show_metrics(self):
        priv = PrivacyMetric()
        util = UtilityMetric()
        result = "---Privacy metrics---\n"
        result = result + priv.print_info()
        result = result + "---Utility metrics---\n"
        result = result + util.print_info()
        print(result)
    
    ## Create a data protection level for a new use case
    def create_protection_level(self, protection_name: str, epsilon: Optional[float] = None, privacy_metric: Optional[PrivacyMetric] = None, privacy_val: Optional[float] = None, utility_metric: Optional[UtilityMetric] = None, utility_val: Optional[float] = None, range: Optional[float] = None):
        return ProtectionLevel(protection_name, epsilon, privacy_metric, privacy_val, utility_metric, utility_val, range)
    
    ## Show all data protection levels for the use cases added to the system
    def show_protection_levels(self):
        result = "---Protection levels--- \n"
        for name, level in self.protection_levels.items():
            result = result + level.show_level()
            result = result + "----\n"
        print(result)
    
    ## Manually add a new generator to the system for a specific use case (specific privacy/utility), new generator should be tested manually for the requirement
    def add_generator(self, generator: Plugin, protection_level: ProtectionLevel):
        if not generator.fitted:
            generator.fit(self.private_data)
        self.generators[protection_level.name] = generator
        self.protection_levels[protection_level.name] = protection_level
        ## Store generator
        generator.save_model()
        ## Generate information file of synthetic data generator
        synthetic_data = self.generate(generator, 10000)
        self.evaluationInfo.generateInfo(self.private_data, synthetic_data, generator, protection_level)

    ## Create new generator to the system for specific use case (specific privacy/utility), system autonomously finds a suitable generator
    def create_generator(self, protection_level: ProtectionLevel):
        if protection_level.epsilon is None and (protection_level.privacy_metric is None or protection_level.utility_metric is None):
            raise ValueError("Data protection level must have an epsilon value or privacy and utility metrics and values")
        ## Create a generator that generates synthetic data that meets the requirements
        generatorCreator = GeneratorCreator(self.private_data, self.pluginClass, self.privacyCalculator, self.utilityCalculator)
        generator = generatorCreator.create_generator(self.generators, protection_level)
        if generator is not None:
            self.generators[protection_level.name] = generator
            self.protection_levels[protection_level.name] = protection_level
            ## Store generator
            generator.save_model()
            ## Generate information file of synthetic data generator
            synthetic_data = self.generate(generator, 10000)
            self.evaluationInfo.generateInfo(self.private_data, synthetic_data, generator, protection_level)
        else:
            raise ValueError('Automatically creating a generator for this protection level did not work. try to add it manually.')

    def generate(self, generator, size) -> DataLoader:
        try:
            synthetic_data = generator.generate(count = size)
        except RuntimeError as e:
            if e.message == 'Fit the generator first':
                generator.fit(self.private_data)
                synthetic_data = generator.generate(count = size)
            else:
                raise RuntimeError("Something went wrong, try adding the generators again")
        return synthetic_data

    def generate_synthetic_data(self, size: int, protection_level: Union[str, ProtectionLevel]) -> pd.DataFrame:
        print(" ")
        print("---Generating synthetic data for protection level: " + protection_level.name + "---")
        synthetic_data = None
        if type(protection_level) is str:
            ## The protection level name has to be linked to a generator
            if protection_level in self.protection_levels.keys():
                ## Check if there is previously generated synthetic data for this protection level
                if protection_level.name in self.syn_data.keys() and self.syn_data[protection_level.name].dataframe().size >= size:
                    print("using previous generated data")
                    synthetic_data = self.create_data_loader(self.syn_data[protection_level.name].dataframe().sample(size))
                else:
                    generator = self.generators[protection_level]
                    ## Generate data
                    synthetic_data = self.generate(generator, size)
            else:
                raise ValueError("This protection level name is not linked to a generator. Try adding a generator with this proteciton level first.")
        else:
            ## Check if the there is a baseline generator with this protection level
            if protection_level.name in self.protection_levels.keys():
                ## Check if there is previously generated synthetic data for this protection level
                if protection_level.name in self.syn_data.keys() and self.syn_data[protection_level.name].dataframe().size >= size:
                    print("using previous generated data")
                    synthetic_data = self.create_data_loader(self.syn_data[protection_level.name].dataframe().sample(size))
                else:
                    generator = self.generators[protection_level.name]
                    ## Generate data
                    synthetic_data = self.generate(generator, size)
            else:
                ## No generator exists for this requirement
                print("No generator has been configured for these requirements, please come back later. The system will come back to you if it's ready to serve your request.")
                synthetic_data = self.handle_new_protection_level(protection_level, size)
                ## First try to create synthetic data using the existing generators to respond on the request as fast as possible
                ## Do this in a seperate thread so the system can handle other requests
                # thread_fine_tune = threading.Thread(target=self.fine_tune_generators, args=(protection_level, count, ))
                # thread_fine_tune.start()
                ## Create a new generator for this specific requirement and add to generator repository
                ## Do this in a seperate thread so the system can handle other requests
                # thread_create_generator = threading.Thread(target=self.create_generator, args=(protection_level,))
                # thread_create_generator.start()
        
        ## Store synthetic data such that the data can be delivered faster on the next request
        if protection_level.name not in self.syn_data.keys():
            self.syn_data[protection_level.name] = synthetic_data

        ## Decode syn data
        if self.encode:
            synthetic_data = self.decode_data(synthetic_data)
        
        print("---Generated synthetic data---")
        return synthetic_data
    
    def handle_new_protection_level(self, protection_level: ProtectionLevel, size: int):
        ## if protection level is specified by an epsilon value, we can easily create the right generator
        if protection_level.epsilon is not None:
            generator = self.pluginClass(epsilon=protection_level.epsilon, protection_level=protection_level)
            generator.fit(self.private_data)
            self.generators[protection_level.name] = generator
            syn = self.generate(generator, size)
            ## Store generator
            generator.save_model()
            ## Generate information file of synthetic data generator
            synthetic_data = self.generate(generator, 10000)
            self.evaluationInfo.generateInfo(self.private_data, synthetic_data, generator, protection_level)
        else:
            ## First try to create synthetic data using the existing generators to respond on the request as fast as possible
            syn = self.fine_tune_generators(protection_level, size)
            ## Do this in a seperate thread so the system can handle other requests
            # thread_fine_tune = threading.Thread(target=self.fine_tune_generators, args=(protection_level, count, ))
            # thread_fine_tune.start()

            ## Create a new generator for this specific requirement and add to generator repository
            self.create_generator(protection_level)
            ## Do this in a seperate thread so the system can handle other requests
            # thread_create_generator = threading.Thread(target=self.create_generator, args=(protection_level,))
            # thread_create_generator.start()
            if syn is None:
                if protection_level.name in self.protection_levels.keys():
                    generator = self.generators[protection_level.name]
                    syn = self.generate(generator, size)
        return syn
    
    def fine_tune_generators(self, protection_level: ProtectionLevel, size: int):
        ## Create synthetic data using the existing generators to respond on the request as fast as possible
        return self.fineTuningInstance.fine_tune(self.private_data, self.generators, protection_level, size)