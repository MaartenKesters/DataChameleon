## Synthcity imports
# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

from protectionLevel import ProtectionLevel
from plugin import Plugin
from generatorCreator import GeneratorCreator

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from importlib import import_module
import pickle as pickle
from typing import Optional, Union, Tuple
import threading

from configurationParser import ConfigHandler
from privacyMetrics import PrivacyMetric
from utilityMetrics import UtilityMetric
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from evaluation import Evaluation

from dataEncoderDecoder import DataEncoderDecoder


class Controller():
    """
    Main class for the data chameleon.
    
    """

    def __init__(self):
        self.generators = {}
        self.protection_levels = {}
        self.syn_data = {}

        self.validation_size = 10000
        self.global_privacy_metric = PrivacyMetric()
        self.global_utility_metric = UtilityMetric()

        self.data_encoders = None

        self.configparser = ConfigHandler()

        self.privacyCalculator = PrivacyCalculator()
        self.utilityCalculator = UtilityCalculator()
        self.evaluation = Evaluation()

        self.handleConfigs()

    ########## CONTROLLER SETUP ########### 
        
    def handleConfigs(self):
        self.configparser.parseConfigs()
        self.handleGenerator()
        self.handleGenerationTechniqueMetrics()
        self.handleGenerationTechnique()
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
    
    def handleGenerationTechniqueMetrics(self):
        ## Get privacy metrics with weights
        privacyMetrics = self.configparser.getPrivacyMetrics()
        self.privacyCalculator.setMetrics(privacyMetrics)
        ## Get utility metrics with weights
        utilityMetrics = self.configparser.getUtilityMetrics()
        self.utilityCalculator.setMetrics(utilityMetrics)
    
    def handleGenerationTechnique(self):
        ## Get the generation class
        module = import_module(self.configparser.getGenerationModule())
        className = self.configparser.getGenerationClass()
        if module == 'none':
            self.generationTechniqueModule = None
        else:
            self.generationTechniqueModule = module
            self.generationTechniqueInstance = getattr(module, className)(self.privacyCalculator, self.utilityCalculator)

    def handleEncoding(self):
        ## Get information if we need to encode the data or not
        self.encode = self.configparser.getEncoding()

    def handleEvaluationMetrics(self):
        ## Get privacy and utility metrics
        privacyMetrics, utilityMetrics = self.configparser.getEvaluationMetrics()
        self.evaluation.setMetrics(privacyMetrics, utilityMetrics)
    
    def show_metrics(self):
        print(self.evaluation.showMetrics())

    ########## DATA LOADING ########### 
    
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

    ########## PROTECTION LEVELS ###########    
    
    ## Create a data protection level for a new use case
    def create_protection_level(self, protection_name: str, epsilon: Optional[float] = None, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: Optional[float] = None):
        if privacy is not None and utility is not None:
            return ProtectionLevel(protection_name, epsilon, [privacy], [utility], range)
        elif privacy is not None:
            return ProtectionLevel(protection_name, epsilon, [privacy], utility, range)
        elif utility is not None:
            return ProtectionLevel(protection_name, epsilon, privacy, [utility], range)
        else:
            return ProtectionLevel(protection_name, epsilon, privacy, utility, range)
    
    ## Show all data protection levels for the use cases added to the system
    def show_protection_levels(self):
        result = "---Protection levels--- \n"
        for _, level in self.protection_levels.items():
            result = result + level.show_level()
            result = result + "----\n"
        print(result)

    ########## PREPARATION PHASE ###########
    
    ## Manually add a new generator to the system for a specific use case (specific privacy/utility), new generator should be tested manually for the requirement
    def add_generator(self, generator: Plugin, protection_level: ProtectionLevel):
        ## check if there already exists a generator for this protection level
        if protection_level.name in self.protection_levels.keys():
            raise ValueError("The system already provides service for this protection level.")
        
        ## fit generator
        if not generator.fitted:
            generator.fit(self.private_data)

        ## validate generator
        validation_data = self.generate(generator, self.validation_size)
        self.evaluation.generateEvalutionInfo(self.private_data, validation_data, generator, protection_level)

        ## evaluate generator with different metrics
        if protection_level.epsilon is None:
            protection_level.add_epsilon(generator.get_dp_epsilon)
        privacy = []
        for priv in self.evaluation.getPrivacyMetrics():
            privacy.append((priv, priv.calculate(self.private_data, validation_data)))
            protection_level.set_privacy(privacy)
        utility = []
        for util in self.evaluation.getUtilityMetrics():
            utility.append((util, util.calculate(self.private_data, validation_data)))
            protection_level.set_utility(utility)

        ## add generator to repository and link to protection level
        generator.set_protection_level(protection_level)
        self.generators[protection_level.name] = generator
        self.protection_levels[protection_level.name] = protection_level

        ## Store generator on disk
        generator.save_model()

    ## Create new generator for the system for specific use case (specific privacy/utility), system autonomously finds a suitable generator
    def create_generator(self, protection_level: ProtectionLevel):
        ## check input, epsilon value or 1 requirement for privacy and/or utility should be given
        if protection_level.epsilon is None and ((not protection_level.privacy and not protection_level.utility) or len(protection_level.privacy) > 1 or len(protection_level.utility) > 1):
            raise ValueError("Data protection level must have an epsilon value or 1 requirement for privacy and/or utility.")
        
        ## check if there already exists a generator for this protection level
        if protection_level.name in self.protection_levels.keys():
            raise ValueError("The system already provides service for this protection level.")
        
        ## Create a generator that generates synthetic data that meets the requirements
        generatorCreator = GeneratorCreator(self.private_data, self.pluginClass, self.privacyCalculator, self.utilityCalculator)
        generator = None
        if not protection_level.privacy and not protection_level.utility:
            generator = generatorCreator.create_generator(self.generators, protection_level, None, None, None, None, None)
        elif not protection_level.utility:
            generator = generatorCreator.create_generator(self.generators, protection_level, protection_level.privacy[0][0], protection_level.privacy[0][1], None, None, protection_level.range)
        elif not protection_level.privacy:
            generator = generatorCreator.create_generator(self.generators, protection_level, None, None, protection_level.utility[0][0], protection_level.utility[0][1], protection_level.range)
        else:
            generator = generatorCreator.create_generator(self.generators, protection_level, protection_level.privacy[0][0], protection_level.privacy[0][1], protection_level.utility[0][0], protection_level.utility[0][1], protection_level.range)

        ## system automatically created a generator
        if generator is not None:
            ## validate generator
            validation_data = self.generate(generator, self.validation_size)
            self.evaluation.generateEvalutionInfo(self.private_data, validation_data, generator, protection_level)

            ## evaluate generator with different metrics
            if protection_level.epsilon is None:
                protection_level.add_epsilon(generator.get_dp_epsilon)
            privacy = []
            for priv in self.evaluation.getPrivacyMetrics():
                privacy.append((priv, priv.calculate(self.private_data, validation_data)))
                protection_level.set_privacy(privacy)
            utility = []
            for util in self.evaluation.getUtilityMetrics():
                utility.append((util, util.calculate(self.private_data, validation_data)))
                protection_level.set_utility(utility)
            
            ## add generator to repository and link to protection level
            generator.set_protection_level(protection_level)
            self.generators[protection_level.name] = generator
            self.protection_levels[protection_level.name] = protection_level

            ## Store generator on disk
            generator.save_model()

        ## system was not able to automatically create a generator
        else:
            raise ValueError('Automatically creating a generator for this protection level did not work. try to add it manually.')
    
    ## Create synthetic data for a specific use case (specific privacy/utility) by merging synthetic data from existing generators
    def create_by_merging(self, protection_level: ProtectionLevel):
        ## If no requirements are given for both privacy and utility, merging is not possible
        if (not protection_level.privacy and not protection_level.utility) or len(protection_level.privacy) > 1 or len(protection_level.utility) > 1:
            raise ValueError("Data protection level must have 1 requirement for privacy and/or utility. If you want to specify a protection level with an epsilon value, use the 'create generator' method.")
        else:
            ## Create synthetic data using the existing generators to respond on the request as fast as possible
            if not protection_level.utility:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, protection_level.privacy[0][0], protection_level.privacy[0][1], None, None, protection_level.range, self.validation_size)
            elif not protection_level.privacy:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, None, None, protection_level.utility[0][0], protection_level.utility[0][1], protection_level.range, self.validation_size)
            else:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, protection_level.privacy[0][0], protection_level.privacy[0][1], protection_level.utility[0][0], protection_level.utility[0][1], protection_level.range, self.validation_size)

            ## system created synthetic data by merging existing generators
            if synthetic_data is not None:
                ## evaluate generator with different metrics
                privacy = []
                for priv in self.evaluation.getPrivacyMetrics():
                    privacy.append((priv, priv.calculate(self.private_data, synthetic_data)))
                    protection_level.set_privacy(privacy)
                utility = []
                for util in self.evaluation.getUtilityMetrics():
                    utility.append((util, util.calculate(self.private_data, synthetic_data)))
                    protection_level.set_utility(utility)
            
                ## add synthetic data to repository and link to protection level
                self.syn_data[protection_level.name] = synthetic_data
                self.protection_levels[protection_level.name] = protection_level

            ## system was not able to automatically merge generators
            else:
                raise ValueError('Merging existing generators for this protection level did not work. try to add a new generator.')
    
    ## remove generator from system
    def remove_generator(self, generator_name: str, protection_name: str):
        ## The protection level name has to be linked to a generator
        self.protection_levels.pop(protection_name, None)
        self.generators.pop(protection_name, None)
        self.syn_data.pop(protection_name, None)

    ########## OPERATION PHASE ###########

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

    def request_synthetic_data(self, size: int, protection_level: Union[str, ProtectionLevel]) -> pd.DataFrame:
        print("---requesting synthetic data for protection level: " + protection_level.name + "---")
        synthetic_data = None
        protection_level_name = ""

        ## Protection level given as string (name of protection level)
        if type(protection_level) is str:
            protection_level_name = protection_level
        ## Protection level given as full protection level instance
        else:
            protection_level_name = protection_level.name

        ## Check if there is previously generated synthetic data for this protection level
        if protection_level_name in self.syn_data.keys():
            print("using previous generated data for this protection level")
            synthetic_data = self.create_data_loader(self.syn_data[protection_level_name].dataframe().sample(size))
        ## Check if there is a generator trained for this protection level
        elif protection_level_name in self.generators.keys():
            print("using trained generator for this protection level")
            generator = self.generators[protection_level_name]
            synthetic_data = self.generate(generator, size)
        ## if protection level instance is given, search suitable generator in repository (based on privacy/utility of protection level)
        elif isinstance(protection_level, ProtectionLevel):
            ## search previously generated synthetic data for this protection level
            synthetic_data = self.find_synthetic_data(protection_level)
            if synthetic_data is not None:
                synthetic_data = self.create_data_loader(synthetic_data.dataframe().sample(size))
            if synthetic_data is None:
                ## search trained generator
                generator = self.find_trained_generator(protection_level)
                if generator is not None:
                    ## use trained generator to create synthetic data
                    synthetic_data = self.generate(generator, size)
                else:
                    ## no trained generator exists, system is not able to create synthetic data at this point for this privacy level, go to preparation phase
                    print("No generator has been configured for these requirements, please come back later. The system will notify you if it's ready to serve your request.")
                    thread_preparation = threading.Thread(target=self.handle_new_protection_level, args=(protection_level, ))
                    thread_preparation.start()
        else: 
            raise ValueError("This protection level name is not linked to a generator. Try adding a generator with this proteciton level first.")
        
        if synthetic_data is not None:
            ## Store synthetic data such that the data can be delivered faster on the next request
            if protection_level.name not in self.syn_data.keys():
                self.syn_data[protection_level.name] = synthetic_data
        else:
            return None

        ## Decode syn data
        if self.encode:
            synthetic_data = self.decode_data(synthetic_data)
        
        print("---Generated synthetic data---")
        return synthetic_data
    
    def find_synthetic_data(self, protection_level: ProtectionLevel) -> DataLoader:
        for name, level in self.protection_levels.items():
            if protection_level == level:
                if name in self.syn_data.keys():
                    print('synthetic data found, the protection level of this synthetic data is close to the requested protection level')
                    print(level.show_level())
                    return self.syn_data[name]
        return None
    
    def find_trained_generator(self, protection_level: ProtectionLevel) -> Plugin:
        for name, level in self.protection_levels.items():
            if protection_level == level:
                if name in self.generators.keys():
                    print('Trained generator found, the protection level of this trained generator is close to the requested protection level')
                    print(level.show_level())
                    return self.generators[name]
        return None
    
    def handle_new_protection_level(self, protection_level: ProtectionLevel):
        ## Let the system create a new generator (preparation phase)
        try:
            self.create_generator(protection_level)
            ## System was able to create generator, notify data consumer that system is ready to handle its request
            print('The system is now ready to handle a request for protection level: ' + protection_level.name)
        except ValueError as e:
            ## System was not able to create generator, notify system operator to add it manually
            print(e)