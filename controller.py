## Synthcity imports
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from importlib import import_module
from typing import Optional, Tuple, List
import threading
from timeit import default_timer as timer

from protectionLevel import ProtectionLevel
from generative_model_classes.plugin import Plugin
from generatorCreator import GeneratorCreator
from configurationParser import ConfigHandler
from metrics.privacyMetrics import PrivacyMetric
from metrics.utilityMetrics import UtilityMetric
from metrics.privacyCalculator import PrivacyCalculator
from metrics.utilityCalculator import UtilityCalculator
from evaluation import Evaluation


class Controller():
    """
    Main class for the Data Chameleon.
    
    """

    def __init__(self):
        self.generators = {}
        self.protection_levels = {}
        self.syn_data = {}

        self.validation_size_min = 1000
        self.validation_size_max = 10000
        self.global_privacy_metric = PrivacyMetric()
        self.global_utility_metric = UtilityMetric()

        self.data_encoders = None

        self.configparser = ConfigHandler()

        self.privacyCalculator = PrivacyCalculator()
        self.utilityCalculator = UtilityCalculator()
        self.evaluation = Evaluation()

        self.handle_configs()

    ########## CONTROLLER SETUP ########### 
        
    def handle_configs(self):
        self.configparser.parse_configs()
        self.handle_generator()
        self.handle_generation_technique_metrics()
        self.handle_generation_technique()
        self.handle_encoding()
        self.handle_evaluation_metrics()

    def handle_generator(self):
        ## Get plugin class generator
        module = import_module('generative_model_classes.' + self.configparser.get_plugin_module())
        className = self.configparser.get_plugin_class()
        if module == 'none':
            raise ValueError("Specify the module name and class name of the plugin in the config file.")
        else:
            self.pluginModule = module
            self.pluginClass = getattr(module, className)
    
    def handle_generation_technique_metrics(self):
        ## Get privacy metrics with weights
        privacyMetrics = self.configparser.get_privacy_metrics()
        self.privacyCalculator.setMetrics(privacyMetrics)
        ## Get utility metrics with weights
        utilityMetrics = self.configparser.get_utility_metrics()
        self.utilityCalculator.setMetrics(utilityMetrics)
    
    def handle_generation_technique(self):
        ## Get the generation class
        module = import_module(self.configparser.get_generation_module())
        className = self.configparser.get_generation_class()
        if module == 'none':
            self.generationTechniqueModule = None
        else:
            self.generationTechniqueModule = module
            self.generationTechniqueInstance = getattr(module, className)(self.privacyCalculator, self.utilityCalculator)

    def handle_encoding(self):
        ## Get information if we need to encode the data or not
        self.encode = self.configparser.get_encoding()

    def handle_evaluation_metrics(self):
        ## Get privacy and utility metrics
        privacyMetrics, utilityMetrics = self.configparser.get_evaluation_metrics()
        self.evaluation.setMetrics(privacyMetrics, utilityMetrics)

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
    
    def decode_data(self, data: DataLoader):
        ## Use decoder of Dataloader class
        if self.private_data_encoders is not None:
            decoded = data.decode(self.private_data_encoders)

        return decoded
    
    ########## METRICS ###########   

    ## Add a new privacy metric to the system
    def add_privacy_metric(self, metric: PrivacyMetric):
        ## Add metric to evaluation
        self.evaluation.add_privacy_metric(metric)
        ## Evaluate trained generators and synthetic data with new metric and add evaluation to protection level
        for name, gen in self.generators.items():
            value_min = metric.calculate(self.private_data, self.generate(gen, self.validation_size_min))
            value_max = metric.calculate(self.private_data, self.generate(gen, self.validation_size_max))
            self.protection_levels[name].add_privacy((metric, (value_min,value_max)))
        for name, syn in self.syn_data.items():
            value_min = metric.calculate(self.private_data, self.generate(gen, self.validation_size_min))
            value_max = metric.calculate(self.private_data, self.generate(gen, self.validation_size_max))
            self.protection_levels[name].add_privacy((metric, (value_min,value_max)))

    ## Add a new utility metric to the system
    def add_utility_metric(self, metric: UtilityMetric):
        ## Add metric to evaluation
        self.evaluation.add_utility_metric(metric)
        ## Evaluate trained generators and synthetic data with new metric and add evaluation to protection level
        for name, gen in self.generators.items():
            value_min = metric.calculate(self.private_data, self.generate(gen, self.validation_size_min))
            value_max = metric.calculate(self.private_data, self.generate(gen, self.validation_size_max))
            self.protection_levels[name].add_utility((metric, (value_min,value_max)))
        for name, syn in self.syn_data.items():
            value_min = metric.calculate(self.private_data, self.generate(gen, self.validation_size_min))
            value_max = metric.calculate(self.private_data, self.generate(gen, self.validation_size_max))
            self.protection_levels[name].add_utility((metric, (value_min,value_max)))

    ## Show available metrics for evaluation
    def show_metrics(self):
        print(self.evaluation.show_metrics())

    ########## PROTECTION LEVELS ###########    
    
    ## Create a data protection level for a new use case
    def create_protection_level(self, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, Tuple[float,float]]] = None, utility: Optional[Tuple[UtilityMetric, Tuple[float,float]]] = None):
        if privacy is not None and utility is not None:
            return ProtectionLevel(protection_name, [privacy], [utility])
        elif privacy is not None:
            return ProtectionLevel(protection_name, [privacy], utility)
        elif utility is not None:
            return ProtectionLevel(protection_name, privacy, [utility])
        else:
            return ProtectionLevel(protection_name, privacy, utility)
    
    ## Show all data protection levels for the use cases added to the system
    def show_protection_levels(self):
        result = "---Protection levels--- \n"
        for _, level in self.protection_levels.items():
            result = result + level.show_level()
            result = result + "----\n"
        print(result)

    ########## PREPARATION PHASE ###########
    
    ## Manually add a new generator to the system for a specific use case (specific privacy/utility), new generator should be tested manually for the requirement
    def add_generator(self, generator: Plugin, protection_name: str):
        ## check if there already exists a generator for this protection level
        if protection_name in self.protection_levels.keys():
            raise ValueError("The system already provides service for this protection level.")
        
        ## create protection level for this generator
        protection_level = self.create_protection_level(protection_name)
        generator.set_protection_level(protection_level)
        
        ## fit generator
        if not generator.fitted:
            generator.fit(self.private_data)

        ## validate generator
        validation_data_min = self.generate(generator, self.validation_size_min)
        validation_data_max = self.generate(generator, self.validation_size_max)
        self.evaluation.generate_evalution_info(self.private_data, validation_data_min, validation_data_max, generator, protection_name)

        ## evaluate generator with different metrics
        for priv in self.evaluation.get_privacy_metrics():
            ## take average of evaluating 5 times, evaluate with small and large dataset
            value_min = 0
            for i in range(0, 5):
                value_min = value_min + priv.calculate(self.private_data, validation_data_min)
            privacy_min = value_min / 5
            value_max = 0
            for i in range(0, 5):
                value_max = value_max + priv.calculate(self.private_data, validation_data_max)
            privacy_max = value_max / 5
            protection_level.add_privacy((priv, (privacy_min, privacy_max)))
        for util in self.evaluation.get_utility_metrics():
            ## take average of evaluating 5 times, evaluate with small and large dataset
            value_min = 0
            for i in range(0, 5):
                value_min = value_min + util.calculate(self.private_data, validation_data_min)
            utility_min = value_min / 5
            value_max = 0
            for i in range(0, 5):
                value_max = value_max + util.calculate(self.private_data, validation_data_max)
            utility_max = value_max / 5
            protection_level.add_utility((util, (utility_min, utility_max)))

        ## add generator to repository and link to protection level
        generator.set_protection_level(protection_level)
        self.generators[protection_level.name] = generator
        self.protection_levels[protection_level.name] = protection_level

        ## Store generator on disk
        generator.save_model()

    ## Create new generator for the system for specific use case (specific privacy/utility), system autonomously finds a suitable generator
    def create_generator(self, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None):
        ## check input, 1 requirement for privacy and/or utility should be given
        if privacy is None and utility is None:
            raise ValueError("1 requirement for privacy and/or utility must be given.")
        
        ## check if there already exists a generator for this protection level
        if protection_name in self.protection_levels.keys():
            ## check if the existing generator meets the requirements
            generator = self.generators[protection_name]
            x = 0
            while x < 5:
                x = x + 1
                validation_data = self.generate(generator, (self.validation_size_min + self.validation_size_max) / 2)
                priv_satisfied = False
                util_satisfied = False
                ## Calculate the privacy with given metric for requirement
                if privacy is not None:
                    priv_val = privacy[0].calculate(self.private_data, validation_data)
                    print('calc priv req: ' + str(priv_val))
                    if privacy[0].satisfied(privacy[1], priv_val, range):
                        priv_satisfied = True
                else:
                    priv_satisfied = True
                ## Calculate the utility with given metric for requirement
                if utility is not None:
                    util_val = utility[0].calculate(self.private_data, validation_data)
                    print('calc util req: ' + str(util_val))
                    if utility[0].satisfied(utility[1], util_val, range):
                        util_satisfied = True
                else:
                    util_satisfied = True
                if priv_satisfied and util_satisfied:
                    return
            raise ValueError("The system already provides service for this protection level.")
        
        ## Create a generator that generates synthetic data that meets the requirements
        generatorCreator = GeneratorCreator(self.private_data, self.pluginClass, self.privacyCalculator, self.utilityCalculator)
        generator = generatorCreator.create_generator(self.generators, protection_name, privacy, utility, range)

        ## system automatically created a generator
        if generator is not None:
            ## validate generator
            validation_data_min = self.generate(generator, self.validation_size_min)
            validation_data_max = self.generate(generator, self.validation_size_max)
            self.evaluation.generate_evalution_info(self.private_data, validation_data_min, validation_data_max, generator, protection_name)

            ## evaluate generator with different metrics
            protection_level = self.create_protection_level(protection_name=protection_name)
            for priv in self.evaluation.get_privacy_metrics():
                ## take average of evaluating 5 times, evaluate with small and large dataset
                value_min = 0
                for i in range(0, 5):
                    value_min = value_min + priv.calculate(self.private_data, validation_data_min)
                privacy_min = value_min / 5
                value_max = 0
                for i in range(0, 5):
                    value_max = value_max + priv.calculate(self.private_data, validation_data_max)
                privacy_max = value_max / 5
                protection_level.add_privacy((priv, (privacy_min, privacy_max)))
            for util in self.evaluation.get_utility_metrics():
                ## take average of evaluating 5 times, evaluate with small and large dataset
                value_min = 0
                for i in range(0, 5):
                    value_min = value_min + util.calculate(self.private_data, validation_data_min)
                utility_min = value_min / 5
                value_max = 0
                for i in range(0, 5):
                    value_max = value_max + util.calculate(self.private_data, validation_data_max)
                utility_max = value_max / 5
                protection_level.add_utility((util, (utility_min, utility_max)))
            
            ## add generator to repository and link to protection level
            generator.set_protection_level(protection_level)
            self.generators[protection_level.name] = generator
            self.protection_levels[protection_level.name] = protection_level

            ## Store generator on disk
            generator.save_model()

        ## system was not able to automatically create a generator
        else:
            print('Automatically creating a generator for this protection level did not work. try to add it manually.')
    
    ## Create synthetic data for a specific use case (specific privacy/utility) by merging synthetic data from existing generators
    def create_by_merging(self, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None):
        ## check input, 1 requirement for privacy and/or utility should be given
        if privacy is None and utility is None:
            raise ValueError("1 requirement for privacy and/or utility must be given.")
        else:
            ## Create synthetic data using the existing generators to respond on the request as fast as possible
            if utility is None:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, privacy[0], privacy[1], None, None, range, (self.validation_size_min + self.validation_size_max) / 2)
            elif privacy is None:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, None, None, utility[0], utility[1], range, (self.validation_size_min + self.validation_size_max) / 2)
            else:
                synthetic_data = self.generationTechniqueInstance.create(self.private_data, self.generators, privacy[0], privacy[1], utility[0], utility[1], range, (self.validation_size_min + self.validation_size_max) / 2)

            ## system created synthetic data by merging existing generators
            if synthetic_data is not None:
                ## evaluate generator with different metrics
                protection_level = self.create_protection_level(protection_name=protection_name)
                for priv_metric in self.evaluation.get_privacy_metrics():
                    privacy = priv_metric.calculate(self.private_data, synthetic_data)
                    protection_level.add_privacy((priv_metric, (privacy,privacy)))
                for util_metric in self.evaluation.get_utility_metrics():
                    utility = util_metric.calculate(self.private_data, synthetic_data)
                    protection_level.add_utility((util_metric, (utility,utility)))
            
                ## add synthetic data to repository and link to protection level
                self.syn_data[protection_level.name] = synthetic_data
                self.protection_levels[protection_level.name] = protection_level

            ## system was not able to automatically merge generators
            else:
                raise ValueError('Merging existing generators for these requirements did not work. try to add a new generator.')
    
    ## remove generator from system
    def remove_generator(self, protection_name: str):
        ## The protection level name has to be linked to a generator
        self.protection_levels.pop(protection_name, None)
        self.generators.pop(protection_name, None)
        self.syn_data.pop(protection_name, None)

    ########## OPERATION PHASE ###########

    ## Generate synthetic data with generator
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

    ## Request handler: Handle incoming request for synthetic data
    def request_synthetic_data(self, size: int, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None) -> pd.DataFrame:
        synthetic_data = None

        ## check input, 1 requirement for privacy and/or utility should be given
        if privacy is None and utility is None:
            raise ValueError("1 requirement for privacy and/or utility must be given.")

        ## Check if there is a protection level in the system that meets these requirements
        levels = self.find_protection_levels(privacy, utility, range)
        for level in levels:
            ## Check if there is previously generated synthetic data for this protection level
            if level.name in self.syn_data.keys():
                syn = self.create_data_loader(self.syn_data[level.name].dataframe().sample(size))
                if len(syn.dataframe()) == size:
                    ## Evaluate synthetic data with right size with metrics from requirements
                    if self.evaluate_synthetic_data(syn, privacy, utility, range):
                        # print("using previous generated data for these requirements")
                        synthetic_data = syn
                        break
            ## Check if there is a generator trained that meets these requirements
            elif level.name in self.generators.keys(): 
                generator = self.generators[level.name]
                ## Evaluate synthetic data with right size with metrics from requirements
                x = 0
                while x < 10:
                    x = x + 1
                    syn = self.generate(generator, size)
                    if self.evaluate_synthetic_data(syn, privacy, utility, range):
                        # print("using trained generator for this protection level: " + generator.get_protection_level().name)
                        synthetic_data = syn
                        ## Store synthetic data such that the data can be delivered faster on the next request
                        self.syn_data[level.name] = synthetic_data
                        break
                if synthetic_data is not None:
                    break

        if synthetic_data is None:
            ## no trained generator exists and no cached synthetic data is available that meets these requirements, go to preparation phase
            print("No generator has been configured for these requirements, please come back later. The system will notify you if it's ready to serve your request.")
            thread_preparation = threading.Thread(target=self.handle_new_requirements, args=(protection_name, privacy, utility, range))
            thread_preparation.start()
            return None
        
        for priv in self.evaluation.get_privacy_metrics():
            print(priv.name() + str(priv.calculate(self.private_data, synthetic_data)))
        for util in self.evaluation.get_utility_metrics():
            print(util.name() + str(util.calculate(self.private_data, synthetic_data)))
        
        ## Decode syn data
        if self.encode:
            synthetic_data = self.decode_data(synthetic_data)
        
        return synthetic_data.dataframe()
    
    ## Match requirements to protection level
    def find_protection_levels(self, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None) -> List[ProtectionLevel]:
        levels = []
        for _, level in self.protection_levels.items():
            if privacy is not None and utility is not None:
                priv_eq = False
                util_eq = False
                for metric, value in level.privacy:
                    ## Check if the name of the metric in the protection level is equal to the metric in the requirement
                    if metric.name() == privacy[0].name():
                        ## Check if the required value falls within the range of the protection level
                        if abs(privacy[1] - value[0]) <= range or abs(privacy[1] - value[1]) <= range:
                            priv_eq = True
                            break
                        else:
                            break
                for metric, value in level.utility:
                    ## Check if the name of the metric in the protection level is equal to the metric in the requirement
                    if metric.name() == utility[0].name():
                        ## Check if the required value falls within the range of the protection level
                        if abs(utility[1] - value[0]) <= range or abs(utility[1] - value[1]) <= range:
                            util_eq = True
                            break
                        else:
                            break
                if priv_eq and util_eq:
                    levels.append(level)
            elif privacy is not None:
                for metric, value in level.privacy:
                    if metric.name() == privacy[0].name():
                        if abs(privacy[1] - value[0]) <= range or abs(privacy[1] - value[1]) <= range:
                            levels.append(level)
                        else:
                            break
            elif utility is not None:
                for metric, value in level.utility:
                    if metric.name() == utility[0].name():
                        if abs(utility[1] - value[0]) <= range or abs(utility[1] - value[1]) <= range:
                            levels.append(level)
                        else:
                            break
        return levels

    ## Evaluate synthetic data against requirements
    def evaluate_synthetic_data(self, syn: DataLoader, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None) -> bool:
        priv_satisfied = False
        util_satisfied = False
        if privacy is not None:
            if abs(privacy[1] - privacy[0].calculate(self.private_data, syn)) <= range:
                priv_satisfied = True
        else:
            priv_satisfied = True
        if utility is not None:
            if abs(utility[1] - utility[0].calculate(self.private_data, syn)) <= range:
                util_satisfied = True
        else:
            util_satisfied = True
        if priv_satisfied and util_satisfied:
            return True
        else:
            return False
    
    ## Handle new requiremetns by creating a new generator or mering existing ones
    def handle_new_requirements(self, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None):
        ## Let the system create a new generator (preparation phase)
        try:
            self.create_by_merging(protection_name, privacy, utility, range)
            ## System was able to create synthetic data, notify data consumer that system is ready to handle its request
            print('The system is now ready to handle a request for protection level: ' + protection_name)
        except ValueError as e:
            ## System was not able to merge synthetic data, try creating a new generator to meet the requirements
            try:
                self.create_generator(protection_name, privacy, utility, range)
                ## System was able to create generator, notify data consumer that system is ready to handle its request
                print('The system is now ready to handle a request for protection level: ' + protection_name)
            except ValueError as e:
                print('The system was unable to automatically create a generator or merge synthetic data for these requirements. Add a generator manually.')