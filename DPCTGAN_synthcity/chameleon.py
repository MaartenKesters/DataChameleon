## Synthcity imports
# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder

from user import User
from privacyLevel import PrivacyLevels
from plugin import Plugin

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from importlib import import_module
import pickle as pickle

from configurationParser import ConfigHandler
import privacyKnowledge
import utilityKnowledge
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from evaluationReport import EvaluationReport

from dataEncoderDecoder import DataEncoderDecoder


class Chameleon():
    """
    Main class for the data chameleon.

    Constructor Args:
        
    """

    def __init__(self):
        self.generators = {}
        # self.syn_data = {}
        self.users = []
        # self.synthetic_data_requests = []
        self.data_encoders = None

        self.configparser = ConfigHandler()
        self.privacyCalculator = PrivacyCalculator()
        self.utilityCalculator = UtilityCalculator()
        self.evaluationReport = EvaluationReport()

        # self.encoder = DataEncoderDecoder()

        # self.syn_data_length = 1000

        self.handleConfigs()
        
    def handleConfigs(self):
        self.configparser.parseConfigs()
        self.handleSynDataRequirements()
        self.handleFineTuningMetrics()
        self.handleFineTuningMethod()
        self.handleEncoding()
        self.handleEvaluationMetrics()

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
        range = self.configparser.getRange()
        if range == 'none':
            self.requirementRrange = None
        else:
            self.requirementRrange = float(range)
    
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
            self.fineTuningInstance = getattr(module, className)(self.generators, self.privacyCalculator, self.utilityCalculator)

    def handleEncoding(self):
        ## Get information if we need to encode the data or not
        self.encode = self.configparser.getEncoding()

    def handleEvaluationMetrics(self):
        ## Get privacy and utility metrics for evaluation report
        metrics = self.configparser.getEvaluationMetrics()
        self.evaluationReport.setMetrics(metrics)


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

        ## Encode real data
        if self.encode:
            self.encode_data(self.loader)
    
    def create_data_loader(self, data):
        return GenericDataLoader(data, sensitive_features=self.sensitive_features)
    
    # def fit_encoder(self, data):
    #     ## Create data encoder and fit to real data
    #     self.encoder = TabularEncoder(categorical_limit=30).fit(data.dataframe())
    
    def encode_data(self, data):
        # self.real_encoded = self.encoder.transform(data.dataframe())
        # self.real_encoded_loader = GenericDataLoader(self.real_encoded, sensitive_features=self.sensitive_features)

        ## Use encoder of Dataloader class
        self.loader_encoded, self.data_encoders = data.encode()

        print('### encoded')
        print(self.loader_encoded)
    
    def decode_data(self, data):
        # decoded = self.encoder.inverse_transform(encoded.dataframe())

        ## Use decoder of Dataloader class
        if self.data_encoders is not None:
            data = data.decode(self.data_encoders)

        print('### decoded')
        print(data)
        return data

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

    def add_generator(self, generator):
        if isinstance(generator, Plugin):
            self.generators[generator.get_privacy_level().level] = generator
            # self.syn_data[generator.get_privacy_level().level] = pd.DataFrame
        else:
            raise ValueError("generator must be an instance of the Plugin class")

    def train_generators(self):
        print(" ")
        ## Fit generators
        for level, generator in self.generators.items():
            if self.encode:
                generator.fit(self.loader_encoded)
            else:
                generator.fit(self.loader)

        ## Generate synthetic data such that the data can be delivered faster on request
        # for level, generator in self.generators.items():
        #     self.syn_data[level] = self.generate(generator, self.syn_data_length)

    def generate(self, generator, count):
        try:
            synthetic_data = generator.generate(count = count)
        except RuntimeError as e:
            if e.message == 'Fit the generator first':
                if self.encode:
                    generator.fit(self.loader_encoded)
                else:
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

        ## TODO remove
        print('### generated encoded syn')
        print(synthetic_data.dataframe())
        # print('### encoded')
        # encoded = suitable_generator.encode(synthetic_data)
        # print(encoded)
        # real_encoded = suitable_generator.encode(self.loader)
        # print(self.utilityCalculator.calculateUtility(real_encoded, encoded))
        # self.fit_encoder(self.loader)
        # encoded = self.encode_data(synthetic_data)
        decoded = self.decode_data(synthetic_data)
        # self.decode_data(encoded)

        
        ## Delete null rows
        # null_rows = synthetic_data.dataframe().isnull().any(axis=1)
        # synthetic_data = self.create_data_loader(synthetic_data.dataframe().drop(np.where(null_rows)[0]))


        ## Encode syn data with same encoder used for real data
        # syn_encoded = self.encode(synthetic_data.dataframe())
        # syn_encoded = self.create_data_loader(syn_encoded)

        ## Fine tune the data if specific privacy and/or utility requirements are included
        if self.privacyMetricRequirement is not None or self.utilityMetricRequirement is not None:
            if self.fineTuningModule is None:
                raise RuntimeError("No fine tuning method specified, please specify a method in the config file.")
            else:
                if self.encode:
                    synthetic_data = self.create_data_loader(self.fineTuningInstance.fine_tune(suitable_generator, self.loader_encoded, synthetic_data, count, self.privacyMetricRequirement, self.privacyValueRequirement, self.utilityMetricRequirement, self.utilityValueRequirement, self.requirementRrange))
                else:
                    synthetic_data = self.create_data_loader(self.fineTuningInstance.fine_tune(suitable_generator, self.loader, synthetic_data, count, self.privacyMetricRequirement, self.privacyValueRequirement, self.utilityMetricRequirement, self.utilityValueRequirement, self.requirementRrange))
        
        if synthetic_data == None:
            print('The privacy and/or utility can not be reached. Try retraining the generators, adding a new generator or adapting the privacy or utility requirements.')
            return None

        ## Generate evaluation report of synthetic data
        report = self.evaluationReport.generateReport(self.loader, synthetic_data)
        print(report)

        ## Decode syn data
        if self.encode:
            synthetic_data = self.decode_data(synthetic_data)
        
        print("---Releasing synthetic data---")
        return synthetic_data