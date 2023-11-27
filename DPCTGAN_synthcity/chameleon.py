# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader

from user import User
from privacyLevel import PrivacyLevels, RequirementLevels, EvaluationLevels
from knowledgeComponent import KnowledgeComponent
from dataRequest import DataRequest
from privacyLevelMetric import PrivacyLevelMetric, NearestNeighborDistance, CommonRows, kAnonymity, DataLeakage, DataMismatch, ReIdentification
from plugin import Plugin

import os
import pandas as pd

class Chameleon():
    """
    Main class for the data chameleon.

    Constructor Args:
        
    """

    def __init__(self, privacy_level_metric = NearestNeighborDistance()):
        self.generators = {}
        self.users = []
        self.synthetic_data_requests = []

        self.privacy_level_metric = privacy_level_metric
        self.knowledge_component = KnowledgeComponent(privacy_level_metric)
    
    def load_real_data(self, data, sensitive_features):
        self.loader = GenericDataLoader(data, sensitive_features=sensitive_features)
    
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
        else:
            raise ValueError("generator must be an instance of the Plugin class")
        
    def train_generators(self):
        for level, generator in self.generators.items():
            generator.fit(self.loader)

        self.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.HIGH, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.SECRET, 1000)

        # for level, generator in self.generators.items():
        #     generator.update(self.loader, increase_privacy=True)

        # self.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.HIGH, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.SECRET, 1000)

    def generate_synthetic_data(self, user_name: str, requested_level: PrivacyLevels, count: int):
        user = self.get_user(user_name)
        if user is None:
            raise RuntimeError("There is no user with this name.")
        
        # Find the user's privacy level
        user_privacy_level = user.get_privacy_level()

        # Check if user is allowed to request data from the specified privacy level (only allowed to request data more private than their own privacy level)
        if user_privacy_level.level > requested_level.level:
            raise RuntimeError("The privacy level of the user must be lower than the privacy level of the requested data.")

        # Check if there is previously generated data for this privacy level if the user did not yet received data from this level
        cwd = os.getcwd()
        if not user.get_data_requested():
            if os.path.isdir(cwd + '/data'):
                # directory exists
                file = cwd + '/data/synthetic_' + requested_level.__str__() + '.csv'
                if os.path.exists(file):
                    print('Using previously generated data.')
                    synthetic_data = pd.read_csv(file)
                    return synthetic_data
            else:
                # create directory
                os.makedirs(cwd + '/data')

        # Find the appropriate generator based on the requested privacy level
        if requested_level.level in self.generators:
            suitable_generator = self.generators.get(requested_level.level)
        else:
            raise RuntimeError("No suitable generator found, first add generators for all privacy levels")

        # Generate synthetic data, fine tune generators untill required privacy level is reached
        correct_privacy_level = False
        # Keep track of used generators in the fine tuning process
        used_generators = {}
        while not correct_privacy_level:
            # Add current generator to used_generators
            used_generators[suitable_generator.get_privacy_level().level] = suitable_generator
            
            # Generate synthetic data
            synthetic_data = self.generate(suitable_generator, count)
            
            # Check if the generated data fits the requested privacy level
            syn_data_privacy_level = self.check_metrics_data(synthetic_data)
            # if  syn_data_privacy_level.level > requested_level.level:
            #     # Privacy level is too high
            #     suitable_generator = self.decrease_privacy(suitable_generator, used_generators, requested_level)
            # elif syn_data_privacy_level.level < requested_level.level:
            #     # Privacy level is too low
            #     suitable_generator = self.increase_privacy(suitable_generator, used_generators, requested_level)
            # else:
            #     correct_privacy_level = True
            
            # TODO remove
            correct_privacy_level = True
        
        # Save synthetic dataset with privacy level
        # synthetic_data.dataframe().to_csv(cwd + '/data/synthetic_' + requested_level.__str__() + '.csv')
        user.set_data_requested(True)

        # Calculate metrics for synthetic data
        print(self.privacy_level_metric.evaluate(self.loader, synthetic_data))
        
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
    
    def check_metrics_data(self, syn_data):
        metric_value = self.privacy_level_metric.evaluate(self.loader, syn_data)
        print(metric_value)
        return self.get_evaluation_level(metric_value)
    
    def increase_privacy(self, generator, used_generators, requested_level):
        gen_privacy_level = generator.get_privacy_level()
        
        # Try generator with higher privacy level 
        new_gen_privacy_level = gen_privacy_level.level + 1
        if new_gen_privacy_level in self.generators:
            # get generator with higher privacy level
            new_generator = self.generators.get(new_gen_privacy_level)
            # if new generator is not yet used before, try this generator
            if new_gen_privacy_level not in used_generators:
                return new_generator
            # if it is used, it means that it generated a too high privacy level. So we need to fine tune the privacy of the current generator
            else:
                # Increase the privacy of the current generator by small steps
                generator.update(self.loader, increase_privacy=True)
                return generator
        else:
            # there is no generator with a higher privacy level
            # Increase the privacy of the current generator by small steps
            generator.update(self.loader, increase_privacy=True)
            return generator

    def decrease_privacy(self, generator, used_generators, requested_level):
        gen_privacy_level = generator.get_privacy_level()
        
        # Try generator with lower privacy level 
        new_gen_privacy_level = gen_privacy_level.level - 1
        if new_gen_privacy_level in self.generators:
            # get generator with lower privacy level
            new_generator = self.generators.get(new_gen_privacy_level)
            # if generator is not yet used before, try this generator
            if new_gen_privacy_level not in used_generators:
                return new_generator
            # if it is used, it means that it generated a too low privacy level. So we need to fine tune the privacy of the new generator
            else:
                # Increase the privacy of the new generator by small steps
                new_generator.update(self.loader, increase_privacy=True)
                return generator
        else:
            # there is no generator with a lower privacy level
            # TODO decrease the privacy of the current generator by small steps
            generator.update(self.loader, increase_privacy=True)
            return generator
    
    def add_syn_data_request(self, request):
        if isinstance(request, DataRequest):
            self.synthetic_data_requests.append(request)
        else:
            raise ValueError("request must be an instance of the DataRequest class")