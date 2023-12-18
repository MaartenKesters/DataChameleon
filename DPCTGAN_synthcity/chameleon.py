# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader

from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator,  InferenceEvaluator

from user import User
from privacyLevel import PrivacyLevels, RequirementLevels, EvaluationLevels
from knowledgeComponent import KnowledgeComponent
from dataRequest import DataRequest
from privacyLevelMetric import PrivacyLevelMetric, NearestNeighborDistance, CommonRows, kAnonymity, DataLeakage, DataMismatch, ReIdentification
from plugin import Plugin

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import copy


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
        self.original_data = data
        self.sensitive_features = sensitive_features
        self.aux_cols = list(data.sample(n=3,axis='columns').columns.values)
        self.train_data, self.control_data = train_test_split(data, test_size=0.2)
        self.loader = GenericDataLoader(self.train_data, sensitive_features=sensitive_features)

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
        else:
            raise ValueError("generator must be an instance of the Plugin class")
        
    def train_generators(self):
        print(" ")
        for level, generator in self.generators.items():
            generator.fit(self.loader)
            # self.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)
            # generator.update(self.loader)
            # self.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)


        # self.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.HIGH, 1000)
        # self.generate_synthetic_data("User2", PrivacyLevels.SECRET, 1000)

    def update_utility(self, generator: Plugin):
        generator.update(self.loader)

    def generate_synthetic_data(self, user_name: str, requested_level: PrivacyLevels, count: int, utility_func: any=None):
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
            
            # TODO remove
            # correct_privacy_level = True
        
        # Save synthetic dataset with privacy level
        # synthetic_data.dataframe().to_csv(cwd + '/data/synthetic_' + requested_level.__str__() + '.csv')
        user.set_data_requested(True)

        ## Generate data
        synthetic_data = self.generate(suitable_generator, count)

        ## Generate synthetic data, fine tune generators untill required utility is reached if utility function is included
        if utility_func is not None:
            # Need to copy the baseline generator to avoid that we update it and lose the baseline model for future generations
            updated_generator = copy.deepcopy(suitable_generator)
            utility_satisfied = False
            while not utility_satisfied:
                print("---Testing utility of synthetic data---")
                print(self.privacy_level_metric.name() + ": " + str(self.privacy_level_metric.evaluate(self.loader, synthetic_data)))
                utility = utility_func(synthetic_data.dataframe())
                print("Utility satisfied? " + str(utility))
                print("Current privacy budjet: " + str(updated_generator.get_privacy_budget()))
                if utility:
                    utility_satisfied = True
                else:
                    self.update_utility(updated_generator)
                    privacy_budjet = updated_generator.get_privacy_budget()
                    # TODO
                    # if privacy_budjet >= requested_level.epsilon:
                    #     raise RuntimeError("This utility can not be achieved for this privacy level.")
                    synthetic_data = self.generate(updated_generator, count)
        

        ## Ask user to confirm the synthetic data
        final_data = self.confirm_synthetic_data(suitable_generator, requested_level, synthetic_data, count)
        
        print("---Releasing synthetic data---")
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
                generator.update(self.loader)
                return generator
        else:
            # there is no generator with a higher privacy level
            # Increase the privacy of the current generator by small steps
            generator.update(self.loader)
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
                new_generator.update(self.loader)
                return generator
        else:
            # there is no generator with a lower privacy level
            # TODO decrease the privacy of the current generator by small steps
            generator.update(self.loader)
            return generator
        
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