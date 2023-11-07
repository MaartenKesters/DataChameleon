# from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader

from user import User
from privacyLevel import PrivacyLevels
from dataRequest import DataRequest
from plugin import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader

from synthcity.metrics.eval_attacks import DataLeakageLinear
from synthcity.metrics.eval_privacy import kAnonymization
from synthcity.metrics.eval_sanity import DataMismatchScore, CommonRowsProportion, NearestSyntheticNeighborDistance

import os
import pandas as pd

class Chameleon():
    """
    Main class for the data chameleon.

    Constructor Args:
        
    """

    def __init__(self):
        self.generators = []
        self.users = []
        self.synthetic_data_requests = []
    
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

    def add_generator(self, generator):
        if isinstance(generator, Plugin):
            self.generators.append(generator)
        else:
            raise ValueError("generator must be an instance of the Plugin class")
        
    def train_generators(self):
        for generator in self.generators:
            generator.fit(self.loader)

    # def update_generators(self):
    #     for plugin in self.generators:
    #         plugin.generator.update(new_real_data)

    def generate_synthetic_data(self, user_name: str, requested_level: PrivacyLevels, count: int):
        user = self.get_user(user_name)
        if user is None:
            raise RuntimeError("There is no user with this name.")
        
        # Find the user's privacy level
        user_privacy_level = user.get_privacy_level()

        # Check if user is allowed to request data from the specified privacy level (only allowed to request data more private than their own privacy level)
        if user_privacy_level > requested_level:
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

        # Find the appropriate generator based on the user's privacy level
        suitable_generator = None
        for generator in self.generators:
            if generator.get_privacy_level() == requested_level:
                suitable_generator = generator
                break

        if suitable_generator is None:
            raise RuntimeError("No suitable generator found, first add generators for all privacy levels")

        try:
            print('Generating new data.')
            synthetic_data = suitable_generator.generate(count = count)
        except RuntimeError as e:
            if e.message == 'Fit the generator first':
                suitable_generator.fit(self.loader)
                synthetic_data = suitable_generator.generate(count = count)
            else:
                raise RuntimeError("Something went wrong, try adding the generators again")
            
        # Check if the generated data fits the requested privacy level
        syn_data_privacy_level = self.check_privacy_data(synthetic_data, requested_level)
        if  syn_data_privacy_level > requested_level:
            # Privacy level is too high
            self.decrease_privacy(suitable_generator, requested_level)
        elif syn_data_privacy_level < requested_level:
            # Privacy level is too low
            self.increase_privacy(suitable_generator, requested_level)
        
        synthetic_data.dataframe().to_csv(cwd + '/data/synthetic_' + requested_level.__str__() + '.csv')
        user.set_data_requested(True)

        # Calculate metrics for synthetic data
        print(self.metrics(synthetic_data))
        
        return synthetic_data
    
    def check_privacy_data(self, syn_data, requested_level):
        return True
    
    def increase_privacy(self, generator: Plugin, requested_level):
        return True

    def decrease_privacy(self, generator: Plugin, requested_level):
        return True
    
    def add_syn_data_request(self, request):
        if isinstance(request, DataRequest):
            self.synthetic_data_requests.append(request)
        else:
            raise ValueError("request must be an instance of the DataRequest class")
    
    def metrics(self, syn_data):
        attacks = self.evaluate_attack(syn_data)
        common_rows_prop = self.common_rows_prop(syn_data)
        k_anonymity = self.k_anonymization(syn_data)
        sanity = self.data_sanity(syn_data)
        neighbor = self.nearest_neighbor(syn_data)

        return neighbor

    def common_rows_prop(self, syn_data):
        common_rows_prop_calc = CommonRowsProportion()
        return common_rows_prop_calc.evaluate(self.loader, syn_data)
    
    def k_anonymization(self, syn_data):
        k_anonymyzation_calc = kAnonymization()
        return k_anonymyzation_calc._evaluate(self.loader, syn_data)
    
    def evaluate_attack(self, syn_data):
        evaluator = DataLeakageLinear()
        return evaluator.evaluate(self.loader, syn_data)
    
    def data_sanity(self, syn_data):
        sanity_evaluator = DataMismatchScore()
        return sanity_evaluator.evaluate(self.loader, syn_data)
    
    def nearest_neighbor(self, syn_data):
        nn = NearestSyntheticNeighborDistance()
        return nn.evaluate(self.loader, syn_data)