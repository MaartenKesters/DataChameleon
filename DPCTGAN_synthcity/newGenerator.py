from fineTuningMethod import FineTuningMethod
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator

from synthcity.plugins.core.dataloader import DataLoader
from plugin import Plugin

import os

class NewGenerator(FineTuningMethod):
    def __init__(
            self,
            trained_generators: dict,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        super().__init__(trained_generators, privacyCalculator, utilityCalculator)
        self.eps_high_priv_bound = 0.1
        self.eps_low_priv_bound = 10
        self.eps_high_util_bound = 10
        self.eps_low_util_bound = 0.1

        self.directory_name = "generators"
        self.cwd = os.getcwd()
        if not os.path.exists(self.cwd + "/generators"):
            os.makedirs(self.cwd + '/generators')

    @staticmethod
    def name() -> str:
        return "newgenerator"
    
    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase privacy')
        ## set current generator eps as low bound
        self.eps_low_priv_bound = self.current_generator.get_dp_epsilon()

        if self.eps_high_priv_bound >= self.eps_low_priv_bound:
            ## No improvements can be made
            return None
        
        ## Increase privacy by decreasing epsilon
        self.set_new_eps(round(self.current_generator.get_dp_epsilon() - ((self.current_generator.get_dp_epsilon() - self.eps_high_priv_bound) * amount), 1))

        print(self.current_generator.get_dp_epsilon())
        
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the privacy increased
        new_privacy = self.privacy_calc.calculatePrivacy(self.real, new_syn)
        if new_privacy > self.current_privacy:
            self.current_privacy = new_privacy
            return new_syn
        else:
            ## Further increase the privacy
            return self.increase_privacy(syn, amount * 2)

    # def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Decrease privacy')
    #     ## set current generator eps as high bound
    #     self.eps_high_priv_bound = self.current_generator.get_dp_epsilon()

    #     if self.eps_low_priv_bound >= self.eps_high_priv_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Decrease privacy by increasing epsilon
    #     self.set_new_eps(round(self.current_generator.get_dp_epsilon() + ((self.eps_low_priv_bound - self.current_generator.get_dp_epsilon()) * amount), 1))

    #     print(self.current_generator.get_dp_epsilon())
        
    #     new_syn = self.current_generator.generate(count=self.count)

    #     ## Check if the privacy decreased
    #     new_privacy = self.privacy_calc.calculatePrivacy(self.real, new_syn)
    #     if new_privacy < self.current_privacy:
    #         self.current_privacy = new_privacy
    #         return new_syn
    #     else:
    #         ## Further decrease the privacy
    #         return self.decrease_privacy(syn, amount * 2)

    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase utility')
        ## set current generator eps as high bound
        self.eps_low_util_bound = self.current_generator.get_dp_epsilon()

        if self.eps_high_util_bound <= self.eps_low_util_bound:
            ## No improvements can be made
            return None
        
        ## Increase utility by increasing epsilon
        self.set_new_eps(round(self.current_generator.get_dp_epsilon() + ((self.eps_high_util_bound - self.current_generator.get_dp_epsilon()) * amount), 1))

        print(self.current_generator.get_dp_epsilon())
        
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the utility increased
        new_utility = self.utility_calc.calculateUtility(self.real, new_syn)
        if new_utility > self.current_utility:
            self.current_utility = new_utility
            return new_syn
        else:
            ## Further increase the utility
            return self.increase_utility(syn, amount * 2)

    # def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Decrease utility')
    #     ## set current generator eps as low bound
    #     self.eps_high_util_bound = self.current_generator.get_dp_epsilon()

    #     if self.eps_high_util_bound <= self.eps_low_util_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Decrease utility by decreasing epsilon
    #     self.set_new_eps(self.current_generator.get_dp_epsilon() - ((self.current_generator.get_dp_epsilon() - self.eps_low_util_bound) * amount))

    #     print(self.current_generator.get_dp_epsilon())

    #     new_syn = self.current_generator.generate(count=self.count)

    #     ## Check if the utility decreased
    #     new_utility = self.utility_calc.calculateUtility(self.real, new_syn)
    #     if new_utility < self.current_utility:
    #         self.current_utility = new_utility
    #         return new_syn
    #     else:
    #         ## Further decrease the utility
    #         return self.decrease_utility(syn, amount * 2)
        
    def set_new_eps(self, eps: float):
        if not self.check_saved_generators(eps):
            ## Need to fit generator if model was not saved before
            self.current_generator.set_dp_epsilon(eps)
            self.current_generator.fit(self.real)
            ## Save trained model
            self.save_generator(self.current_generator)
        
    def save_generator(self, generator: Plugin):
        generator.save_model(self.cwd + '\\' + self.directory_name + '\\' + str(generator.get_dp_epsilon()))

    def check_saved_generators(self, eps: float) -> Plugin:
        if os.path.exists(self.cwd + '\\' + self.directory_name + '\\' + str(eps)):
            ## Load trained model in current generator
            self.current_generator.load_model(self.cwd + '\\' + self.directory_name + '\\' + str(eps))
            return True
        else:
            return False
    
    # def get_epsilon_range(self, eps: float) -> Plugin:
    #     ...