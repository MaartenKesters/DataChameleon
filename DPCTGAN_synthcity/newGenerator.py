from fineTuningMethod import FineTuningMethod
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator

from synthcity.plugins.core.dataloader import DataLoader

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

    @staticmethod
    def name() -> str:
        return "newgenerator"
    
    def increase_privacy(self, real: DataLoader, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase privacy')
        ## set current generator eps as low bound
        self.eps_low_priv_bound = self.current_generator.get_dp_epsilon()

        if self.eps_high_priv_bound >= self.eps_low_priv_bound:
            ## No improvements can be made
            return None
        
        ## Increase privacy by decreasing epsilon
        self.current_generator.set_dp_epsilon(self.current_generator.get_dp_epsilon() - ((self.current_generator.get_dp_epsilon() - self.eps_high_priv_bound) * amount))
        print(self.current_generator.get_dp_epsilon())
        self.current_generator.fit(real)
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the privacy increased
        new_privacy = self.privacy_calc.calculatePrivacy(real, new_syn)
        if new_privacy > self.current_privacy:
            return new_syn
        else:
            ## Further increase the privacy
            return self.increase_privacy(real, syn, amount * 2)

    def decrease_privacy(self, real: DataLoader, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease privacy')
        ## set current generator eps as high bound
        self.eps_high_priv_bound = self.current_generator.get_dp_epsilon()

        if self.eps_low_priv_bound >= self.eps_high_priv_bound:
            ## No improvements can be made
            return None
        
        ## Decrease privacy by increasing epsilon
        self.current_generator.set_dp_epsilon(self.current_generator.get_dp_epsilon() + ((self.eps_low_priv_bound - self.current_generator.get_dp_epsilon()) * amount))
        print(self.current_generator.get_dp_epsilon())
        self.current_generator.fit(real)
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the privacy decreased
        new_privacy = self.privacy_calc.calculatePrivacy(real, new_syn)
        if new_privacy < self.current_privacy:
            self.current_privacy = new_privacy
            return new_syn
        else:
            ## Further increase the privacy
            return self.increase_privacy(real, syn, amount * 2)

    def increase_utility(self, real: DataLoader, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase utility')
        ## set current generator eps as high bound
        self.eps_low_util_bound = self.current_generator.get_dp_epsilon()

        if self.eps_high_util_bound <= self.eps_low_util_bound:
            ## No improvements can be made
            return None
        
        ## Increase utility by increasing epsilon
        self.current_generator.set_dp_epsilon(self.current_generator.get_dp_epsilon() + ((self.eps_high_util_bound - self.current_generator.get_dp_epsilon()) * amount))
        print(self.current_generator.get_dp_epsilon())
        self.current_generator.fit(real)
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the utility increased
        new_utility = self.utility_calc.calculateUtility(real, new_syn)
        if new_utility > self.current_utility:
            self.current_utility = new_utility
            return new_syn
        else:
            ## Further increase the utility
            return self.increase_utility(real, syn, amount * 2)

    def decrease_utility(self, real: DataLoader, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease utility')
        ## set current generator eps as low bound
        self.eps_high_util_bound = self.current_generator.get_dp_epsilon()

        if self.eps_high_util_bound <= self.eps_low_util_bound:
            ## No improvements can be made
            return None
        
        ## Decrease utility by decreasing epsilon
        self.current_generator.set_dp_epsilon(self.current_generator.get_dp_epsilon() - ((self.current_generator.get_dp_epsilon() - self.eps_low_util_bound) * amount))
        print(self.current_generator.get_dp_epsilon())
        self.current_generator.fit(real)
        new_syn = self.current_generator.generate(count=self.count)

        ## Check if the utility decreased
        new_utility = self.utility_calc.calculateUtility(real, new_syn)
        if new_utility < self.current_utility:
            self.current_utility = new_utility
            return new_syn
        else:
            ## Further decrease the utility
            return self.decrease_utility(real, syn, amount * 2)