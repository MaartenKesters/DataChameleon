from abc import abstractmethod
import pandas as pd

from plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from protectionLevel import ProtectionLevel

class FineTuningTechnique():
    """
    Base class for all fine tuning techniques.

    Each derived class must implement the following methods:
        name() - a static method that returns the name of the method. e.g., merge, new_gen, etc..
        increase_privacy() - mehtod that handles the increase in privacy
        decrease_privacy() - method that handles the decrease in privacy
        increase_utility() - mehtod that handles the increase in utility
        decrease_utility() - method that handles the decrease in utility

    If any method implementation is missing, the class constructor will fail.

    """
    def __init__(
            self,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        self.privacy_calc = privacyCalculator
        self.utility_calc = utilityCalculator

    def fine_tune(self, private_data: DataLoader, generators, protection_level: ProtectionLevel, size: int) -> DataLoader:
        print("fine tune")
        self.private_data = private_data
        self.generators = generators
        self.protection_level = protection_level
        self.size = size

        ## Find the generator with the most similar protection level to use as a starting point
        self.generator = self.find_initial_generator(generators, protection_level)
        if self.generator is None:
            ## No suitable generator exists, can not fine-tune the synthetic data from other generators, need to create a new generator specific for this protection level
            return None

        ## Generate synthetic data as the starting point to check the privacy/utility
        syn = self.generator.generate(count=self.size)
        
        ## Adapt synthetic data untill privacy/utility are met
        counter = 0
        while counter < 10:
            ## Fine tune privacy
            ## Calculate the current privacy to use during fine tuning
            self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)

            priv_satisfied = False
            while not priv_satisfied:
                ## Calculate the privacy with given metric for requirement
                val = protection_level.privacy_metric.calculate(self.private_data, syn)
                print('calc req: ' + str(val))

                if (protection_level.privacy_metric.privacy() == 1 and val > (protection_level.privacy_value - protection_level.range)) or (protection_level.privacy_metric.privacy() == 0 and val < (protection_level.privacy_value + protection_level.range)):
                    priv_satisfied = True
                    break

                ## Check how much the privacy has to change
                amount = protection_level.privacy_metric.amount(protection_level.privacy_value, val)
                new = self.increase_privacy(syn, amount)

                ## If new is None, no improvements are made
                if new is not None:
                    syn = new
                else:
                    break

            ## Fine tune utility
            ## Calculate the current utility to use during fine tuning
            self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

            util_satisfied = False
            while not util_satisfied:
                ## Calculate the utility
                val = protection_level.utility_metric.calculate(self.private_data, syn)
                print('calc req: ' + str(val))

                if (protection_level.utility_metric.utility() == 1 and val > (protection_level.utility_value - protection_level.range)) or (protection_level.utility_metric.utility() == 0 and val < (protection_level.utility_value + protection_level.range)):
                    util_satisfied = True
                    break
                    
                ## Check how much the utility has to change
                amount = protection_level.utility_metric.amount(protection_level.utility_value, val)
                new = self.increase_utility(syn, amount)

                ## If new is None, no improvements are made
                if new is not None:
                    syn = new
                else:
                    break

            ## Calculate the privacy and utility, check if both values are still larger than the required value
            print('final check')
            priv_val = protection_level.privacy_metric.calculate(self.private_data, syn)
            util_val = protection_level.utility_metric.calculate(self.private_data, syn)
            print('calc priv req: ' + str(priv_val))
            print('calc util req: ' + str(util_val))
            if (protection_level.privacy_metric.privacy() == 1 and val > (protection_level.privacy_value - protection_level.range)) or (protection_level.privacy_metric.privacy() == 0 and val < (protection_level.privacy_value + protection_level.range)):
                if (protection_level.utility_metric.utility() == 1 and val > (protection_level.utility_value - protection_level.range)) or (protection_level.utility_metric.utility() == 0 and val < (protection_level.utility_value + protection_level.range)):
                    return syn
            
            counter = counter + 1
                
        return None

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        pass

    @abstractmethod
    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        pass

    def find_initial_generator(self, generators, protection_level: ProtectionLevel) -> Plugin:
        closest_generator = None
        difference = 0
        for level, generator in generators.items():
            syn = generator.generate(count=self.size)
            priv_diff = abs(protection_level.privacy_metric.calculate(self.private_data, syn) - protection_level.privacy_value)
            util_diff = abs(protection_level.utility_metric.calculate(self.private_data, syn) - protection_level.utility_value)
            diff = (priv_diff + util_diff) / 2
            if closest_generator is None:
                closest_generator = generator
                difference = diff
            elif diff < difference:
                closest_generator = generator
                difference = difference
        return closest_generator