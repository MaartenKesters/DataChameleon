from abc import abstractmethod
import pandas as pd

from plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader

from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from privacyMetrics import PrivacyMetric
from utilityMetrics import UtilityMetric

class GenerationTechnique():
    """
    Base class for all generation techniques.

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

    def create(self, private_data: DataLoader, generators, privacy_metric: PrivacyMetric, privacy_value: float, utility_metric: UtilityMetric, utility_value: float, range: float, size: int) -> DataLoader:
        self.private_data = private_data
        self.generators = generators
        self.privacy_metric = privacy_metric
        self.privacy_value = privacy_value
        self.utility_metric = utility_metric
        self.utility_value = utility_value
        self.range = range
        self.size = size

        ## Find the generator with the most similar requirements to use as a starting point
        self.generator = self.find_initial_generator(generators, privacy_metric, privacy_value, utility_metric, utility_value)
        if self.generator is None:
            ## No suitable generator exists, can not create synthetic data from other generators, need to create a new generator specific for these requirements
            return None

        ## Generate synthetic data as the starting point to check the privacy/utility
        syn = self.generator.generate(count=self.size)
        
        ## Adapt synthetic data untill privacy/utility are met
        counter = 0
        while counter < 10:

            new = None
            if privacy_metric is not None:
                ## update privacy
                ## Calculate the current privacy to use during creation
                self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)

                priv_satisfied = False
                while not priv_satisfied:
                    ## Calculate the privacy with given metric for requirement
                    val = privacy_metric.calculate(self.private_data, syn)
                    print('calc req: ' + str(val))

                    if (privacy_metric.privacy() == 1 and val > (privacy_value - range)) or (privacy_metric.privacy() == 0 and val < (privacy_value + range)):
                        priv_satisfied = True
                        break

                    ## Check how much the privacy has to change
                    amount = privacy_metric.amount(privacy_value, val)
                    new = self.increase_privacy(syn, amount)

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        counter = counter + 1
                        break

            new = None
            if utility_metric is not None:
                ## Update utility
                ## Calculate the current utility to use during creation
                self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

                util_satisfied = False
                while not util_satisfied:
                    ## Calculate the utility
                    val = utility_metric.calculate(self.private_data, syn)
                    print('calc req: ' + str(val))

                    if (utility_metric.utility() == 1 and val > (utility_value - range)) or (utility_metric.utility() == 0 and val < (utility_value + range)):
                        util_satisfied = True
                        break
                        
                    ## Check how much the utility has to change
                    amount = utility_metric.amount(utility_value, val)
                    new = self.increase_utility(syn, amount)

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        counter = counter + 1
                        break

            ## Calculate the privacy and utility, check if both values are still larger than the required value
            print('final check')
            priv_satisfied = True
            util_satisfied = True
            if privacy_metric is not None:
                priv_val = privacy_metric.calculate(self.private_data, syn)
                print('calc priv req: ' + str(priv_val))
                if not (privacy_metric.privacy() == 1 and priv_val > (privacy_value - range)) or (privacy_metric.privacy() == 0 and priv_val < (privacy_value + range)):
                    priv_satisfied = False
            if utility_metric is not None:
                util_val = utility_metric.calculate(self.private_data, syn)
                print('calc util req: ' + str(util_val))
                if not (utility_metric.utility() == 1 and util_val > (utility_value - range)) or (utility_metric.utility() == 0 and util_val < (utility_value + range)):
                    util_satisfied = False
            if priv_satisfied and util_satisfied:
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

    def find_initial_generator(self, generators, privacy_metric: PrivacyMetric, privacy_value: float, utility_metric: UtilityMetric, utility_value: float) -> Plugin:
        closest_generator = None
        difference = 0
        for level, generator in generators.items():
            syn = generator.generate(count=self.size)
            priv_diff = abs(privacy_metric.calculate(self.private_data, syn) - privacy_value)
            util_diff = abs(utility_metric.calculate(self.private_data, syn) - utility_value)
            diff = (priv_diff + util_diff) / 2
            if closest_generator is None:
                closest_generator = generator
                difference = diff
            elif diff < difference:
                closest_generator = generator
                difference = difference
        return closest_generator