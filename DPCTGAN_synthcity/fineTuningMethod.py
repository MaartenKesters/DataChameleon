from abc import abstractmethod
import pandas as pd

from plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

from privacyKnowledge import PrivacyKnowledge
from utilityKnowledge import UtilityKnowledge
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator

class FineTuningMethod():
    """
    Base class for all fine tuning methods.

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
            trained_generators: dict,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        self.trained_generators = trained_generators
        self.privacy_calc = privacyCalculator
        self.utility_calc = utilityCalculator

    def fine_tune(self, current_generator: Plugin, real: DataLoader, syn: DataLoader, count: int, priv_metric_req: PrivacyKnowledge, priv_val_req: float, util_metric_req: UtilityKnowledge, util_val_req: float, error_range: float) -> pd.DataFrame:
        print("fine tune")
        self.count = count
        self.real = real
        self.current_generator = current_generator
        self.level = current_generator.get_privacy_level()

        satisfied = False

        while not satisfied:

            ## Fine tune privacy
            if priv_metric_req is not None:
                ## Calculate the current privacy to use during fine tuning
                self.current_privacy = self.privacy_calc.calculatePrivacy(self.real, syn)

                priv_satisfied = False
                while not priv_satisfied:
                    ## Calculate the privacy
                    val = priv_metric_req.calculate(self.real, syn)
                    print('calc req: ' + str(val))

                    if priv_metric_req.satisfied(priv_val_req, val, error_range):
                        priv_satisfied = True
                        break

                    ## Check in what direction the privacy has to change
                    result = priv_metric_req.change(priv_val_req, val)
                    if result['direction'] == 'up':
                        new = self.increase_privacy(syn, result['amount'])
                    elif result['direction'] == 'down':
                        new = self.decrease_privacy(syn, result['amount'])

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        break

            ## Fine tune utility
            if util_metric_req is not None:
                ## Calculate the current utility to use during fine tuning
                self.current_utility = self.utility_calc.calculateUtility(self.real, syn)

                util_satisfied = False
                while not util_satisfied:
                    ## Calculate the utility
                    val = util_metric_req.calculate(self.real, syn)
                    print('calc req: ' + str(val))

                    if util_metric_req.satisfied(util_val_req, val, error_range):
                        util_satisfied = True
                        break
                    
                    ## Check in what direction the utility has to change
                    result = util_metric_req.change(util_val_req, val)
                    if result['direction'] == 'up':
                        new = self.increase_utility(syn, result['amount'])
                    elif result['direction'] == 'down':
                        new = self.decrease_utility(syn, result['amount'])

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        break

            ## Calculate the privacy and utility, check if both values are still larger than the required value
            print('final check')
            priv_val = priv_metric_req.calculate(self.real, syn)
            util_val = util_metric_req.calculate(self.real, syn)
            print('calc priv req: ' + str(priv_val))
            print('calc util req: ' + str(util_val))
            if (priv_metric_req.satisfied(priv_val_req, priv_val, error_range) or priv_val > priv_val_req) and (util_metric_req.satisfied(util_val_req, util_val, error_range) or util_val > util_val_req):
                return syn.dataframe()
            
            ## Merge privacy and utility fine tuned data
            # if priv_metric_req is not None and util_metric_req is not None:
            #     priv_syn = priv_syn.dataframe().sample(int(count/2))
            #     util_syn = util_syn.dataframe().sample(int(count/2))
            #     return pd.concat([priv_syn, util_syn], ignore_index=True)
            # elif priv_metric_req is not None:
            #     return priv_syn.dataframe()
            # elif util_metric_req is not None:
            #     return util_syn.dataframe()
                
        return None

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        pass

    @abstractmethod
    def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        pass

    @abstractmethod
    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        pass

    @abstractmethod
    def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        pass
