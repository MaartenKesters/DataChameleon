from fineTuningMethod import FineTuningMethod
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from privacyKnowledge import PrivacyKnowledge
from utilityKnowledge import UtilityKnowledge

from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from plugin import Plugin

import os
import pandas as pd
import numpy as np

class NoiseFineTuning(FineTuningMethod):
    def __init__(
            self,
            trained_generators: dict,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        super().__init__(trained_generators, privacyCalculator, utilityCalculator)

    @staticmethod
    def name() -> str:
        return "noisefinetuning"
    
    def fine_tune(self, current_generator: Plugin, real: DataLoader, syn: DataLoader, count: int, priv_metric_req: PrivacyKnowledge, priv_val_req: float, util_metric_req: UtilityKnowledge, util_val_req: float, error_range: float) -> pd.DataFrame:
        print("fine tune")
        self.count = count
        self.real = real
        self.current_generator = current_generator
        self.level = current_generator.get_privacy_level()

        new_syn = syn
        
        ## Fine tune privacy
        if priv_metric_req is not None:
            ## Calculate the current privacy to use during fine tuning
            self.current_privacy = self.privacy_calc.calculatePrivacy(self.real, syn)
            priv_satisfied = False
            prev_val = 0
            while not priv_satisfied:
                ## Calculate if the privacy requirement is met
                val = priv_metric_req.calculate(self.real, new_syn)
                print('calc req: ' + str(val))
                if (priv_metric_req.privacy() == 1 and val < prev_val) or (priv_metric_req.privacy() == 0 and val > prev_val):
                    ## Privacy requirement can not be satisfied
                    return None
                else:
                    prev_val = val
                    if priv_metric_req.satisfied(priv_val_req, val, error_range):
                        priv_satisfied = True
                        break
                    else:
                        amount = priv_metric_req.amount(priv_val_req, val)
                        new_syn = self.increase_privacy(new_syn, amount)

        return new_syn.dataframe()
    
    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase privacy')
        ## Add noise to the synthetic data
        # Genearte noise with same size as that of the data.
        noise = np.random.normal(0, amount, syn.dataframe().shape)
        # Add the noise to the data. 
        syn_noised = syn.dataframe() + noise
        new_syn = GenericDataLoader(syn_noised)

        ## Check if the privacy increased
        new_privacy = self.privacy_calc.calculatePrivacy(self.real, new_syn)
        if new_privacy > self.current_privacy:
            self.current_privacy = new_privacy
            return new_syn
        else:
            ## Further increase the privacy
            return self.increase_privacy(syn, amount * 2)

    # def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
    #     ...

    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        ...

    # def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
    #     ...