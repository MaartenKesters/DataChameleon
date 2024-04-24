from generationTechnique import GenerationTechnique
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from protectionLevel import ProtectionLevel

from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from plugin import Plugin

import os
import pandas as pd
import numpy as np

class NoiseFineTuningTechnique(GenerationTechnique):
    def __init__(
            self,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        super().__init__(privacyCalculator, utilityCalculator)

    @staticmethod
    def name() -> str:
        return "noisefinetuning"
    
    def fine_tune(self, private_data: DataLoader, generators, protection_level: ProtectionLevel) -> pd.DataFrame:
        print("fine tune")
        self.private_data = private_data
        self.generators = generators

        ## Find the generator with the most similar requirements to use as a starting point
        self.generator = self.find_initial_generator(generators, protection_level)
        if self.generator is None:
            ## No suitable generator exists, can not fine-tune the synthetic data from other generators, need to create a new generator specific for this requirement
            return None
        
        ## Generate synthetic data as the starting point to check the requirements
        syn = self.generator.generate(count=self.size)
        
        ## Fine tune privacy
        ## Calculate the current privacy to use during fine tuning
        self.current_privacy = self.privacy_calc.calculatePrivacy(self.real, syn)
        priv_satisfied = False
        prev_val = 0
        while not priv_satisfied:
            ## Calculate if the privacy requirement is met
            val = protection_level.privacy_metric.calculate(self.real, syn)
            print('calc req: ' + str(val))
            if (protection_level.privacy_metric.privacy() == 1 and val < prev_val) or (protection_level.privacy_metric.privacy() == 0 and val > prev_val):
                ## Privacy requirement can not be satisfied
                return None
            else:
                prev_val = val
                if (protection_level.privacy_metric.privacy() == 1 and val > (protection_level.privacy_value - protection_level.range)) or (protection_level.privacy_metric.privacy() == 0 and val < (protection_level.privacy_value + protection_level.range)):
                    priv_satisfied = True
                    break
                else:
                    amount = protection_level.privacy_metric.amount(protection_level.privacy_value, val)
                    syn = self.increase_privacy(syn, amount)

        return syn.dataframe()
    
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