from fineTuningMethod import FineTuningMethod
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator

from synthcity.plugins.core.dataloader import DataLoader

class TrainOnSynthetic(FineTuningMethod):
    def __init__(
            self,
            trained_generators: dict,
            privacyCalculator: PrivacyCalculator,
            utilityCalculator: UtilityCalculator
    ) -> None:
        super().__init__(trained_generators, privacyCalculator, utilityCalculator)

    @staticmethod
    def name() -> str:
        return "trainonsynthetic"
    
    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase privacy')
        new_privacy = self.privacy_calc.calculatePrivacy(self.real, syn)
        new_utility = self.utility_calc.calculateUtility(self.real, syn)
        self.current_generator.fit(syn)
        new_syn = self.current_generator.generate(count=self.count)
        new_privacy = self.privacy_calc.calculatePrivacy(self.real, new_syn)
        new_utility = self.utility_calc.calculateUtility(self.real, new_syn)


    def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease privacy')
        ...

    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase utility')
        ...

    def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease utility')
        ...