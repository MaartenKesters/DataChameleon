from abc import abstractmethod
from synthcity.plugins.core.dataloader import DataLoader

from synthcity.metrics.eval_statistical import InverseKLDivergence

## dictionary to get the class name from the name given in the configuration file
CLASS_NAME_FILE = {
    "inversekldivergence" : "inverseKLDivergenceMetric"
}

class UtilityKnowledge():
    """
    Base class for all utility metrics. Each utility class holds the knowledge of the utility metric. The utility metrics can be used to specify the requirements of the synthetic data or during the fine tuning process of the data.

    Each derived class must implement the following methods:
        name() - returns the name of the metric.
        calculate() - returns the calculated value of the utility metric.
        range() - returns the range of possible values.
        utility() - returns 0 if small value in the range is high utility, 1 if high utility is a large value.
        change() - returns a dict with 'direction' 0 or 1 if utility needs to decrease or increase, 'amount' to indicate how much it needs to change.

    If any method implementation is missing, the class constructor will fail.
        
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    def satisfied(self, required: float, val: float, error: float) -> bool:
        if val >= required - error and val <= required + error:
            return True
        else:
            return False
    
    @abstractmethod
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        pass
    
    @abstractmethod
    def range(self) -> [float]:
        pass
    
    @abstractmethod
    def utility(self) -> int:
        pass
    
    @abstractmethod
    def change(self, required: float, val: float) -> dict:
        pass
    
class inverseKLDivergenceMetric(UtilityKnowledge):
    def __init__(self) -> None:
        self.evaluator = InverseKLDivergence()
        self.rangeLow = 0
        self.rangeHigh = 1.0

    @staticmethod
    def name() -> str:
        return "inversekldivergence"
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        return self.evaluator.evaluate(X_gt, X_syn).get('marginal')
    
    def range(self) -> [float]:
        return [self.rangeLow, self.rangeHigh]
    
    def utility(self) -> int:
        return 1

    def change(self, required: float, val: float) -> dict:
        if val < required:
            amount = (required - val) / (self.rangeHigh - self.rangeLow)
            return {'direction': 'up', 'amount': amount}
        else:
            amount = (val - required) / (self.rangeHigh - self.rangeLow)
            return {'direction': 'down', 'amount': amount}
    