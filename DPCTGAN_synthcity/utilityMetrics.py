from abc import abstractmethod
from typing import List

from synthcity.plugins.core.dataloader import DataLoader

from synthcity.metrics.eval_statistical import InverseKLDivergence

## dictionary to get the class name from the name given in the configuration file
CLASS_NAME_FILE = {
    "inversekldivergence" : "InverseKLDivergenceMetric"
}

class UtilityMetric():
    """
    Base class for all utility metrics. Each utility class holds the knowledge of the utility metric. The utility metrics can be used to specify the requirements of the synthetic data or during the fine tuning process of the data.

    Each derived class must implement the following methods:
        name() - returns the name of the metric.
        information() - returns the information of the metric.
        calculate() - returns the calculated value of the utility metric.
        range() - returns the range of possible values.
        utility() - returns 0 if small value in the range is high utility, 1 if high utility is a large value.
        amount() - returns a float that indicates how much the value still needs to change to satisfy the required value.

    If any method implementation is missing, the class constructor will fail.
        
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    @staticmethod
    def information() -> str:
        raise NotImplementedError()
    
    def satisfied(self, required: float, val: float, error: float) -> bool:
        # if (self.utility() == 1 and val >= required - error) or ((self.utility() == 0 and val <= required + error)):
        ## Satisfied within interval
        if val >= required - error and val <= required + error:
            return True
        else:
            return False
    
    @abstractmethod
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        pass
    
    @abstractmethod
    def range(self) -> List[float]:
        pass
    
    @abstractmethod
    def utility(self) -> int:
        pass
    
    def amount(self, required: float, val: float) -> float:
        if self.utility() == 1:
            amount = (required - val) / (self.range()[1] - self.range()[0])
            return amount
        else:
            amount = (val - required) / (self.range()[1] - self.range()[0])
            return amount
    
    def print_info(self) -> str:
        result = ""
        for _, m in CLASS_NAME_FILE.items():
            metric = globals().get(m)
            result = result + "###"
            result = result + "Metric: " + metric.name() + "\n"
            result = result + "Information: " + metric.information() + "\n"
        return result
    
    def get_all_utility_metrics(self):
        metrics = []
        for _, m in CLASS_NAME_FILE.items():
            metrics.extend(globals().get(m))
        return metrics
    
class InverseKLDivergenceMetric(UtilityMetric):
    def __init__(self) -> None:
        self.evaluator = InverseKLDivergence()
        self.rangeLow = 0
        self.rangeHigh = 1.0

    @staticmethod
    def name() -> str:
        return "inversekldivergence"
    
    @staticmethod
    def information() -> str:
        return "Returns the average inverse of the Kullback–Leibler Divergence metric."
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        return self.evaluator.evaluate(X_gt, X_syn).get('marginal')
    
    def range(self) -> List[float]:
        return [self.rangeLow, self.rangeHigh]
    
    def utility(self) -> int:
        return 1
    