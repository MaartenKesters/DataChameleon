from abc import abstractmethod
from synthcity.plugins.core.dataloader import DataLoader

from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance
from synthcity.metrics.eval_privacy import IdentifiabilityScore

## dictionary to get the class name from the name given in the configuration file
CLASS_NAME_FILE = {
    "nearestneighbordistance" : "NearestNeighborDistance",
    "identifiabilityscore" : "Identifiability"
}

class PrivacyKnowledge():
    """
    Base class for all privacy metrics. Each privacy class holds the knowledge of the privacy metric. The privacy metrics can be used to specify the requirements of the synthetic data or during the fine tuning process of the data.

    Each derived class must implement the following methods:
        name() - returns the name of the metric.
        calculate() - returns the calculated value of the privacy metric.
        range() - returns the range of possible values.
        privacy() - returns 0 if small value in the range is high privacy, 1 if high privacy is a large value.
        change() - returns a dict with 'direction' up or down if privacy needs to increase or decrease, 'amount' to indicate how much it needs to change.

    If any method implementation is missing, the class constructor will fail.
        
    """
    def __init__(self) -> None:
        pass
    
    def satisfied(self, required: float, val: float, error: float) -> bool:
        if val >= required - error and val <= required + error:
            return True
        else:
            return False

    @staticmethod
    def name(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        pass
    
    @abstractmethod
    def range(self) -> [float]:
        pass
    
    @abstractmethod
    def privacy(self) -> int:
        pass
    
    def change(self, required: float, val: float) -> dict:
        if val < required:
            amount = (required - val) / (self.rangeHigh - self.rangeLow)
            return {'direction': 'up', 'amount': amount}
        else:
            amount = (val - required) / (self.rangeHigh - self.rangeLow)
            return {'direction': 'down', 'amount': amount}
    
class NearestNeighborDistance(PrivacyKnowledge):
    def __init__(self) -> None:
        self.evaluator = NearestSyntheticNeighborDistance()
        self.rangeLow = 0
        self.rangeHigh = 1.0

    @staticmethod
    def name(self) -> str:
        return "nearestneighbordistance"
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        return self.evaluator.evaluate(X_gt, X_syn).get('mean')
    
    def range(self) -> [float]:
        return [self.rangeLow, self.rangeHigh]
    
    def privacy(self) -> int:
        return 1
        
class Identifiability(PrivacyKnowledge):
    def __init__(self) -> None:
        self.evaluator = IdentifiabilityScore()
        self.rangeLow = 0
        self.rangeHigh = 1.0

    @staticmethod
    def name(self) -> str:
        return "identifiabilityscore"
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        return self.evaluator.evaluate(X_gt, X_syn).get('score_OC')
    
    def range(self) -> [float]:
        return [self.rangeLow, self.rangeHigh]
    
    def privacy(self) -> int:
        return 0