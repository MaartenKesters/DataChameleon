from abc import abstractmethod
from synthcity.plugins.core.dataloader import DataLoader

from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance
from synthcity.metrics.eval_privacy import IdentifiabilityScore
from synthcity.metrics.eval_attacks import DataLeakageMLP

## dictionary to get the class name from the name given in the configuration file
CLASS_NAME_FILE = {
    "nearestneighbordistance" : "NearestNeighborDistance",
    "identifiabilityscore" : "Identifiability",
    "dataleakage" : "DataLeakage"
}

class PrivacyKnowledge():
    """
    Base class for all privacy metrics. Each privacy class holds the knowledge of the privacy metric. The privacy metrics can be used to specify the requirements of the synthetic data or during the fine tuning process of the data.

    Each derived class must implement the following methods:
        name() - returns the name of the metric.
        information() - returns the information of the metric.
        calculate() - returns the calculated value of the privacy metric.
        range() - returns the range of possible values.
        privacy() - returns 0 if small value in the range is high privacy, 1 if high privacy is a large value.
        change() - returns a dict with 'direction' up or down if privacy needs to increase or decrease, 'amount' to indicate how much it needs to change.

    If any method implementation is missing, the class constructor will fail.
        
    """
    def __init__(self) -> None:
        pass
    
    def satisfied(self, required: float, val: float, error: float) -> bool:
        # if val >= required - error and val <= required + error:
        if (self.privacy() == 1 and val >= required - error) or ((self.privacy() == 0 and val <= required + error)):
            return True
        else:
            return False

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    @staticmethod
    def information() -> str:
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
    def name() -> str:
        return "nearestneighbordistance"
    
    @staticmethod
    def information() -> str:
        return "Computes the <reduction>(distance) from the real data to the closest neighbor in the synthetic data"
    
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
    def name() -> str:
        return "identifiabilityscore"
    
    @staticmethod
    def information() -> str:
        return "Returns the re-identification score on the real dataset from the synthetic dataset."
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        return self.evaluator.evaluate(X_gt, X_syn).get('score_OC')
    
    def range(self) -> [float]:
        return [self.rangeLow, self.rangeHigh]
    
    def privacy(self) -> int:
        return 0
    
class DataLeakage(PrivacyKnowledge):
    def __init__(self) -> None:
        self.evaluator = DataLeakageMLP()
        self.rangeLow = 0
        self.rangeHigh = 1.0

    @staticmethod
    def name() -> str:
        return "dataleakage"
    
    @staticmethod
    def information() -> str:
        return "Data leakage test using a neural net."
    
    def calculate(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        if len(X_gt.sensitive_features) == 0:
            return 0
        return self.evaluator.evaluate(X_gt, X_syn).get('mean')
    
    def range(self) -> [float]:
        return [self.rangeLow, self.rangeHigh]
    
    def privacy(self) -> int:
        return 0