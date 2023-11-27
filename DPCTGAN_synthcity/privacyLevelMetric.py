from abc import abstractmethod
from typing import Dict

from synthcity.plugins.core.dataloader import DataLoader

from synthcity.metrics.eval_attacks import DataLeakageLinear
from synthcity.metrics.eval_privacy import kAnonymization, IdentifiabilityScore
from synthcity.metrics.eval_sanity import DataMismatchScore, CommonRowsProportion, NearestSyntheticNeighborDistance

class PrivacyLevelMetric():
    """Base class for all metrics.

    Each derived class must implement the following methods:
        name() - name of metric
        info() - information about metric
        borders() - borders/thersholds between the different privacy levels
        comparison() - indicate how values of the metric for the different privacy levels should be compared
        _evaluate() - compare two datasets and return a dictionary of metrics.

    If any method implementation is missing, the class constructor will fail.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        raise NotImplementedError()
    
    @staticmethod
    def info() -> str:
        raise NotImplementedError()
    
    @staticmethod
    def borders() -> [float]:
        # The borders should be the values between the different privacy levels. Thus the value between the low and medium level, medium and high level, ...
        raise NotImplementedError()
    
    @staticmethod
    def comparison() -> bool:
        # Do the values of the metric increase or decrease if the privacy level inceases. True if the value of the low level is lower than the value of the medium level and so on.
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        ...

    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        # The _evaluate method should return a comparable float value
        results = self._evaluate(X_gt, X_syn)
        return results

class NearestNeighborDistance(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = NearestSyntheticNeighborDistance()

    @staticmethod
    def name() -> str:
        return "Nearest Neighbor Distance"
    
    @staticmethod
    def info() -> str:
        return "Computes the <reduction>(distance) from the real data to the closest neighbor in the synthetic data."
    
    @staticmethod
    def borders() -> [float]:
        return [1.0, 0.9, 0.8]
    
    @staticmethod
    def comparison() -> bool:
        return False
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data).get('mean')
    
class CommonRows(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = CommonRowsProportion()
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data)
    
class kAnonymity(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = kAnonymization()
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data)
    
class DataLeakage(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = DataLeakageLinear()
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data)
    
class DataMismatch(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = DataMismatchScore()
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data)
    
class ReIdentification(PrivacyLevelMetric):
    def __init__(self) -> None:
        self.evaluator = IdentifiabilityScore()
    
    def _evaluate(self, real_data, syn_data):
        return self.evaluator.evaluate(real_data, syn_data)