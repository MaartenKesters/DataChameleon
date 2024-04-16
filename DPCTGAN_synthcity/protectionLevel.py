from typing import Optional

from privacyKnowledge import PrivacyMetric
from utilityKnowledge import UtilityMetric

class ProtectionLevel():
    def __init__(self, protection_name: str, epsilon: Optional[float] = None, privacy_metric: Optional[PrivacyMetric] = None, privacy_val: Optional[float] = None, utility_metric: Optional[UtilityMetric] = None, utility_val: Optional[float] = None, range: Optional[float] = None):
        self._name = protection_name
        self._epsilon = epsilon
        self._privacy_metric = privacy_metric
        self._privacy_value = privacy_val
        self._utility_metric = utility_metric
        self._utility_value = utility_val
        self._range = range
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def privacy_metric(self) -> PrivacyMetric:
        return self._privacy_metric
    
    @property
    def privacy_value(self) -> float:
        return self._privacy_value
    
    @property
    def utility_metric(self) -> UtilityMetric:
        return self._utility_metric
    
    @property
    def utility_value(self) -> float:
        return self._utility_value
    
    @property
    def range(self) -> float:
        return self._range
    
    def show_level(self) -> str:
        result = "Protection level: " + self.name + "\n"
        if self.epsilon is not None:
            result = result + "- Epsilon value: " + str(self.epsilon) + "\n"
        else: 
            result = result + "- Privacy metric: " + str(self.privacy_metric.name()) + "\n"
            result = result + "- Privacy value: " + str(self.privacy_value) + "\n"
            result = result + "- Utility metric: " + str(self.utility_metric.name()) + "\n"
            result = result + "- Utility value: " + str(self.utility_value) + "\n"
            result = result + "- range: " + str(self.range) + "\n"
        return result
