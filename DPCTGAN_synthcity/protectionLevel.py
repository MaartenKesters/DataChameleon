from typing import Optional, List, Tuple

from metrics.privacyMetrics import PrivacyMetric
from metrics.utilityMetrics import UtilityMetric

class ProtectionLevel():
    def __init__(self, protection_name: str,
                 privacy: Optional[List[Tuple[PrivacyMetric, float]]] = None,
                 utility: Optional[List[Tuple[UtilityMetric, float]]] = None):
        self._name = protection_name
        self._privacy = privacy if privacy is not None else []
        self._utility = utility if utility is not None else []
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def privacy(self) -> List[Tuple[PrivacyMetric, float]]:
        return self._privacy
    
    @property
    def utility(self) -> List[Tuple[UtilityMetric, float]]:
        return self._utility
    
    def add_privacy(self, priv: Tuple[PrivacyMetric, float]):
        self._privacy.append(priv)

    def add_utility(self, util: Tuple[UtilityMetric, float]):
        self._utility.append(util)
    
    def show_level(self) -> str:
        result = "Protection level: " + self.name + "\n"
        if self.privacy:
            for privacy_metric, privacy_value in self.privacy:
                result = result + "- Privacy metric: " + str(privacy_metric.name()) + " with value: " + str(privacy_value) + "\n"
        if self.utility:
            for utility_metric, utility_value in self.utility:
                result = result + "- Utility metric: " + str(utility_metric.name()) + " with value: " + str(utility_value) + "\n"
        return result
