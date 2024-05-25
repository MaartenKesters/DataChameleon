from typing import Optional, List, Tuple

from metrics.privacyMetrics import PrivacyMetric
from metrics.utilityMetrics import UtilityMetric

class ProtectionLevel():
    def __init__(self, protection_name: str,
                 privacy: Optional[List[Tuple[PrivacyMetric, Tuple[float,float]]]] = None,
                 utility: Optional[List[Tuple[UtilityMetric, Tuple[float,float]]]] = None):
        self._name = protection_name
        self._privacy = privacy if privacy is not None else []
        self._utility = utility if utility is not None else []
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def privacy(self) -> List[Tuple[PrivacyMetric, Tuple[float,float]]]:
        return self._privacy
    
    @property
    def utility(self) -> List[Tuple[UtilityMetric, Tuple[float,float]]]:
        return self._utility
    
    def add_privacy(self, priv: Tuple[PrivacyMetric, Tuple[float,float]]):
        self._privacy.append(priv)

    def add_utility(self, util: Tuple[UtilityMetric, Tuple[float,float]]):
        self._utility.append(util)
    
    def show_level(self) -> str:
        result = "Protection level: " + self.name + "\n"
        if self.privacy:
            for privacy_metric, privacy_value in self.privacy:
                result = result + "- Privacy metric: " + str(privacy_metric.name()) + " with values for different size datasets: " + str(privacy_value[0]) + " and " + str(privacy_value[1]) + "\n"
        if self.utility:
            for utility_metric, utility_value in self.utility:
                result = result + "- Utility metric: " + str(utility_metric.name()) + " with values for different size datasets: " + str(utility_value[0]) + " and " + str(utility_value[1]) + "\n"
        return result
