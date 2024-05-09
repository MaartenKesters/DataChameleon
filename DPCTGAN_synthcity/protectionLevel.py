from typing import Optional, List, Tuple

from privacyMetrics import PrivacyMetric
from utilityMetrics import UtilityMetric

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
    
    # def __eq__(self, other) -> bool:
    #     priv_eq = False
    #     util_eq = False

    #     # Check privacy metrics
    #     for privacy_metric, privacy_value in self.privacy:
    #         for other_privacy_metric, other_privacy_value in other.privacy:
    #             if privacy_metric.name() == other_privacy_metric.name():
    #                 if abs(privacy_value - other_privacy_value) <= self._range:
    #                     priv_eq = True

    #     # Check utility metrics
    #     for utility_metric, utility_value in self._utility:
    #         for other_utility_metric, other_utility_value in other.utility:
    #             if utility_metric.name() == other_utility_metric.name():
    #                 if abs(utility_value - other_utility_value) <= self._range:
    #                     util_eq = True
        
    #     ## Both privacy and utility has to be the same
    #     if not self.privacy and not self.utility:
    #         if priv_eq and util_eq:
    #             return True
    #     ## Only privacy has to be the same
    #     if not self.privacy:
    #         if priv_eq:
    #             return True
    #     ## Only utility has to be the same
    #     if not self.utility:
    #         if util_eq:
    #             return True

    #     # If no check passes, return False
    #     return False
