from privacyLevel import PrivacyLevels, RequirementLevels, EvaluationLevels
from privacyLevelMetric import PrivacyLevelMetric

class KnowledgeComponent():
    """
    Class that holds the knowledge of the mappings between the privacy levels and metrics of the data chameleon.

    Constructor Args:
        
    """

    def __init__(self, metric = PrivacyLevelMetric):
        self.privacy_level_metric = metric

    def level_by_requirement(self, value):
        if self.privacy_level_metric.comparison():
            if value < self.privacy_level_metric.borders()[0]:
                return PrivacyLevels.LOW
            elif value < self.privacy_level_metric.borders()[1]:
                return PrivacyLevels.MEDIUM
            elif value < self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.HIGH
            elif value >= self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.SECRET
        else:
            if value >= self.privacy_level_metric.borders()[0]:
                return PrivacyLevels.LOW
            elif value >= self.privacy_level_metric.borders()[1]:
                return PrivacyLevels.MEDIUM
            elif value >= self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.HIGH
            elif value < self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.SECRET
        # if value >= EvaluationLevels.LOW_MEDIUM.border_value:
        #     return PrivacyLevels.LOW
        # elif value >= EvaluationLevels.MEDIUM_HIGH.border_value:
        #     return PrivacyLevels.MEDIUM
        # elif value >= EvaluationLevels.HIGH_SECRET.border_value:
        #     return PrivacyLevels.HIGH
        # elif value < EvaluationLevels.HIGH_SECRET.border_value:
        #     return PrivacyLevels.SECRET
    
    def level_by_evaluation(self, value):
        if self.privacy_level_metric.comparison():
            if value < self.privacy_level_metric.borders()[0]:
                return PrivacyLevels.LOW
            elif value < self.privacy_level_metric.borders()[1]:
                return PrivacyLevels.MEDIUM
            elif value < self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.HIGH
            elif value >= self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.SECRET
        else:
            if value >= self.privacy_level_metric.borders()[0]:
                return PrivacyLevels.LOW
            elif value >= self.privacy_level_metric.borders()[1]:
                return PrivacyLevels.MEDIUM
            elif value >= self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.HIGH
            elif value < self.privacy_level_metric.borders()[2]:
                return PrivacyLevels.SECRET
        # if value >= EvaluationLevels.LOW_MEDIUM.border_value:
        #     return PrivacyLevels.LOW
        # elif value >= EvaluationLevels.MEDIUM_HIGH.border_value:
        #     return PrivacyLevels.MEDIUM
        # elif value >= EvaluationLevels.HIGH_SECRET.border_value:
        #     return PrivacyLevels.HIGH
        # elif value < EvaluationLevels.HIGH_SECRET.border_value:
        #     return PrivacyLevels.SECRET
