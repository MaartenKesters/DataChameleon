from privacyLevel import PrivacyLevels, RequirementLevels, EvaluationLevels

class KnowledgeComponent():
    """
    Class that holds the knowledge of the mappings between the privacy levels and metrics of the data chameleon.

    Constructor Args:
        
    """

    def __init__(self):
        ## mapping between privacy levels and requirement metrics
        # TODO requirements metrics
        self.requirement_metric_mapping = {RequirementLevels.LOW : PrivacyLevels.LOW, RequirementLevels.MEDIUM : PrivacyLevels.MEDIUM, RequirementLevels.HIGH : PrivacyLevels.HIGH, RequirementLevels.SECRET : PrivacyLevels.SECRET}
        ## mapping between privacy levels and evaluation metrics
        # TODO evaluation metrics
        self.evaluation_metric_mapping = {EvaluationLevels.LOW : PrivacyLevels.LOW, EvaluationLevels.MEDIUM : PrivacyLevels.MEDIUM, EvaluationLevels.HIGH : PrivacyLevels.HIGH, EvaluationLevels.SECRET : PrivacyLevels.SECRET}

    def level_by_requirement(self, requirement_level):
        return self.requirement_metric_mapping.get(requirement_level)
    
    def level_by_evaluation(self, evaluation_level):
        return self.evaluation_metric_mapping.get(evaluation_level)
