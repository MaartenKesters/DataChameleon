from enum import Enum

class PrivacyLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    def __init__(self, level, eps):
        self.level = level
        self.epsilon = eps
    
    ## Level = id level, epsilon value
    LOW = 1, 10
    MEDIUM = 2, 5
    HIGH = 3, 1
    SECRET = 4, 0.5

class RequirementLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    def __init__(self, level, border):
        self.level = level
        self.border_value = border
    
    # Prototype: let the user specify how close the synthetic data can be to the real data. The closer the synthetic data to the real data, the more privacy risk. There are potential points of privacy leakage and a source of useful information for inference attacks.
    # This can be measured by calculating nearest neighbor between the two datasets.
    # Think of this as exact overlap between the datasets but with range instead.

    # These values can be adapted if needed to better fit the requirements or when other metrics are used

    ## Level = id level, border
    LOW_MEDIUM = 1, 1.0
    MEDIUM_HIGH = 2, 0.9
    HIGH_SECRET = 3, 0.8

class EvaluationLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    def __init__(self, level, border):
        self.level = level
        self.border_value = border

    # Prototype: for now, use nearest neighbor metric between the two datasets as evaluation
    
    ## Level = id level, border
    LOW_MEDIUM = 1, 1.0
    MEDIUM_HIGH = 2, 0.9
    HIGH_SECRET = 3, 0.8