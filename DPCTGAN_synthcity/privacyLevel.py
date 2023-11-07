from enum import Enum

class PrivacyLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    def __init__(self, id, eps):
        self.id = id
        self.epsilon = eps
    
    ## Level = id, epsilon value
    LOW = 1, 10
    MEDIUM = 2, 4
    HIGH = 3, 1
    SECRET = 4, 0.1

class RequirementLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    ## Level = id
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SECRET = 4

class EvaluationLevels(Enum):
    def __str__(self):
        return str(self.name)
    
    ## Level = id
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SECRET = 4