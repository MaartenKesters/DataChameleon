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