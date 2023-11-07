from privacyLevel import PrivacyLevels

class DataRequest():
    """
    Class that holds all information of a request: privacy level, synthetic data, performance report, etc.

    Constructor Args:
        
    """

    def __init__(self, level):
        if isinstance(level, PrivacyLevels):
            self.privacy_level = level
        else:
            raise ValueError("level must be an instance of the PrivacyLevel class")

    def privacy_level(self):
        return self.privacy_level

    def add_synthetic_data(self, path):
        self.data_path = path

    def synthetic_data(self):
        return self.data_path
    
    def add_performance_report(self, path):
        self.report = path

    def performance_report(self):
        return self.report