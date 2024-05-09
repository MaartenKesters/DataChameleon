import privacyMetrics
import utilityMetrics
from privacyMetrics import PrivacyMetric
from utilityMetrics import UtilityMetric
from plugin import Plugin
from protectionLevel import ProtectionLevel

from synthcity.plugins.core.dataloader import DataLoader
import os
from typing import List

class Evaluation():
    """
    Class that performs evaluation of synthetic data with different privacy and utility metrics and creates an evaluation report.
    """
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.directory = "information_files"

    def setMetrics(self, priv: List[PrivacyMetric], util: List[UtilityMetric]):
        self.privacyMetrics = []
        for metric in priv:
            self.privacyMetrics.append(getattr(privacyMetrics, privacyMetrics.CLASS_NAME_FILE[metric])())
        
        self.utilityMetrics = []
        for metric in util:
            self.utilityMetrics.append(getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metric])())

    def add_privacy_metric(self, metric: PrivacyMetric):
        self.privacyMetrics.append(metric)

    def add_utility_metric(self, metric: UtilityMetric):
        self.utilityMetrics.append(metric)
    
    def getPrivacyMetrics(self) -> List[PrivacyMetric]:
        return self.privacyMetrics
    
    def getUtilityMetrics(self) -> List[UtilityMetric]:
        return self.utilityMetrics
    
    def showMetrics(self) -> str:
        info = "---Metrics info--- \n"
        info = info + "### Privacy metrics:\n"
        for priv in self.privacyMetrics:
            info = info + "- Metric: " + priv.name() + "\n"
            info = info + "- Information: " + priv.information() + "\n"
            info = info + "\n"
        info = info + "### Utility metrics:\n"
        for util in self.utilityMetrics:
            info = info + "- Metric: " + util.name() + "\n"
            info = info + "- Information: " + util.information() + "\n"
            info = info + "\n"
        return info

    def generateEvalutionInfo(self, real: DataLoader, syn: DataLoader, generator: Plugin, protection_name: str):
        info = "---Evaluation info: " + generator.name() + ", " + protection_name + "--- \n"
        info = info + "### Privacy metrics:\n"
        for priv in self.privacyMetrics:
            info = info + "Metric: " + priv.name() + "\n"
            info = info + "Information: " + priv.information() + "\n"
            info = info + "Value: " + str(priv.calculate(real, syn)) + "\n"
            info = info + "\n"
        info = info + "### Utility metrics:\n"
        for util in self.utilityMetrics:
            info = info + "Metric: " + util.name() + "\n"
            info = info + "Information: " + util.information() + "\n"
            info = info + "Value: " + str(util.calculate(real, syn)) + "\n"
            info = info + "\n"
        directory_path = os.path.join(self.cwd, self.directory)
        os.makedirs(directory_path, exist_ok=True)
        file_name = generator.name() + "_" + protection_name + "_info.txt"
        full_file_path = os.path.join(directory_path, file_name)
        with open(full_file_path, "w") as file:
            file.write(info)


