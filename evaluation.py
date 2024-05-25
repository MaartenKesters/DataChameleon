import metrics.privacyMetrics
import metrics.utilityMetrics
from metrics.privacyMetrics import PrivacyMetric
from metrics.utilityMetrics import UtilityMetric
from generative_model_classes.plugin import Plugin
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
            self.privacyMetrics.append(getattr(metrics.privacyMetrics, metrics.privacyMetrics.CLASS_NAME_FILE[metric])())
        
        self.utilityMetrics = []
        for metric in util:
            self.utilityMetrics.append(getattr(metrics.utilityMetrics, metrics.utilityMetrics.CLASS_NAME_FILE[metric])())

    def add_privacy_metric(self, metric: PrivacyMetric):
        self.privacyMetrics.append(metric)

    def add_utility_metric(self, metric: UtilityMetric):
        self.utilityMetrics.append(metric)
    
    def get_privacy_metrics(self) -> List[PrivacyMetric]:
        return self.privacyMetrics
    
    def get_utility_metrics(self) -> List[UtilityMetric]:
        return self.utilityMetrics
    
    def show_metrics(self) -> str:
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

    def generate_evalution_info(self, real: DataLoader, syn_min: DataLoader, syn_max: DataLoader, generator: Plugin, protection_name: str):
        info = "---Evaluation info: " + generator.name() + ", " + protection_name + "--- \n"
        info = info + "### Privacy metrics:\n"
        for priv in self.privacyMetrics:
            info = info + "Metric: " + priv.name() + "\n"
            info = info + "Information: " + priv.information() + "\n"
            info = info + "Values: " + str(priv.calculate(real, syn_min)) + " and " +  str(priv.calculate(real, syn_max)) + "\n"
            info = info + "\n"
        info = info + "### Utility metrics:\n"
        for util in self.utilityMetrics:
            info = info + "Metric: " + util.name() + "\n"
            info = info + "Information: " + util.information() + "\n"
            info = info + "Value: " + str(util.calculate(real, syn_min)) + " and " +  str(util.calculate(real, syn_max)) + "\n"
            info = info + "\n"
        directory_path = os.path.join(self.cwd, self.directory)
        os.makedirs(directory_path, exist_ok=True)
        file_name = generator.name() + "_" + protection_name + "_info.txt"
        full_file_path = os.path.join(directory_path, file_name)
        with open(full_file_path, "w") as file:
            file.write(info)


