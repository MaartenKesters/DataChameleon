import privacyMetrics
import utilityMetrics
from plugin import Plugin
from protectionLevel import ProtectionLevel

from synthcity.plugins.core.dataloader import DataLoader
import os

class EvaluationInfo():
    """
    Class that creates the the evaluation report of the synthetic data with privacy and utility metrics.
    """
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.directory = "information_files"

    def setMetrics(self, metrics: dict):
        self.privMetrics = []
        self.privMetrics.append(getattr(privacyMetrics, privacyMetrics.CLASS_NAME_FILE[metrics['privmetric1']])())
        if metrics['privmetric2'] != 'none':
            self.privMetrics.append(getattr(privacyMetrics, privacyMetrics.CLASS_NAME_FILE[metrics['privmetric2']])())
        if metrics['privmetric3'] != 'none':  
            self.privMetrics.append(getattr(privacyMetrics, privacyMetrics.CLASS_NAME_FILE[metrics['privmetric3']])())
        
        self.utilMetrics = []
        self.utilMetrics.append(getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['utilmetric1']])())
        if metrics['utilmetric2'] != 'none':
            self.utilMetrics.append(getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['utilmetric2']])())
        if metrics['utilmetric3'] != 'none':  
            self.utilMetrics.append(getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['utilmetric3']])())

    def generateInfo(self, real: DataLoader, syn: DataLoader, generator: Plugin, protection_level: ProtectionLevel) -> str:
        info = "---Evaluation info: " + generator.name() + ", " + protection_level.name + "--- \n"
        info = info + "### Privacy metrics:\n"
        for priv in self.privMetrics:
            info = info + "Metric: " + priv.name() + "\n"
            info = info + "Information: " + priv.information() + "\n"
            info = info + "Value: " + str(priv.calculate(real, syn)) + "\n"
            info = info + "\n"
        info = info + "### Utility metrics:\n"
        for util in self.utilMetrics:
            info = info + "Metric: " + util.name() + "\n"
            info = info + "Information: " + util.information() + "\n"
            info = info + "Value: " + str(util.calculate(real, syn)) + "\n"
            info = info + "\n"
        directory_path = os.path.join(self.cwd, self.directory)
        os.makedirs(directory_path, exist_ok=True)
        file_name = generator.name() + "_" + protection_level.name + "_info.txt"
        full_file_path = os.path.join(directory_path, file_name)
        with open(full_file_path, "w") as file:
            file.write(info)


