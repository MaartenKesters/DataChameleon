import privacyKnowledge
import utilityKnowledge

from synthcity.plugins.core.dataloader import DataLoader

class EvaluationReport():
    """
    Class that creates the the evaluation report of the synthetic data with privacy and utility metrics.
    """
    def __init__(self) -> None:
        pass

    def setMetrics(self, metrics: dict):
        self.privMetrics = []
        self.privMetrics.append(getattr(privacyKnowledge, privacyKnowledge.CLASS_NAME_FILE[metrics['privmetric1']])())
        if metrics['privmetric2'] != 'none':
            self.privMetrics.append(getattr(privacyKnowledge, privacyKnowledge.CLASS_NAME_FILE[metrics['privmetric2']])())
        if metrics['privmetric3'] != 'none':  
            self.privMetrics.append(getattr(privacyKnowledge, privacyKnowledge.CLASS_NAME_FILE[metrics['privmetric3']])())
        
        self.utilMetrics = []
        self.utilMetrics.append(getattr(utilityKnowledge, utilityKnowledge.CLASS_NAME_FILE[metrics['utilmetric1']])())
        if metrics['utilmetric2'] != 'none':
            self.utilMetrics.append(getattr(utilityKnowledge, utilityKnowledge.CLASS_NAME_FILE[metrics['utilmetric2']])())
        if metrics['utilmetric3'] != 'none':  
            self.utilMetrics.append(getattr(utilityKnowledge, utilityKnowledge.CLASS_NAME_FILE[metrics['utilmetric3']])())

    def generateReport(self, real: DataLoader, syn: DataLoader) -> str:
        report = "---Evaluation report--- \n"
        report = report + "### Privacy metrics:\n"
        for priv in self.privMetrics:
            report = report + "Metric: " + priv.name() + "\n"
            report = report + "Information: " + priv.information() + "\n"
            report = report + "Value: " + str(priv.calculate(real, syn)) + "\n"
            report = report + "\n"
        report = report + "### Utility metrics:\n"
        for util in self.utilMetrics:
            report = report + "Metric: " + util.name() + "\n"
            report = report + "Information: " + util.information() + "\n"
            report = report + "Value: " + str(util.calculate(real, syn)) + "\n"
            report = report + "\n"
        return report


