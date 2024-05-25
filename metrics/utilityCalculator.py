import metrics.utilityMetrics as utilityMetrics

from synthcity.plugins.core.dataloader import DataLoader

class UtilityCalculator():
    """
    Class that handles the calculation of the utility of the synthetic data. 
    Keeps track of the metrics that need to be used and calculates the weighted sum of results.
    """
    def __init__(self) -> None:
        pass

    def setMetrics(self, metrics: dict):
        self.metric1 = getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['metric1']])()
        self.weight1 = float(metrics['weight1'])
        if metrics['metric2'] != 'none':
            self.metric2 = getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['metric2']])()
            self.weight2 = float(metrics['weight2'])
        else:
            self.metric2 = None
            self.weight2 = None
        if metrics['metric3'] != 'none':  
            self.metric3 = getattr(utilityMetrics, utilityMetrics.CLASS_NAME_FILE[metrics['metric3']])()
            self.weight3 = float(metrics['weight3'])
        else:
            self.metric3 = None
            self.weight3 = None

    def calculateUtility(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
        value = 0
        value = value + self.normalize(self.metric1, self.metric1.calculate(X_gt, X_syn)) * self.weight1
        if self.metric2 is not None:
            value = value + self.normalize(self.metric2, self.metric2.calculate(X_gt, X_syn)) * self.weight2
        if self.metric3 is not None:
            value = value + self.normalize(self.metric3, self.metric3.calculate(X_gt, X_syn)) * self.weight3
        # print('calc util: ' + str(value))
        return value
    
    def normalize(self, metric: utilityMetrics, val: float) -> float:
        ## normalize each value to range 0-1
        normalized = (val - metric.range()[0]) / (metric.range()[1] - metric.range()[0])
        
        ## check if high utility is when the metric has a high or low value
        if metric.utility() == 1:
            return normalized
        else:
            ## high utility should be close to 1
            return 1 - normalized