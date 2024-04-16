import configparser

class ConfigHandler():
    """
    Class to handle all configurations. Requirement configs, fine tuning metrics configs, fine tuning process configs

    """
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def parseConfigs(self):
        self.parseGenerator()
        self.parseFineTuningMetrics()
        self.parseFineTuningMethod()
        self.parseEncoding()
        self.parseEvaluationMetrics()

    def parseGenerator(self):
        self.pluginModule = self.config.get('Plugin', 'module')
        self.pluginClass = self.config.get('Plugin', 'className')
    
    def parseFineTuningMetrics(self):
        self.privacyMetric1 = self.config.get('Metrics', 'privacymetric1')
        self.privacyMetric1Weight = self.config.get('Metrics', 'privacymetric1weight')
        self.privacyMetric2 = self.config.get('Metrics', 'privacymetric2')
        self.privacyMetric2Weight = self.config.get('Metrics', 'privacymetric2weight')
        self.privacyMetric3 = self.config.get('Metrics', 'privacymetric3')
        self.privacyMetric3Weight = self.config.get('Metrics', 'privacymetric3weight')
        self.utilityMetric1 = self.config.get('Metrics', 'utilitymetric1')
        self.utilityMetric1Weight = self.config.get('Metrics', 'utilitymetric1weight')
        self.utilityMetric2 = self.config.get('Metrics', 'utilitymetric2')
        self.utilityMetric2Weight = self.config.get('Metrics', 'utilitymetric2weight')
        self.utilityMetric3 = self.config.get('Metrics', 'utilitymetric3')
        self.utilityMetric3Weight = self.config.get('Metrics', 'utilitymetric3weight')

    def parseFineTuningMethod(self):
        self.fineTuningModule = self.config.get('FineTuning', 'module')
        self.fineTuningClass = self.config.get('FineTuning', 'className')

    def parseEncoding(self):
        encode = self.config.get('Encoding', 'encode')
        if encode == '1':
            self.encode = True
        else:
            self.encode = False

    def parseEvaluationMetrics(self):
        self.evalPrivacyMetric1 = self.config.get('Evaluation', 'privacymetric1')
        self.evalPrivacyMetric2 = self.config.get('Evaluation', 'privacymetric2')
        self.evalPrivacyMetric3 = self.config.get('Evaluation', 'privacymetric3')
        self.evalUtilityMetric1 = self.config.get('Evaluation', 'utilitymetric1')
        self.evalUtilityMetric2 = self.config.get('Evaluation', 'utilitymetric2')
        self.evalUtilityMetric3 = self.config.get('Evaluation', 'utilitymetric3')
    
    def getPluginModule(self):
        return self.pluginModule
    
    def getPluginClass(self):
        return self.pluginClass
    
    def getPrivacyMetrics(self):
        return {'metric1':self.privacyMetric1, 'weight1':self.privacyMetric1Weight, 'metric2':self.privacyMetric2, 'weight2':self.privacyMetric2Weight, 'metric3':self.privacyMetric3, 'weight3':self.privacyMetric3Weight}
    
    def getUtilityMetrics(self):
        return {'metric1':self.utilityMetric1, 'weight1':self.utilityMetric1Weight, 'metric2':self.utilityMetric2, 'weight2':self.utilityMetric2Weight, 'metric3':self.utilityMetric3, 'weight3':self.utilityMetric3Weight}
    
    def getFineTuningModule(self):
        return self.fineTuningModule
    
    def getFineTuningClass(self):
        return self.fineTuningClass
    
    def getEncoding(self):
        return self.encode
    
    def getEvaluationMetrics(self):
        return {'privmetric1':self.evalPrivacyMetric1, 'privmetric2':self.evalPrivacyMetric2, 'privmetric3':self.evalPrivacyMetric3, 'utilmetric1':self.evalUtilityMetric1, 'utilmetric2':self.evalUtilityMetric2, 'utilmetric3':self.evalUtilityMetric3}