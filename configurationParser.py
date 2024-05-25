import configparser

class ConfigHandler():
    """
    Class to handle all configurations. Requirement configs, fine tuning metrics configs, fine tuning process configs

    """
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def parse_configs(self):
        self.parse_generator()
        self.parse_generation_metrics()
        self.parse_generation_technique()
        self.parse_encoding()
        self.parse_evaluation_metrics()

    def parse_generator(self):
        self.pluginModule = self.config.get('Plugin', 'module')
        self.pluginClass = self.config.get('Plugin', 'className')
    
    def parse_generation_metrics(self):
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

    def parse_generation_technique(self):
        self.generationModule = self.config.get('Generation', 'module')
        self.generationClass = self.config.get('Generation', 'className')

    def parse_encoding(self):
        encode = self.config.get('Encoding', 'encode')
        if encode == '1':
            self.encode = True
        else:
            self.encode = False

    def parse_evaluation_metrics(self):
        self.privacy_metrics = self.config['Evaluation']['privacymetrics'].split(', ')
        self.utility_metrics = self.config['Evaluation']['utilitymetrics'].split(', ')
    
    def get_plugin_module(self):
        return self.pluginModule
    
    def get_plugin_class(self):
        return self.pluginClass
    
    def get_privacy_metrics(self):
        return {'metric1':self.privacyMetric1, 'weight1':self.privacyMetric1Weight, 'metric2':self.privacyMetric2, 'weight2':self.privacyMetric2Weight, 'metric3':self.privacyMetric3, 'weight3':self.privacyMetric3Weight}
    
    def get_utility_metrics(self):
        return {'metric1':self.utilityMetric1, 'weight1':self.utilityMetric1Weight, 'metric2':self.utilityMetric2, 'weight2':self.utilityMetric2Weight, 'metric3':self.utilityMetric3, 'weight3':self.utilityMetric3Weight}
    
    def get_generation_module(self):
        return self.generationModule
    
    def get_generation_class(self):
        return self.generationClass
    
    def get_encoding(self):
        return self.encode
    
    def get_evaluation_metrics(self):
        return self.privacy_metrics, self.utility_metrics