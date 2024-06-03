from controller import Controller
from metrics.privacyMetrics import NearestNeighborDistance, Identifiability, SinglingOut
from metrics.utilityMetrics import InverseKLDivergenceMetric
from generative_model_classes.dpgan import DPGANPlugin
from generative_model_classes.ctganPlugin import CTGANPlugin
from generative_model_classes.adsgan import AdsGANPlugin
from generative_model_classes.ctgansdv import CTGANSDV

from timeit import default_timer as timer
import psutil

class SystemValidator():
    """
    Class to perform the validation of the system.
    
    """

    def __init__(self):
        self.nn_metric = NearestNeighborDistance()
        self.identify_metric = Identifiability()
        self.singling_out_metric = SinglingOut()
        self.kl_metric = InverseKLDivergenceMetric()
    
    #### R1: Self-adaptive requirement ####
    def validate_R1(self, controller: Controller):
        fraud_detection = DPGANPlugin(epsilon=8)
        personalized_marketing = DPGANPlugin(epsilon=6)
        inventory_management = DPGANPlugin(epsilon=2)
        trend_analysis_DPGAN = DPGANPlugin(epsilon=0.5)

        controller.add_generator(generator=fraud_detection, protection_name="Fraud detection")
        controller.add_generator(generator=personalized_marketing, protection_name="Personalized marketing")
        controller.add_generator(generator=inventory_management, protection_name="Inventory management")
        controller.add_generator(generator=trend_analysis_DPGAN, protection_name="Trend analysis")

        ## Show protection levels (Note: this should not be shown to the data consumers as this might reveal too much information about the system.)
        controller.show_protection_levels()

        ## start time to measure response time
        start = timer()

        ## Generate synthetic data for a use case with specific privacy/utility requirements
        syn = controller.request_synthetic_data(size=1000, protection_name="Fraud detection", privacy=None, utility=(self.kl_metric, 0.75), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Personalized marketing", privacy=(self.identify_metric, 0.15), utility=(self.kl_metric, 0.6), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Inventory management", privacy=(self.identify_metric, 0.1), utility=(self.kl_metric, 0.4), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Trend analysis", privacy=(self.identify_metric, 0.05), utility=None, range=0.05)

        ## end time to measure response time
        end = timer()
        print('Response time: ' + str(end - start))

        ## Extra number of generators
        gen1 = CTGANPlugin(batch_size=50)
        gen2 = CTGANPlugin(batch_size=200)
        gen3 = DPGANPlugin(epsilon=1)
        gen4 = DPGANPlugin(epsilon=2)

        controller.add_generator(generator=gen1, protection_name="Gen 1")
        controller.add_generator(generator=gen2, protection_name="Gen 2")
        controller.add_generator(generator=gen3, protection_name="Gen 3")
        controller.add_generator(generator=gen4, protection_name="Gen 4")

        ## start time to measure response time
        start = timer()

        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.identify_metric, 0.1), utility=(self.kl_metric, 0.6), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.nn_metric, 0.5), utility=(self.kl_metric, 0.5), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.nn_metric, 0.3), utility=(self.kl_metric, 0.5), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=None, utility=(self.kl_metric, 0.8), range=0.05)

        ## end time to measure response time
        end = timer()
        print('Response time: ' + str(end - start))

        ## Print received data
        print(syn)

    #### R2: Extensible requirement ####
    def validate_R2(self, controller: Controller): 
        extended_gen = CTGANSDV(discrete_columns=['Description', 'Country'])
        controller.add_generator(generator=extended_gen, protection_name="extended")

        ## Check the working of new generator
        controller.show_protection_levels()
        syn = extended_gen.generate(count=1000)
        print(syn)

    #### R3: Cost-effective requirement ####
    def validate_R3(self, controller: Controller):
        fraud_detection = DPGANPlugin(epsilon=0.5)
        personalized_marketing = DPGANPlugin(epsilon=4)
        inventory_management = DPGANPlugin(epsilon=8)
        trend_analysis_DPGAN = DPGANPlugin(epsilon=10)

        controller.add_generator(generator=fraud_detection, protection_name="Fraud detection")
        controller.add_generator(generator=personalized_marketing, protection_name="Personalized marketing")
        controller.add_generator(generator=inventory_management, protection_name="Inventory management")
        controller.add_generator(generator=trend_analysis_DPGAN, protection_name="Trend analysis")

        ## Generate synthetic data for a use case with specific privacy/utility requirements
        syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.identify_metric, 0.05), utility=None, range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.identify_metric, 0.2), utility=(self.kl_metric, 0.6), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(self.identify_metric, 0.3), utility=(self.kl_metric, 0.7), range=0.05)
        # syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=None, utility=(self.kl_metric, 0.8), range=0.05)

        print('The CPU usage is: ', psutil.cpu_percent(1))

        ## Print received data
        print(syn)