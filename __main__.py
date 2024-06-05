import os
import pandas as pd
from timeit import default_timer as timer
import psutil

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from controller import Controller
from metrics.privacyMetrics import NearestNeighborDistance, Identifiability, SinglingOut
from metrics.utilityMetrics import InverseKLDivergenceMetric
from generative_model_classes.dpgan import DPGANPlugin
from generative_model_classes.ctganPlugin import CTGANPlugin
from generative_model_classes.adsgan import AdsGANPlugin
from generative_model_classes.ctgansdv import CTGANSDV
from systemValidator import SystemValidator

def main():
    controller = Controller()
    controller.handle_configs()

    print("##########")
    print("PREPARATION PHASE")
    print('##########')

    ## Get the dataset
    cwd = os.getcwd()

    ## Dataset 1: kag_risk_factors_cervical_cancer.csv dataset
    # csv = pd.read_csv(cwd + '/data/kag_risk_factors_cervical_cancer.csv')
    # data = csv.sample(n=10000,replace="False")
    # data = data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    # data.drop(data[data.apply(lambda x: '?' in x.values, axis=1)].index, inplace=True)
    # data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    # data = data.sample(n=500)

    ## Dataset 2: adult dataset
    # adult = fetch_ucirepo(id=2)
    # X = adult.data.features 
    # data = X.sample(n=1000)
    # data.drop(data[data.apply(lambda x: '?' in x.values, axis=1)].index, inplace=True)

    ## Dataset 3: iris dataset
    # data, y = load_iris(as_frame=True, return_X_y=True)
    # data, test = train_test_split(data, test_size=0.2)

    ## Dataset 4: Online retail 
    online_retail = pd.read_csv(cwd + '/data/Online Retail.csv')
    X = online_retail.dropna()
    data = X.sample(n=1000)
    print(data)

    ## Get the sensitive columns
    print('Give the sensitive colums, one by one. If there are no sensitive colums left, type: /')
    sensitive_features = []
    more_sensitive_features = True
    while more_sensitive_features:
        sensitive = input('Enter a sensitive column: ')
        if sensitive != '/':
            if sensitive in data:
                sensitive_features.append(sensitive)
            else:
                print('This column name does not exist.')
        else:
            more_sensitive_features = False

    ## Create data loader
    controller.load_private_data(data, sensitive_features=sensitive_features)

    ## Show available metrics
    controller.show_metrics()

    ## Load metric
    nn_metric = NearestNeighborDistance()
    identify_metric = Identifiability()
    singling_out_metric = SinglingOut()
    kl_metric = InverseKLDivergenceMetric()

    ## Create new baseline generators

    ## Option 1: manual configuration
    fraud_detection = DPGANPlugin(epsilon=8)
    personalized_marketing = DPGANPlugin(epsilon=6)
    inventory_management = DPGANPlugin(epsilon=2)
    trend_analysis_DPGAN = DPGANPlugin(epsilon=0.5)

    controller.add_generator(generator=fraud_detection, protection_name="Fraud detection")
    controller.add_generator(generator=personalized_marketing, protection_name="Personalized marketing")
    controller.add_generator(generator=inventory_management, protection_name="Inventory management")
    controller.add_generator(generator=trend_analysis_DPGAN, protection_name="Trend analysis")
    
    ## Option 2: automatic configuration
    # controller.create_generator(protection_name="Level 1", privacy=(privacy_metric, 0.5), utility=(utility_metric, 0.5), range=0.1)

    ## Option 3: Merge synthetic data of existing generators
    # controller.create_by_merging(protection_name="Level 2", privacy=(privacy_metric, 0.5), utility=(utility_metric, 0.5), range=0.1)

    print("##########")
    print("OPERATION PHASE")
    print('##########')

    ## Add custom metric that is not available in the system
    # controller.add_privacy_metric(singling_out_metric)

    ## Show protection levels (Note: this should not be shown to the data consumers as this might reveal too much information about the system.)
    controller.show_protection_levels()

    ## Generate synthetic data for a use case with specific privacy/utility requirements
    syn = controller.request_synthetic_data(size=1000, protection_name="Test", privacy=(identify_metric, 0.15), utility=(kl_metric, 0.6), range=0.05)

    ## Print received data
    print(syn)

    #### Validation ####
    validator = SystemValidator()
    # validator.validate_R1(controller)
    # validator.validate_R2(controller)
    # validator.validate_R3(controller)


if __name__ == '__main__':
    main()
