import os
import pandas as pd
import numpy as np

from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score

from controller import Controller
from privacyMetrics import NearestNeighborDistance, DataLeakage
from utilityMetrics import InverseKLDivergenceMetric
from dpgan import DPGANPlugin
from adsgan import AdsGANPlugin
from ctganPlugin import CTGANPlugin

from ucimlrepo import fetch_ucirepo

def main():
    controller = Controller()

    controller.handleConfigs()

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
    # fetch dataset 
    online_retail = fetch_ucirepo(id=352) 
    X = online_retail.data.features
    X = X.dropna()
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

    ## Take existing metric
    privacy_metric = NearestNeighborDistance()
    utility_metric = InverseKLDivergenceMetric()

    ## Create new baseline generators
    ## Option 1
    gen1 = DPGANPlugin(epsilon=0.5)
    gen2 = DPGANPlugin(epsilon=1)
    gen3 = DPGANPlugin(epsilon=4)
    #gen4 = CTGANPlugin(discrete_columns=['Description', 'Country'])
    controller.add_generator(generator=gen1, protection_name="Level 1")
    controller.add_generator(generator=gen2, protection_name="Level 2")
    controller.add_generator(generator=gen3, protection_name="Level 3")
    #controller.add_generator(generator=gen4, protection_name="Level 4")
    ## Option 2
    # controller.create_generator(protection_name="Level 4", privacy=(privacy_metric, 0.5), utility=(utility_metric, 0.5), range=0.1)
    # controller.create_generator(protection_name="Level 5", privacy=(privacy_metric, 0.6), utility=(utility_metric, 0.6), range=0.1)
    ## Option 3
    # controller.create_by_merging(protection_name="Level 6", privacy=(privacy_metric, 0.5), utility=(utility_metric, 0.5), range=0.1)

    print("##########")
    print("OPERATION PHASE")
    print('##########')

    ## Show protection levels (Note: this should not be shown to the data consumers as this might reveal too much information about the system.)
    controller.show_protection_levels()

    ## Add custom metric that is not available in the system
    

    ## Generate synthetic data for a use case with specific privacy/utility requirements
    # syn = controller.request_synthetic_data(size=1000, protection_name="Level 10", privacy=(privacy_metric, 0.4), utility=(utility_metric, 0.4), range=0.1)
    syn = controller.request_synthetic_data(size=1000, protection_name="custom", privacy=(privacy_metric, 0.2), utility=(utility_metric, 0.34), range=0.1)
    # syn = controller.request_synthetic_data(size=1000, protection_level=controller.create_protection_level(protection_name="Level 7", privacy=(privacy_metric, 0.4), utility=(utility_metric, 0.4), range=0.1))

    print(syn)


if __name__ == '__main__':
    main()
