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
from protectionLevel import ProtectionLevel
from privacyMetrics import NearestNeighborDistance
from utilityMetrics import InverseKLDivergenceMetric
from dpgan import DPGANPlugin

from ucimlrepo import fetch_ucirepo

def main():
    controller = Controller()

    controller.handleConfigs()

    print("##########")
    print("TRAINING PHASE")
    print('##########')

    privacy_metric = NearestNeighborDistance()
    utility_metric = InverseKLDivergenceMetric()

    ## Get the dataset
    cwd = os.getcwd()

    # correct_file = False
    # while not correct_file:
    #     file_name = input('Enter a file name: ')
    #     path = cwd + '/data/' + file_name
    #     print(path)

    #     if os.path.exists(cwd + '/data/' + file_name):
    #         print('The file exists')
    #         data = pd.read_csv(path)
    #         print(data.head())
    #         correct_file = True
    #     else:
    #         print('The specified file does NOT exist, make sure to include it in the data folder')


    ## Get kag_risk_factors_cervical_cancer.csv dataset
    # csv = pd.read_csv(cwd + '/data/kag_risk_factors_cervical_cancer.csv')
    # data = csv.sample(n=5000,replace="False")
    
    ## Preprocess the data
    # data = data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    # data.drop(data[data.apply(lambda x: '?' in x.values, axis=1)].index, inplace=True)
    # data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    # data = data.sample(n=500)

    ## Get a update dataset, data that is new when generators are already trained
    # update_data = csv.sample(n=500,replace="False")
    # update_data = update_data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    # update_data = update_data.replace('?', '-1')
    # update_data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = update_data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    

    ## Get the adult dataset
    # adult = fetch_ucirepo(id=2)
    # X = adult.data.features 
    # data = X.sample(n=1000)
    # data.drop(data[data.apply(lambda x: '?' in x.values, axis=1)].index, inplace=True)

    ## Get the iris dataset
    data, y = load_iris(as_frame=True, return_X_y=True)
    data, test = train_test_split(data, test_size=0.2)

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
    # chameleon.encode_data(data_loader.dataframe())

    ## Show available metrics
    controller.show_metrics()

    ## Create new protection levels for different use cases
    # level1 = chameleon.create_protection_level("Level 1", privacy_metric=privacy_metric, privacy_val=0.4, utility_metric=utility_metric, utility_val=0.4, range=0.1)
    level2 = controller.create_protection_level("Level 2", epsilon=3.0)
    level3 = controller.create_protection_level("Level 3", epsilon=2.0)
    level4 = controller.create_protection_level("Level 4", epsilon=1.0)
    level5 = controller.create_protection_level("Level 5", epsilon=0.5)

    ## Create new baseline generators
    controller.create_generator(protection_level=level2)
    controller.create_generator(protection_level=level3)
    controller.create_generator(protection_level=level4)
    controller.create_generator(protection_level=level5)
    # chameleon.add_generator(generator=DPGANPlugin(epsilon=1.0), protection_level=level3)

    print("##########")
    print("OPERATIONS PHASE")
    print('##########')

    ## Show protection levels available for a use case
    controller.show_protection_levels()

    ## Generate synthetic data for a use case with protection level
    syn = controller.generate_synthetic_data(size=1000, protection_level=level2)
    syn = controller.generate_synthetic_data(size=1000, protection_level=controller.create_protection_level('Level 6', privacy_metric=privacy_metric, privacy_val=0.4, utility_metric=utility_metric, utility_val=0.4, range=0.1))

    print(syn)


if __name__ == '__main__':
    main()
