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

from chameleon import Chameleon
from dpctgan import DPCTGANPlugin
from user import User
from privacyLevel import PrivacyLevels

import utility_functions
import privacy_functions

from ucimlrepo import fetch_ucirepo

def main():
    chameleon = Chameleon()

    chameleon.handleConfigs()
    chameleon.handleSynDataRequirements()

    chameleon.add_user("User1", True)
    chameleon.add_user("User2", False, PrivacyLevels.LOW)
    chameleon.add_user("User3", False, PrivacyLevels.SECRET)

    ## Create baseline models for chameleon
    model1 = DPCTGANPlugin(privacy_level=PrivacyLevels.LOW)
    model2 = DPCTGANPlugin(privacy_level=PrivacyLevels.MEDIUM)
    model3 = DPCTGANPlugin(privacy_level=PrivacyLevels.HIGH)
    model4 = DPCTGANPlugin(privacy_level=PrivacyLevels.SECRET)
    chameleon.add_generator(model1)
    chameleon.add_generator(model2)
    chameleon.add_generator(model3)
    chameleon.add_generator(model4)

    cwd = os.getcwd()


    ## Get the dataset from the user
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
    # data = data.replace('?', '-1')
    # data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    # print(data.head())

    ## Get a update dataset, data that is new when generators are already trained
    # update_data = csv.sample(n=500,replace="False")
    # update_data = update_data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    # update_data = update_data.replace('?', '-1')
    # update_data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = update_data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    

    ## Get the adult dataset
    # adult = fetch_ucirepo(id=2)
    # X = adult.data.features 
    # data = X.sample(n=1000)

    # data2 = X.sample(n=1000)

    # sim = jaccard_score(data, data2)
    # print(sim)


    ## Get the iris dataset
    data, y = load_iris(as_frame=True, return_X_y=True)
    data, test = train_test_split(data, test_size=0.2)
    
    print(data)


    ## Get the sensitive columns from the user
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
    data_loader = chameleon.load_real_data(data, sensitive_features=sensitive_features)

    ## Train baseline models of chameleon
    chameleon.train_generators()

    ## Get the requirements for the synthetic data from the user
    # print(" ")
    # print('Now, indicate the privacy requirements for the synthetic data. \nThe privacy metric that is used to specify the requirements is: ' + chameleon.get_privacy_level_metric().name() + '\n-- info about the metric: ' + chameleon.get_privacy_level_metric().info() + '\nThese are the border values between each privacy level: \n - LOW-MEDIUM: ' + str(chameleon.get_privacy_level_metric().borders()[0]) + '\n - MEDIUM-HIGH: ' + str(chameleon.get_privacy_level_metric().borders()[1]) + '\n - HIGH-SECRET: ' + str(chameleon.get_privacy_level_metric().borders()[2]))
    # requirement_value = input('Enter the value (achieved with the above metric) of the required privacy: ')
    # requirement_level = chameleon.get_requirement_level(float(requirement_value))


    ## Generate synthetic data

    # syn_data = chameleon.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000, privacy_func=privacy_functions.nearestNeighborDistanceMetric)
    # print(syn_data)

    # syn_data = chameleon.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000, utility_func=utility_functions.inverseKLDivergenceMetric)
    # print(syn_data)

    syn_data = chameleon.generate_synthetic_data("User2", 1000)
    print(syn_data)

    # syn_data = chameleon.generate_synthetic_data("User2", PrivacyLevels.MEDIUM, 1000)
    # print(syn_data)


if __name__ == '__main__':
    main()
