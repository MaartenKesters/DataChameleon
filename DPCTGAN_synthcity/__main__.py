import os
import pandas as pd

from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.datasets import load_diabetes

from chameleon import Chameleon
from dpctgan import DPCTGANPlugin
from user import User
from privacyLevel import PrivacyLevels


def main():
    chameleon = Chameleon()

    chameleon.add_user("User1", True)
    chameleon.add_user("User2", False, PrivacyLevels.LOW)
    chameleon.add_user("User3", False, PrivacyLevels.HIGH)

    ## Create baseline models for chameleon
    model1 = DPCTGANPlugin(privacy_level=PrivacyLevels.LOW)
    # model2 = DPCTGANPlugin(privacy_level=PrivacyLevels.MEDIUM)
    # model3 = DPCTGANPlugin(privacy_level=PrivacyLevels.HIGH)
    # model4 = DPCTGANPlugin(privacy_level=PrivacyLevels.SECRET)
    chameleon.add_generator(model1)
    # chameleon.add_generator(model2)
    # chameleon.add_generator(model3)
    # chameleon.add_generator(model4)

    cwd = os.getcwd()

    # Get the dataset from the user
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

    data = pd.read_csv(cwd + '/data/kag_risk_factors_cervical_cancer.csv')
    data = data.sample(n=1000,replace="False")
    
    ## Preprocess the data
    data = data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    data = data.replace('?', '-1')
    data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']] = data[['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV']].astype(float)
    print(data.head())

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
    chameleon.load_real_data(data, sensitive_features=sensitive_features)

    ## Train baseline models of chameleon
    chameleon.train_generators()

    ## Get the requirements for the synthetic data from the user
    print('Now, indicate the privacy requirements for the synthetic data. \n The privacy metric that is used to specify the requirements is: ' + chameleon.get_privacy_level_metric().name() + '\n-- info about the metric: ' + chameleon.get_privacy_level_metric().info() + 'These are the border values between each privacy level: \n - LOW-MEDIUM: ' + str(chameleon.get_privacy_level_metric().borders()[0]) + '\n - MEDIUM-HIGH: ' + str(chameleon.get_privacy_level_metric().borders()[1]) + '\n - HIGH-SECRET: ' + str(chameleon.get_privacy_level_metric().borders()[2]))
    requirement_value = input('Enter the value (achieved with the above metric) of the required privacy: ')
    requirement_level = chameleon.get_requirement_level(float(requirement_value))


    ## Generate synthetic data
    # syn_data = chameleon.generate_synthetic_data("User2", PrivacyLevels.LOW, 1000)
    # chameleon.generate_synthetic_data("User3", PrivacyLevels.HIGH, 1000)

    # print(syn_data)

    # print(syn_data.dataframe().dtypes)
    # print(data.dtypes)


    # syn_model = DPCTGANPlugin()

    # cwd = os.getcwd()

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

    # loader = GenericDataLoader(data)
    # print(loader.dataframe())

    # syn_model.fit(loader)
    # print(syn_model.generate(count = 10).dataframe())

    # score = Benchmarks.evaluate(
    #     [("CTGAN", "ctgan", {})],
    #     loader,
    #     synthetic_size=500,
    #     repeats=2,
    #     synthetic_reuse_if_exists=False
    # )

    # Benchmarks.print(score)

    # score = Benchmarks.evaluate(
    #     [("DPCTGAN", "dpctgan", {"epsilon": 4})],
    #     loader,
    #     synthetic_size=500,
    #     repeats=2,
    #     synthetic_reuse_if_exists=False
    # )

    # Benchmarks.print(score)


if __name__ == '__main__':
    main()
