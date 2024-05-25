
# Data Chameleon: A self-adaptive privacy-enhancement system for structured data.
Data Chameleon is a self-adaptive synthetic data generation system designed to facilitate data sharing in collaborative environments. It dynamically adjusts its privacy and utility measures to meet the unique requirements of different use cases, minimizing privacy risks while optimizing data utility.

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Sample usage](#sample-usage)
- [Authors](#authors)

## Features
- Self-adaptive: dynamically adjusts its process to meet varying privacy/utility requirements
- Extensible: Easy to extend with new generative models and metrics.
- Cost-effective: Minimal added cost with respect to response time, storage needs and CPU usage

## Dependencies
- Python 3.8.8
- [Synthcity](https://github.com/vanderschaarlab/synthcity): A library for generating synthetic data using a variety of generative models. Used for the baseline of the generative models in Data Chameleon.
- [Anonymeter](https://github.com/statice/anonymeter): A unified statistical framework to jointly quantify different types of privacy risks in synthetic tabular datasets. Used to include metrics in Data Chameleon.
- [Synthetic Data Vault](https://github.com/sdv-dev/SDV/tree/main): A library designed to be your one-stop shop for creating tabular synthetic data. Used to include generative models in Data Chameleon.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/MaartenKesters/datachameleon.git
   ```
2. Navigate to the project directory:
    ```sh
    cd datachameleon
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Sample usage
You can add baseline generators to the system in the following way:
- Load a private dataset (e.g. data from an online retail store)
    ```sh
    online_retail = pd.read_csv(cwd + '/data/Online Retail.csv')
    X = online_retail.dropna()
    controller.load_private_data(data, sensitive_features=["sensitive features"])
    ```
- Show available privacy/utility metrics in the system
    ```sh
    controller.show_metrics()
    ```
- Load the existing metrics you want to use to specify your requirements
    ```sh
    nn_metric = NearestNeighborDistance()
    identify_metric = Identifiability()
    kl_metric = InverseKLDivergenceMetric()
    ```
- Create baseline generators
    ```sh
    DPGAN_generator = DPGANPlugin(epsilon=0.5)
    controller.add_generator(generator=DPGAN_generator, protection_name="fraud detection")

    controller.create_generator(protection_name="personalized marketing", privacy=(identify_metric, 0.2), utility=(kl_metric, 0.6), range=0.05)

    controller.create_by_merging(protection_name="trend analysis", utility=(kl_metric, 0.8), range=0.05)
    ```

You can request synthetic data with specific privacy/utility requirements in the following way:
- Create a request with the required size, privacy, and utility of the data
    ```sh
    syn = controller.request_synthetic_data(size=1000, protection_name="personalized marketing", privacy=(identify_metric, 0.2), utility=(kl_metric, 0.6), range=0.05)
    ```

## Authors
- [@MaartenKesters](https://www.github.com/MaartenKesters)
- Promoters: Prof. dr. ir. W. Joosen and Dr. D. Van Landuyt
- Assistant: Q. Liao