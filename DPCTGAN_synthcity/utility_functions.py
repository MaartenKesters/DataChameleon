## The utility functions can be created by the data users
## The functions can be included in the request for synthetic data
## If such a function is included, the system tests the utility of the synthetic data before releasing it to the user 
## The system possibly adapts the generator by increasing the utility within the boundries of the privacy level

## Each utility funcitons should return true if the utility is satisfied and take a synthetic dataset (dataframe) as input

from synthcity.metrics.eval_statistical import InverseKLDivergence, KolmogorovSmirnovTest

import numpy as np

def utility(data):
    print("Testing utility of synthetic data...")
    return True

def inverseKLDivergenceMetric(real_data, syn_data, error):
    evaluator = InverseKLDivergence()

    score = evaluator.evaluate(real_data, syn_data).get('marginal')
    print('Inverse kL divergence: ' + str(score))
    if score > (0.75 - error):
        return True
    else:
        return False
    
def kolmogorovSmirnovTestMetric(real_data, syn_data, error):
    evaluator = KolmogorovSmirnovTest()

    score = evaluator.evaluate(real_data, syn_data)
    print('Kolmogorov Smirnov Test: ' + str(score.get('marginal')))
    if score.get('marginal') > (0.75 - error):
        return True
    else:
        return False