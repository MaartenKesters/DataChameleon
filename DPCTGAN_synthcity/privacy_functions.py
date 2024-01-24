## The privacy functions can be created by the data users
## The functions can be included in the request for synthetic data
## If such a function is included, the system tests the privacy of the synthetic data before releasing it to the user 
## The system possibly adapts the generator to increase or decrease the utility within the boundries of the privacy level

## Each privacy funcitons should return true if the privacy is satisfied and take a dataloader with the real dataset (dataframe) and a dataloader with the synthetic dataset (dataframe) as input

from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance
from synthcity.metrics.eval_privacy import IdentifiabilityScore, DeltaPresence
from nn_adversarial_accuracy import NearestNeighborMetrics
from synthcity.metrics.eval_attacks import DataLeakageMLP

def privacy(real_data, syn_data):
    print("Testing privacy of synthetic data...")
    return True

def nearestNeighborDistanceMetric(real_data, syn_data):
    evaluator = NearestSyntheticNeighborDistance()
    dist = evaluator.evaluate(real_data, syn_data).get('mean')
    print('Nearest neighbor distance: ' + str(dist))
    if dist > 0.5:
        return True
    else:
        return False
    
def identifiabilityScoreMetric(real_data, syn_data):
    evaluator = IdentifiabilityScore()
    score = evaluator.evaluate(real_data, syn_data)
    print('Identifiability score: ' + str(score))
    # if score > 0.2:
    #     return True
    # else:
    #     return False
    
def deltaPresenceMetric(real_data, syn_data):
    evaluator = DeltaPresence()
    score = evaluator.evaluate(real_data, syn_data)
    print(score)
    if score > 0.2:
        return True
    else:
        return False
    
def dataLeakageMLP(real_data, syn_data):
    evaluator = DataLeakageMLP()
    score = evaluator.evaluate(real_data, syn_data)
    print(score)
    if score > 0.2:
        return True
    else:
        return False
    
def nearestNeighborAccuracyMetric(real_data, test, syn_data):
    evaluator = NearestNeighborMetrics(real_data.dataframe(), test, syn_data.dataframe())
    evaluator.compute_nn()
    score = evaluator.compute_adversarial_accuracy()
    print('Nearest neighbor accuracy score: ' + str(score))
    # if score > 0.2:
    #     return True
    # else:
    #     return False