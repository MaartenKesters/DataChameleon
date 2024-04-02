## The privacy functions can be created by the data users
## The functions can be included in the request for synthetic data
## If such a function is included, the system tries to meet the privacy requirements by adapting the synthetic data before releasing it to the user

## Each privacy funcitons should take as input: 
## - a dataloader with the real dataset (dataframe) 
## - a dataloader with the synthetic dataset (dataframe) 
## - an error value that is used when the initial requirement can not be reached
## Return true if the privacy is satisfied

from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance
from synthcity.metrics.eval_privacy import IdentifiabilityScore, DeltaPresence
from synthcity.metrics.eval_attacks import DataLeakageMLP

def privacy(real_data, syn_data, error):
    print("Testing privacy of synthetic data...")
    return True

def nearestNeighborDistanceMetric(real_data, syn_data, error):
    evaluator = NearestSyntheticNeighborDistance()
    dist = evaluator.evaluate(real_data, syn_data).get('mean')
    print('Nearest neighbor distance: ' + str(dist))
    if dist > (0.5 - error):
        return True
    else:
        return False
    
def identifiabilityScoreMetric(real_data, syn_data, error):
    evaluator = IdentifiabilityScore()
    score = evaluator.evaluate(real_data, syn_data)
    print('Identifiability score: ' + str(score))
    # if score > 0.2:
    #     return True
    # else:
    #     return False
    
def deltaPresenceMetric(real_data, syn_data, error):
    evaluator = DeltaPresence()
    score = evaluator.evaluate(real_data, syn_data)
    print(score)
    if score > 0.2:
        return True
    else:
        return False
    
def dataLeakageMLP(real_data, syn_data, error):
    evaluator = DataLeakageMLP()
    score = evaluator.evaluate(real_data, syn_data)
    print(score)
    if score > 0.2:
        return True
    else:
        return False