## The utility functions can be created by the data users
## The functions can be included in the request for synthetic data
## If such a function is included, the system tests the utility of the synthetic data before releasing it to the user 
## The system ossibly adapts the generator by increasing the utility within the boundries of the privacy level

## Each utility funcitons should return true if the utility is satisfied and take a synthetic dataset (dataframe) as input

def utility(data):
    print("Testing utility of synthetic data...")
    return True