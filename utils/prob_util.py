import numpy as np

def compute_entropy(prob):
    """
    Input: Probability distribution (e.g. [.2, .4, .1, .1]]).
    Output: 
    Corresponding entropy
    A list of strategies. Each strategy is a list with its elements denoting the probability of choosing 1, given the state corresponding to its index.
    A numpy array with elements denoting the probabilities of choosing a strategy corresponding to its index. 
    """
    entropy = 0
    for prob_val in prob:
        if prob_val != 0:
            entropy += - prob_val * np.log(prob_val)
    return entropy