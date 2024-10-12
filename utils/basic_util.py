import numpy as np
import random
from matplotlib import pyplot as plt

def binary_patterns(no_binary_patterns):
    """
    Input: Number of binary patterns needed (e.g. 4).
    Output: A list containing lists of possible patterns (e.g. [[0,0],[0,1],[1,0],[1,1]]).
    """
    return [list(np.int_(list(bin(decimal_no)[2:].zfill(no_binary_patterns)))) for decimal_no in range(2**no_binary_patterns)]

def prob_vector(no_outcomes):
    """
    Input: Number of possible outcomes (e.g. 4).
    Output: A numpy array with randomly generated probabilities for each outcome. (e.g. [.2, .4, .1, .1]).
    """
    probs = np.random.uniform(0,1,no_outcomes)
    probs /= np.sum(probs)
    return probs

def sample_distributions_for_fixed_strategies(strategies = None, no_probs_for_strategies = 1, memory_size = 0, no_policies = 1, all_deterministic = False):
    """
    Input: 
    Strategies, if there is a preferred one.
    Number of distributions over strategies needed.
    Memory size (e.g. 2).
    Number of policies to be used for getting strategies, in the case it is not provided. 
    Output: 
    A list of strategies. Each strategy is a list with its elements denoting the probability of choosing 1, given the state corresponding to its index.
    An array where each element is a probability distribution over strategies.
    """
    if all_deterministic is True:
        random_nos = random.sample(range(2**(2*memory_size)), no_policies) #without replacement
        strategies = [list(np.int_(list(bin(random_no)[2:].zfill(2**(2*memory_size))))) for random_no in random_nos] if strategies is None else strategies
    else:
        strategies = np.random.uniform(0,1,[no_policies,2**(2*memory_size)]) if strategies is None else strategies
    probs_for_strategies_list = []
    for _ in range(no_probs_for_strategies):

        # probs_for_strategies_list.append(prob_vector(len(strategies))) # changed to Dirichlet case temporarily
        probs_for_strategies_list.append(np.random.dirichlet(prob_vector(len(strategies))))
    return strategies, probs_for_strategies_list

def plot_agent_comparison(mean_smart_entropy_progress, mean_naive_entropy_progress, learnability_index, learnability, plot_choice):
    if plot_choice == 0:
        # Average entropy
        # plt.figure()
        color_list = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
        plt.plot(mean_smart_entropy_progress, color_list[learnability_index], label = f'Conditional entropy, learnability {learnability[learnability_index]}')
        plt.plot(mean_naive_entropy_progress, color_list[learnability_index]+'-.', label = f'Random sampling, learnability {learnability[learnability_index]}')
        plt.xlabel('time/games')
        plt.ylabel('Average entropy')
        plt.legend()
        plt.title('Comparing different experimental strategies')
        # plt.show()

    elif plot_choice == 1:
        # # Log of average entropy
        # # plt.figure()
        color_list = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
        plt.plot(np.log(mean_smart_entropy_progress), color_list[learnability_index], label = f'Conditional entropy, learnability {learnability[learnability_index]}')
        plt.plot(np.log(mean_naive_entropy_progress), color_list[learnability_index]+'-.', label = f'Random sampling, learnability {learnability[learnability_index]}')
        plt.xlabel('time/games')
        plt.ylabel('Log of average entropy')
        plt.legend()
        plt.title('Comparing different experimental strategies')
        # # plt.show()

        
    elif plot_choice == 2:        
        # # Difference of log of average entropies
        # # plt.figure()
        color_list = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
        plt.plot(np.log(mean_smart_entropy_progress)-np.log(mean_naive_entropy_progress), color_list[learnability_index], label = f'Conditional entropy, learnability {learnability[learnability_index]}')
        plt.xlabel('time/games')
        plt.ylabel('log(smart avg entropy"/"naive avg entropy)')
        plt.legend()
        plt.title('Comparing different experimental strategies')
        # # plt.show()

    elif plot_choice == 3:
        # # Difference of average entropies
        # # plt.figure()
        color_list = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
        plt.plot(mean_smart_entropy_progress-mean_naive_entropy_progress, color_list[learnability_index], label = f'Conditional entropy, learnability {learnability[learnability_index]}')
        plt.xlabel('time/games')
        plt.ylabel('smart avg entropy-naive avg entropy')
        plt.legend()
        plt.title('Comparing different experimental strategies')
        # # plt.show()
