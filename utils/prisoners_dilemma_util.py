import numpy as np
from prisoners_dilemma import Prisoners_Dilemma
from matplotlib import pyplot as plt
from utils.prob_util import compute_entropy

def compute_KL_divergence(prob1, prob2):
    """
    Input: Two probability distributions (e.g. ([.2, .8], [.9, .1])).
    Output: KL divergence between those distributions. (e.g. 3.2)
    """
    if len(prob1) != len(prob2):
        raise Exception('Both probabilities must have the same length.\n')
    return np.sum([prob1[ind] * np.log(prob1[ind]/prob2[ind]) for ind in range(len(prob1))])

def compute_weighted_sum_of_KL(strategies, probs_for_strategies):
    """
    Input: 
    A list of strategies. Each strategy is a list with its elements denoting the probability of choosing 1, given the state corresponding to its index.
    A numpy array with elements denoting the probabilities of choosing a strategy corresponding to its index. 
    Output: Weighted sum of KL divergences between the different strategies. Weight is the probability of the strategy with respect to which the KL divergence is being calculated.
    """
    learnability = 0
    for strategy_ind1 in range(len(strategies)):
        sum_val = 0
        for strategy_ind2 in range(len(strategies)):
            if strategy_ind2 != strategy_ind1:
                for state_ind in range(len(strategies[0])):
                    sum_val += compute_KL_divergence([strategies[strategy_ind1][state_ind],1-strategies[strategy_ind1][state_ind]], [strategies[strategy_ind2][state_ind], 1-strategies[strategy_ind2][state_ind]])
        learnability += probs_for_strategies[strategy_ind1] * sum_val
    learnability *= 1/(len(strategies[0]))
    return learnability

def play_multiple_sessions(strategies, probs_for_strategies, total_no_sessions = 100, total_time = 100, random_sampling = False):
    entropy_progress_lists = []
    prob_of_correct_list = []
    for _ in range(total_no_sessions):
        session = Prisoners_Dilemma(strategies = strategies, probs_for_strategies = probs_for_strategies, random_sampling = random_sampling, total_time = total_time, do_visualize = False)
        entropy_progress_list, prob_of_correct = session.play_prisoners_dilemma()
        entropy_progress_lists.append(entropy_progress_list)
        prob_of_correct_list.append(prob_of_correct)
    mean_entropy_progress  = np.mean(entropy_progress_lists, axis = 0)
    mean_prob_of_correct  = np.mean(prob_of_correct_list)
    return mean_entropy_progress, mean_prob_of_correct

def increasingly_simple_distributions_for_fixed_strategies(strategies, probs_for_strategies_list, random_sampling = True, learnability = None, total_no_sessions = 100, total_time = 100):
    """
    Input: 
    Number of distributions over strategies needed.
    Agent memory size.
    Number of policies to be used for getting strategies, in the case it is not provided. 
    Strategies, if there is a preferred one.
    Order, in the increasing order of learnability (ease) if computed through other measures.
    Output: 
    Strategies.
    An array where each element is a probability distribution over strategies.
    An array of corresponding learnability values (by default measured as 'probability of being correct').
    Mean entropy progress (for random sampling experimentalist in the default case).
    """
    if learnability is None:
        mean_entropy_progress_list = []
        mean_prob_of_correct_list = []
        for ind in range(len(probs_for_strategies_list)):
            mean_entropy_progress, mean_prob_of_correct = play_multiple_sessions(strategies, probs_for_strategies_list[ind], random_sampling = random_sampling, total_no_sessions = total_no_sessions, total_time = total_time)
            mean_entropy_progress_list.append(mean_entropy_progress)
            mean_prob_of_correct_list.append(mean_prob_of_correct)
        order = np.argsort(np.array(mean_prob_of_correct_list))
        mean_entropy_progress_list = np.array(mean_entropy_progress_list)[order]
        learnability = mean_prob_of_correct_list
    else:
        mean_entropy_progress_list = None
        order = np.argsort(np.array(learnability))
    learnability = np.array(learnability)[order]
    probs_for_strategies_list = np.array(probs_for_strategies_list)[order]
    return strategies, probs_for_strategies_list, learnability, mean_entropy_progress_list

def compute_weighted_sum_of_KL_for_distributions(strategies, probs_for_strategies_list, need_increasing = False):
    """
    Input: 
    Number of distributions over strategies needed.
    Agent memory size.
    Number of policies to be used for getting strategies, in the case it is not provided. 
    Strategies, if there is a preferred one.
    Output: 
    Strategies.
    An array where each element is a probability distribution over strategies.
    An array of corresponding learnability scores.
    """
    weighted_sum_of_KL_values = []
    for ind in range(len(probs_for_strategies_list)):
            weighted_sum_of_KL_values.append(compute_weighted_sum_of_KL(strategies, probs_for_strategies_list[ind]))
    if need_increasing:
        strategies, probs_for_strategies_list, weighted_sum_of_KL_values, _ = increasingly_simple_distributions_for_fixed_strategies(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, learnability = weighted_sum_of_KL_values)
    return strategies, probs_for_strategies_list, weighted_sum_of_KL_values

def compute_entropy_for_distributions(strategies, probs_for_strategies_list, need_increasing = False):
    """
    Input: 
    Number of distributions over strategies needed.
    Agent memory size.
    Number of policies to be used for getting strategies, in the case it is not provided. 
    Strategies, if there is a preferred one.
    Output: 
    Strategies.
    An array where each element is a probability distribution over strategies.
    An array of corresponding learnability scores.
    """
    entropy_values = []
    for ind in range(len(probs_for_strategies_list)):
            entropy_values.append(compute_entropy(probs_for_strategies_list[ind]))
    if need_increasing:
        strategies, probs_for_strategies_list, entropy_values, _ = increasingly_simple_distributions_for_fixed_strategies(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, learnability = entropy_values)
    return strategies, probs_for_strategies_list, entropy_values

def plot_learnability_trends(min_memory_size = 0, max_memory_size = 6, memory_step = 2, min_no_policies = 1, max_no_policies = 5, no_policies_step = 1, averaged_over = 10, fixed_strategies = False, no_probs_for_strategies = 100, memory_size = 3, no_policies = 6, strategies = None):
    if fixed_strategies is False:
        agent_memeory_list = [memory for memory in range(min_memory_size, max_memory_size, memory_step)]
        no_policies_list = [memory for memory in range(min_no_policies, max_no_policies, no_policies_step)]
        learnability_array_mean = np.zeros((len(agent_memeory_list),len(no_policies_list)))
        learnability_array_std = np.zeros((len(agent_memeory_list),len(no_policies_list)))
        for agent_memory_size in agent_memeory_list:
            for no_policies in no_policies_list:
                learnability = []
                for _ in range(averaged_over):
                    strategies, probs_for_strategies = generate_policies(memory_size = agent_memory_size, no_policies = no_policies)
                    learnability.append(compute_agent_learnability(strategies, probs_for_strategies)) 
                learnability_array_mean[agent_memeory_list.index(agent_memory_size), no_policies_list.index(no_policies)] = np.mean(learnability)
                learnability_array_std[agent_memeory_list.index(agent_memory_size), no_policies_list.index(no_policies)] = np.std(learnability)
        
        plt.figure()
        plt.imshow(learnability_array_mean)
        plt.xlabel('Number of policies')
        plt.xticks(range(len(no_policies_list)), labels=['{:g}'.format(v) for v in no_policies_list])
        plt.ylabel('Agent memory size')
        plt.yticks(range(len(agent_memeory_list)), labels = ['{:g}'.format(v) for v in agent_memeory_list])
        plt.colorbar(label='learnability mean')
        plt.title('learnability mean')

        plt.figure()
        plt.imshow(learnability_array_std)
        plt.xlabel('Number of policies')
        plt.xticks(range(len(no_policies_list)), labels=['{:g}'.format(v) for v in no_policies_list])
        plt.ylabel('Agent memory size')
        plt.yticks(range(len(agent_memeory_list)), labels = ['{:g}'.format(v) for v in agent_memeory_list])
        plt.colorbar(label='learnability std')
        plt.title(label='learnability std')

        _, ax = plt.subplots(len(no_policies_list))
        for no_policies_ind in range(len(no_policies_list)):
            ax[no_policies_ind].errorbar(range(len(agent_memeory_list)),learnability_array_mean[:,no_policies_ind],learnability_array_std[:,no_policies_ind])
            ax[no_policies_ind].set(xlabel = 'Memory size', ylabel = 'learnability', title = f'no of policies was {no_policies_list[no_policies_ind]}')
            ax[no_policies_ind].set_xticks(range(len(agent_memeory_list)), labels = ['{:g}'.format(v) for v in agent_memeory_list])
        plt.show()
    else:
        _, probs_for_strategies_list, learnability_list = increasingly_simple_distributions_for_fixed_strategies(no_probs_for_strategies = no_probs_for_strategies, memory_size = memory_size, no_policies = no_policies, strategies = strategies)
        entropy_list = [compute_entropy(probs_for_strategies) for probs_for_strategies in probs_for_strategies_list]
        plt.figure()
        plt.scatter(entropy_list, learnability_list)
        plt.xlabel('Entropy')
        plt.ylabel('learnability')
        plt.title('Entropy Vs learnability')
        plt.show()
    

if __name__ == '__main__':
    plot_learnability_trends(fixed_strategies = True)
    plot_learnability_trends(fixed_strategies = False)