import numpy as np
from matplotlib import pyplot as plt
from utils.basic_util import sample_distributions_for_fixed_strategies, plot_agent_comparison
from utils.prisoners_dilemma_util import play_multiple_sessions, increasingly_simple_distributions_for_fixed_strategies, compute_weighted_sum_of_KL_for_distributions, compute_entropy_for_distributions

agent_memory_size =3
no_policies = 5
all_deterministic = False
total_time = 60
total_no_sessions = 200
no_probs_for_strategies = 4

# # To plot using probabibility of correct as the learnability score
# strategies, probs_for_strategies_list = sample_distributions_for_fixed_strategies(strategies = None, no_probs_for_strategies = no_probs_for_strategies, memory_size = agent_memory_size, no_policies = no_policies, all_deterministic = all_deterministic)
# strategies, probs_for_strategies_list, learnability_for_naive, mean_entropy_progress = increasingly_simple_distributions_for_fixed_strategies(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, random_sampling = True, learnability = None, total_no_sessions = total_no_sessions, total_time = total_time)
# _, _, weighted_sum_of_KL_values = compute_weighted_sum_of_KL_for_distributions(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, need_increasing = False)
# print(f' probability of correct for naive  is {learnability_for_naive}\n')
# print(f' KL measure is {weighted_sum_of_KL_values}\n')
# learnability_for_smart = []
# for ind in range(no_probs_for_strategies):
#     mean_smart_entropy_progress, mean_smart_prob_of_correct = play_multiple_sessions(strategies, probs_for_strategies_list[ind], total_no_sessions = total_no_sessions, total_time = total_time)
#     learnability_for_smart.append(mean_smart_prob_of_correct)
#     mean_naive_entropy_progress = mean_entropy_progress[ind]
#     plot_agent_comparison(mean_smart_entropy_progress, mean_naive_entropy_progress, ind, learnability_for_naive)
# plt.show()
# print(f' probability of correct for smart is {learnability_for_smart}\n')

# # To plot using KL metric as the learnability score
# strategies, probs_for_strategies_list = sample_distributions_for_fixed_strategies(strategies = None, no_probs_for_strategies = no_probs_for_strategies, memory_size = agent_memory_size, no_policies = no_policies, all_deterministic = all_deterministic)
# strategies, probs_for_strategies_list, weighted_sum_of_KL_values = compute_weighted_sum_of_KL_for_distributions(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, need_increasing = True)

# # To plot using entropy of strategy as the learnability score
strategies, probs_for_strategies_list = sample_distributions_for_fixed_strategies(strategies = None, no_probs_for_strategies = no_probs_for_strategies, memory_size = agent_memory_size, no_policies = no_policies, all_deterministic = all_deterministic)
strategies, probs_for_strategies_list, entropy_values = compute_entropy_for_distributions(strategies = strategies, probs_for_strategies_list = probs_for_strategies_list, need_increasing = True)

mean_smart_entropy_progress_list, mean_naive_entropy_progress_list = [], []
for ind in range(no_probs_for_strategies):
    mean_smart_entropy_progress, mean_smart_prob_of_correct = play_multiple_sessions(strategies, probs_for_strategies_list[ind], total_no_sessions = total_no_sessions, total_time = total_time)
    mean_naive_entropy_progress, mean_naive_prob_of_correct = play_multiple_sessions(strategies, probs_for_strategies_list[ind], total_no_sessions = total_no_sessions, total_time = total_time, random_sampling = True)
    mean_smart_entropy_progress_list.append(mean_smart_entropy_progress)
    mean_naive_entropy_progress_list.append(mean_naive_entropy_progress)    

for plot_choice in [0,3]:
    plt.figure()
    for ind in range(no_probs_for_strategies):
        plot_agent_comparison(mean_smart_entropy_progress_list[ind], mean_naive_entropy_progress_list[ind], ind, entropy_values, plot_choice)
plt.show()