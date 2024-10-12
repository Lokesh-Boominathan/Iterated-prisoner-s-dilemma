import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from experimentalist import Experimentalist
from utils.basic_util import sample_distributions_for_fixed_strategies

class Prisoners_Dilemma():
    def __init__(self, strategies = None, probs_for_strategies = None, agent_memory_size = None, no_policies = None, is_batch = False, random_sampling = False, all_deterministic = False, total_time = 50, do_visualize = True) -> None:
        self.is_batch = is_batch # If it's a batch game, set to True. If sequential, set to False.
        self.random_sampling = random_sampling
        self.total_time =  (total_time - total_time % (self.agent_memory_size + 2)) if (self.is_batch is True and (total_time % (self.agent_memory_size + 2)) != 0) else total_time
        self.do_visualize = do_visualize
        if strategies is None:
            if agent_memory_size is not None:
                self.agent_memory_size = agent_memory_size
            else:
                raise Exception('Please specify agent memory size\n')
            if no_policies is None:
                raise Exception('Please specify number of policies\n')
            print('Generating random policies as they were not provided.\n')
            self.strategies, probs_for_strategies_list = sample_distributions_for_fixed_strategies(no_probs_for_strategies = 1, memory_size = self.agent_memory_size, no_policies = no_policies, all_deterministic = all_deterministic)
            self.probs_for_strategies = probs_for_strategies_list[0]
        else:
            self.strategies = strategies
            self.agent_memory_size = int(np.log2(len(strategies[0]))/2)
            if probs_for_strategies is None:
                print('Assigning all strategies to be equally likely as the distibution over the strategies was not provided.\n')
                self.probs_for_strategies = np.ones(len(strategies))/len(strategies)
            else:
                self.probs_for_strategies = probs_for_strategies


    def play_prisoners_dilemma(self):
        # Agent's policy
        agent_policy_choice = np.random.choice(np.arange(len(self.strategies)), p=self.probs_for_strategies)
        agent_policy = self.strategies[agent_policy_choice]
        assumed_memory_size = self.agent_memory_size
        human = Experimentalist(assumed_memory_size = assumed_memory_size, strategies = self.strategies, probs_for_strategies = self.probs_for_strategies, random_sampling = self.random_sampling)
        exp_action_list = []
        agent_action_list = []
        entropy_progress_list = []
        probs_for_strategies_progress_list = []
        for time in range(self.total_time):
            if self.is_batch is True:
                game_time = time % (self.agent_memory_size + 2)
            else:
                game_time = time
            if game_time == 0:
                monkey = Agent(memory_size = self.agent_memory_size, policy = agent_policy)
                previous_agent_action = None
                previous_exp_action = None
                if time != 0:
                    entropy_progress_list.append(human.entropy_progress)
                    probs_for_strategies_progress_list.append(human.probs_for_strategies_progress)
                    human = Experimentalist(assumed_memory_size = assumed_memory_size, strategies = self.strategies, probs_for_strategies = human.info_model.probs_for_strategies, random_sampling = self.random_sampling, state_action_hist = human.state_action_hist)
            exp_action = human.exp_choose(game_time,previous_agent_action)
            agent_action = monkey.agent_choose(game_time,previous_exp_action)
            exp_action_list.append(exp_action)
            agent_action_list.append(agent_action)
            previous_agent_action = agent_action
            previous_exp_action = exp_action
        agent_policy = monkey.policy
        
        if self.is_batch is False:
            entropy_progress_list = human.entropy_progress
            probs_for_strategies_progress_list = human.probs_for_strategies_progress
        
        if self.do_visualize:
            self.visualize_exp_analysis(human, monkey, entropy_progress_list, probs_for_strategies_progress_list, exp_action_list, agent_action_list, agent_policy_choice, agent_policy)

        prob_of_correct = probs_for_strategies_progress_list[-1][agent_policy_choice]
        
        return entropy_progress_list, prob_of_correct

    def visualize_exp_analysis(self, human, monkey, entropy_progress_list, probs_for_strategies_progress_list, exp_action_list, agent_action_list, agent_policy_choice, agent_policy):

        estimated_policy_using_hist = human.estimate_policy_using_hist()
        estimated_probs_for_strategies = human.info_model.probs_for_strategies
        estimated_mode_policy = human.strategies[np.argmax(estimated_probs_for_strategies)]
        counted_state_occurences = human.count_state_occurences()
    
        plt.figure()
        plt.stem([ind for ind in range(len(agent_action_list))], agent_action_list,  markerfmt='r', linefmt = ':', label='Monkey\'s actions', basefmt=" ")
        plt.stem([ind for ind in range(len(exp_action_list))], exp_action_list, markerfmt='b', linefmt = ':', label='Human\'s actions', basefmt=" ")
        plt.title('Action choices Vs Time')
        plt.xlabel("Time")
        plt.ylabel("Action choice")
        plt.legend()

        plt.figure()
        plt.stem(entropy_progress_list, basefmt="black")
        plt.title('Entropy Vs Time/Games')
        plt.xlabel('Time/Games')
        plt.ylabel('Entropy')

        plt.figure()
        plt.imshow(np.array(probs_for_strategies_progress_list).T)
        plt.title('Evolution of posterior over strategies')
        plt.xlabel('Time/Games')
        plt.ylabel('Strategy index')
        plt.colorbar(label='Probability', shrink = .2)

        if len(monkey.states) == len(human.states):
            _, (ax1, ax2) = plt.subplots(2, 1)
            ax1.stem([ind for ind in range(len(monkey.states))], agent_policy,  markerfmt='r', linefmt = ':', label='Monkey\'s policy', basefmt="black")
            ax1.stem([ind for ind in range(len(human.states))], estimated_policy_using_hist, markerfmt='b', linefmt = ':', label='Policy using histogram', basefmt="black")
            ax1.stem([ind for ind in range(len(human.states))], estimated_mode_policy, markerfmt='g', linefmt = ':', label='Mode of policy posterior', basefmt="black")
            ax1.set(xlabel = "Past index", ylabel = "Prob. of agent choosing 1", title = "Histogram Vs True Policy\n (Negative implies not seen)")
            ax1.legend()
            ax2.stem([ind for ind in range(len(human.states))], counted_state_occurences, linefmt = ':', basefmt="black")
            ax2.set(xlabel = "Past index", ylabel = "Count", title = 'Counts of past occurences')
        else:
            plt.figure()
            plt.stem([ind for ind in range(len(monkey.states))], agent_policy,  markerfmt='r*', linefmt = ':', label='Monkey\'s policy', basefmt="black")
            plt.xlabel("Past index")
            plt.ylabel("Prob. of agent choosing 1")
            plt.legend()
            plt.figure()
            plt.stem([ind for ind in range(len(human.states))], estimated_policy_using_hist, markerfmt='b*', linefmt = ':', label='Histogram policy', basefmt="black")
            plt.xlabel("Past index")
            plt.ylabel("Prob. of agent choosing 1")
            plt.title('Histogram Vs True Policy\n (Negative implies not seen)')
            plt.legend()
            plt.figure()
            plt.stem([ind for ind in range(len(human.states))], counted_state_occurences, linefmt = ':', basefmt="black")
            plt.title('Counts of past occurences')
            plt.xlabel("Past index")
            plt.ylabel("Count")
        plt.show()


        print(f'Estimated probs for strategies is {estimated_probs_for_strategies}\n')
        print(f'Original agent policy choice is {agent_policy_choice}, and estimated policy choice is {np.argmax(estimated_probs_for_strategies)}\n')

        # print('\n___Monkey policy is as follows___\n')
        # for ind in range(len(monkey.states)):
        #     print(f'When past is {monkey.states[ind]}, prob. of choosing 1 is {agent_policy[ind]}')
        # print('\n___Estimated policy is as follows___\n')
        # for ind in range(len(human.states)):
        #     print(f'When past is {human.states[ind]}, prob. of choosing 1 is {estimated_policy_using_hist[ind]}')
        

if __name__ == '__main__':
    game = Prisoners_Dilemma(agent_memory_size = 2, no_policies = 4, total_time = 100)
    game.play_prisoners_dilemma()