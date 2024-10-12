import numpy as np
from utils.basic_util import binary_patterns
from utils.info_model import InfoModel
# from utils.prisoners_dilemma_util import compute_entropy
from utils.prob_util import compute_entropy

class Experimentalist:
    def __init__(self, assumed_memory_size = 0, strategies = None, probs_for_strategies = None, states = None, random_sampling = False, state_action_hist = None) -> None:
        self.assumed_memory_size = assumed_memory_size 
        self.states = binary_patterns(2*self.assumed_memory_size) if states is None else states
        self.state_action_hist = {ind: np.zeros(2) for ind in range(len(self.states))} if state_action_hist is None else state_action_hist
        self.random_sampling = random_sampling
        self.entropy_progress = []
        self.probs_for_strategies_progress = []
        if strategies is not None:
            self.strategies = strategies
            self.probs_for_strategies = np.ones(len(strategies))/len(strategies) if probs_for_strategies is None else probs_for_strategies
        else:
            self.random_sampling = True
        # States is a list of all possible buffer values of the agent. 
        # State_action_hist is a dictionary with key as the index corresponding to the assumed state space of the agent, and values as the [# times action 0 was taken, # times action 1 was taken]
        self.buffer = [] # Note buffer storage is [agent_action_0, exp_action_0, agent_action_1, exp_action_1,...., agent_action_recent, exp_action_recent]

    def exp_choose(self, time_step, previous_agent_action = None):
        if time_step != 0:
                self.buffer.append(previous_agent_action)
                self.buffer.append(self.previous_exp_action)
        if time_step < self.assumed_memory_size:
            exp_action = self.random_action()
        elif time_step == self.assumed_memory_size:
            self.info_model = InfoModel(self.strategies, self.probs_for_strategies, self.states, memory = self.buffer[-2*self.assumed_memory_size:])
            exp_action = self.random_action()
        else:
            self.info_model.post_game_update(previous_agent_action, self.previous_exp_action)
            self.state_action_hist[self.states.index(self.buffer[-2*(self.assumed_memory_size+1):-2])][self.buffer[-2]] += 1
            exp_action = self.info_model.find_best_action() if self.random_sampling is False else self.random_action()
            self.entropy_progress.append(compute_entropy(self.info_model.probs_for_strategies))
            self.probs_for_strategies_progress.append(self.info_model.probs_for_strategies)
        self.previous_exp_action = exp_action
        return exp_action
 
    def random_action(self):
        return np.random.binomial(size=1, n=1, p= 0.5)[0]

    def estimate_policy_using_hist(self):
        return [(self.state_action_hist[ind][1])/np.sum(self.state_action_hist[ind]) if np.sum(self.state_action_hist[ind])!=0 else -1 for ind in range(len(self.states))]

    def count_state_occurences(self):
        return [np.sum(self.state_action_hist[ind]) for ind in range(len(self.states))]