import numpy as np

class InfoModel():
    def __init__(self, strategies, probs_for_strategies, states, memory = None) -> None:
        self.strategies = strategies
        self.probs_for_strategies = probs_for_strategies
        self.states = states
        self.memory = memory

    def compute_cond_entropy(self, current_exp_action):
        random_variable_list = [[current_agent_action, future_agent_action, strategy_ind] for current_agent_action in range(2) for future_agent_action in range(2) for strategy_ind in range(len(self.strategies))]
        joint_prob = np.zeros((2,2,len(self.strategies)))
        for [current_agent_action, future_agent_action, strategy_ind] in random_variable_list:
            joint_prob[current_agent_action, future_agent_action, strategy_ind] = self.compute_joint_prob(current_agent_action, future_agent_action, strategy_ind, current_exp_action) 
        marginal_prob = np.sum(joint_prob, axis=2)
        cond_entropy = 0
        for [current_agent_action, future_agent_action, strategy_ind] in random_variable_list:
            if joint_prob[current_agent_action, future_agent_action, strategy_ind] != 0:
                cond_entropy += - joint_prob[current_agent_action, future_agent_action, strategy_ind] * np.log(joint_prob[current_agent_action, future_agent_action, strategy_ind]/marginal_prob[current_agent_action, future_agent_action])
        return cond_entropy

    def compute_joint_prob(self, current_agent_action, future_agent_action, strategy_ind, current_exp_action):
        joint_prob = self.find_prob_agent_action(future_agent_action, np.concatenate((self.memory[2:], current_agent_action*np.ones(1),current_exp_action*np.ones(1))), strategy_ind)
        joint_prob *= self.find_prob_agent_action(current_agent_action, self.memory, strategy_ind)
        joint_prob *= self.probs_for_strategies[strategy_ind]
        return joint_prob

    def find_best_action(self):
        cond_entropies = [self.compute_cond_entropy(current_exp_action) for current_exp_action in range(2)]
        return cond_entropies.index(min(cond_entropies))

    def post_game_update(self, current_agent_action, current_exp_action):
        self.probs_for_strategies = np.array([self.find_prob_agent_action(current_agent_action, self.memory, strategy_ind) * self.probs_for_strategies[strategy_ind] for strategy_ind in range(len(self.strategies))])
        self.probs_for_strategies /= np.sum(self.probs_for_strategies)
        self.memory = np.concatenate((self.memory[2:], current_agent_action * np.ones(1), current_exp_action * np.ones(1)))
        
    def find_prob_agent_action(self, action, state, strategy_ind):
        strategy = self.strategies[strategy_ind]
        state_index = self.states.index([int(state_elt) for state_elt in list(state)])
        prob = action * strategy[state_index] + (1-action) * (1-strategy[state_index])
        return prob