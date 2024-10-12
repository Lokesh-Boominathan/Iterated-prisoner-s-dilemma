import numpy as np
from utils.basic_util import binary_patterns

class Agent:
    def __init__(self, memory_size = 0, prefernce_for_1 = 0.5, states = None,  policy = None) -> None:
        self.memory_size = memory_size # Number of previous games used in decision making
        self.buffer = [] # Agent's memory buffer. Data points on the right is more recent than ones on the left.
        # Note buffer storage is [agent_action_t-(memory_size-1), exp_action_t-(memory_size-1), agent_action_t-(memory_size-2), exp_action_t-(memory_size-2),.... ,agent_action_t, exp_action_t]
        self.prefernce_for_1 = prefernce_for_1 # Agent's preference to choose action 1
        self.states = binary_patterns(2*self.memory_size) if states is None else states #list of all possible values buffer takes.
        self.policy = self.random_policy() if policy is None else policy
        self.previous_agent_action = None

    def random_policy(self):
        return np.random.uniform(0,1,len(self.states))
    
    def agent_choose(self, time_step, previous_exp_action = None):
        if time_step == 0:
            agent_action = self.random_action()
        elif time_step < self.memory_size:
            self.buffer.append(int(self.previous_agent_action))
            self.buffer.append(int(previous_exp_action))
            agent_action = self.random_action()
        elif time_step == self.memory_size:
            self.buffer.append(int(self.previous_agent_action))
            self.buffer.append(int(previous_exp_action))
            agent_action = self.sample_action()
        else:
            if len(self.buffer) != 0:
                self.buffer.pop(0)
                self.buffer.pop(0)
                self.buffer.append(int(self.previous_agent_action)) #Note integer
                self.buffer.append(int(previous_exp_action))
            agent_action = self.sample_action()
        self.previous_agent_action = agent_action
        return agent_action
        
    def sample_action(self):
        if len(self.buffer) != 0:
            return np.random.binomial(size=1, n=1, p= self.policy[self.states.index(self.buffer)])[0]
        else:
            return self.random_action()

    def random_action(self):
        return np.random.binomial(size=1, n=1, p= self.prefernce_for_1)[0]