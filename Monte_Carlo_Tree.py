import numpy as np

# Define the node class
class Node:
    def __init__(self, env_buffer_idx):
        self.ucb_value = 0
        self.n_visits = 0
        self.reward = 0 # reward until that point
        self.q_value = 0 # q value predicted by value netwrok
        self.policy_value = 0 # prior, action prob predicted by policy netwrok
        self.ts_value = 0
        self.env_buffer_idx = env_buffer_idx
        self.children = None
        self.parent = None
    
    def compute_ucb(self):
        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.        
        '''
        
        # Unexplored nodes have maximum values so we favour exploration
        if self.n_visits == 0:
            return float('inf')
        
        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
                        
        value_score = (self.reward / self.n_visits) 
        c = 1.0
            
        prior_score = c * self.policy_value * sqrt(log(top_node.n_visits) / self.n_visits))
        
        # We use both the Value(s) and Policy from the neural network estimations for calculating the node value
        return value_score + prior_score

    def explore_childs(self):
        


    def get_child_env_buffer_idx(self):
        
        pass
