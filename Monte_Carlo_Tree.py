import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from math import *
import random
import jax
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action


c = 1.0

"""
Tracked robot specific actions.
"""

DO_NOTHING = -1
FORWARD = 0
BACKWARD = 1
CLOCK = 2
ANTICLOCK = 3
CABIN_CLOCK = 4
CABIN_ANTICLOCK = 5
EXTEND_ARM = 6
RETRACT_ARM = 7
DO = 8

GAME_ACTIONS = [DO_NOTHING, FORWARD, BACKWARD, CLOCK, ANTICLOCK, CABIN_CLOCK, CABIN_ANTICLOCK, EXTEND_ARM, RETRACT_ARM, DO]
GAME_ACTIONS_LABELS = ["DO_NOTHING", "FORWARD", "BACKWARD", "CLOCK", "ANTICLOCK", "CABIN_CLOCK", "CABIN_ANTICLOCK", "EXTEND_ARM", "RETRACT_ARM", "DO"]


class Node:    
    
    '''
    The Node class represents a node of the MCTS tree. 
    It contains the information needed for the algorithm to run its search.
    It stores extra information about neural network value and policy for that state.
    '''

    def __init__(self, env, done, parent, timestep, action_index, rng, model, model_params, rl_config, immediate_reward):
          
        # child nodes
        self.child = None
        self.children = []  # To track child nodes
        
        # total rewards from MCTS exploration
        self.T = 0
        self.immediate_reward = immediate_reward
        
        # visit count
        self.N = 0        
        
        # the environment
        self.env = env
        self.rng = rng
        
        # observation of the environment
        self.timestep = timestep
        
        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent
        
        # action index that leads to this node
        self.action_index = action_index

        # model params
        self.model_params = model_params

        # model
        self.model = model

        # rl config
        self.rl_config = rl_config
        
        # the value of the node according to nn
        self.nn_v = 0
        
        # the next probabilities
        self.nn_p = None    
        self.ucb = None 
        
        
    def getUCBscore(self):        
        
        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.        
        '''
        
        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')
        
        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
                        
        value_score = (self.T / self.N) 
            
        prior_score = c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)
                
        # We use both the Value(s) and Policy from the neural network estimations for calculating the node value
        ucb = value_score + prior_score + self.immediate_reward
        self.ucb = ucb
        return ucb
    
    
    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None
            
    def create_child(self):
        
        '''
        We create one children for each possible action of the game, 
        then we apply such action to a copy of the current node enviroment 
        and create such child node with proper information returned from the action executed
        '''
        
        if self.done:
            return
    
        child = {} 
        for action_index in range(len(GAME_ACTIONS)):
            rgn, rng_step = jax.random.split(self.rng, 2)
            rng_step = jax.random.split(rng_step, 1)

            timestep_child = self.env.step(self.timestep, wrap_action(jnp.array([action_index]), self.env.batch_cfg.action_type), rng_step)

            observation = timestep_child.observation
            reward = timestep_child.reward
            done = timestep_child.done
            # print(reward, action_index)

            new_node = Node(env=self.env, done=done, parent=self, timestep=timestep_child, action_index=action_index, rng=rng_step[0], model=self.model, model_params=self.model_params, rl_config=self.rl_config, immediate_reward=reward[0])                        
            self.children.append(new_node)
            child[action_index] = new_node

        self.child = child
                
            
    def explore(self):
        
        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the AlphaZero formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''
        
        # find a leaf node by choosing nodes with max U.
        
        current = self
        
        while current.child:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)                      
            action = random.choice(actions)
            current = child[action]
            
        # play a random game, or expand if needed          
            
        if current.N < 1:
            current.nn_v, current.nn_p = current.rollout()
            # current.T = current.T + current.nn_v
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.nn_v, current.nn_p = current.rollout()
            # current.T = current.T + current.nn_v 
            
        current.N += 1      
                
        # update statistics and backpropagate
            
        parent = current
            
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T           
              
    def rollout(self):
        
        '''
        The rollout is where we use the neural network estimations to approximate the Value and Policy of a given node.
        With the trained neural network, it will give us a good approximation even in large state spaces searches.
        '''
        
        if self.done:
            return 0, None        
        else:
            obs_model = obs_to_model_input(self.timestep.observation, self.rl_config)
            v, logits_pi = self.model.apply(self.model_params, obs_model)
            action_probabilities = jnp.exp(logits_pi)
            row_sums = action_probabilities.sum(axis=1, keepdims=True)
            action_probabilities = action_probabilities / row_sums

            return v[0], action_probabilities[0]
                   
    def next(self):
        
        ''' 
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        child = self.child
        
        max_N = max(node.N for node in child.values())
        
        probs = [ node.N / max_N for node in child.values() ]
        probs /= np.sum(probs)
        
        next_children = random.choices(list(child.values()), weights=probs)[0]
        
        return next_children.action_index, probs