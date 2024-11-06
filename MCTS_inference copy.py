import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
from terra.env import TerraEnvBatch
import jax
from utils.utils_ppo import obs_to_model_input, wrap_action
from Monte_Carlo_Tree import Node

def init_timestep_mcts(timestep, n_env_per_step):
    # Apply a custom function to each element in the nested structure of timestep
    timestep_expanded = jax.tree_map(
        lambda x: jnp.concatenate([x[[i], ...].repeat(n_env_per_step, axis=0) for i in range(x.shape[0])], axis=0),
        timestep
    )
    return timestep_expanded


def total_nodes_in_k_ary_tree(levels, children_per_node):
    if levels < 1:
        return 0
    # Calculate the total using the sum of a geometric series:
    # Sum = a * (r^n - 1) / (r - 1) where a = 1, r = children_per_node, n = levels
    total = (children_per_node**levels - 1) // (children_per_node - 1)
    return total

def get_best_action(
    env_origin: TerraEnvBatch,
    model,
    model_params,
    timestep_origin,
    rng, 
    rl_config,
    env_cfgs_1,
    epsilon=0.9,  # Probability of choosing a random action
    num_rollouts=10,  # Number of rollouts to perform for each action decision
    n_steps=4,  # Number of future steps to simulate
    n_envs=8
    ):
    obs = timestep_origin.observation
    best_action = np.full(n_envs, -1, dtype=np.int32)  # Initialize with -1 (invalid action)
    best_total_reward = -np.inf * np.ones(n_envs)

    # create new env buffer to search tree of possibilites
    n_actions = 4
    depth = 3 # how many step in the future
    n_env_per_tree = total_nodes_in_k_ary_tree(depth, n_actions)
    n_envs_mmcts = n_envs*n_env_per_tree # n_action^depth ?
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs_mmcts, 0), env_cfgs_1
    )  # take first config and replicate
    print(env_cfgs)
    shuffle_maps = True
    env_mcts = TerraEnvBatch(rendering=False, shuffle_maps=shuffle_maps)

    # initial monte carlo tree env buffer time steps
    timestep_mcts = init_timestep_mcts(timestep_origin,n_env_per_tree)
    print("LETSGOO")

    # init MC trees for each env
    roots = [Node(env_buffer_idx=i*n_env_per_step) for i in range(0, n_envs)]
    
    # expand to all possible states the root node to create initial tree
    for r in roots:
        r.explore_childs()

    # SELECTION

    # EXPANSION

    # BACK UP
    


    return best_action
