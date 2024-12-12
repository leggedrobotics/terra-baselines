import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
from terra.env import TerraEnvBatch
import jax
from Monte_Carlo_Tree import Node, GAME_ACTIONS, GAME_ACTIONS_LABELS
from copy import deepcopy

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

import pydot

def add_node_edges(graph, node, parent=None):
    """ Recursively add nodes and edges to the graph. """
    prob = 0
    if node.parent is not None:
        prob = node.parent.nn_p[node.action_index]
    
    node_label = f"Action: {GAME_ACTIONS_LABELS[node.action_index]}\nProb: {prob:.2f}\nVisits: {node.N}\nReward: {node.immediate_reward}\nUCB: {node.ucb}"
    node_id = id(node)  # Unique identifier for the node in the graph
    graph_node = pydot.Node(node_id, label=node_label, shape="ellipse", style="filled", fillcolor="lightblue")
    graph.add_node(graph_node)

    if parent:
        # Connect this node to its parent
        parent_id = id(parent)
        edge = pydot.Edge(parent_id, node_id)
        graph.add_edge(edge)

    # Recurse on child nodes
    if node.children:
        for child in node.children:
            add_node_edges(graph, child, node)

def visualize_mcts(root_node):
    """ Visualize the MCTS using PyDot. """
    graph = pydot.Dot(graph_type='digraph', fontname="Helvetica", fontsize="10")
    add_node_edges(graph, root_node)

    # Create PNG image
    graph.write_png('mcts_tree.png')  # Save to file
    # from IPython.display import Image, display
    # display(Image(graph.create_png()))  # Display in Jupyter Notebook or similar environment


def get_best_action(
    env_origin,
    model,
    model_params,
    timestep_origin,
    rng,
    rl_config
    ):

    env_mcts = deepcopy(env_origin) 
    timestep_init = deepcopy(timestep_origin)
    mcts_tree = Node(
        env=env_mcts, 
        done=False, 
        parent=None, 
        timestep=timestep_init, 
        action_index=0, 
        rng=rng, 
        model=model, 
        model_params=model_params, 
        rl_config=rl_config,
        immediate_reward=-1
    )

    MCTS_POLICY_EXPLORE = 100
    for i in range(MCTS_POLICY_EXPLORE):
        mcts_tree.explore()

    
    # visualize_mcts(mcts_tree)

    action_idx, probs = mcts_tree.next()
    action = GAME_ACTIONS[action_idx]
    print("action idx", action_idx)
    print("probs", probs)

    # destroy mcts tree and env
    del mcts_tree
    del env_mcts
    jax.clear_caches()    # Add this line
    return jnp.array([action_idx])
