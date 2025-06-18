import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    return obs


def obs_to_model_input(obs, prev_actions_1, prev_actions_2, train_cfg):
    # Feature engineering
    if train_cfg.clip_action_maps:
        obs = clip_action_map_in_obs(obs)

    # Create input list with indexed comments for easy reference
    obs = [
        obs["agent_state"],              # [0] - Primary agent state
        obs["local_map_action_neg"],     # [1] - Primary agent negative action local map
        obs["local_map_action_pos"],     # [2] - Primary agent positive action local map
        obs["local_map_target_neg"],     # [3] - Primary agent negative target local map
        obs["local_map_target_pos"],     # [4] - Primary agent positive target local map
        obs["local_map_dumpability"],    # [5] - Primary agent dumpability local map
        obs["local_map_obstacles"],      # [6] - Primary agent obstacles local map
        obs["action_map"],               # [7] - Global action map
        obs["target_map"],               # [8] - Global target map
        obs["traversability_mask"],      # [9] - Traversability mask
        obs["dumpability_mask"],         # [10] - Dumpability mask
        prev_actions_1,                  # [11] - Agent 1 previous actions history
        obs["agent_state_2"],            # [12] - Secondary agent state features
        obs["local_map_action_neg_2"],   # [13] - Secondary agent negative action local map
        obs["local_map_action_pos_2"],   # [14] - Secondary agent positive action local map
        obs["local_map_target_neg_2"],   # [15] - Secondary agent negative target local map
        obs["local_map_target_pos_2"],   # [16] - Secondary agent positive target local map
        obs["local_map_dumpability_2"],  # [17] - Secondary agent dumpability local map
        obs["local_map_obstacles_2"],    # [18] - Secondary agent obstacles local map
        prev_actions_2,                  # [19] - Agent 2 previous actions history  
    ]
    return obs


def policy(
    apply_fn,
    params,
    obs,
):
    value, logits_pi = apply_fn(params, obs)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    return value, pi


def select_action_ppo(
    train_state,
    obs: jnp.ndarray,
    prev_actions_1: jnp.ndarray,
    prev_actions_2: jnp.ndarray,
    rng: jax.random.PRNGKey,
    config,
    action_type=None,  # Add action_type as parameter
):
    # Prepare policy input from Terra State
    obs = obs_to_model_input(obs, prev_actions_1, prev_actions_2, config)

    value, pi = policy(train_state.apply_fn, train_state.params, obs)
    action = pi.sample(seed=rng)
    
    # Get action type - try multiple sources
    if action_type is not None:
        num_actions = action_type.get_num_actions()
    elif hasattr(config, 'action_type'):
        num_actions = config.action_type.get_num_actions()
    else:
        # Default fallback - assume 7 actions (TrackedAction/WheeledAction)
        num_actions = 7
    
    # Check if we have dual-agent logits
    total_logits = pi.logits.shape[-1]
    if total_logits == 2 * num_actions:
        # Split actions for two agents - split along the last dimension (action dimension)
        logits_1 = pi.logits[..., :num_actions]
        logits_2 = pi.logits[..., num_actions:]
        
        pi_1 = tfp.distributions.Categorical(logits=logits_1)
        pi_2 = tfp.distributions.Categorical(logits=logits_2)
        
        action_1 = pi_1.sample(seed=rng)
        action_2 = pi_2.sample(seed=rng)
        
        log_prob_1 = pi_1.log_prob(action_1)
        log_prob_2 = pi_2.log_prob(action_2)
    else:
        # Single agent case - duplicate the action
        action_1 = action
        action_2 = action
        log_prob_1 = pi.log_prob(action)
        log_prob_2 = pi.log_prob(action)
    
    return (action_1, action_2), (log_prob_1, log_prob_2), value[:, 0], pi


def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action
