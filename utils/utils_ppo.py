import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    return obs


def obs_to_model_input(obs, prev_actions_1, prev_actions_2, train_cfg):
    """
    Convert observations to model input for centralized 2-agent policy.
    
    Expected obs structure:
    - agent_state_1, agent_state_2: Both agents' states
    - local_map_*: Local maps for agent 1
    - local_map_*_2: Local maps for agent 2  
    - Global maps: action_map, target_map, etc.
    """
    # Feature engineering
    if train_cfg.clip_action_maps:
        obs = clip_action_map_in_obs(obs)

    # Build input in expected order for CentralizedTwoAgentNet
    model_input = [
        # Agent states
        obs["agent_state_1"],
        obs["agent_state_2"],
        # Agent 1 local maps
        obs["local_map_action_neg"],
        obs["local_map_action_pos"],
        obs["local_map_target_neg"],
        obs["local_map_target_pos"],
        obs["local_map_dumpability"],
        obs["local_map_obstacles"],
        # Agent 2 local maps
        obs["local_map_action_neg_2"],
        obs["local_map_action_pos_2"],
        obs["local_map_target_neg_2"],
        obs["local_map_target_pos_2"],
        obs["local_map_dumpability_2"],
        obs["local_map_obstacles_2"],
        # Global maps
        obs["action_map"],
        obs["target_map"],
        obs["traversability_mask"],
        obs["dumpability_mask"],
        # Previous actions for both agents
        prev_actions_1,
        prev_actions_2,
    ]
    return model_input


def policy(
    apply_fn,
    params,
    obs,
):
    value, (logits_pi1, logits_pi2) = apply_fn(params, obs)
    pi1 = tfp.distributions.Categorical(logits=logits_pi1)
    pi2 = tfp.distributions.Categorical(logits=logits_pi2)
    return value, (pi1, pi2)


def select_action_ppo(
    train_state,
    obs: jnp.ndarray,
    prev_actions_1: jnp.ndarray,
    prev_actions_2: jnp.ndarray,
    rng: jax.random.PRNGKey,
    config,
):
    """
    Select actions for both agents simultaneously using centralized policy.
    """
    # Prepare policy input from Terra State
    obs = obs_to_model_input(obs, prev_actions_1, prev_actions_2, config)

    # Get value and both agent policies
    rng1, rng2 = jax.random.split(rng)
    value, (pi1, pi2) = policy(train_state.apply_fn, train_state.params, obs)
    
    # Sample actions for both agents
    action1 = pi1.sample(seed=rng1)
    action2 = pi2.sample(seed=rng2)
    
    # Get log probabilities
    log_prob1 = pi1.log_prob(action1)
    log_prob2 = pi2.log_prob(action2)
    
    return (action1, action2), (log_prob1, log_prob2), value[:, 0], (pi1, pi2)


def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action
