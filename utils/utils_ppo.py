import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    return obs


def obs_to_model_input(obs, prev_actions, train_cfg):
    # Feature engineering
    if train_cfg.clip_action_maps:
        obs = clip_action_map_in_obs(obs)

    # Create input list with indexed comments for easy reference
    # Note: This is still a single-agent problem; agent_2 data is just additional features
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
        prev_actions,                    # [11] - Previous actions history
        # New features (still single-agent problem, just additional features)
        obs["agent_state_2"],            # [12] - Secondary agent state features
        obs["local_map_action_neg_2"],   # [13] - Secondary agent negative action local map
        obs["local_map_action_pos_2"],   # [14] - Secondary agent positive action local map
        obs["local_map_target_neg_2"],   # [15] - Secondary agent negative target local map
        obs["local_map_target_pos_2"],   # [16] - Secondary agent positive target local map
        obs["local_map_dumpability_2"],  # [17] - Secondary agent dumpability local map
        obs["local_map_obstacles_2"],    # [18] - Secondary agent obstacles local map
        obs["interaction_mask"],          # [19] - Interaction map
        obs["id"]                   # [20] - Agent ID
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
    prev_actions: jnp.ndarray,
    rng: jax.random.PRNGKey,
    config,
):
    # Prepare policy input from Terra State
    obs = obs_to_model_input(obs, prev_actions, config)

    value, pi = policy(train_state.apply_fn, train_state.params, obs)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], pi


def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action
