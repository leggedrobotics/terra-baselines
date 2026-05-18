import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    return {
        **obs,
        "action_map": jnp.clip(obs["action_map"], a_min=-1, a_max=1),
    }


def obs_to_model_input(obs, prev_actions, train_cfg):
    # Feature engineering
    if train_cfg.clip_action_maps:
        obs = clip_action_map_in_obs(obs)
    edge_features_dim = (
        train_cfg.get("edge_features_dim", 0)
        if isinstance(train_cfg, dict)
        else getattr(train_cfg, "edge_features_dim", 0)
    )
    include_episode_progress = (
        train_cfg.get("include_episode_progress", False)
        if isinstance(train_cfg, dict)
        else getattr(train_cfg, "include_episode_progress", False)
    )
    batch_size = obs["agent_states"].shape[0]
    if edge_features_dim:
        edge_features = obs["edge_features"].reshape((batch_size, -1))[
            :, :edge_features_dim
        ]
    else:
        edge_features = jnp.zeros((batch_size, 0), dtype=jnp.float32)

    if include_episode_progress:
        episode_progress = obs["episode_progress"].reshape((batch_size, -1))[:, :1]
    else:
        episode_progress = jnp.zeros((batch_size, 0), dtype=jnp.float32)

    action_mask = obs["action_mask"]

    # Create input list with indexed comments for easy reference
    # Updated to match the new observation structure from env.py
    # Exclude agent_width and agent_height from the list as they're scalars and not used in the model
    obs = [
        obs["agent_states"],             # [0] - All agent states (ordered: active first)
        obs["agent_active"],             # [1] - Agent active mask
        obs["num_agents"],              # [2] - Number of active agents
        obs["local_map_action_neg"],     # [3] - Primary agent negative action local map
        obs["local_map_action_pos"],     # [4] - Primary agent positive action local map
        obs["local_map_target_neg"],     # [5] - Primary agent negative target local map
        obs["local_map_target_pos"],     # [6] - Primary agent positive target local map
        obs["local_map_dumpability"],    # [7] - Primary agent dumpability local map
        obs["local_map_obstacles"],      # [8] - Primary agent obstacles local map
        obs["local_map_border_workspace"],    # [9] - Border tiles currently in workspace
        obs["local_map_edge_alignment_error"],# [10] - Workspace border alignment error sum
        obs["local_map_border_diggable"],     # [11] - Workspace border tiles currently diggable
        obs["traversability_mask"],      # [12] - Traversability mask
        obs["reachability_mask"],        # [13] - Reachability mask (optional; zero when disabled)
        obs["action_map"],               # [14] - Global action map
        obs["target_map"],               # [15] - Global target map
        obs["agent_width"],              # [16] - Agent width (scalar)
        obs["agent_height"],             # [17] - Agent height (scalar)
        obs["padding_mask"],             # [18] - Padding mask
        obs["dumpability_mask"],         # [19] - Dumpability mask
        obs["interaction_mask"],         # [20] - Interaction map
        prev_actions,                    # [21] - Previous actions history
        action_mask,                     # [22] - Coarse primitive action availability
        edge_features,                   # [23] - Edge affordance/progress features
        episode_progress,                # [24] - Episode progress scalar
    ]
    return obs


def policy(apply_fn, params, obs):
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

    value, pi = policy(
        train_state.apply_fn,
        train_state.params,
        obs,
    )
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], pi


def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action
