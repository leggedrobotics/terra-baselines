import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


_LOCAL_MAP_KEYS = (
    "local_map_action_neg",
    "local_map_action_pos",
    "local_map_target_neg",
    "local_map_target_pos",
    "local_map_dumpability",
    "local_map_obstacles",
    "local_map_border_workspace",
    "local_map_edge_alignment_error",
    "local_map_border_diggable",
)


def _config_option(config, name: str, default):
    try:
        value = getattr(config, name)
    except (AttributeError, KeyError):
        try:
            value = config[name]
        except (TypeError, KeyError, IndexError):
            return default
    return value if value is not None else default


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    return obs


def scale_local_maps_in_obs(obs, scale):
    """Scale local workspace-sum observations before model preprocessing."""
    scale = float(scale)
    if scale == 1.0:
        return obs
    obs = dict(obs)
    scale_arr = jnp.asarray(scale, dtype=jnp.float32)
    for key in _LOCAL_MAP_KEYS:
        obs[key] = obs[key].astype(jnp.float32) / scale_arr
    return obs


def obs_to_model_input(obs, prev_actions, train_cfg):
    # Feature engineering
    if _config_option(train_cfg, "clip_action_maps", True):
        obs = clip_action_map_in_obs(obs)
    obs = scale_local_maps_in_obs(
        obs,
        _config_option(train_cfg, "local_map_area_scale", 1.0),
    )

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
