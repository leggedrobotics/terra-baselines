import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


MOVE_TO_POSE_ACTION = 8
SET_CABIN_ANGLE_ACTION = 9
SET_BASE_ANGLE_ACTION = 10
ACTION_PACK_SIZE = 4
DISABLED_MACRO_POLICY_ACTIONS = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)


class MacroMovePolicyDistribution:
    def __init__(
        self,
        action_logits,
        move_xy_logits,
        move_angle_logits,
        cabin_angle_logits,
        conditional_entropy_scale=0.05,
    ):
        self.action_logits = action_logits
        self.move_xy_logits = move_xy_logits
        self.move_angle_logits = move_angle_logits
        self.cabin_angle_logits = cabin_angle_logits
        self.conditional_entropy_scale = conditional_entropy_scale
        self.action_dist = tfp.distributions.Categorical(logits=action_logits)
        self.move_xy_dist = tfp.distributions.Categorical(logits=move_xy_logits)
        self.move_angle_dist = tfp.distributions.Categorical(logits=move_angle_logits)
        self.cabin_angle_dist = tfp.distributions.Categorical(logits=cabin_angle_logits)

    def sample(self, seed):
        key_action, key_xy, key_angle, key_cabin = jax.random.split(seed, 4)
        action = self.action_dist.sample(seed=key_action)
        move_xy = self.move_xy_dist.sample(seed=key_xy)
        move_angle = self.move_angle_dist.sample(seed=key_angle)
        cabin_angle = self.cabin_angle_dist.sample(seed=key_cabin)
        return jnp.stack([action, move_xy, move_angle, cabin_angle], axis=-1).astype(jnp.int32)

    def mode(self):
        action = jnp.argmax(self.action_logits, axis=-1)
        move_xy = jnp.argmax(self.move_xy_logits, axis=-1)
        move_angle = jnp.argmax(self.move_angle_logits, axis=-1)
        cabin_angle = jnp.argmax(self.cabin_angle_logits, axis=-1)
        return jnp.stack([action, move_xy, move_angle, cabin_angle], axis=-1).astype(jnp.int32)

    def log_prob(self, action):
        action = self._ensure_action_pack(action)
        action_type = action[..., 0]
        move_xy = action[..., 1]
        move_angle = action[..., 2]
        cabin_angle = action[..., 3]
        is_move = action_type == MOVE_TO_POSE_ACTION
        is_set_cabin = action_type == SET_CABIN_ANGLE_ACTION
        is_set_base = action_type == SET_BASE_ANGLE_ACTION
        return (
            self.action_dist.log_prob(action_type)
            + jnp.where(is_move, self.move_xy_dist.log_prob(move_xy), 0.0)
            + jnp.where(is_move, self.move_angle_dist.log_prob(move_angle), 0.0)
            + jnp.where(is_set_base, self.move_angle_dist.log_prob(move_angle), 0.0)
            + jnp.where(is_set_cabin, self.cabin_angle_dist.log_prob(cabin_angle), 0.0)
        )

    def entropy(self):
        prob_move = self.action_dist.probs_parameter()[..., MOVE_TO_POSE_ACTION]
        prob_set_cabin = self.action_dist.probs_parameter()[..., SET_CABIN_ANGLE_ACTION]
        prob_set_base = self.action_dist.probs_parameter()[..., SET_BASE_ANGLE_ACTION]
        return (
            self.action_dist.entropy()
            + self.conditional_entropy_scale
            * (
                prob_move * (self.move_xy_dist.entropy() + self.move_angle_dist.entropy())
                + prob_set_cabin * self.cabin_angle_dist.entropy()
                + prob_set_base * self.move_angle_dist.entropy()
            )
        )

    @staticmethod
    def _ensure_action_pack(action):
        action = jnp.asarray(action)
        if action.ndim > 0 and action.shape[-1] == ACTION_PACK_SIZE:
            return action
        if action.ndim > 0 and action.shape[-1] == 3:
            zeros = jnp.zeros_like(action[..., :1])
            return jnp.concatenate([action, zeros], axis=-1).astype(jnp.int32)
        zeros = jnp.zeros_like(action)
        return jnp.stack([action, zeros, zeros, zeros], axis=-1).astype(jnp.int32)


def clip_action_map_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    return obs


def obs_to_model_input(obs, prev_actions, train_cfg):
    # Feature engineering
    if train_cfg.clip_action_maps:
        obs = clip_action_map_in_obs(obs)

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
    macro_conditional_entropy_scale=0.05,
):
    value, policy_out = apply_fn(params, obs)
    action_logits, move_xy_logits, move_angle_logits, cabin_angle_logits = policy_out
    disabled_action_mask = jnp.any(
        jnp.arange(action_logits.shape[-1])[:, None] == DISABLED_MACRO_POLICY_ACTIONS[None, :],
        axis=-1,
    )
    action_logits = jnp.where(
        disabled_action_mask,
        jnp.full_like(action_logits, -1e9),
        action_logits,
    )
    reachability_mask = obs[13].reshape((*obs[13].shape[:-2], -1)).astype(jnp.bool_)
    masked_move_xy_logits = jnp.where(
        reachability_mask,
        move_xy_logits,
        jnp.full_like(move_xy_logits, -1e9),
    )
    no_reachable = ~jnp.any(reachability_mask, axis=-1, keepdims=True)
    masked_move_xy_logits = jnp.where(no_reachable, move_xy_logits, masked_move_xy_logits)
    pi = MacroMovePolicyDistribution(
        action_logits,
        masked_move_xy_logits,
        move_angle_logits,
        cabin_angle_logits,
        conditional_entropy_scale=macro_conditional_entropy_scale,
    )
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
        macro_conditional_entropy_scale=getattr(config, "macro_conditional_entropy_scale", 0.05),
    )
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], pi


def wrap_action(action, action_type, map_size: int = 64):
    action = jnp.asarray(action)
    if action.ndim > 0 and action.shape[-1] in (3, ACTION_PACK_SIZE):
        action_idx = action[..., 0]
        move_xy = action[..., 1]
        move_angle = action[..., 2]
        if action.shape[-1] == ACTION_PACK_SIZE:
            cabin_angle = action[..., 3]
        else:
            cabin_angle = jnp.zeros_like(move_angle)
        x = move_xy // map_size
        y = move_xy % map_size
        target_pose = jnp.stack([x, y, move_angle, cabin_angle], axis=-1)
        return action_type.new(action_idx[..., None], target_pose=target_pose)
    return action_type.new(action[:, None])


def action_type_from_policy_action(action):
    action = jnp.asarray(action)
    if action.ndim > 0 and action.shape[-1] in (3, ACTION_PACK_SIZE):
        return action[..., 0]
    return action
