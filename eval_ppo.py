# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple
from utils.utils_ppo import select_action_ppo, wrap_action


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(NamedTuple):
    max_reward: jax.Array = jnp.asarray(-100)
    min_reward: jax.Array = jnp.asarray(100)
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)
    positive_terminations: jax.Array = jnp.asarray(0)  # Count of positive terminations
    terminations: jax.Array = jnp.asarray(0)  # Count of terminations
    positive_terminations_steps: jax.Array = jnp.asarray(0)

    action_0: jax.Array = jnp.asarray(0)
    action_1: jax.Array = jnp.asarray(0)
    action_2: jax.Array = jnp.asarray(0)
    action_3: jax.Array = jnp.asarray(0)
    action_4: jax.Array = jnp.asarray(0)
    action_5: jax.Array = jnp.asarray(0)
    action_6: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    config,
) -> RolloutStats:
    num_envs = config.num_envs_per_device
    num_rollouts = config.num_rollouts_eval

    def _cond_fn(carry):
        # carry is (rng, timestep, prev_actions_1, prev_actions_2, stats)
        _, _, _, _, stats = carry # Unpack all 5 elements
        return stats.length < num_rollouts

    def _body_fn(carry):
        rng, timestep, prev_actions_1, prev_actions_2, stats = carry

        rng_model_1, rng_model_2, rng_step_base, rng_next_loop = jax.random.split(rng, 4)

        obs_current = timestep.observation
        # --- Action for Agent 1 ---
        action1_raw, _, _, _ = select_action_ppo(
            train_state, obs_current, prev_actions_1, prev_actions_2, rng_model_1, config
        )
        action_env_1 = wrap_action(action1_raw, env.batch_cfg.action_type)

        # --- Action for Agent 2 (policy-based, rotated perspective) ---
        obs_for_agent2_persp = obs_current.copy()
        obs_for_agent2_persp["agent_state_1"] = obs_current["agent_state_2"]
        obs_for_agent2_persp["agent_state_2"] = obs_current["agent_state_1"]
        local_map_keys_agent1 = [
            "local_map_action_neg", "local_map_action_pos", "local_map_target_neg", 
            "local_map_target_pos", "local_map_dumpability", "local_map_obstacles"
        ]
        local_map_keys_agent2 = [key + "_2" for key in local_map_keys_agent1]
        for key1, key2 in zip(local_map_keys_agent1, local_map_keys_agent2):
            obs_for_agent2_persp[key1] = obs_current[key2]
            obs_for_agent2_persp[key2] = obs_current[key1]
        
        action2_raw, _, _, _ = select_action_ppo(
            train_state, obs_for_agent2_persp, prev_actions_2, prev_actions_1, rng_model_2, config
        )
        action_env_2 = wrap_action(action2_raw, env.batch_cfg.action_type)
        
        _rng_step_split = jax.random.split(rng_step_base, num_envs)
        timestep = env.step(timestep, action_env_1, action_env_2, _rng_step_split)

        # Update action statistics (still based on action1_raw for consistency with training logs)
        action_stats = jnp.array([
            jnp.sum(action1_raw == 0),  # FORWARD
            jnp.sum(action1_raw == 1),  # BACKWARD
            jnp.sum(action1_raw == 2),  # CLOCK
            jnp.sum(action1_raw == 3),  # ANTICLOCK
            jnp.sum(action1_raw == 4),  # CABIN_CLOCK
            jnp.sum(action1_raw == 5),  # CABIN_ANTICLOCK
            jnp.sum(action1_raw == 6),  # DO
        ])

        # Safely get info values with defaults if keys are missing
        positive_termination = jnp.zeros_like(timestep.done)
        termination = jnp.zeros_like(timestep.done)
        
        # Check if keys exist in info dictionary
        has_pos_term = "positive_termination" in timestep.info
        has_term = "termination" in timestep.info
        
        # Use the values if they exist, otherwise use zeros
        if has_pos_term:
            positive_termination = timestep.info["positive_termination"]
        if has_term:
            termination = timestep.info["termination"]

        # Update stats
        new_stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, jnp.max(timestep.reward)),
            min_reward=jnp.minimum(stats.min_reward, jnp.min(timestep.reward)),
            reward=stats.reward + jnp.sum(timestep.reward),
            length=stats.length + 1,
            episodes=stats.episodes + jnp.sum(timestep.done),
            positive_terminations=stats.positive_terminations + jnp.sum(positive_termination),
            terminations=stats.terminations + jnp.sum(termination),
            positive_terminations_steps=stats.positive_terminations_steps + jnp.sum(
                positive_termination * stats.length
            ),
            action_0=stats.action_0 + action_stats[0],
            action_1=stats.action_1 + action_stats[1],
            action_2=stats.action_2 + action_stats[2],
            action_3=stats.action_3 + action_stats[3],
            action_4=stats.action_4 + action_stats[4],
            action_5=stats.action_5 + action_stats[5],
            action_6=stats.action_6 + action_stats[6],
        )

        # UPDATE PREVIOUS ACTIONS for both agents
        new_prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=-1).at[..., 0].set(action1_raw)
        new_prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=-1).at[..., 0].set(action2_raw)

        return rng_next_loop, timestep, new_prev_actions_1, new_prev_actions_2, new_stats

    # Initialize evaluation
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    timestep = env.reset(env_params, reset_rng)
    
    # Initialize previous actions for both agents
    prev_actions_1 = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    prev_actions_2 = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    
    initial_stats = RolloutStats()
    carry = (rng, timestep, prev_actions_1, prev_actions_2, initial_stats)
    
    _, _, _, _, final_stats = jax.lax.while_loop(_cond_fn, _body_fn, carry)
    
    return final_stats
