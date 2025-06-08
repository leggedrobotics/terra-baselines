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
    """Perform a rollout in the environment using the policy."""
    
    # Initialize the env - pass env_params as the first argument
    key_reset, rng = jax.random.split(rng)
    num_envs = config.num_envs_per_device
    reset_keys = jax.random.split(key_reset, num_envs)
    timestep = env.reset(env_params, reset_keys)
    
    # Initialize previous action arrays
    prev_actions_1 = jnp.zeros((num_envs, config.num_prev_actions))
    prev_actions_2 = jnp.zeros((num_envs, config.num_prev_actions))

    # Initialize statistics with correct batch dimension to match rewards
    stats = RolloutStats(
        max_reward=jnp.full(num_envs, -100.0),
        min_reward=jnp.full(num_envs, 100.0),
        reward=jnp.zeros(num_envs),
        length=jnp.zeros(num_envs),
        episodes=jnp.zeros(num_envs),
        positive_terminations=jnp.zeros(num_envs),
        terminations=jnp.zeros(num_envs),
        positive_terminations_steps=jnp.zeros(num_envs),
        action_0=jnp.zeros(num_envs),
        action_1=jnp.zeros(num_envs),
        action_2=jnp.zeros(num_envs),
        action_3=jnp.zeros(num_envs),
        action_4=jnp.zeros(num_envs),
        action_5=jnp.zeros(num_envs),
        action_6=jnp.zeros(num_envs),
    )

    # While loop condition: continue until done
    def _cond_fn(carry):
        _, _, timestep, _, _, _ = carry
        return ~jnp.all(timestep.done)

    # While loop body: step the environment
    def _body_fn(carry):
        rng, env_state, timestep, prev_actions_1, prev_actions_2, stats = carry

        # Get a random key for action selection
        rng, key_action = jax.random.split(rng)

        # Get actions for both agents from a single policy call
        actions_tuple, _, _, _ = select_action_ppo(
            train_state,
            timestep.observation,
            prev_actions_1,
            prev_actions_2,
            key_action,
            config,
        )
        action1_raw, action2_raw = actions_tuple

        # Wrap actions for environment
        action_env_1 = wrap_action(action1_raw, env.batch_cfg.action_type)
        action_env_2 = wrap_action(action2_raw, env.batch_cfg.action_type)

        # Step the environment
        key_step, rng = jax.random.split(rng)
        env_step_keys = jax.random.split(key_step, num_envs)
        timestep = env.step(timestep, action_env_1, action_env_2, env_step_keys)

        # Update previous actions
        new_prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=1).at[:, 0].set(action1_raw)
        new_prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=1).at[:, 0].set(action2_raw)

        # Update stats with consistent shapes
        new_stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward),
            min_reward=jnp.minimum(stats.min_reward, timestep.reward),
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes,
            positive_terminations=stats.positive_terminations,
            terminations=stats.terminations,
            positive_terminations_steps=stats.positive_terminations_steps,
            action_0=stats.action_0,
            action_1=stats.action_1,
            action_2=stats.action_2,
            action_3=stats.action_3,
            action_4=stats.action_4,
            action_5=stats.action_5,
            action_6=stats.action_6,
        )

        return rng, env_state, timestep, new_prev_actions_1, new_prev_actions_2, new_stats

    # Run the episode
    carry = (rng, None, timestep, prev_actions_1, prev_actions_2, stats)
    _, _, _, _, _, final_stats = jax.lax.while_loop(_cond_fn, _body_fn, carry)
    
    # Aggregate stats across batch dimension for final output
    return RolloutStats(
        max_reward=jnp.max(final_stats.max_reward),
        min_reward=jnp.min(final_stats.min_reward),
        reward=jnp.mean(final_stats.reward),
        length=jnp.mean(final_stats.length),
        episodes=jnp.sum(final_stats.episodes),
        positive_terminations=jnp.sum(final_stats.positive_terminations),
        terminations=jnp.sum(final_stats.terminations),
        positive_terminations_steps=jnp.sum(final_stats.positive_terminations_steps),
        action_0=jnp.mean(final_stats.action_0),
        action_1=jnp.mean(final_stats.action_1),
        action_2=jnp.mean(final_stats.action_2),
        action_3=jnp.mean(final_stats.action_3),
        action_4=jnp.mean(final_stats.action_4),
        action_5=jnp.mean(final_stats.action_5),
        action_6=jnp.mean(final_stats.action_6),
    )
