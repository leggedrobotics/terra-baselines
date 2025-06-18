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


# @partial(jax.pmap, axis_name="devices")
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
        _, stats, _ = carry
        # Check if the number of steps has been reached
        return jnp.less(stats.length, num_rollouts + 1)

    def _body_fn(carry):
        rng, stats, timestep, prev_actions_1, prev_actions_2 = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        (action_1, action_2), _, _, _ = select_action_ppo(
            train_state, timestep.observation, prev_actions_1, prev_actions_2, _rng_model, config
        )
        
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env_1 = wrap_action(action_1, env.batch_cfg.action_type)
        action_env_2 = wrap_action(action_2, env.batch_cfg.action_type)
        timestep = env.step(timestep, action_env_1, action_env_2, _rng_step)

        prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=-1)
        prev_actions_1 = prev_actions_1.at[..., 0].set(action_1)
        prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=-1)
        prev_actions_2 = prev_actions_2.at[..., 0].set(action_2)

        terminations_update = timestep.done.sum()
        positive_termination_update = timestep.info["task_done"].sum()
        positive_termination_steps_update = (stats.length + 1) * positive_termination_update

        # Update stats (combine actions from both agents)
        combined_actions = jnp.concatenate([action_1, action_2], axis=0)
        
        stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward.max()),
            min_reward=jnp.minimum(stats.min_reward, timestep.reward.min()),
            reward=stats.reward + timestep.reward.sum(),  # Ensure correct aggregation
            length=stats.length + 1,
            episodes=stats.episodes + timestep.done.any(),
            positive_terminations=stats.positive_terminations
            + positive_termination_update,
            terminations=stats.terminations + terminations_update,
            positive_terminations_steps=stats.positive_terminations_steps
            + positive_termination_steps_update,
            action_0=stats.action_0 + (combined_actions == 0).sum(),
            action_1=stats.action_1 + (combined_actions == 1).sum(),
            action_2=stats.action_2 + (combined_actions == 2).sum(),
            action_3=stats.action_3 + (combined_actions == 3).sum(),
            action_4=stats.action_4 + (combined_actions == 4).sum(),
            action_5=stats.action_5 + (combined_actions == 5).sum(),
            action_6=stats.action_6 + (combined_actions == 6).sum(),
        )
        carry = (rng, stats, timestep, prev_actions_1, prev_actions_2)
        return carry

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    prev_actions_1 = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    prev_actions_2 = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    init_carry = (rng, RolloutStats(), timestep, prev_actions_1, prev_actions_2)

    # final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    final_carry = jax.lax.fori_loop(
        0, num_rollouts, lambda i, carry: _body_fn(carry), init_carry
    )
    return final_carry[1]
