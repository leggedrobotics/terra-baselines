# utilities for PPO training and evaluation
from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple
from utils.utils_ppo import select_action_ppo, wrap_action


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(NamedTuple):
    max_reward: jax.Array = jnp.asarray(-100)
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    successes: jax.Array = jnp.asarray(0)
    success_steps: jax.Array = jnp.asarray(0)
    return_sum: jax.Array = jnp.asarray(0.0)
    return_sq_sum: jax.Array = jnp.asarray(0.0)
    return_min: jax.Array = jnp.asarray(jnp.inf)
    return_max: jax.Array = jnp.asarray(-jnp.inf)
    return_count: jax.Array = jnp.asarray(0)
    current_return: jax.Array = jnp.asarray(0.0)
    success_return_sum: jax.Array = jnp.asarray(0.0)
    success_return_count: jax.Array = jnp.asarray(0)
    failure_return_sum: jax.Array = jnp.asarray(0.0)
    failure_return_count: jax.Array = jnp.asarray(0)
    action_counts: jax.Array = jnp.zeros((8,), dtype=jnp.int32)


def _rollout_impl(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    config,
) -> RolloutStats:
    num_envs = config.num_envs_per_device
    num_rollouts = config.num_rollouts_eval

    def _body_fn(carry):
        rng, stats, timestep, prev_actions = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        action, _, _, _ = select_action_ppo(
            train_state, timestep.observation, prev_actions, _rng_model, config
        )
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        timestep = env.step(timestep, action_env, _rng_step)

        prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
        prev_actions = prev_actions.at[..., 0].set(action)

        successes_update = timestep.info["task_done"].sum()
        success_steps_update = (stats.length + 1) * successes_update
        next_episode_return = stats.current_return + timestep.reward
        finished_return = jnp.where(timestep.done, next_episode_return, 0.0)
        return_count = timestep.done.sum()
        success_done = jnp.logical_and(timestep.done, timestep.info["task_done"])
        failure_done = jnp.logical_and(
            timestep.done,
            jnp.logical_not(timestep.info["task_done"]),
        )
        success_return = jnp.where(success_done, next_episode_return, 0.0)
        failure_return = jnp.where(failure_done, next_episode_return, 0.0)
        return_min = jnp.min(
            jnp.where(timestep.done, next_episode_return, jnp.inf)
        )
        return_max = jnp.max(
            jnp.where(timestep.done, next_episode_return, -jnp.inf)
        )

        stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward.max()),
            reward=stats.reward + timestep.reward.sum(),
            length=stats.length + 1,
            successes=stats.successes + successes_update,
            success_steps=stats.success_steps + success_steps_update,
            return_sum=stats.return_sum + finished_return.sum(),
            return_sq_sum=stats.return_sq_sum + jnp.square(finished_return).sum(),
            return_min=jnp.minimum(stats.return_min, return_min),
            return_max=jnp.maximum(stats.return_max, return_max),
            return_count=stats.return_count + return_count,
            current_return=jnp.where(
                timestep.done,
                0.0,
                next_episode_return,
            ),
            success_return_sum=stats.success_return_sum + success_return.sum(),
            success_return_count=stats.success_return_count + success_done.sum(),
            failure_return_sum=stats.failure_return_sum + failure_return.sum(),
            failure_return_count=stats.failure_return_count + failure_done.sum(),
            action_counts=stats.action_counts + jnp.bincount(
                action.reshape(-1),
                length=8,
            ),
        )
        carry = (rng, stats, timestep, prev_actions)
        return carry

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)

    prev_actions = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    init_stats = RolloutStats(
        current_return=jnp.zeros_like(timestep.reward),
    )
    init_carry = (rng, init_stats, timestep, prev_actions)

    # final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    final_carry = jax.lax.fori_loop(
        0, num_rollouts, lambda i, carry: _body_fn(carry), init_carry
    )
    return final_carry[1]


# Cache of pmapped rollout functions keyed by (id(env), id(config)). `env` and
# `config` are closed over (not passed as pmap args) because they are not
# pytrees and `MixedAgentTrainConfig` is not hashable, which would trip
# `static_broadcasted_argnums`. Closure means each new (env, config) pair
# retraces once; identical pairs reuse the compiled pmap.
_PMAPPED_ROLLOUT_CACHE: dict = {}


def rollout(rng, env, env_params, train_state, config):
    key = (id(env), id(config))
    pmapped = _PMAPPED_ROLLOUT_CACHE.get(key)
    if pmapped is None:
        def _closure(rng_, env_params_, train_state_):
            return _rollout_impl(rng_, env, env_params_, train_state_, config)

        pmapped = jax.pmap(_closure, axis_name="devices")
        _PMAPPED_ROLLOUT_CACHE[key] = pmapped
    return pmapped(rng, env_params, train_state)
