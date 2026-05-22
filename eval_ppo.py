# utilities for PPO training and evaluation
from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple
from utils.utils_ppo import select_action_ppo, wrap_action


def _strip_reward_components(timestep):
    """Keep eval timestep pytrees small and independent of training-only logs."""
    if isinstance(timestep.info, dict) and "reward_components" in timestep.info:
        return timestep._replace(
            info={k: v for k, v in timestep.info.items() if k != "reward_components"}
        )
    return timestep


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
    action_7: jax.Array = jnp.asarray(0)


def _rollout_impl(
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
        rng, stats, timestep, prev_actions = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        action, _, _, _ = select_action_ppo(
            train_state, timestep.observation, prev_actions, _rng_model, config
        )
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        timestep = env.step(timestep, action_env, _rng_step)
        timestep = _strip_reward_components(timestep)

        prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
        prev_actions = prev_actions.at[..., 0].set(action)

        terminations_update = timestep.done.sum()
        positive_termination_update = timestep.info["task_done"].sum()
        positive_termination_steps_update = (stats.length + 1) * positive_termination_update

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
            action_0=stats.action_0 + (action == 0).sum(),
            action_1=stats.action_1 + (action == 1).sum(),
            action_2=stats.action_2 + (action == 2).sum(),
            action_3=stats.action_3 + (action == 3).sum(),
            action_4=stats.action_4 + (action == 4).sum(),
            action_5=stats.action_5 + (action == 5).sum(),
            action_6=stats.action_6 + (action == 6).sum(),
            action_7=stats.action_7 + (action == 7).sum(),
        )
        carry = (rng, stats, timestep, prev_actions)
        return carry

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    timestep = _strip_reward_components(timestep)
    
    prev_actions = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    init_carry = (rng, RolloutStats(), timestep, prev_actions)

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


def rollout_single_device(rng, env, env_params, train_state, config):
    return _rollout_impl(rng, env, env_params, train_state, config)


def rollout(rng, env, env_params, train_state, config):
    key = (id(env), id(config))
    pmapped = _PMAPPED_ROLLOUT_CACHE.get(key)
    if pmapped is None:
        def _closure(rng_, env_params_, train_state_):
            return _rollout_impl(rng_, env, env_params_, train_state_, config)

        pmapped = jax.pmap(_closure, axis_name="devices")
        _PMAPPED_ROLLOUT_CACHE[key] = pmapped
    return pmapped(rng, env_params, train_state)
