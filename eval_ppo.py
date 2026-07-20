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
    initial_episode_successes: jax.Array = jnp.asarray(0)
    initial_episode_terminations: jax.Array = jnp.asarray(0)

    action_0: jax.Array = jnp.asarray(0)
    action_1: jax.Array = jnp.asarray(0)
    action_2: jax.Array = jnp.asarray(0)
    action_3: jax.Array = jnp.asarray(0)
    action_4: jax.Array = jnp.asarray(0)
    action_5: jax.Array = jnp.asarray(0)
    action_6: jax.Array = jnp.asarray(0)
    action_7: jax.Array = jnp.asarray(0)


def _positive_episode_step_sum(episode_steps_before, task_done):
    """Sum true lengths of successful episodes completed on this env step."""
    completed_episode_lengths = jnp.asarray(episode_steps_before) + 1
    return jnp.sum(
        completed_episode_lengths
        * jnp.asarray(task_done, dtype=completed_episode_lengths.dtype)
    )


def _success_events(done, task_done):
    """Keep task successes a strict subset of terminal episode events."""
    return jnp.logical_and(
        jnp.asarray(done, dtype=jnp.bool_),
        jnp.asarray(task_done, dtype=jnp.bool_),
    )


def episode_success_rate(successful_episodes, completed_episodes):
    """Return success/completion in [0, 1], or NaN when no episode completed."""
    successful = jnp.asarray(successful_episodes, dtype=jnp.float32)
    completed = jnp.asarray(completed_episodes, dtype=jnp.float32)
    rate = successful / jnp.maximum(completed, jnp.float32(1.0))
    return jnp.where(
        completed > 0,
        rate,
        jnp.full_like(rate, jnp.nan),
    )


def _initial_episode_outcomes(already_done, done, task_done):
    """Count only each environment's first terminal event in an eval rollout."""
    already_done = jnp.asarray(already_done, dtype=jnp.bool_)
    done = jnp.asarray(done, dtype=jnp.bool_)
    task_done = jnp.asarray(task_done, dtype=jnp.bool_)
    first_termination = jnp.logical_and(jnp.logical_not(already_done), done)
    first_success = jnp.logical_and(first_termination, task_done)
    return (
        jnp.logical_or(already_done, done),
        jnp.sum(first_success),
        jnp.sum(first_termination),
    )


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
        _, stats, _, _, _ = carry
        # Check if the number of steps has been reached
        return jnp.less(stats.length, num_rollouts + 1)

    def _body_fn(carry):
        rng, stats, timestep, prev_actions, initial_episode_done = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        action, _, _, _ = select_action_ppo(
            train_state, timestep.observation, prev_actions, _rng_model, config
        )
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        episode_steps_before = timestep.state.env_steps
        timestep = env.step(timestep, action_env, _rng_step)
        timestep = _strip_reward_components(timestep)

        prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
        prev_actions = prev_actions.at[..., 0].set(action)
        prev_actions = jnp.where(
            timestep.done[..., None],
            jnp.zeros_like(prev_actions),
            prev_actions,
        )

        terminations_update = timestep.done.sum()
        success_events = _success_events(
            timestep.done,
            timestep.info["task_done"],
        )
        positive_termination_update = success_events.sum()
        positive_termination_steps_update = _positive_episode_step_sum(
            episode_steps_before,
            success_events,
        )
        (
            initial_episode_done,
            initial_episode_successes_update,
            initial_episode_terminations_update,
        ) = _initial_episode_outcomes(
            initial_episode_done,
            timestep.done,
            timestep.info["task_done"],
        )

        stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward.max()),
            min_reward=jnp.minimum(stats.min_reward, timestep.reward.min()),
            reward=stats.reward + timestep.reward.sum(),  # Ensure correct aggregation
            length=stats.length + 1,
            episodes=stats.episodes + terminations_update,
            positive_terminations=stats.positive_terminations
            + positive_termination_update,
            terminations=stats.terminations + terminations_update,
            positive_terminations_steps=stats.positive_terminations_steps
            + positive_termination_steps_update,
            initial_episode_successes=stats.initial_episode_successes
            + initial_episode_successes_update,
            initial_episode_terminations=stats.initial_episode_terminations
            + initial_episode_terminations_update,
            action_0=stats.action_0 + (action == 0).sum(),
            action_1=stats.action_1 + (action == 1).sum(),
            action_2=stats.action_2 + (action == 2).sum(),
            action_3=stats.action_3 + (action == 3).sum(),
            action_4=stats.action_4 + (action == 4).sum(),
            action_5=stats.action_5 + (action == 5).sum(),
            action_6=stats.action_6 + (action == 6).sum(),
            action_7=stats.action_7 + (action == 7).sum(),
        )
        carry = (rng, stats, timestep, prev_actions, initial_episode_done)
        return carry

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    timestep = _strip_reward_components(timestep)
    
    prev_actions = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    initial_episode_done = jnp.zeros((num_envs,), dtype=jnp.bool_)
    init_carry = (
        rng,
        RolloutStats(),
        timestep,
        prev_actions,
        initial_episode_done,
    )

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
_SINGLE_DEVICE_STEP_CACHE: dict = {}
_PMAPPED_STEP_CACHE: dict = {}


def aggregate_device_stats(stats: RolloutStats) -> RolloutStats:
    """Merge per-device rollout stats from `rollout` into scalar totals."""
    return RolloutStats(
        max_reward=jnp.max(stats.max_reward),
        min_reward=jnp.min(stats.min_reward),
        reward=jnp.sum(stats.reward),
        length=jnp.max(stats.length),
        episodes=jnp.sum(stats.episodes),
        positive_terminations=jnp.sum(stats.positive_terminations),
        terminations=jnp.sum(stats.terminations),
        positive_terminations_steps=jnp.sum(stats.positive_terminations_steps),
        initial_episode_successes=jnp.sum(stats.initial_episode_successes),
        initial_episode_terminations=jnp.sum(stats.initial_episode_terminations),
        action_0=jnp.sum(stats.action_0),
        action_1=jnp.sum(stats.action_1),
        action_2=jnp.sum(stats.action_2),
        action_3=jnp.sum(stats.action_3),
        action_4=jnp.sum(stats.action_4),
        action_5=jnp.sum(stats.action_5),
        action_6=jnp.sum(stats.action_6),
        action_7=jnp.sum(stats.action_7),
    )


def rollout_single_device(rng, env, env_params, train_state, config):
    # Keep eval out of one large XLA control-flow graph. The RTX 4090 crash is
    # triggered in this path, so smaller step-level compiles make it easier to
    # isolate while preserving inline eval.
    num_envs = config.num_envs_per_device
    num_rollouts = config.num_rollouts_eval

    key = (id(env), id(config))
    eval_step = _SINGLE_DEVICE_STEP_CACHE.get(key)
    if eval_step is None:
        def _eval_step(
            rng_,
            stats_,
            timestep_,
            prev_actions_,
            initial_episode_done_,
            train_state_,
        ):
            rng_, _rng_step, _rng_model = jax.random.split(rng_, 3)

            action, _, _, _ = select_action_ppo(
                train_state_,
                timestep_.observation,
                prev_actions_,
                _rng_model,
                config,
            )
            _rng_step = jax.random.split(_rng_step, num_envs)
            action_env = wrap_action(action, env.batch_cfg.action_type)
            episode_steps_before = timestep_.state.env_steps
            timestep_ = env.step(timestep_, action_env, _rng_step)
            timestep_ = _strip_reward_components(timestep_)

            prev_actions_ = jnp.roll(prev_actions_, shift=1, axis=-1)
            prev_actions_ = prev_actions_.at[..., 0].set(action)
            prev_actions_ = jnp.where(
                timestep_.done[..., None],
                jnp.zeros_like(prev_actions_),
                prev_actions_,
            )

            terminations_update = timestep_.done.sum()
            success_events = _success_events(
                timestep_.done,
                timestep_.info["task_done"],
            )
            positive_termination_update = success_events.sum()
            positive_termination_steps_update = _positive_episode_step_sum(
                episode_steps_before,
                success_events,
            )
            (
                initial_episode_done_,
                initial_episode_successes_update,
                initial_episode_terminations_update,
            ) = _initial_episode_outcomes(
                initial_episode_done_,
                timestep_.done,
                timestep_.info["task_done"],
            )

            stats_ = RolloutStats(
                max_reward=jnp.maximum(stats_.max_reward, timestep_.reward.max()),
                min_reward=jnp.minimum(stats_.min_reward, timestep_.reward.min()),
                reward=stats_.reward + timestep_.reward.sum(),
                length=stats_.length + 1,
                episodes=stats_.episodes + terminations_update,
                positive_terminations=stats_.positive_terminations
                + positive_termination_update,
                terminations=stats_.terminations + terminations_update,
                positive_terminations_steps=stats_.positive_terminations_steps
                + positive_termination_steps_update,
                initial_episode_successes=stats_.initial_episode_successes
                + initial_episode_successes_update,
                initial_episode_terminations=stats_.initial_episode_terminations
                + initial_episode_terminations_update,
                action_0=stats_.action_0 + (action == 0).sum(),
                action_1=stats_.action_1 + (action == 1).sum(),
                action_2=stats_.action_2 + (action == 2).sum(),
                action_3=stats_.action_3 + (action == 3).sum(),
                action_4=stats_.action_4 + (action == 4).sum(),
                action_5=stats_.action_5 + (action == 5).sum(),
                action_6=stats_.action_6 + (action == 6).sum(),
                action_7=stats_.action_7 + (action == 7).sum(),
            )
            return rng_, stats_, timestep_, prev_actions_, initial_episode_done_

        eval_step = jax.jit(_eval_step)
        _SINGLE_DEVICE_STEP_CACHE[key] = eval_step

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    timestep = _strip_reward_components(timestep)

    prev_actions = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)
    stats = RolloutStats()
    initial_episode_done = jnp.zeros((num_envs,), dtype=jnp.bool_)

    for _ in range(num_rollouts):
        rng, stats, timestep, prev_actions, initial_episode_done = eval_step(
            rng,
            stats,
            timestep,
            prev_actions,
            initial_episode_done,
            train_state,
        )

    return jax.block_until_ready(stats)


def rollout_from_timestep(rng, env, timestep, train_state, config):
    """Evaluate from an already-reset pmapped timestep.

    This keeps eval in the same pmap/env.step shape regime as training, but
    keeps the rollout loop in Python so XLA does not lower one large eval graph.
    """
    num_devices = rng.shape[0]
    num_envs = config.num_envs_per_device
    num_rollouts = config.num_rollouts_eval

    key = (id(env), id(config))
    eval_step = _PMAPPED_STEP_CACHE.get(key)
    if eval_step is None:
        @partial(jax.pmap, axis_name="devices")
        def _eval_step(
            rng_,
            stats_,
            timestep_,
            prev_actions_,
            initial_episode_done_,
            train_state_,
        ):
            rng_, _rng_step, _rng_model = jax.random.split(rng_, 3)

            action, _, _, _ = select_action_ppo(
                train_state_,
                timestep_.observation,
                prev_actions_,
                _rng_model,
                config,
            )
            _rng_step = jax.random.split(_rng_step, num_envs)
            action_env = wrap_action(action, env.batch_cfg.action_type)
            episode_steps_before = timestep_.state.env_steps
            timestep_ = env.step(timestep_, action_env, _rng_step)
            timestep_ = _strip_reward_components(timestep_)

            prev_actions_ = jnp.roll(prev_actions_, shift=1, axis=-1)
            prev_actions_ = prev_actions_.at[..., 0].set(action)
            prev_actions_ = jnp.where(
                timestep_.done[..., None],
                jnp.zeros_like(prev_actions_),
                prev_actions_,
            )

            terminations_update = timestep_.done.sum()
            success_events = _success_events(
                timestep_.done,
                timestep_.info["task_done"],
            )
            positive_termination_update = success_events.sum()
            positive_termination_steps_update = _positive_episode_step_sum(
                episode_steps_before,
                success_events,
            )
            (
                initial_episode_done_,
                initial_episode_successes_update,
                initial_episode_terminations_update,
            ) = _initial_episode_outcomes(
                initial_episode_done_,
                timestep_.done,
                timestep_.info["task_done"],
            )

            stats_ = RolloutStats(
                max_reward=jnp.maximum(stats_.max_reward, timestep_.reward.max()),
                min_reward=jnp.minimum(stats_.min_reward, timestep_.reward.min()),
                reward=stats_.reward + timestep_.reward.sum(),
                length=stats_.length + 1,
                episodes=stats_.episodes + terminations_update,
                positive_terminations=stats_.positive_terminations
                + positive_termination_update,
                terminations=stats_.terminations + terminations_update,
                positive_terminations_steps=stats_.positive_terminations_steps
                + positive_termination_steps_update,
                initial_episode_successes=stats_.initial_episode_successes
                + initial_episode_successes_update,
                initial_episode_terminations=stats_.initial_episode_terminations
                + initial_episode_terminations_update,
                action_0=stats_.action_0 + (action == 0).sum(),
                action_1=stats_.action_1 + (action == 1).sum(),
                action_2=stats_.action_2 + (action == 2).sum(),
                action_3=stats_.action_3 + (action == 3).sum(),
                action_4=stats_.action_4 + (action == 4).sum(),
                action_5=stats_.action_5 + (action == 5).sum(),
                action_6=stats_.action_6 + (action == 6).sum(),
                action_7=stats_.action_7 + (action == 7).sum(),
            )
            return rng_, stats_, timestep_, prev_actions_, initial_episode_done_

        eval_step = _eval_step
        _PMAPPED_STEP_CACHE[key] = eval_step

    timestep = _strip_reward_components(timestep)
    prev_actions = jnp.zeros(
        (num_devices, num_envs, config.num_prev_actions), dtype=jnp.int32
    )
    stats = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.asarray(x)[None], num_devices, axis=0),
        RolloutStats(),
    )
    initial_episode_done = jnp.zeros(
        (num_devices, num_envs), dtype=jnp.bool_
    )

    for _ in range(num_rollouts):
        rng, stats, timestep, prev_actions, initial_episode_done = eval_step(
            rng,
            stats,
            timestep,
            prev_actions,
            initial_episode_done,
            train_state,
        )

    return jax.block_until_ready(stats)


def rollout(rng, env, env_params, train_state, config):
    key = (id(env), id(config))
    pmapped = _PMAPPED_ROLLOUT_CACHE.get(key)
    if pmapped is None:
        def _closure(rng_, env_params_, train_state_):
            return _rollout_impl(rng_, env, env_params_, train_state_, config)

        pmapped = jax.pmap(_closure, axis_name="devices")
        _PMAPPED_ROLLOUT_CACHE[key] = pmapped
    return pmapped(rng, env_params, train_state)
