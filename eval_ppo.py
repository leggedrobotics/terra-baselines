# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple
from train_debug import select_action_ppo, wrap_action


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

    action_0: jax.Array = jnp.asarray(0)
    action_1: jax.Array = jnp.asarray(0)
    action_2: jax.Array = jnp.asarray(0)
    action_3: jax.Array = jnp.asarray(0)
    action_4: jax.Array = jnp.asarray(0)
    action_5: jax.Array = jnp.asarray(0)
    action_6: jax.Array = jnp.asarray(0)
    action_7: jax.Array = jnp.asarray(0)
    action_8: jax.Array = jnp.asarray(0)

# @partial(jax.pmap, axis_name="devices")
def rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    num_envs: int,
    num_rollouts: int,
) -> RolloutStats:
    def _cond_fn(carry):
        _, stats, _ = carry
        # Check if the number of steps has been reached
        return jnp.less(stats.length, num_rollouts + 1)

    def _body_fn(carry):
        rng, stats, timestep = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        action, _, _, _ = select_action_ppo(train_state, timestep.observation, _rng_model)
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        timestep = env.step(timestep, action_env, _rng_step)

        positive_termination_update = timestep.info['task_done'].sum()
        
        terminations_update = timestep.done.sum()

        # Replace jax.debug.print with:
        # host_callback.id_tap(print_debug, (positive_termination_update, timeout_update), 
        #                     result=(positive_termination_update, timeout_update))

        stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward.max()),
            min_reward=jnp.minimum(stats.min_reward, timestep.reward.min()),
            reward=stats.reward + timestep.reward.sum(),  # Ensure correct aggregation
            length=stats.length + 1,
            episodes=stats.episodes + timestep.done.any(),
            positive_terminations=stats.positive_terminations + positive_termination_update,
            terminations=stats.terminations + terminations_update,
            action_0=stats.action_0 + (action == 0).sum(),
            action_1=stats.action_1 + (action == 1).sum(),
            action_2=stats.action_2 + (action == 2).sum(),
            action_3=stats.action_3 + (action == 3).sum(),
            action_4=stats.action_4 + (action == 4).sum(),
            action_5=stats.action_5 + (action == 5).sum(),
            action_6=stats.action_6 + (action == 6).sum(),
            action_7=stats.action_7 + (action == 7).sum(),
            action_8=stats.action_8 + (action == 8).sum(),
        )
        carry = (rng, stats, timestep)
        return carry
    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    init_carry = (rng, RolloutStats(), timestep)

    # final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    final_carry = jax.lax.fori_loop(0, num_rollouts, lambda i, carry: _body_fn(carry), init_carry)
    return final_carry[1]

