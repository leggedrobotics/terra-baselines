"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

from functools import partial
import optax
import jax
import time
import jax.numpy as jnp
from jax import Array
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
from terra.Transition import Transition
from terra.actions import Action
from terra.env import TerraEnvBatch, TerraEnv
from terra.maps_buffer import init_maps_buffer, MapsBuffer
from terra.state import State
from terra.wrappers import TraversabilityMaskWrapper, LocalMapWrapper

from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from utils.helpers import save_pkl_object
from terra.config import EnvConfig, BatchConfig
from typing import NamedTuple

from utils.ppo import ConstantRewardNormalizer, AdaptiveRewardNormalizer, AdaptiveScaledRewardNormalizer, \
    obs_to_model_input, loss_actor_and_critic, clip_action_maps_in_obs, cut_local_map_layers, wrap_action, update


def create_train_state(params, apply_fn, tx):
    @jax.pmap
    def _create_train_state(p):
        return TrainState.create(
            apply_fn=apply_fn,
            params=p,
            tx=tx,
        )

    return _create_train_state(params)

class TrainingConfig(NamedTuple):
    num_train_cycles: int
    lr_warmup: float
    lr_begin: float
    lr_end: float
    max_grad_norm: float
    reward_normalizer: int # constant:0, adaptive:1, adaptive-scaled": 2
    gamma: float
    gae_lambda: float
    num_train_envs: int
    clip_action_maps: bool
    mask_out_arm_extension: bool
    n_steps: int
    epoch_ppo: int
    max_reward: int
    minibatch_size: int
    n_minibatch: int
    clip_eps:int
    entropy_coeff: int

class StepStateUnvectorized(NamedTuple):
    train_state: TrainState
    action_mask: Any
    env: TerraEnv
    reward_normalizer: Any
    rng: jax.random.PRNGKey
    train_config: Any
    batch_config: BatchConfig
    map_buffer: MapsBuffer


class StepState(NamedTuple):
    env_state: State
    obs: Any
    maps_buffer_keys: Any
    env_config: EnvConfig


class Transition2(NamedTuple):
    done: Any
    action: Any
    value: Any
    reward: Any
    log_prob: Any
    obs: Any


# info: Any

def env_step(env: TerraEnv, states: State, actions: Action, maps_buffer_keys: jax.random.PRNGKey,
             maps_buffer: MapsBuffer, env_config: EnvConfig):
    (
        target_maps,
        padding_masks,
        trench_axes,
        trench_type,
        dumpability_mask_init,
        maps_buffer_keys,
    ) = maps_buffer.get_map(maps_buffer_keys, env_config)
    transistion = env.step(states,
                           actions,
                           target_maps,
                           padding_masks,
                           trench_axes,
                           trench_type,
                           dumpability_mask_init,
                           env_config)
    return transistion, maps_buffer_keys


def forward(_step_state: StepState, _unvectorized_step_state: StepStateUnvectorized, rng: jax.random.PRNGKey):
    obs = _step_state.obs
    _train_state: TrainState = _unvectorized_step_state.train_state
    model_input_obs = obs_to_model_input(obs)

    print("input obs shape", model_input_obs[1].shape)
    value, logits_pi = _train_state.apply_fn(_train_state.params, model_input_obs,
                                             _unvectorized_step_state.action_mask)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    value = value[:, 0]
    wrapped_action = wrap_action(action, _unvectorized_step_state.batch_config.action_type)
    return log_prob, value, wrapped_action


def generateObservation(step_state_tuple: (StepState, StepStateUnvectorized), _):
    _step_state, unvectorized_step_state = step_state_tuple
    next_rng, current_rng = jax.random.split(unvectorized_step_state.rng, 2)

    # Vectorized forward
    log_prob, value, wrapped_action = forward(_step_state, unvectorized_step_state, current_rng)

    vectorized_step = jax.vmap(env_step, in_axes=(None, 0, 0, 0, None, 0))
    transition, next_map_buffer_keys = vectorized_step(unvectorized_step_state.env,
                                                       _step_state.env_state,
                                                       wrapped_action,
                                                       _step_state.maps_buffer_keys,
                                                       unvectorized_step_state.map_buffer,
                                                       _step_state.env_config)

    # Todo Vectorization needed?
    reward_normalizer = unvectorized_step_state.reward_normalizer.update(transition.reward)
    new_unvectorized_step_state = StepStateUnvectorized(unvectorized_step_state.train_state,
                                                        transition.infos["action_mask"],
                                                        unvectorized_step_state.env,
                                                        reward_normalizer,
                                                        next_rng,
                                                        unvectorized_step_state.train_config,
                                                        unvectorized_step_state.batch_config,
                                                        unvectorized_step_state.map_buffer
                                                        )
    obs = transition.obs
    if unvectorized_step_state.train_config.clip_action_maps:
        obs = clip_action_maps_in_obs(obs)
    if unvectorized_step_state.train_config.mask_out_arm_extension:
        obs = cut_local_map_layers(obs)

    new_step_state = StepState(transition.next_state,
                               obs,
                               next_map_buffer_keys,
                               _step_state.env_config)
    terminated = transition.infos["done_task"]
    record = Transition2(terminated, wrapped_action.action, value, transition.reward, log_prob,
                         obs_to_model_input(_step_state.obs))
    return (new_step_state, new_unvectorized_step_state), record


def _loss_fn(params, traj_batch, gae, targets, unvectorized_step_state):
    def _forward(_train_state, action_mask, action):
        value, logits_pi = _train_state.apply_fn(params, traj_batch.obs,
                                                 action_mask)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        log_prob = pi.log_prob(action)
        value = value[:, 0]
        return log_prob, value, pi.entropy()

    log_prob, value, entropy = _forward(unvectorized_step_state.train_state, unvectorized_step_state.action_mask,
                                        traj_batch.action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
    ).clip(-unvectorized_step_state.train_config.clip_eps, unvectorized_step_state.train_config.clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - unvectorized_step_state.train_config.clip_eps,
                1.0 + unvectorized_step_state.train_config.clip_eps,
            )
            * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = entropy.mean()

    total_loss = (
            loss_actor
            + unvectorized_step_state.train_config.max_grad_norm * value_loss
            - unvectorized_step_state.train_config.entropy_coeff * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def _update_minbatch(t_state, batch_info):
    train_state, unvectorized_step_state = t_state
    traj_batch, advantages, targets = batch_info

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params, traj_batch, advantages, targets, unvectorized_step_state
    )
    return (train_state, unvectorized_step_state), grads


def _update_epoch(update_state, unused):
    (train_state, unvectorized_step_state), traj_batch, advantages, targets, rng = update_state

    rng, _rng = jax.random.split(rng)
    batch_size = unvectorized_step_state.train_config.minibatch_size * unvectorized_step_state.train_config.n_minibatch
    assert (
            batch_size == unvectorized_step_state.train_config.n_steps * unvectorized_step_state.train_config.num_train_envs
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(_rng, batch_size)
    batch = (traj_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [unvectorized_step_state.train_config.n_minibatch, -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    (train_state, unvectorized_step_state), grads = jax.lax.scan(
        _update_minbatch, (train_state, unvectorized_step_state), minibatches
    )
    update_state = ((train_state, unvectorized_step_state), traj_batch, advantages, targets, rng)
    summed_grads = jax.tree_util.tree_map(lambda xs: jax.numpy.sum(xs, 0), grads)
    return update_state, summed_grads


def _state_to_obs_dict(state: State) -> dict[str, Array]:
    """
    Transforms a State object to an observation dictionary.
    """
    agent_state = jnp.hstack(
        [
            state.agent.agent_state.pos_base,  # pos_base is encoded in traversability_mask
            state.agent.agent_state.angle_base,
            state.agent.agent_state.angle_cabin,
            state.agent.agent_state.arm_extension,
            state.agent.agent_state.loaded,
        ]
    )
    return {
        "agent_state": agent_state,
        "local_map_action_neg": state.world.local_map_action_neg.map,
        "local_map_action_pos": state.world.local_map_action_pos.map,
        "local_map_target_neg": state.world.local_map_target_neg.map,
        "local_map_target_pos": state.world.local_map_target_pos.map,
        "local_map_dumpability": state.world.local_map_dumpability.map,
        "local_map_obstacles": state.world.local_map_obstacles.map,
        "traversability_mask": state.world.traversability_mask.map,
        "action_map": state.world.action_map.map,
        "target_map": state.world.target_map.map,
        "agent_width": state.agent.width,
        "agent_height": state.agent.height,
        "padding_mask": state.world.padding_mask.map,
        "dig_map": state.world.dig_map.map,
        "dumpability_mask": state.world.dumpability_mask.map,
    }


def reset_env(maps_buffer: MapsBuffer, seed: jax.random.PRNGKey, env_config: EnvConfig):
    key_1, key_2 = jax.random.split(seed)
    target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, maps_buffer_keys = maps_buffer.get_map_init(
        key_1, env_config)
    state = State.new(
        key_2, env_config, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
    )
    state = TraversabilityMaskWrapper.wrap(state)
    state = LocalMapWrapper.wrap(state)
    observation = _state_to_obs_dict(state)
    observation["do_preview"] = state._handle_do().world.action_map.map

    return state, observation, maps_buffer_keys


def train_ppo(rng, config, model, model_params, mle_log, env: TerraEnvBatch, curriculum: Curriculum, run_name: str,
              n_devices: int):
    if config["reward_normalizer"] == "constant":
        reward_normalizer_nr = 0
    if config["reward_normalizer"] == "adaptive":
        reward_normalizer_nr = 1
    if config["reward_normalizer"] == "adaptive-scaled":
        reward_normalizer_nr = 2
    converted_config = TrainingConfig(
        config["num_train_steps"] // config["n_steps"] // config["num_train_envs"],
        config["lr_warmup"],
        config["lr_begin"],
        config["lr_end"],
        config["max_grad_norm"],
        reward_normalizer_nr,
        config["gamma"],
        config["gae_lambda"],
        config["num_train_envs"],
        config["clip_action_maps"],
        config["mask_out_arm_extension"],
        config["n_steps"],
        config["epoch_ppo"],
        config["max_reward"],
        config["minibatch_size"],
        config["n_minibatch"],
        config["clip_eps"],
        config["entropy_coeff"],
    )
    num_steps_warm_up = int(converted_config.num_train_cycles * converted_config.lr_warmup * converted_config.n_steps * converted_config.num_train_envs)
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_init()

    schedule_fn = optax.linear_schedule(
        init_value=-float(converted_config.lr_begin),
        end_value=-float(converted_config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(converted_config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    if converted_config.reward_normalizer == 0:
        reward_normalizer = ConstantRewardNormalizer(max_reward=converted_config.max_reward)
    elif converted_config.reward_normalizer == 1:
        reward_normalizer = AdaptiveRewardNormalizer()
    elif converted_config.reward_normalizer == 2:
        reward_normalizer = AdaptiveScaledRewardNormalizer(max_reward=converted_config.max_reward)
    else:
        raise (ValueError(f"{converted_config.reward_normalizer=}"))

    def _calculate_gae(traj_batch, last_val):

        # Inherently vectorized, thus no vmap?
        def _get_advantages(gae_and_next_value, transition: Transition2):
            gae, next_value = gae_and_next_value

            delta = transition.reward + converted_config.gamma * next_value * (1 - transition.done) - transition.value
            gae = (
                    delta
                    + converted_config.gamma * converted_config.gae_lambda * (1 - transition.done) * gae
            )
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def _update_step():

        def _individual_gradients(_env: TerraEnv, _env_config: EnvConfig, _rng, _train_state: TrainState,
                                  maps_buffer: MapsBuffer, batch_config: BatchConfig):
            print("in _individual_gradients")

            current_1, current_2, current_3, next_rng = jax.random.split(_rng, 4)
            # Init batch over multiple devices with different env seeds
            k_dev_envs = jax.random.split(current_1, converted_config.num_train_envs)

            vectorized_env_reset = jax.vmap(reset_env, in_axes=(None, 0, 0))
            env_state, obs, maps_buffer_keys = vectorized_env_reset(maps_buffer, k_dev_envs, _env_config)
            # Vectorized reset
            # env_state, obs, maps_buffer_keys = _envBatch.reset(k_dev_envs, _env_config)

            if converted_config.clip_action_maps:
                obs = clip_action_maps_in_obs(obs)
            if converted_config.mask_out_arm_extension:
                obs = cut_local_map_layers(obs)

            multi_stepstate = jax.vmap(StepState, in_axes=(0, 0, 0, 0))
            step_state = multi_stepstate(env_state, obs, maps_buffer_keys, _env_config)
            step_state_unvectorized = StepStateUnvectorized(_train_state,
                                                            None,
                                                            _env,
                                                            reward_normalizer,
                                                            current_2,
                                                            converted_config,
                                                            batch_config,
                                                            maps_buffer)
            carry, progress = jax.lax.scan(generateObservation, (step_state, step_state_unvectorized), None,
                                           length=converted_config.n_steps)
            last_step_state, last_unvectorized_step_state = carry
            log_prob, value, wrapped_action = forward(last_step_state, last_unvectorized_step_state, current_3)

            advantages, targets = _calculate_gae(progress, value)
            train_state = last_unvectorized_step_state.train_state
            update_state = ((train_state, last_unvectorized_step_state), progress, advantages, targets, rng)

            # UPDATE NETWORK
            update_state, grads = jax.lax.scan(
                _update_epoch, update_state, None, converted_config.epoch_ppo
            )
            next_train_state = update_state[0]
            avg_grads = jax.lax.pmean(grads, axis_name="data")
            next_train_state[0].apply_gradients(avg_grads)
            return (_env, _env_config, next_rng, next_train_state), grads

        # Initialize the models on multiple GPUs with the same params
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)
        train_state = create_train_state(
            params=replicated_params,
            apply_fn=model.apply,
            tx=tx,
        )
        rng_step = jax.random.split(rng, n_devices)
        print("Here")
        parallel_gradients = jax.pmap(_individual_gradients, axis_name="data", static_broadcasted_argnums=(0, 4, 5))
        something, grads = parallel_gradients(env.terra_env,
                                              env_cfgs,
                                              rng_step,
                                              train_state,
                                              env.maps_buffer,
                                              env.batch_cfg)
        # avg_grads = jax.lax.pmean(grads, axis_name="data")
        # temp_ts = TrainState.create(
        #     apply_fn=model.apply,
        #     params=model_params,
        #     tx=tx,
        # )
        # temp_ts = temp_ts.apply_gradients(avg_grads)
        return something

    for _ in range(converted_config.num_train_cycles):
        model_params = _update_step()
