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
from terra.config import EnvConfig, BatchConfig, ImmutableMapsConfig
from typing import NamedTuple
import wandb

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
    ppo2_num_training_cycles: int
    ppo2_num_epochs: int
    ppo2_num_env_started: int
    lr_warmup: float
    lr_begin: float
    lr_end: float
    max_grad_norm: float
    reward_normalizer: int  # constant:0, adaptive:1, adaptive-scaled": 2
    gamma: float
    gae_lambda: float
    num_training_envs: int
    clip_action_maps: bool
    mask_out_arm_extension: bool
    ppo2_num_steps: int
    max_reward: int
    ppo2_minibatch_size: int
    clip_eps: int
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
    value, logits_pi = _train_state.apply_fn(_train_state.params, model_input_obs,
                                             _unvectorized_step_state.action_mask)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    value = value[:, 0]
    wrapped_action = wrap_action(action, _unvectorized_step_state.batch_config.action_type)
    return log_prob, value, wrapped_action


def generateObservation(step_state_tuple: (StepState, StepStateUnvectorized), _, obs_conv: Callable):
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
                                                        unvectorized_step_state.map_buffer)
    obs = obs_conv(transition.obs)

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
    return (train_state, unvectorized_step_state), (grads, total_loss)


def _update_epoch(update_state, unused, train_config: TrainingConfig):
    (train_state, unvectorized_step_state), traj_batch, advantages, targets, rng = update_state

    rng, _rng = jax.random.split(rng)

    # assert (
    #         batch_size == unvectorized_step_state.train_config.n_steps * unvectorized_step_state.train_config.num_train_envs
    # ), "batch size must be equal to number of steps * number of envs"
    num_obs = train_config.ppo2_num_steps * train_config.num_training_envs
    permutation = jax.random.permutation(_rng, train_config.ppo2_minibatch_size)
    batch = (traj_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((num_obs,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [num_obs // train_config.ppo2_minibatch_size, -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    (next_train_state, unvectorized_step_state), (grads, loss) = jax.lax.scan(
        _update_minbatch, (train_state, unvectorized_step_state), minibatches
    )
    update_state = ((next_train_state, unvectorized_step_state), traj_batch, advantages, targets, rng)

    summed_grads = jax.tree_util.tree_map(lambda xs: jax.numpy.sum(xs, 0), grads)

    avg_loss = jax.tree_util.tree_map(
        lambda xs: jax.numpy.mean(xs, axis=0),
        loss
    )

    return update_state, (summed_grads, avg_loss)


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


def _reset(key_2, env_config, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init):
    state = State.new(
        key_2, env_config, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
    )
    state = TraversabilityMaskWrapper.wrap(state)
    state = LocalMapWrapper.wrap(state)
    observation = _state_to_obs_dict(state)
    observation["do_preview"] = state._handle_do().world.action_map.map
    return state, observation


def reset_env(maps_buffer: MapsBuffer, rng: jax.random.PRNGKey, env_config: EnvConfig, num_train_envs: int):
    key_1, key_2 = jax.random.split(rng)
    k_dev_envs = jax.random.split(key_1, num_train_envs)
    k_dev_envs_2 = jax.random.split(key_2, num_train_envs)
    target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, maps_buffer_keys = jax.vmap(
        maps_buffer.get_map_init, in_axes=(0,0))(k_dev_envs, env_config)

    state, observation = jax.vmap(_reset)(
        k_dev_envs_2,
        env_config,
        target_map,
        padding_mask,
        trench_axes,
        trench_type,
        dumpability_mask_init)
    return state, observation, maps_buffer_keys


def _calculate_gae(traj_batch, last_val, converted_config: TrainingConfig):
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

def conv_obs(obs, clip: bool, mask: bool):
    if clip:
        obs = clip_action_maps_in_obs(obs)
    if mask:
        obs = cut_local_map_layers(obs)
    return obs


@partial(jax.jit, static_argnums=(2,3))
def step_through_env(carry, _, converted_config: TrainingConfig, batch_config: BatchConfig):
    (_env, _env_config, _rng, _train_state, maps_buffer, reward_normalizer) = carry

    print("cycle")

    current_1, current_2, current_3, current_4, next_rng = jax.random.split(_rng, 5)
    # Init batch over multiple devices with different env seeds
    env_state, obs, maps_buffer_keys = reset_env(maps_buffer, current_1, _env_config, converted_config.num_training_envs)
    # Vectorized reset
    # env_state, obs, maps_buffer_keys = _envBatch.reset(k_dev_envs, _env_config)
    conv_obs_filled = partial(conv_obs, clip=converted_config.clip_action_maps, mask=converted_config.mask_out_arm_extension)
    obs = conv_obs_filled(obs)

    initial_action_mask = jax.vmap(State._get_action_mask, in_axes=(0, None))(env_state, batch_config.action_type)

    multi_stepstate = jax.vmap(StepState, in_axes=(0, 0, 0, 0))
    step_state = multi_stepstate(env_state, obs, maps_buffer_keys, _env_config)
    step_state_unvectorized = StepStateUnvectorized(_train_state,
                                                    initial_action_mask,
                                                    _env,
                                                    reward_normalizer,
                                                    current_2,
                                                    converted_config,
                                                    batch_config,
                                                    maps_buffer)
    generateObservationPartial = partial(generateObservation, obs_conv=conv_obs_filled)

    ## STEPPING ##
    carry, progress = jax.lax.scan(generateObservationPartial, (step_state, step_state_unvectorized), None,
                                   length=converted_config.ppo2_num_steps)
    last_step_state, last_unvectorized_step_state = carry
    log_prob, value, wrapped_action = forward(last_step_state, last_unvectorized_step_state, current_3)

    advantages, targets = _calculate_gae(progress, value, converted_config)
    train_state = last_unvectorized_step_state.train_state
    update_state = ((train_state, last_unvectorized_step_state), progress, advantages, targets, current_4)

    update_epoch_prefilled = partial(_update_epoch, train_config=converted_config)
    # UPDATE NETWORK
    update_state, (summed_grads, avg_loss) = jax.lax.scan(
        update_epoch_prefilled, update_state, None, converted_config.ppo2_num_epochs
    )
    summed_grads = jax.tree_util.tree_map(lambda xs: jax.numpy.sum(xs, 0), summed_grads)
    avg_loss = jax.tree_util.tree_map(
        lambda xs: jax.numpy.mean(xs, axis=0),
        avg_loss
    )
    (next_train_state, next_unvectorized_step_state) = update_state[0]

    avg_grads = jax.lax.pmean(summed_grads, axis_name="data")
    updated_train_state = next_train_state.apply_gradients(grads=avg_grads)

    # updated_train_state = next_train_state.apply_gradients(grads=summed_grads)

    updated_carry = (_env, _env_config, next_rng, updated_train_state, maps_buffer, reward_normalizer)
    return updated_carry, avg_loss


def _individual_gradients(_env: TerraEnv, _env_config: EnvConfig, _rng, _train_state: TrainState,
                          maps_buffer: MapsBuffer, batch_config: BatchConfig, converted_config: TrainingConfig, reward_normalizer):
    initial_carry = (_env, _env_config, _rng, _train_state, maps_buffer, reward_normalizer)
    step_through_env_fixed = partial(step_through_env, converted_config=converted_config, batch_config=batch_config)
    final_carry, loss = jax.lax.scan(step_through_env_fixed, initial_carry, None, length=converted_config.ppo2_num_env_started)


    (_env, _env_config, next_rng, updated_train_state, _, _) = final_carry
    return (_env, _env_config, next_rng, updated_train_state), loss


def train_ppo(rng, config, model, model_params, mle_log, env: TerraEnvBatch, curriculum: Curriculum, run_name: str,
              n_devices: int):
    if config["reward_normalizer"] == "constant":
        reward_normalizer_nr = 0
    if config["reward_normalizer"] == "adaptive":
        reward_normalizer_nr = 1
    if config["reward_normalizer"] == "adaptive-scaled":
        reward_normalizer_nr = 2
    converted_config = TrainingConfig(
        config["ppo2_num_training_cycles"],
        config["ppo2_num_epochs"],
        config["ppo2_num_env_started"],
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
        config["ppo2_num_steps"],
        config["max_reward"],
        config["ppo2_minibatch_size"],
        config["clip_eps"],
        config["entropy_coeff"],
    )
    print(f"Launching {converted_config.ppo2_num_training_cycles} training cycles")
    num_steps_warm_up = int(
        converted_config.ppo2_num_training_cycles * converted_config.lr_warmup * converted_config.ppo2_num_steps * converted_config.num_training_envs)
    env_cfgs, dof = curriculum.get_cfgs_init()
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

    # Initialize the models on multiple GPUs with the same params
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)
    train_state = create_train_state(
        params=replicated_params,
        apply_fn=model.apply,
        tx=tx,
    )

    rng_step = jax.random.split(rng, n_devices)

    prefiled_individual_gradients = partial(_individual_gradients,
                                            batch_config=env.batch_cfg,
                                            converted_config=converted_config,
                                            reward_normalizer=reward_normalizer)

    parallel_gradients = jax.pmap(prefiled_individual_gradients, axis_name="data",
                                  static_broadcasted_argnums=(0, 4))

    timer = time.time()
    for i in range(converted_config.ppo2_num_training_cycles):
        something, loss = parallel_gradients(env.terra_env,
                                              env_cfgs,
                                              rng_step,
                                              train_state,
                                              env.maps_buffer)
        avg_loss = jax.tree_util.tree_map(
            lambda xs: jax.numpy.mean(xs, axis=0),
            loss
        )
        done = time.time()
        train_state = something[3]
        rng_step = something[2]
        print(f"---Cycle {i}/{converted_config.ppo2_num_training_cycles - 1}:---")
        num_steps = config['num_train_envs'] * config['ppo2_num_env_started'] * config['ppo2_num_steps']
        secs = done - timer
        print(f"{num_steps} steps in {secs:.2f} seconds: {num_steps/secs:.4f} steps/sec")
        print("Total:  ",avg_loss[0])
        print("Value:  ",avg_loss[1][0])
        print("Actor:  ",avg_loss[1][1])
        print("Entropy:",avg_loss[1][2])
        timer = done

    # something, grads = _individual_gradients(env.terra_env,
    #                                       env_cfgs,
    #                                       rng,
    #                                       train_state,
    #                                       env.maps_buffer,
    #                                       env.batch_cfg,
    #                                       converted_config,
    #                                       reward_normalizer)

    return (None, None, train_state.params, None)
