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
from terra.env import TerraEnvBatch, TerraEnv
from terra.maps_buffer import init_maps_buffer
from terra.state import State

from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from utils.helpers import save_pkl_object
from terra.config import EnvConfig
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


class StepStateUnvectorized(NamedTuple):
    train_state: TrainState
    action_mask: Any
    env: TerraEnvBatch
    maps_buffer_keys: Any
    rng: jax.random.PRNGKey
    reward_normalizer: Any


class StepState(NamedTuple):
    env_state: State
    obs: Any


def forward(_step_state: StepState, _unvectorized_step_state: StepStateUnvectorized):
    current, next = jax.random.split(_unvectorized_step_state.rng, 2)

    obs = _step_state.obs
    _train_state: TrainState = _unvectorized_step_state.train_state
    model_input_obs = obs_to_model_input(obs)

    value, logits_pi = _train_state.apply_fn(_train_state.params, model_input_obs,
                                             _unvectorized_step_state.action_mask)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    action = pi.sample(seed=current)
    log_prob = pi.log_prob(action)
    value = value[:, 0]
    wrapped_action = wrap_action(action, _unvectorized_step_state.env.batch_cfg.action_type)
    return log_prob, value, wrapped_action



def train_ppo(rng, config, model, model_params, mle_log, env: TerraEnvBatch, curriculum: Curriculum, run_name: str,
              n_devices: int):
    num_updates = config["num_train_steps"] // config["n_steps"] // config["num_train_envs"]
    num_steps_warm_up = int(config["num_train_steps"] * config["lr_warmup"])
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_init()

    schedule_fn = optax.linear_schedule(
        init_value=-float(config["lr_begin"]),
        end_value=-float(config["lr_end"]),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    if config["reward_normalizer"] == "constant":
        reward_normalizer = ConstantRewardNormalizer(max_reward=config["max_reward"])
    elif config["reward_normalizer"] == "adaptive":
        reward_normalizer = AdaptiveRewardNormalizer()
    elif config["reward_normalizer"] == "adaptive-scaled":
        reward_normalizer = AdaptiveScaledRewardNormalizer(max_reward=config["max_reward"])
    else:
        raise (ValueError(f"{config['reward_normalizer']=}"))

    def _calculate_gae(traj_batch, last_val):

        # Inherently vectorized, thus no vmap?
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition[3],
                transition[5],
                transition[2],
            )
            delta = reward + config["gamma"] * next_value * (1 - done) - value
            gae = (
                    delta
                    + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch[5]

    def _update_step():

        @partial(jax.pmap, axis_name="data", static_broadcasted_argnums=(0,))
        def _individual_gradients(_envBatch: TerraEnvBatch, _env_config: EnvConfig, _rng, _train_state: TrainState):



            def generateObservation(step_state_tuple: (StepState, StepStateUnvectorized), _):
                _step_state, unvectorized_step_state = step_state_tuple

                # Vectorized forward
                log_prob, value, wrapped_action = forward(_step_state, unvectorized_step_state)

                # Vectorized env step
                transition, next_map_buffer_keys = unvectorized_step_state.env.step(_step_state.env_state,
                                                                                    wrapped_action, _env_config,
                                                                                    unvectorized_step_state.maps_buffer_keys)

                # Todo Vectorization needed?
                reward_normalizer = unvectorized_step_state.reward_normalizer.update(transition.reward)

                new_unvectorized_step_state = StepStateUnvectorized(unvectorized_step_state.train_state,
                                                                    transition.infos["action_mask"],
                                                                    unvectorized_step_state.env,
                                                                    unvectorized_step_state.maps_buffer_keys,
                                                                    next, reward_normalizer)
                new_step_state = StepState(transition.next_state,
                                           transition.obs)
                terminated = transition.infos["done_task"]

                return (new_step_state, new_unvectorized_step_state), (transition.obs,
                                                                       wrapped_action.action,
                                                                       transition.reward,
                                                                       transition.done,
                                                                       log_prob,
                                                                       value,
                                                                       transition.infos["action_mask"],
                                                                       reward_normalizer)

            current, next = jax.random.split(_rng, 2)

            # Init batch over multiple devices with different env seeds
            k_dev_envs = jax.random.split(current, config["num_train_envs"])

            # Vectorized reset
            env_state, obs, maps_buffer_keys = _envBatch.reset(k_dev_envs, _env_config)

            if config["clip_action_maps"]:
                obs = clip_action_maps_in_obs(obs)
            if config["mask_out_arm_extension"]:
                obs = cut_local_map_layers(obs)

            multi_stepstate = jax.vmap(StepState, in_axes=(0, 0))
            step_state = multi_stepstate(env_state, obs)
            step_state_unvectorized = StepStateUnvectorized(_train_state, None, _envBatch, maps_buffer_keys, next,
                                                            reward_normalizer)
            carry, progress = jax.lax.scan(generateObservation, (step_state, step_state_unvectorized), None,
                                           length=config["n_steps"])

            last_step_state, last_unvectorized_step_state = carry
            log_prob, value, wrapped_action = forward(last_step_state, last_unvectorized_step_state)


            advantages, targets = _calculate_gae(progress, value)
            update_state = (carry, progress, advantages, targets, rng)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (train_state, unvectorized_step_state), traj_batch, advantages, targets, rng = update_state

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):

                        def _forward(_train_state, action_mask, action):
                            model_input_obs = obs_to_model_input(traj_batch.obs)

                            value, logits_pi = _train_state.apply_fn(_train_state.params, model_input_obs,
                                                                     action_mask)
                            pi = tfp.distributions.Categorical(logits=logits_pi)
                            log_prob = pi.log_prob(action)
                            value = value[:, 0]
                            return log_prob, value, pi.entropy()

                        log_prob, value, entropy = _forward(train_state, unvectorized_step_state.action_mask, traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
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
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = entropy.mean()

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
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
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss


            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

        # Initialize the models on multiple GPUs with the same params
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)
        train_state = create_train_state(
            params=replicated_params,
            apply_fn=model.apply,
            tx=tx,
        )
        rng_step = jax.random.split(rng, n_devices)
        total_loss, grads = _individual_gradients(env, env_cfgs, rng_step, train_state)
        print(grads.shape())

    _update_step()
