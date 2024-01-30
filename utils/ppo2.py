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
from terra.state import State

from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from utils.helpers import save_pkl_object
from terra.config import EnvConfig
from typing import NamedTuple

from utils.ppo import ConstantRewardNormalizer, AdaptiveRewardNormalizer, AdaptiveScaledRewardNormalizer, \
    obs_to_model_input, loss_actor_and_critic


def create_train_state(params, apply_fn, tx):
    @jax.pmap
    def _create_train_state(p):
        return TrainState.create(
            apply_fn=apply_fn,
            params=p,
            tx=tx,
        )

    return _create_train_state(params)



class StepState(NamedTuple):
    train_state: TrainState
    env_state: State
    obs: Any
    action_mask: Any
    env: TerraEnv
    maps_buffer_keys: Any
    rng: jax.random.PRNGKey
    reward_normalizer: Any


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

    def calculate_gae(value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = config["gamma"] * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]

    def _update_step():

        @partial(jax.pmap, axis_name="data", static_broadcasted_argnums=(0,))
        def _individual_gradients(env: TerraEnvBatch, _env_config: EnvConfig, _rng, _train_state: TrainState):

            @jax.vmap
            def _env_step(step_state: StepState, _) -> (StepState, Transition):
                current, next = jax.random.split(step_state.rng, 2)
                obs = step_state.obs
                _train_state: TrainState = step_state.train_state
                model_input_obs = obs_to_model_input(obs)
                value, logits_pi = _train_state.apply_fn(_train_state.params, model_input_obs, step_state.action_mask)
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=current)
                log_prob = pi.log_prob(action)
                value = value[:,0]
                transition = step_state.env.step(step_state.env_state,action, step_state.env_state.env_cfg, step_state.maps_buffer_keys)
                reward_normalizer = step_state.reward_normalizer.update(transition.reward)

                new_step_state = StepState(step_state.train_state,
                                           transition.next_state,
                                           transition.obs,
                                           transition.infos["action_mask"],
                                           step_state.maps_buffer_keys,
                                           next, reward_normalizer)
                terminated = transition.infos["done_task"]

                return new_step_state, (transition.obs,
                                        action.action,
                                        transition.reward,
                                        transition.done,
                                        log_prob,
                                        value,
                                        transition.infos["action_mask"],
                                        reward_normalizer)


            current, next = jax.random.split(_rng, 2)
            # Init batch over multiple devices with different env seeds
            k_dev_envs = jax.random.split(current, config["num_train_envs"])
            env_state, obs, maps_buffer_keys = env.reset(k_dev_envs, _env_config)
            # multi_reset = jax.vmap(env.reset, in_axes=(0, None))
            # env_state, obs, maps_buffer_keys = multi_reset(k_dev_envs, env_config)
            multi_stepstate = jax.vmap(StepState, in_axes=(None, 0, 0, None, None, None, None, None))
            step_state = multi_stepstate(_train_state, env_state, obs, None, env.terra_env, maps_buffer_keys, next, reward_normalizer)
            carry, progress = jax.lax.scan(_env_step, step_state, None, length=config["n_steps"])
            grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
            value = [step[5] for step in progress]
            log_pi = [step[4] for step in progress]
            action_mask = [step[6] for step in progress]
            actions = [step[1] for step in progress]
            gae, target = calculate_gae(value, [step[2] for step in progress], [step[3] for step in progress])
            total_loss, grads = grad_fn(_train_state.params,
                                        _train_state.apply_fn,
                                        obs=[step[0] for step in progress],
                                        target=target,
                                        value_old=value,
                                        log_pi_old=log_pi,
                                        gae=gae,
                                        action_mask=action_mask,
                                        action=actions,
                                        clip_eps=config["clip_eps"],
                                        critical_coeff=config["critic_coeff"],
                                        entropy_coeff=config["entropy_coeff"],
                                        )
            return total_loss, grads

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