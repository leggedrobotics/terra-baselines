"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

from functools import partial
import optax
import jax
# from jax import config
# config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax import Array
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
from terra.env import TerraEnvBatch
from utils.helpers import append_to_pkl_object
from utils.curriculum import Curriculum
from utils.reset_manager import ResetManager
from tensorflow_probability.substrates import jax as tfp


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
    ):
        self.num_envs = num_envs
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda
    
    # TODO jit
    # @partial(jax.jit, static_argnums=0)
    def reset(self, action_size, observation_shapes, num_actions):
        return {
            "states": {
                key: jnp.empty(
                (self.n_steps, self.num_envs, *value)
                )
                for key, value in observation_shapes.items()
            },
            "action_mask": jnp.empty(
                (self.n_steps, self.num_envs, num_actions),
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, obs, action, reward, done, log_pi, value, action_mask):
        return {
                "states": {
                    "agent_states": buffer["states"]["agent_states"].at[buffer["_p"]].set(obs["agent_state"]),
                    "local_map_action": buffer["states"]["local_map_action"].at[buffer["_p"]].set(obs["local_map_action"]),
                    "local_map_target": buffer["states"]["local_map_target"].at[buffer["_p"]].set(obs["local_map_target"]),
                    "action_map": buffer["states"]["action_map"].at[buffer["_p"]].set(obs["action_map"]),
                    "target_map": buffer["states"]["target_map"].at[buffer["_p"]].set(obs["target_map"]),
                    "traversability_mask": buffer["states"]["traversability_mask"].at[buffer["_p"]].set(obs["traversability_mask"]),
                    "do_preview": buffer["states"]["do_preview"].at[buffer["_p"]].set(obs["do_preview"]),
                    "dig_map": buffer["states"]["dig_map"].at[buffer["_p"]].set(obs["dig_map"]),
                },
                "action_mask": buffer["action_mask"].at[buffer["_p"]].set(action_mask.squeeze()),
                "actions": buffer["actions"].at[buffer["_p"]].set(action.squeeze()),
                "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
                "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
                "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
                "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
                "_p": (buffer["_p"] + 1) % self.n_steps,
            }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = (
            (
                buffer["states"]["agent_states"][:-1],
                buffer["states"]["local_map_action"][:-1],
                buffer["states"]["local_map_target"][:-1],
                buffer["states"]["action_map"][:-1],
                buffer["states"]["target_map"][:-1],
                buffer["states"]["traversability_mask"][:-1],
                buffer["states"]["do_preview"][:-1],
                buffer["states"]["dig_map"][:-1],
            ),
            buffer["action_mask"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
            buffer["rewards"][:-1],  # just for logging
            buffer["dones"][:-1],  # just for logging
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


class RolloutManager(object):
    def __init__(self, env, n_envs: int, rl_config, observation_shapes, model = None):
        # Setup functionalities for vectorized batch rollout
        self.env: TerraEnvBatch = env
        # self.apply_fn = model.apply
        self.select_action = self.select_action_ppo
        self.select_action_deterministic = self.select_action_ppo_deterministic
        self.n_envs = n_envs
        self.reset_manager_evaluate = ResetManager(rl_config, observation_shapes, eval=True)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RolloutManager) and self.env == __o.env

    def __hash__(self) -> int:
        return hash((RolloutManager, self.env))

    @partial(jax.jit, static_argnums=0)
    def select_action_ppo(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        action_mask: Array,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        # Prepare policy input from Terra State
        obs = obs_to_model_input(obs)

        value, pi = policy(train_state.apply_fn, train_state.params, obs, action_mask)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0]
    
    @partial(jax.jit, static_argnums=0)
    def select_action_ppo_deterministic(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        action_mask: Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        # Prepare policy input from Terra State
        obs = obs_to_model_input(obs)

        value, action = policy_deterministic(train_state.apply_fn, train_state.params, obs, action_mask)
        return action, value[:, 0]

    # @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, keys, env_cfgs):
        seeds = jnp.array([k[0] for k in keys])
        return self.env.reset(seeds, env_cfgs)

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, states, actions, env_cfgs, maps_buffer_keys, force_resets):
        return self.env.step(states, actions, env_cfgs, maps_buffer_keys, force_resets)
    
    @partial(jax.jit, static_argnums=(0, 3, 6))
    def _batch_evaluate(self, rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs):
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        state, obs, maps_buffer_keys = self.batch_reset(jax.random.split(rng_reset, num_envs), env_cfgs)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask, done, action_mask, maps_buffer_keys = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _ = self.select_action_deterministic(train_state, obs, action_mask)
            jax.debug.print("bicount action = {x}", x=jnp.bincount(action, length=9))
            force_resets_dummy = self.reset_manager_evaluate.dummy()
            next_s, (next_o, reward, done, infos), maps_buffer_keys = self.batch_step(
                state,
                wrap_action(action.squeeze(), self.env.batch_cfg.action_type),
                env_cfgs,
                maps_buffer_keys,
                force_resets_dummy,  # TODO make it real
            )
            action_mask = infos["action_mask"]
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = [
                next_o,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
                done,
                action_mask,
                maps_buffer_keys
            ], [{k: v[:n_evals_save] for k, v in obs.items()}]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),  # cum reward
                jnp.array(num_envs * [1.0]),  # valid mask
                jnp.array(num_envs * [False]),  # dones
                action_mask_init[None].repeat(num_envs, 0),
                maps_buffer_keys
            ],
            (),
            600, #num_steps_test_rollouts,
        )

        cum_return = carry_out[4].squeeze()
        dones = carry_out[6].squeeze()
        return jnp.mean(cum_return), dones, scan_out[0]
    
    
    def batch_evaluate(self, rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs, folder_name):
        """Rollout an episode with lax.scan. and save outputs to pkl"""
        cum_return_mean, dones, obs_log = self._batch_evaluate(rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs)
        # Append sample to pkl file
        jax.experimental.io_callback(partial(append_to_pkl_object, foldername=folder_name), None, obs_log, step)
        return cum_return_mean, dones

@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    action_mask: Array
):
    value, logits_pi = apply_fn(params, obs, action_mask)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    return value, pi

@partial(jax.jit, static_argnums=0)
def policy_deterministic(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    action_mask: Array
):
    value, logits_pi = apply_fn(params, obs, action_mask)
    return value, np.argmax(logits_pi, axis=-1)

def train_ppo(rng, config, model, params, mle_log, env: TerraEnvBatch, curriculum: Curriculum, reset_manager: ResetManager):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    if config["wandb"]:
        import wandb
    
    num_total_epochs = int(config["num_train_steps"] // config["num_train_envs"] + 1)
    num_steps_warm_up = int(config["num_train_steps"] * config["lr_warmup"])
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

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(env, config["num_train_envs"], config, env.observation_shapes)
    num_actions = env.batch_cfg.action_type.get_num_actions()

    batch_manager = BatchManager(
        discount=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_steps=config["n_steps"] + 1,
        num_envs=config["num_train_envs"],
    )

    @jax.jit
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        batch,
        action_mask: Array,
        rng: jax.random.PRNGKey,
        env_cfgs,
        maps_buffer_keys: jax.random.PRNGKey,
    ):
        new_key, key_step = jax.random.split(rng)
        action, log_pi, value = rollout_manager.select_action(
            train_state, obs, action_mask, key_step
        )
        action = wrap_action(action, rollout_manager.env.batch_cfg.action_type)
        
        force_resets = reset_manager.update(obs, action)
        next_state, (next_obs, reward, done, infos), maps_buffer_keys = rollout_manager.batch_step(state, action, env_cfgs, maps_buffer_keys, force_resets)
        batch = batch_manager.append(
            batch, obs, action.action, reward, done, log_pi, value, infos["action_mask"]
        )
        return train_state, next_obs, next_state, batch, new_key, infos["action_mask"], done, maps_buffer_keys

    batch = batch_manager.reset(
        action_size=rollout_manager.env.actions_size,
        observation_shapes=rollout_manager.env.observation_shapes,
        num_actions=num_actions
    )

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_init()
    state, obs, maps_buffer_keys = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config["num_train_envs"]), env_cfgs
    )

    total_steps = 0
    log_steps, log_return = [], []
    action_mask_init = jnp.ones((num_actions,), dtype=jnp.bool_)
    action_mask = action_mask_init.copy()[None].repeat(config["num_train_envs"], 0)
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    dones_after_update = np.zeros(config["num_train_envs"], dtype=np.bool_)  # reset dones
    for step in t:
        train_state, obs, state, batch, rng_step, action_mask, dones, maps_buffer_keys = get_transition(
            train_state,
            obs,
            state,
            batch,
            action_mask,
            rng_step,
            env_cfgs,
            maps_buffer_keys
        )
        dones_after_update = dones_after_update | dones
        total_steps += config["num_train_envs"]
        if step % (config["n_steps"] + 1) == 0:
            metric_dict, train_state, rng_update = update(
                train_state,
                batch_manager.get(batch),
                config["num_train_envs"],
                config["n_steps"],
                config["n_minibatch"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                rng_update,
                config["fast_compile"]
            )

            if config["wandb"]:
                wandb.log({**metric_dict, **dofs_count_dict})
            
            env_cfgs, dofs_count_dict = curriculum.get_cfgs(metric_dict, dones_after_update)

            batch = batch_manager.reset(
                action_size=rollout_manager.env.actions_size,
                observation_shapes=rollout_manager.env.observation_shapes,
                num_actions=num_actions
            )

            dones_after_update = np.zeros(config["num_train_envs"], dtype=np.bool_)  # reset dones


        if (step + 1) % config["evaluate_every_epochs"] == 0:
            rng, rng_eval = jax.random.split(rng)
            env_cfgs_eval, dofs_count_dict_eval = curriculum.get_cfgs_eval()
            rewards, dones = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config["num_test_rollouts"],
                step,
                action_mask_init,
                config["n_evals_save"],
                env_cfgs_eval,
                "agents/Terra/" + config["run_name"]
            )
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()

            if mle_log is not None:
                mle_log.update(
                    {"num_steps": total_steps},
                    {"return": rewards},
                    model=train_state.params,
                    save=True,
                )
            if config["wandb"]:
                wandb.log({"eval - cum_reward": rewards, "eval - dones %": 100 * dones.sum() / dones.shape[0]})

    return (
        log_steps,
        log_return,
        train_state.params,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])

def obs_to_model_input(obs):
    """
    Need to convert Dict to List to make it usable by JAX.
    """
    return [
        obs["agent_state"],
        obs["local_map_action"],
        obs["local_map_target"],
        obs["action_map"],
        obs["target_map"],
        obs["traversability_mask"],
        obs["do_preview"],
        obs["dig_map"],
    ]

    # Legacy: Categorical MLP
    # num_train_envs = obs["local_map"].shape[0]
    # print(f"{num_train_envs=}")
    # obs = jnp.hstack([v.reshape(num_train_envs, -1) for v in obs.values()])
    # return obs

def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action

# def obs_to_model_input_batch(obs):
#     obs = [v.swapaxes(0, 1).reshape(v.shape[0] * v.shape[1], -1) for v in obs]
#     return jnp.hstack(obs)

def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    obs: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action_mask: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> jnp.ndarray:

    value_pred, logits_pi = apply_fn(params_model, obs, action_mask)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    value_pred = value_pred[:, 0]

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    log_prob = pi.log_prob(action[..., -1])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_losses_individual = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
    value_loss = value_losses_individual.mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    gae = (gae - gae_mean) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor + critic_coeff * value_loss - entropy_coeff * entropy
    )

    return total_loss, (
        value_loss,
        loss_actor,
        entropy,
        value_pred.mean(),
        target.mean(),
        gae_mean,
    )


def update(
    train_state: TrainState,
    batch: Tuple,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    epoch_ppo: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    rng: jax.random.PRNGKey,
    fast_compile
):
    """Perform multiple epochs of updates with multiple updates."""
    obs, action_mask, action, log_pi_old, value, target, gae, rewards, dones = batch
    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch
    idxes = jnp.arange(num_envs * n_steps)
    avg_metrics_dict = defaultdict(int)

    for _ in range(epoch_ppo):
        if fast_compile:
            idxes = jax.random.permutation(rng, idxes)
            idxes_list = jnp.array([
                idxes[start : start + size_minibatch]
                for start in jnp.arange(0, size_batch, size_minibatch)
            ])

            # Scan option
            train_state, total_loss = update_epoch_scan(
                    train_state,
                    idxes_list,
                    [flatten_dims(el) for el in obs],
                    flatten_dims(action_mask),
                    flatten_dims(action),
                    flatten_dims(log_pi_old),
                    flatten_dims(value),
                    jnp.array(flatten_dims(target)),
                    jnp.array(flatten_dims(gae)),
                    clip_eps,
                    entropy_coeff,
                    critic_coeff,
                    n_minibatch
                )
            
            # Fori_loop option
            # train_state, total_loss = update_epoch_fori_loop(
            #         train_state,
            #         idxes_list,
            #         [flatten_dims(el) for el in obs],
            #         flatten_dims(action_mask),
            #         flatten_dims(action),
            #         flatten_dims(log_pi_old),
            #         flatten_dims(value),
            #         jnp.array(flatten_dims(target)),
            #         jnp.array(flatten_dims(gae)),
            #         clip_eps,
            #         entropy_coeff,
            #         critic_coeff,
            #         n_minibatch
            #     )
        else:
            idxes = jax.random.permutation(rng, idxes)
            idxes_list = [
                idxes[start : start + size_minibatch]
                for start in jnp.arange(0, size_batch, size_minibatch)
            ]
            train_state, total_loss = update_epoch(
                train_state,
                idxes_list,
                [flatten_dims(el) for el in obs],
                flatten_dims(action_mask),
                flatten_dims(action),
                flatten_dims(log_pi_old),
                flatten_dims(value),
                jnp.array(flatten_dims(target)),
                jnp.array(flatten_dims(gae)),
                clip_eps,
                entropy_coeff,
                critic_coeff,
            )

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            value_pred,
            target_val,
            gae_val,
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)
        avg_metrics_dict["max_reward"] += np.asarray(rewards.max())
        avg_metrics_dict["min_reward"] += np.asarray(rewards.min())

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (epoch_ppo)

    avg_metrics_dict["avg dones %"] += np.asarray(100 * dones.mean())
    avg_metrics_dict["values_individual"] += np.asarray(value.mean(0))
    avg_metrics_dict["targets_individual"] += np.asarray(target.mean(0))

    return avg_metrics_dict, train_state, rng


@jax.jit
def update_epoch(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action_mask,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
):
    for idx in idxes:
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            train_state.apply_fn,
            obs=[el[idx] for el in obs], #obs[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            action_mask=action_mask[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        train_state = train_state.apply_gradients(grads=grads)

    # Note: the total_loss returned is w.r.t. the last minibatch only
    return train_state, total_loss

@partial(jax.jit, static_argnames=("n_minibatch",))
def update_epoch_fori_loop(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action_mask,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    n_minibatch: int  # needs to be static for better optimization of the for loop
):
    def _update_epoch(i, carry):
        train_state, _ = carry
        idx = idxes[i]
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            train_state.apply_fn,
            obs=[el[idx] for el in obs],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            action_mask=action_mask[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        train_state = train_state.apply_gradients(grads=grads)
        carry = train_state, total_loss
        return carry

    carry = train_state, (0., (0., 0., 0., 0., 0., 0.))
    carry = jax.lax.fori_loop(
        0,
        n_minibatch,
        _update_epoch,
        carry
    )
    train_state, total_loss = carry

    # Note: the total_loss returned is w.r.t. the last minibatch only
    return train_state, total_loss


@partial(jax.jit, static_argnames=("n_minibatch",))
def update_epoch_scan(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action_mask,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    n_minibatch: int  # needs to be static for better optimization of the for loop
):
    grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
    def _update_epoch(train_state, idx):
        total_loss, grads = grad_fn(
            train_state.params,
            train_state.apply_fn,
            obs=[el[idx] for el in obs],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            action_mask=action_mask[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        l = jnp.array([total_loss[0], *total_loss[1]])
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, l

    train_state, total_loss_l = jax.lax.scan(
        _update_epoch,
        train_state,
        idxes,
        length=n_minibatch,
        # unroll=1,
    )
    total_loss = total_loss_l[-1][0], tuple(total_loss_l[-1, 1:])
    # Note: the total_loss returned is w.r.t. the last minibatch only
    return train_state, total_loss
