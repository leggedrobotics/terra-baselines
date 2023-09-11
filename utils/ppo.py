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
from typing import Any, Callable, Tuple, Union
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
from terra.env import TerraEnvBatch
from utils.curriculum import Curriculum
from utils.curriculum_testbench import CurriculumTestbench
from utils.reset_manager import ResetManager
from tensorflow_probability.substrates import jax as tfp
from utils.helpers import save_pkl_object
from terra.config import EnvConfig
from typing import NamedTuple


class AdaptiveRewardNormalizer(NamedTuple):
    mean: float = 0.0
    var: float = 1.0
    alpha: float = 0.99

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rewards: Array) -> "AdaptiveRewardNormalizer":
        # Update running mean and variance
        new_mean = jnp.mean(rewards)
        new_var = jnp.var(rewards)
        mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        var = self.alpha * self.var + (1 - self.alpha) * new_var
        return AdaptiveRewardNormalizer(
            mean=mean,
            var=var,
        )

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, rewards: Array) -> Array:
        # Normalize based on running statistics
        normalized_rewards = (rewards - self.mean) / (jnp.sqrt(self.var) + 1e-8)
        return normalized_rewards


class ConstantRewardNormalizer(NamedTuple):
    max_reward: float 
    
    @partial(jax.jit, static_argnums=(0,))
    def update(self, rewards: Array) -> "ConstantRewardNormalizer":
        return ConstantRewardNormalizer(self.max_reward)

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, rewards: Array) -> Array:
        return rewards / self.max_reward
    

class AdaptiveScaledRewardNormalizer(NamedTuple):
    max_reward: float
    mean: float = 0.0
    var: float = 1.0
    alpha: float = 0.99

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rewards: Array) -> "AdaptiveRewardNormalizer":
        # Update running mean and variance
        new_mean = jnp.mean(rewards)
        new_var = jnp.var(rewards)
        mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        var = self.alpha * self.var + (1 - self.alpha) * new_var
        return AdaptiveRewardNormalizer(
            mean=mean,
            var=var,
        )

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, rewards: Array) -> Array:
        rewards /= self.max_reward
        # Normalize based on running statistics
        normalized_rewards = (rewards - self.mean) / (jnp.sqrt(self.var) + 1e-8)
        return normalized_rewards


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        mask_out_arm_extension: bool,
    ):
        self.num_envs = num_envs
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.mask_out_arm_extension = mask_out_arm_extension
    
    # TODO jit
    # @partial(jax.jit, static_argnums=0)
    def reset(self, action_size, observation_shapes, num_actions):
        local_maps_obs_shape = observation_shapes["local_map_action"]
        if self.mask_out_arm_extension:
            local_maps_obs_shape = list(observation_shapes["local_map_action"])
            local_maps_obs_shape[-1] = 1
            local_maps_obs_shape = tuple(local_maps_obs_shape)
            
        return {
            "states": {
                "agent_states": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["agent_states"])),
                    dtype=jnp.int16,    
                ),
                "local_map_action": jnp.empty(
                    (self.n_steps, self.num_envs, *local_maps_obs_shape),
                    dtype=jnp.int8,    
                ),
                "local_map_target": jnp.empty(
                    (self.n_steps, self.num_envs, *local_maps_obs_shape),
                    dtype=jnp.int8,    
                ),
                "action_map": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["action_map"])),
                    dtype=jnp.int8,    
                ),
                "target_map": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["target_map"])),
                    dtype=jnp.int8,    
                ),
                "traversability_mask": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["traversability_mask"])),
                    dtype=jnp.int8,    
                ),
                "do_preview": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["do_preview"])),
                    dtype=jnp.int8,    
                ),
                "dig_map": jnp.empty(
                    (self.n_steps, self.num_envs, *(observation_shapes["dig_map"])),
                    dtype=jnp.int8,
                ),
            },
            "action_mask": jnp.empty(
                (self.n_steps, self.num_envs, num_actions),
                dtype=jnp.uint8,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *action_size),
                dtype=jnp.int8,
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs),
                dtype=jnp.float32
            ),
            "dones": jnp.empty(
                (self.n_steps, self.num_envs),
                dtype=jnp.uint8
                ),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs),
                dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs),
                dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, obs, action, reward, done, log_pi, value, action_mask, reward_normalizer):
        # Normalize the rewards
        reward = reward_normalizer.normalize(reward)
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
    
    @partial(jax.jit, static_argnums=(0, 3, 6, 8, 9))
    def _batch_evaluate(self, rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs, clip_action_maps, mask_out_arm_extension):
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        state, obs, maps_buffer_keys = self.batch_reset(jax.random.split(rng_reset, num_envs), env_cfgs)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask, done, action_mask, maps_buffer_keys, episode_length, episode_done_once = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            
            if clip_action_maps:
                obs = clip_action_maps_in_obs(obs)
            if mask_out_arm_extension:
                obs = cut_local_map_layers(obs)
            
            action, _ = self.select_action_deterministic(train_state, obs, action_mask)
            # jax.debug.print("bicount action = {x}", x=jnp.bincount(action, length=9))
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

            # Update episode length
            episode_done_once = episode_done_once | done
            episode_length += ~episode_done_once

            carry, y = [
                next_o,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
                done,
                action_mask,
                maps_buffer_keys,
                episode_length,
                episode_done_once,
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
                maps_buffer_keys,
                jnp.array(num_envs * [0], dtype=jnp.int32),  # episode_length
                jnp.array(num_envs * [0], dtype=jnp.bool_),  # episode done once
            ],
            (),
            300, #num_steps_test_rollouts,
        )

        cum_return = carry_out[4].squeeze()
        dones = carry_out[6].squeeze()
        episode_length = carry_out[9].squeeze()
        obs_log = scan_out[0]
        return jnp.mean(cum_return), dones, obs_log, episode_length
    
    
    def batch_evaluate(self, rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs, clip_action_maps, mask_out_arm_extension):
        """Rollout an episode with lax.scan"""
        cum_return_mean, dones, obs_log, episode_length = self._batch_evaluate(rng_input, train_state, num_envs, step, action_mask_init, n_evals_save, env_cfgs, clip_action_maps, mask_out_arm_extension)
        jax.debug.print("dones.shape = {x}", x=dones.shape)
        return cum_return_mean, dones, obs_log, episode_length

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

def train_ppo(rng, config, model, params, mle_log, env: TerraEnvBatch, curriculum: Union[Curriculum, CurriculumTestbench], reset_manager: ResetManager, run_name: str):
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
        mask_out_arm_extension=config["mask_out_arm_extension"],
    )

    if config["reward_normalizer"] == "constant":
        reward_normalizer = ConstantRewardNormalizer(max_reward=config["max_reward"])
    elif config["reward_normalizer"] == "adaptive":
        reward_normalizer = AdaptiveRewardNormalizer()
    elif config["reward_normalizer"] == "adaptive-scaled":
        reward_normalizer = AdaptiveScaledRewardNormalizer(max_reward=config["max_reward"])
    else:
        raise(ValueError(f"{config['reward_normalizer']=}"))

    @partial(jax.jit, static_argnames=("clip_action_maps", "reward_normalizer"))
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        batch,
        action_mask: Array,
        rng: jax.random.PRNGKey,
        env_cfgs,
        maps_buffer_keys: jax.random.PRNGKey,
        clip_action_maps: bool,
        reward_normalizer,
    ):
        new_key, key_step = jax.random.split(rng)

        if clip_action_maps:
            obs = clip_action_maps_in_obs(obs)
        if config["mask_out_arm_extension"]:
            obs = cut_local_map_layers(obs)

        action, log_pi, value = rollout_manager.select_action(
            train_state, obs, action_mask, key_step
        )
        action = wrap_action(action, rollout_manager.env.batch_cfg.action_type)
        
        force_resets = reset_manager.update(obs, action)
        next_state, (next_obs, reward, done, infos), maps_buffer_keys = rollout_manager.batch_step(state, action, env_cfgs, maps_buffer_keys, force_resets)
        terminated = infos["done_task"]
        reward_normalizer = reward_normalizer.update(reward)
        batch = batch_manager.append(
            batch, obs, action.action, reward, done, log_pi, value, infos["action_mask"], reward_normalizer
        )
        return train_state, next_obs, next_state, batch, new_key, infos["action_mask"], done, terminated, maps_buffer_keys

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
    terminated_aggregate = np.zeros(config["num_train_envs"], dtype=np.bool_)
    timeouts = np.zeros(config["num_train_envs"], dtype=np.bool_)
    all_dones_update = np.zeros(config["num_train_envs"], dtype=np.bool_)
    best_historical_eval_reward = -1e6
    for step in t:
        train_state, obs, state, batch, rng_step, action_mask, done, terminated, maps_buffer_keys = get_transition(
            train_state,
            obs,
            state,
            batch,
            action_mask,
            rng_step,
            env_cfgs,
            maps_buffer_keys,
            int(config["clip_action_maps"]),
            reward_normalizer,
        )
        terminated_aggregate = terminated_aggregate | terminated
        timeouts = timeouts | (done & (~terminated))
        all_dones_update = all_dones_update | done
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
                wandb.log(
                    {
                        **metric_dict,
                        **dofs_count_dict,
                        "envs done %": all_dones_update.mean(),
                        "envs terminated %": terminated_aggregate.mean(),
                        "envs timeouts %": timeouts.mean(),
                    }
                )
            
            env_cfgs, dofs_count_dict = curriculum.get_cfgs(metric_dict, terminated_aggregate, timeouts)

            batch = batch_manager.reset(
                action_size=rollout_manager.env.actions_size,
                observation_shapes=rollout_manager.env.observation_shapes,
                num_actions=num_actions
            )

            terminated_aggregate = np.zeros(config["num_train_envs"], dtype=np.bool_)
            timeouts = np.zeros(config["num_train_envs"], dtype=np.bool_)
            all_dones_update = np.zeros(config["num_train_envs"], dtype=np.bool_)


        if (step + 1) % config["evaluate_every_epochs"] == 0:
            rng, rng_eval = jax.random.split(rng)
            env_cfgs_eval, dofs_count_dict_eval = curriculum.get_cfgs_eval()
            rewards, dones, obs_log, episode_length = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config["num_test_rollouts"],
                step,
                action_mask_init,
                config["n_evals_save"],
                env_cfgs_eval,
                int(config["clip_action_maps"]),
                int(config["mask_out_arm_extension"]),
            )
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()

            if rewards > best_historical_eval_reward:
                best_historical_eval_reward = rewards
                # Save model
                model_dict = {
                    "network": train_state.params,
                    "train_config": config,
                    "curriculum": curriculum.curriculum_dicts,
                    "default_env_cfg": EnvConfig(),
                    "batch_config": env.batch_cfg._asdict(),
                }
                save_pkl_object(
                    model_dict,
                    f"agents/{config['env_name']}/{run_name}_best_model.pkl",
                )
                print(f"~~~~~~~~ New best model checkpoint saved -> reward = {rewards} ~~~~~~~~")

                # Save episode for replay
                if config['save_episode_for_replay']:
                    obs_log_filename = "agents/Terra/" + config["run_name"] + "/eval_best.pkl"
                    save_pkl_object(
                        obs_log,
                        obs_log_filename,
                    )

            if config["wandb"]:
                wandb.log(
                    {
                        "eval - cum_reward": rewards,
                        "eval - dones %": 100 * dones.sum() / dones.shape[0],
                        "eval - first episode length": episode_length.mean(),
                    }
                )

    return (
        log_steps,
        log_return,
        train_state.params,
        obs_log,
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

def clip_action_maps_in_obs(obs):
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    obs["do_preview"] = jnp.clip(obs["do_preview"], a_min=-1, a_max=1)
    obs["dig_map"] = jnp.clip(obs["dig_map"], a_min=-1, a_max=1)
    return obs

def cut_local_map_layers(obs):
    """Only keep the first layer of the local map"""
    obs["local_map_action"] = obs["local_map_action"][..., [0]]
    obs["local_map_target"] = obs["local_map_target"][..., [0]]
    return obs

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
            idxes = idxes.reshape(n_minibatch, -1)

            # Scan option
            train_state, total_loss = update_epoch_scan(
                    train_state,
                    idxes,
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

    # avg_metrics_dict["avg dones %"] += np.asarray(100 * dones.mean())
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
