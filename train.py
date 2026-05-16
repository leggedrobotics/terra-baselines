import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from flax.training.train_state import TrainState
import optax
import wandb
import eval_ppo
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm
from functools import partial
from flax.jax_utils import replicate, unreplicate
from flax import struct
import utils.helpers as helpers
from utils.utils_ppo import (
    select_action_ppo,
    wrap_action,
    obs_to_model_input,
    policy,
)
import os

jax.config.update("jax_threefry_partitionable", True)


#REMARK:
#Helper file, use train_mixed.py for training


@dataclass
class TrainConfig:
    name: str
    num_devices: int = 0
    project: str = "debug"
    group: str = "default"
    num_envs_per_device: int = 4096
    num_steps: int = 32
    update_epochs: int = 5
    num_minibatches: int = 32
    total_timesteps: int = 100_000_000_000
    lr: float = 3e-4
    clip_eps: float = 0.5
    gamma: float = 0.9984
    gae_lambda: float = 0.95
    ent_coef: float = 0.015
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    use_action_mask: bool = False
    edge_features_dim: int = 0
    use_critic_affordances: bool = False
    critic_affordance_dim: int = 0
    include_episode_progress: bool = False
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = (
        50  # Number of updates between running eval and syncing with wandb
    )
    checkpoint_interval: int = 50  # Number of updates between checkpoints
    # model settings
    num_prev_actions = 10
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    maps_net_normalization_bounds = [-10, 10]  # Required field for network initialization
    loaded_max = 100
    num_rollouts_eval = 500  # max length of an episode in Terra for eval (for training it is in Terra's curriculum)
    cache_clear_interval = 1000  # Number of updates between clearing caches

    def __post_init__(self):
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.env_steps_per_update = self.num_steps * self.num_envs
        self.num_updates = self.total_timesteps // self.env_steps_per_update
        self.actual_total_timesteps = self.num_updates * self.env_steps_per_update
        if self.use_critic_affordances and self.edge_features_dim == 0:
            self.edge_features_dim = 10
        self.critic_affordance_dim = (
            self.edge_features_dim + int(self.include_episode_progress)
            if self.use_critic_affordances
            else 0
        )
        print(
            "Num devices: "
            f"{self.num_devices}, Num updates: {self.num_updates}, "
            f"Env steps/update: {self.env_steps_per_update}"
        )

    # make object subscriptable
    def __getitem__(self, key):
        return getattr(self, key)


def make_states(config: TrainConfig, env_params: EnvConfig = EnvConfig()):
    env = TerraEnvBatch()
    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None, None]
        .repeat(num_devices, 0)
        .repeat(num_envs_per_device, 1),
        env_params,
    )
    print(f"{env_params.tile_size.shape=}")

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    network, network_params = get_model_ready(_rng, config, env)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, train_state


class Transition(struct.PyTreeNode):
    done: jax.Array
    task_done: jax.Array
    timeout_done: jax.Array
    bootstrap_value: jax.Array
    bootstrap_mask: jax.Array
    gae_mask: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_actions: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        next_value = jnp.where(
            transition.timeout_done,
            transition.bootstrap_value,
            next_value,
        )
        delta = (
            transition.reward
            + gamma * next_value * transition.bootstrap_mask
            - transition.value
        )
        gae = delta + gamma * gae_lambda * transition.gae_mask * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def value_from_observation(train_state, observation, prev_actions, config):
    model_obs = obs_to_model_input(observation, prev_actions, config)
    value, _ = train_state.apply_fn(train_state.params, model_obs)
    return value[:, 0]


def timeout_bootstrap_value(train_state, info, next_prev_actions, value, config):
    def _bootstrap(_):
        final_value = value_from_observation(
            train_state,
            info["final_observation"],
            next_prev_actions,
            config,
        )
        return jnp.where(info["timeout_done"], final_value, 0.0)

    return jax.lax.cond(
        jnp.any(info["timeout_done"]),
        _bootstrap,
        lambda _: jnp.zeros_like(value),
        operand=0,
    )


def randomize_initial_episode_progress(timestep, rng):
    max_steps = jnp.maximum(
        jnp.asarray(timestep.env_cfg.max_steps_in_episode, dtype=jnp.float32),
        1.0,
    )
    episode_age = jnp.floor(
        jax.random.uniform(rng, timestep.done.shape) * max_steps
    ).astype(jnp.asarray(timestep.state.env_steps).dtype)
    episode_progress = jnp.clip(
        episode_age.astype(jnp.float32) / max_steps,
        0.0,
        1.0,
    )
    state = timestep.state._replace(env_steps=episode_age)
    observation = {
        **timestep.observation,
        "episode_progress": episode_progress,
    }
    info = {
        **timestep.info,
        "episode_progress": episode_progress,
        "final_observation": {
            **timestep.info["final_observation"],
            "episode_progress": episode_progress,
        },
    }
    return timestep._replace(state=state, observation=observation, info=info)


def _finite_mean(value):
    value = value.astype(jnp.float32)
    finite = jnp.isfinite(value)
    return jnp.sum(jnp.where(finite, value, 0.0)) / jnp.maximum(
        jnp.sum(finite.astype(jnp.float32)),
        1.0,
    )


def train_rollout_metrics(transitions, reward_components):
    metrics = {
        "train/reward": jnp.mean(transitions.reward),
        "train/success_rate": jnp.mean(transitions.task_done.astype(jnp.float32)),
        "train/timeout_rate": jnp.mean(transitions.timeout_done.astype(jnp.float32)),
    }

    episode_progress = transitions.obs["episode_progress"].astype(jnp.float32)
    metrics["train/episode_progress"] = jnp.mean(episode_progress)

    if "action_mask" in transitions.obs:
        chosen_valid = jnp.take_along_axis(
            transitions.obs["action_mask"],
            transitions.action[..., None],
            axis=-1,
        )[..., 0]
        metrics["behavior/invalid_action_rate"] = jnp.mean(
            (~chosen_valid).astype(jnp.float32)
        )

    component_metrics = {
        "completion": "progress/completion",
        "core_completion": "progress/core_completion",
        "edge_completion": "progress/edge_completion",
        "do_attempt": "behavior/do_attempt_rate",
        "dig_success_event": "behavior/dig_success_rate",
        "dump_success_event": "behavior/dump_success_rate",
        "collision": "behavior/collision_rate",
        "noop": "behavior/noop_rate",
        "terrain_changed": "behavior/terrain_changed_rate",
        "loaded_at_timeout": "behavior/loaded_at_timeout_rate",
        "terminal": "reward/terminal",
        "dig_success": "reward/dig_success",
        "dig_wrong": "reward/dig_wrong",
        "dump_success": "reward/dump_success",
        "dump_wrong": "reward/dump_wrong",
        "existence": "reward/existence",
    }
    for component, name in component_metrics.items():
        metrics[name] = _finite_mean(reward_components[component])
    return metrics


def optimizer_log_metrics(loss_info):
    metrics = {
        "loss/total": loss_info["total_loss"],
        "loss/value": loss_info["value_loss"],
        "loss/policy": loss_info["actor_loss"],
        "loss/entropy": loss_info["entropy"],
    }
    if "explained_variance" in loss_info:
        metrics["loss/explained_variance"] = loss_info["explained_variance"]
    return metrics


def train_log_metrics(loss_info):
    metrics = optimizer_log_metrics(loss_info)
    for key, value in loss_info.items():
        if key.startswith(("train/", "progress/", "behavior/", "reward/")):
            metrics[key] = value
    return metrics


def eval_log_metrics(stats, config):
    envs = config.num_envs_per_device * config.num_devices
    steps = envs * stats.length
    return_count = jnp.maximum(stats.return_count, 1)
    return_mean = stats.return_sum / return_count
    return_var = stats.return_sq_sum / return_count - jnp.square(return_mean)
    success_count = jnp.maximum(stats.success_return_count, 1)
    failure_count = jnp.maximum(stats.failure_return_count, 1)

    return {
        "eval/success_rate": stats.successes / envs,
        "eval/timeout_rate": jnp.maximum(stats.return_count - stats.successes, 0) / envs,
        "eval/return_mean": return_mean,
        "eval/return_std": jnp.sqrt(jnp.maximum(return_var, 0.0)),
        "eval/return_min": jnp.where(stats.return_count > 0, stats.return_min, 0.0),
        "eval/return_max": jnp.where(stats.return_count > 0, stats.return_max, 0.0),
        "eval/success_return_mean": stats.success_return_sum / success_count,
        "eval/failure_return_mean": stats.failure_return_sum / failure_count,
        "eval/reward_per_step": stats.reward / steps,
        "eval/max_step_reward": stats.max_reward,
        "eval/success_length": jnp.where(
            stats.successes > 0,
            stats.success_steps / stats.successes,
            0.0,
        ),
        "eval/action_do": stats.action_counts[6] / steps,
        "eval/action_wait": stats.action_counts[7] / steps,
    }

def _config_value(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size <= 16:
        return array.reshape(-1).tolist()
    return {
        "shape": list(array.shape),
        "first": array.reshape(-1)[0].item(),
    }


def _flatten_namedtuple(prefix, value, out):
    if hasattr(value, "_asdict"):
        for key, child in value._asdict().items():
            _flatten_namedtuple(f"{prefix}/{key}", child, out)
        return
    out[prefix] = _config_value(value)


def log_effective_env_config(run, env_params):
    flat = {}
    _flatten_namedtuple("env", env_params, flat)
    run.config.update(flat, allow_val_change=True)


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    config,
    ent_coef_override: float | None = None,
):
    clip_eps = config.clip_eps
    vf_coef = config.vf_coef
    # Allow runtime override of entropy coefficient for schedulers
    ent_coef = ent_coef_override if ent_coef_override is not None else config.ent_coef

    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # Terra: Reshape
        # [minibatch_size, seq_len, ...] -> [minibatch_size * seq_len, ...]
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.obs,
        )
        transitions_actions_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.prev_actions,
        )

        # NOTE: can't use select_action_ppo here because it doesn't decouple params from train_state
        obs = obs_to_model_input(transitions_obs_reshaped, transitions_actions_reshaped, config)
        value, dist = policy(
            train_state.apply_fn,
            params,
            obs,
            use_action_mask=getattr(config, "use_action_mask", False),
        )
        value = value[:, 0]
        # action = dist.sample(seed=rng_model)
        transitions_actions_reshaped = jnp.reshape(
            transitions.action, (-1, *transitions.action.shape[2:])
        )
        log_prob = dist.log_prob(transitions_actions_reshaped)

        # Terra: Reshape
        value = jnp.reshape(value, transitions.value.shape)
        log_prob = jnp.reshape(log_prob, transitions.log_prob.shape)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


def get_curriculum_levels(env_cfg, global_curriculum_levels, timestep=None):
    """
    Get curriculum level statistics with enhanced agent type tracking.
    
    Args:
        env_cfg: Environment configuration containing curriculum levels
        global_curriculum_levels: List of curriculum level configurations
        timestep: Optional timestep containing observations with agent state info
    """
    curriculum_stat = {}
    curriculum_levels = env_cfg.curriculum.level
    
    # Original curriculum level tracking
    for i, global_curriculum_level in enumerate(global_curriculum_levels):
        curriculum_stat[f'Level {i}: {global_curriculum_level["maps_path"]}'] = jnp.sum(
            curriculum_levels == i
        ).item()
    
    # Enhanced: Track agent types at each curriculum level
    # Extract agent types from observations if available, otherwise use fallback
    agent_types_1 = None
    agent_types_2 = None
    
    if timestep is not None and hasattr(timestep, 'observation'):
        obs = timestep.observation
        if 'agent_states' in obs and 'num_agents' in obs:
            # Agent type is at index 6 in per-agent feature vector
            agent_types_1 = obs['agent_states'][:, 0, 6].astype(jnp.int32)
            last_idx = jnp.maximum(0, obs['num_agents'] - 1)
            agent_types_2 = obs['agent_states'][jnp.arange(obs['agent_states'].shape[0]), last_idx, 6].astype(jnp.int32)
    
    # Fallback to default mixed agent setup if no observation data
    if agent_types_1 is None or agent_types_2 is None:
        num_envs = curriculum_levels.shape[0]
        # Default: agent 1 = tracked (0), agent 2 = skid steer (2)
        agent_types_1 = jnp.zeros(num_envs, dtype=jnp.int32)  # All tracked
        agent_types_2 = jnp.full(num_envs, 2, dtype=jnp.int32)  # All skid steer
    
    # Agent type names for logging
    agent_type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
    
    # Count agent types at each curriculum level
    for level_idx in range(len(global_curriculum_levels)):
        level_mask = curriculum_levels == level_idx
        num_envs_at_level = jnp.sum(level_mask).item()
        
        if num_envs_at_level > 0:  # Only process levels that have environments
            # Agent 1 types at this level
            agent1_types_at_level = agent_types_1[level_mask]
            # Agent 2 types at this level  
            agent2_types_at_level = agent_types_2[level_mask]
            
            # Count each agent type for both agent 1 and agent 2
            for agent_type_id, agent_name in agent_type_names.items():
                count_agent1 = jnp.sum(agent1_types_at_level == agent_type_id).item()
                count_agent2 = jnp.sum(agent2_types_at_level == agent_type_id).item()
                total_count = count_agent1 + count_agent2
                
                if total_count > 0:  # Only log non-zero counts
                    curriculum_stat[f'Level {level_idx} {agent_name} Agents'] = total_count
                    if count_agent1 > 0:
                        curriculum_stat[f'Level {level_idx} Agent1 {agent_name}'] = count_agent1
                    if count_agent2 > 0:
                        curriculum_stat[f'Level {level_idx} Agent2 {agent_name}'] = count_agent2
    
    # Overall agent type distribution (across all levels)
    for agent_type_id, agent_name in agent_type_names.items():
        count_agent1 = jnp.sum(agent_types_1 == agent_type_id).item()
        count_agent2 = jnp.sum(agent_types_2 == agent_type_id).item()
        total_count = count_agent1 + count_agent2
        
        if total_count > 0:
            curriculum_stat[f'Total {agent_name} Agents'] = total_count
            if count_agent1 > 0:
                curriculum_stat[f'Total Agent1 {agent_name}'] = count_agent1
            if count_agent2 > 0:
                curriculum_stat[f'Total Agent2 {agent_name}'] = count_agent2
    
    return curriculum_stat


def make_train(
    env: TerraEnvBatch,
    env_params: EnvConfig,
    config: TrainConfig,
):
    def train(
        rng: jax.Array,
        train_state: TrainState,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(
            _rng, config.num_envs_per_device * config.num_devices
        )
        reset_rng = reset_rng.reshape(
            (config.num_devices, config.num_envs_per_device, -1)
        )

        # TERRA: Reset envs
        (
            env_params_reset,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        ) = env.prepare_reset(env_params, reset_rng)
        reset_fn_p = jax.pmap(env.reset_prepared, axis_name="devices")
        randomize_episode_age_p = jax.pmap(
            randomize_initial_episode_progress,
            axis_name="devices",
        )
        timestep = reset_fn_p(
            env_params_reset,
            reset_rng,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        )
        rng, _rng_episode_age = jax.random.split(rng)
        episode_age_rng = jax.random.split(_rng_episode_age, config.num_devices)
        timestep = randomize_episode_age_p(timestep, episode_age_rng)
        prev_actions = jnp.zeros(
            (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
        )
        prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))

        # TRAIN LOOP
        @partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
        def _update_step(runner_state):
            """
            Performs a single update step in the training loop.

            This function orchestrates the collection of trajectories from the environment,
            calculation of advantages, and updating of the network parameters based on the
            collected data. It involves stepping through the environment to collect data,
            calculating the advantage estimates for each step, and performing several epochs
            of updates on the network parameters using the collected data.

            Parameters:
            - runner_state: A tuple containing the current state of the RNG, the training state,
                            the previous timestep, the previous action, and the previous reward.
            - _: Placeholder to match the expected input signature for jax.lax.scan.

            Returns:
            - runner_state: Updated runner state after performing the update step.
            - loss_info: A dictionary containing information about the loss and other
                         metrics for this update step.
            """

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                """
                Executes a step in the environment for all agents.

                This function takes the current state of the runners (agents), selects an
                action for each agent based on the current observation using the PPO
                algorithm, and then steps the environment forward using these actions.
                The environment returns the next state, reward, and whether the episode
                has ended for each agent. These are then used to create a transition tuple
                containing the current state, action, reward, and next state, which can
                be used for training the model.

                Parameters:
                - runner_state: Tuple containing the current rng state, train_state,
                                previous timestep, previous action, and previous reward.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - runner_state: Updated runner state after stepping the environment.
                - transition: A namedtuple containing the transition information
                              (current state, action, reward, next state) for this step.
                """
                rng, train_state, prev_timestep, prev_actions, prev_reward = runner_state

                # SELECT ACTION
                rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                action, log_prob, value, _ = select_action_ppo(
                    train_state, prev_timestep.observation, prev_actions, _rng_model, config
                )

                # STEP ENV
                _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                action_env = wrap_action(action, env.batch_cfg.action_type)
                timestep = env.step(prev_timestep, action_env, _rng_env)
                next_prev_actions = jnp.concatenate(
                    [action[..., None], prev_actions[..., :-1]],
                    axis=-1,
                )
                task_done = timestep.info["task_done"]
                timeout_done = timestep.info["timeout_done"]
                bootstrap_value = timeout_bootstrap_value(
                    train_state,
                    timestep.info,
                    next_prev_actions,
                    value,
                    config,
                )
                transition = Transition(
                    # done=timestep.last(),
                    done=timestep.done,
                    task_done=task_done,
                    timeout_done=timeout_done,
                    bootstrap_value=bootstrap_value,
                    bootstrap_mask=(1.0 - task_done.astype(jnp.float32)),
                    gae_mask=(1.0 - timestep.done.astype(jnp.float32)),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_actions=prev_actions,
                    prev_reward=prev_reward,
                )

                # UPDATE PREVIOUS ACTIONS
                prev_actions = next_prev_actions

                runner_state = (rng, train_state, timestep, prev_actions, timestep.reward)
                return runner_state, (transition, timestep.info["reward_components"])

            # transitions: [seq_len, batch_size, ...]
            runner_state, (transitions, reward_components) = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_actions, prev_reward = runner_state
            rng, _rng = jax.random.split(rng)
            _, _, last_val, _ = select_action_ppo(
                train_state, timestep.observation, prev_actions, _rng, config
            )
            advantages, targets = calculate_gae(
                transitions, last_val, config.gamma, config.gae_lambda
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                """
                Performs a single epoch of updates on the network parameters.

                This function iterates over minibatches of the collected data and
                applies updates to the network parameters based on the PPO algorithm.
                It is called multiple times to perform multiple epochs of updates.

                Parameters:
                - update_state: A tuple containing the current state of the RNG,
                                the training state, and the collected transitions,
                                advantages, and targets.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - update_state: Updated state after performing the epoch of updates.
                - update_info: Information about the updates performed in this epoch.
                """

                def _update_minbatch(train_state, batch_info):
                    """
                    Updates the network parameters based on a single minibatch of data.

                    This function applies the PPO update rule to the network
                    parameters using the data from a single minibatch. It is
                    called for each minibatch in an epoch.

                    Parameters:
                    - train_state: The current training state, including the network parameters.
                    - batch_info: A tuple containing the transitions, advantages, and targets for the minibatch.

                    Returns:
                    - new_train_state: The training state after applying the updates.
                    - update_info: Information about the updates performed on this minibatch.
                    """
                    transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        advantages=advantages,
                        targets=targets,
                        config=config,
                    )
                    return new_train_state, update_info

                rng, train_state, transitions, advantages, targets = update_state

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                # [seq_len, batch_size, ...]
                batch = (transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # [num_minibatches, minibatch_size, seq_len, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (config.num_minibatches, -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (rng, train_state, transitions, advantages, targets)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            update_state = (rng, train_state, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
            loss_info = dict(loss_info)
            loss_info.update(train_rollout_metrics(transitions, reward_components))
            loss_info = jtu.tree_map(
                lambda x: jax.lax.pmean(x, axis_name="devices"),
                loss_info,
            )

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)

            runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
            return runner_state, loss_info

        # Setup runner state for multiple devices

        rng, rng_rollout = jax.random.split(rng)
        rng = jax.random.split(rng, num=config.num_devices)
        train_state = replicate(train_state, jax.local_devices()[: config.num_devices])
        runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
        # runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_updates)
        for i in tqdm(range(config.num_updates), desc="Training"):
            start_time = time.time()  # Start time for measuring iteration speed
            runner_state, loss_info = jax.block_until_ready(
                _update_step(runner_state)
            )
            end_time = time.time()

            iteration_duration = end_time - start_time
            iterations_per_second = 1 / iteration_duration
            steps_per_second = iterations_per_second * config.env_steps_per_update
            steps_per_second_per_gpu = steps_per_second / config.num_devices

            tqdm.write(
                f"Steps/s: {steps_per_second:.2f}"
            )  # Display steps and iterations per second

            need_train_log = (
                config.log_train_interval > 0
                and i % config.log_train_interval == 0
            )
            need_checkpoint = (
                config.checkpoint_interval > 0
                and i % config.checkpoint_interval == 0
            )
            need_eval = (
                config.log_eval_interval > 0
                and i % config.log_eval_interval == 0
            )
            need_final_state = i == config.num_updates - 1
            need_host_state = need_train_log or need_checkpoint or need_eval or need_final_state

            if need_host_state:
                # Use data from the first device for stats, checkpointing, and final return.
                loss_info_single = unreplicate(loss_info)
                runner_state_single = unreplicate(runner_state)
                _, train_state_single, timestep, prev_actions = runner_state_single[:4]
                env_params_single = timestep.env_cfg

            if need_train_log:
                curriculum_levels = get_curriculum_levels(
                    env_params_single, env.batch_cfg.curriculum_global.levels
                )
                wandb.log(
                    {
                        "performance/steps_per_second": steps_per_second,
                        "performance/steps_per_second_per_gpu": steps_per_second_per_gpu,
                        "performance/iterations_per_second": iterations_per_second,
                        "performance/env_steps_per_update": config.env_steps_per_update,
                        "performance/actual_env_steps": (i + 1) * config.env_steps_per_update,
                        "curriculum_levels": curriculum_levels,
                        "lr": config.lr,
                        **train_log_metrics(loss_info_single),
                    }
                )

            if need_checkpoint:
                checkpoint = {
                    "train_config": config,
                    "env_config": env_params_single,
                    "model": runner_state_single[1].params,
                    "loss_info": loss_info_single,
                }
                helpers.save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")

            if need_eval:
                train_state_replicated = runner_state[1]
                env_params_replicated = runner_state[2].env_cfg
                rng_rollout_per_device = jax.random.split(
                    rng_rollout, num=config.num_devices
                )
                eval_stats_per_device = eval_ppo.rollout(
                    rng_rollout_per_device,
                    env,
                    env_params_replicated,
                    train_state_replicated,
                    config,
                )
                eval_stats = eval_stats_per_device._replace(
                    max_reward=eval_stats_per_device.max_reward.max(),
                    reward=eval_stats_per_device.reward.sum(),
                    length=eval_stats_per_device.length[0],
                    successes=eval_stats_per_device.successes.sum(),
                    success_steps=eval_stats_per_device.success_steps.sum(),
                    return_sum=eval_stats_per_device.return_sum.sum(),
                    return_sq_sum=eval_stats_per_device.return_sq_sum.sum(),
                    return_min=eval_stats_per_device.return_min.min(),
                    return_max=eval_stats_per_device.return_max.max(),
                    return_count=eval_stats_per_device.return_count.sum(),
                    success_return_sum=eval_stats_per_device.success_return_sum.sum(),
                    success_return_count=eval_stats_per_device.success_return_count.sum(),
                    failure_return_sum=eval_stats_per_device.failure_return_sum.sum(),
                    failure_return_count=eval_stats_per_device.failure_return_count.sum(),
                    action_counts=eval_stats_per_device.action_counts.sum(axis=0),
                )
                wandb.log(eval_log_metrics(eval_stats, config))

            # Clear JAX caches only after a completed interval. Clearing at i == 0
            # forces the second update to retrace/recompile immediately.
            if config.cache_clear_interval > 0 and (i + 1) % config.cache_clear_interval == 0:
                jax.clear_caches()
                import gc
                gc.collect()

        return {"runner_state": runner_state_single, "loss_info": loss_info_single}

    return train


def train(config: TrainConfig):
    run = wandb.init(
        #entity="Terra_MARL1",
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    
    # Log config.py and train.py files to wandb
    train_py_path = os.path.abspath(__file__)  # Path to current train.py file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terra", "terra", "config.py")
    
    code_artifact = wandb.Artifact(name="source_code", type="code")
    
    # Add train.py
    if os.path.exists(train_py_path):
        code_artifact.add_file(train_py_path, name="train.py")
    
    # Add config.py
    if os.path.exists(config_path):
        code_artifact.add_file(config_path, name="config.py")
    
    # Log the artifact if any files were added
    if code_artifact.files:
        run.log_artifact(code_artifact)

    rng, env, env_params, train_state = make_states(config)
    log_effective_env_config(run, env_params)

    train_fn = make_train(env, env_params, config)

    print("Training...")
    try:  # Try block starts here
        t = time.time()
        train_info = jax.block_until_ready(train_fn(rng, train_state))
        elapsed_time = time.time() - t
        print(f"Done in {elapsed_time:.2f}s")
    except KeyboardInterrupt:  # Catch Ctrl+C
        print("Training interrupted. Finalizing...")
    finally:  # Ensure wandb.finish() is called
        run.finish()
        print("wandb session finished.")


if __name__ == "__main__":
    DT = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="experiment",
    )
    parser.add_argument(
        "-m",
        "--machine",
        type=str,
        default="local",
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=0,
        help="Number of devices to use. If 0, uses all available devices.",
    )
    args, _ = parser.parse_known_args()

    name = f"{args.name}-{args.machine}-{DT}"
    train(TrainConfig(name=name, num_devices=args.num_devices))
