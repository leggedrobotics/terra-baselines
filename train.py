import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
from utils.utils_ppo import select_action_ppo, wrap_action, obs_to_model_input, policy
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
        self.num_updates = (
            self.total_timesteps // (self.num_steps * self.num_envs)
        ) // self.num_devices
        print(f"Num devices: {self.num_devices}, Num updates: {self.num_updates}")

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
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


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
        if 'agent_states' in transitions.obs:
            print(f"ppo_update_networks agent_states[0] shape={transitions.obs['agent_states'][:,0,:].shape}")
        print(f"ppo_update_networks {transitions.prev_actions.shape=}")
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.obs,
        )
        transitions_actions_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.prev_actions,
        )
        if 'agent_states' in transitions_obs_reshaped:
            print(f"ppo_update_networks agent_states[0] shape={transitions_obs_reshaped['agent_states'][:,0,:].shape}")
        print(f"ppo_update_networks {transitions_actions_reshaped.shape=}")

        # NOTE: can't use select_action_ppo here because it doesn't decouple params from train_state
        obs = obs_to_model_input(transitions_obs_reshaped, transitions_actions_reshaped, config)
        value, dist = policy(train_state.apply_fn, params, obs)
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
            dumpability_mask_init,
            action_maps,
            distance_maps,
        ) = env.prepare_reset(env_params, reset_rng)
        reset_fn_p = jax.pmap(env.reset_prepared, axis_name="devices")
        timestep = reset_fn_p(
            env_params_reset,
            reset_rng,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        )
        prev_actions = jnp.zeros(
            (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
        )
        prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))

        # TRAIN LOOP
        @partial(jax.pmap, axis_name="devices")
        def _update_step(runner_state, _):
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
                transition = Transition(
                    # done=timestep.last(),
                    done=timestep.done,
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_actions=prev_actions,
                    prev_reward=prev_reward,
                )

                # UPDATE PREVIOUS ACTIONS
                prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
                prev_actions = prev_actions.at[..., 0].set(action)

                runner_state = (rng, train_state, timestep, prev_actions, timestep.reward)
                return runner_state, transition

            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(
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
                _update_step(runner_state, None)
            )
            end_time = time.time()

            iteration_duration = end_time - start_time
            iterations_per_second = 1 / iteration_duration
            steps_per_second = (
                iterations_per_second
                * config.num_steps
                * config.num_envs
                * config.num_devices
            )

            tqdm.write(
                f"Steps/s: {steps_per_second:.2f}"
            )  # Display steps and iterations per second

            # Use data from the first device for stats and eval
            loss_info_single = unreplicate(loss_info)
            runner_state_single = unreplicate(runner_state)
            _, train_state, timestep, prev_actions = runner_state_single[:4]
            env_params_single = timestep.env_cfg

            if i % config.log_train_interval == 0:
                curriculum_levels = get_curriculum_levels(
                    env_params_single, env.batch_cfg.curriculum_global.levels
                )
                wandb.log(
                    {
                        "performance/steps_per_second": steps_per_second,
                        "performance/iterations_per_second": iterations_per_second,
                        "curriculum_levels": curriculum_levels,
                        "lr": config.lr,
                        **loss_info_single,
                    }
                )

            if i % config.checkpoint_interval == 0:
                checkpoint = {
                    "train_config": config,
                    "env_config": env_params_single,
                    "model": runner_state_single[1].params,
                    "loss_info": loss_info_single,
                }
                helpers.save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")

            if i % config.log_eval_interval == 0:
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
                    min_reward=eval_stats_per_device.min_reward.min(),
                    reward=eval_stats_per_device.reward.sum(),
                    length=eval_stats_per_device.length[0],
                    episodes=eval_stats_per_device.episodes.sum(),
                    positive_terminations=eval_stats_per_device.positive_terminations.sum(),
                    terminations=eval_stats_per_device.terminations.sum(),
                    positive_terminations_steps=eval_stats_per_device.positive_terminations_steps.sum(),
                    action_0=eval_stats_per_device.action_0.sum(),
                    action_1=eval_stats_per_device.action_1.sum(),
                    action_2=eval_stats_per_device.action_2.sum(),
                    action_3=eval_stats_per_device.action_3.sum(),
                    action_4=eval_stats_per_device.action_4.sum(),
                    action_5=eval_stats_per_device.action_5.sum(),
                    action_6=eval_stats_per_device.action_6.sum(),
                    action_7=eval_stats_per_device.action_7.sum(),
                )
                n = config.num_envs_per_device * config.num_devices * eval_stats.length
                avg_positive_episode_length = jnp.where(
                    eval_stats.positive_terminations > 0,
                    eval_stats.positive_terminations_steps / eval_stats.positive_terminations,
                    jnp.zeros_like(eval_stats.positive_terminations_steps)
                )
                loss_info_single.update(
                    {
                        "eval/rewards": eval_stats.reward / n,
                        "eval/max_reward": eval_stats.max_reward,
                        "eval/min_reward": eval_stats.min_reward,
                        "eval/lengths": eval_stats.length,
                        "eval/FORWARD %": eval_stats.action_0 / n,
                        "eval/BACKWARD %": eval_stats.action_1 / n,
                        "eval/CLOCK %": eval_stats.action_2 / n,
                        "eval/ANTICLOCK %": eval_stats.action_3 / n,
                        "eval/CABIN_CLOCK %": eval_stats.action_4 / n,
                        "eval/CABIN_ANTICLOCK %": eval_stats.action_5 / n,
                        "eval/DO": eval_stats.action_6 / n,
                        "eval/positive_terminations": eval_stats.positive_terminations
                        / (config.num_envs_per_device * config.num_devices),
                        "eval/total_terminations": eval_stats.terminations
                        / (config.num_envs_per_device * config.num_devices),
                        "eval/avg_positive_episode_length": avg_positive_episode_length
                    }
                )

                wandb.log(loss_info_single)

            # Clear JAX caches and run garbage collection to stabilize memory use
            if i % config.cache_clear_interval == 0:
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
