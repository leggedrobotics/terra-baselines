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
    total_timesteps: int = 30_000_000_000
    lr: float = 3e-4
    clip_eps: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
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
    num_prev_actions = 5
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
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


# Modify the Transition class to store both agents' data
class Transition(struct.PyTreeNode):
    done: jax.Array
    action_1: jax.Array  # Agent 1's action
    action_2: jax.Array  # Agent 2's action
    value_1: jax.Array   # Agent 1's value
    value_2: jax.Array   # Agent 2's value
    reward: jax.Array    # Joint reward
    log_prob_1: jax.Array  # Agent 1's log prob
    log_prob_2: jax.Array  # Agent 2's log prob
    obs: jax.Array       # Original observation
    prev_actions_1: jax.Array  # Agent 1's previous actions
    prev_actions_2: jax.Array  # Agent 2's previous actions
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
            - transition.value_1  # Changed from transition.value to transition.value_1
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value_1), gae  # Changed from transition.value to transition.value_1

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value_1  # Changed from transitions.value to transitions.value_1


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    advantages: tuple,
    targets: tuple,
    config,
):
    clip_eps = config.clip_eps
    vf_coef = config.vf_coef
    ent_coef = config.ent_coef

    # NORMALIZE ADVANTAGES - update for tuple format
    advantages_normalized = (
        (advantages[0] - advantages[0].mean()) / (advantages[0].std() + 1e-8),
        (advantages[1] - advantages[1].mean()) / (advantages[1].std() + 1e-8)
    )
    advantages = advantages_normalized

    def _loss_fn(params):
        # Process Agent 1's data
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.obs,
        )
        transitions_actions_1_reshaped = jnp.reshape(
            transitions.prev_actions_1, (-1, *transitions.prev_actions_1.shape[2:])
        )
        transitions_actions_2_reshaped = jnp.reshape(
            transitions.prev_actions_2, (-1, *transitions.prev_actions_2.shape[2:])
        )
        
        # Agent 1 perspective (original observation)
        obs_agent1 = obs_to_model_input(
            transitions_obs_reshaped, 
            transitions_actions_1_reshaped, 
            transitions_actions_2_reshaped, 
            config
        )
        value1, dists = policy(train_state.apply_fn, params, obs_agent1)
        pi1, _ = dists  # Extract agent1's distribution
        value1 = value1[:, 0]
        
        # Agent 2 perspective (swapped observation)
        obs_for_agent2 = transitions_obs_reshaped.copy()
        obs_for_agent2["agent_state_1"] = transitions_obs_reshaped["agent_state_2"]
        obs_for_agent2["agent_state_2"] = transitions_obs_reshaped["agent_state_1"]
        
        local_map_keys_agent1 = [
            "local_map_action_neg", "local_map_action_pos", "local_map_target_neg", 
            "local_map_target_pos", "local_map_dumpability", "local_map_obstacles"
        ]
        local_map_keys_agent2 = [key + "_2" for key in local_map_keys_agent1]
        for key1, key2 in zip(local_map_keys_agent1, local_map_keys_agent2):
            obs_for_agent2[key1] = transitions_obs_reshaped[key2]
            obs_for_agent2[key2] = transitions_obs_reshaped[key1]
        
        obs_agent2 = obs_to_model_input(
            obs_for_agent2, 
            transitions_actions_2_reshaped, 
            transitions_actions_1_reshaped,  # Swapped
            config
        )
        value2, dists2 = policy(train_state.apply_fn, params, obs_agent2)
        _, pi2 = dists2  # Extract agent2's distribution
        value2 = value2[:, 0]
        
        # Reshape actions for log_prob calculation
        action1_reshaped = jnp.reshape(transitions.action_1, (-1, *transitions.action_1.shape[2:]))
        action2_reshaped = jnp.reshape(transitions.action_2, (-1, *transitions.action_2.shape[2:]))
        
        log_prob1 = pi1.log_prob(action1_reshaped)
        log_prob2 = pi2.log_prob(action2_reshaped)
        
        # Reshape back to original dimensions
        value1 = jnp.reshape(value1, transitions.value_1.shape)
        value2 = jnp.reshape(value2, transitions.value_2.shape)
        log_prob1 = jnp.reshape(log_prob1, transitions.log_prob_1.shape)
        log_prob2 = jnp.reshape(log_prob2, transitions.log_prob_2.shape)
        
        # Calculate value losses for both agents
        value_loss1 = calculate_value_loss(value1, transitions.value_1, targets[0], clip_eps)
        value_loss2 = calculate_value_loss(value2, transitions.value_2, targets[1], clip_eps)
        value_loss = 0.5 * (value_loss1 + value_loss2)
        
        # Calculate actor losses for both agents
        actor_loss1 = calculate_actor_loss(log_prob1, transitions.log_prob_1, advantages[0], clip_eps)
        actor_loss2 = calculate_actor_loss(log_prob2, transitions.log_prob_2, advantages[1], clip_eps)
        actor_loss = -0.5 * (actor_loss1 + actor_loss2)
        
        # Calculate entropy for both distributions
        entropy = 0.5 * (pi1.entropy().mean() + pi2.entropy().mean())
        
        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)
        
    # Helper functions for cleaner code    
    def calculate_value_loss(value, old_value, targets, clip_eps):
        value_pred_clipped = old_value + (value - old_value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        return jnp.maximum(value_loss, value_loss_clipped).mean()
        
    def calculate_actor_loss(log_prob, old_log_prob, advantages, clip_eps):
        ratio = jnp.exp(log_prob - old_log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        return jnp.minimum(actor_loss1, actor_loss2).mean()
    
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


def get_curriculum_levels(env_cfg, global_curriculum_levels):
    curriculum_stat = {}
    curriculum_levels = env_cfg.curriculum.level
    for i, global_curriculum_level in enumerate(global_curriculum_levels):
        curriculum_stat[f'Level {i}: {global_curriculum_level["maps_path"]}'] = jnp.sum(
            curriculum_levels == i
        ).item()
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
        reset_fn_p = jax.pmap(env.reset, axis_name="devices")
        timestep = reset_fn_p(env_params, reset_rng)
        
        # Track previous actions for both agents
        prev_actions_1 = jnp.zeros(
            (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
        )
        prev_actions_2 = jnp.zeros(
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
                rng, train_state, prev_timestep, prev_actions_1, prev_actions_2, prev_reward = runner_state

                # Split RNG key for model policy call and environment step
                rng_model, rng_env_step, rng_next_step = jax.random.split(rng, 3)

                obs_current = prev_timestep.observation
                
                # Get actions, log_probs, and value for BOTH agents from a SINGLE policy call
                # select_action_ppo returns: (actions_tuple, log_probs_tuple), value, dists_tuple
                actions_tuple, log_probs_tuple, value, _ = select_action_ppo(
                    train_state, obs_current, prev_actions_1, prev_actions_2, rng_model, config
                )
                action1_raw, action2_raw = actions_tuple # Unpack actions for each agent
                log_prob1, log_prob2 = log_probs_tuple   # Unpack log_probs for each agent
                # 'value' is the single value from the centralized critic

                # Wrap actions for the environment
                action_env_1 = wrap_action(action1_raw, env.batch_cfg.action_type)
                action_env_2 = wrap_action(action2_raw, env.batch_cfg.action_type)
                
                # Step environment with both agents' actions
                _rng_env_split = jax.random.split(rng_env_step, config.num_envs_per_device)
                # Assuming env.step is modified or designed to take prev_timestep and two separate action arguments
                timestep = env.step(prev_timestep, action_env_1, action_env_2, _rng_env_split)
                
                # Store data for both agents
                # The centralized value is used for both value_1 and value_2
                transition = Transition(
                    done=timestep.done,
                    action_1=action1_raw,
                    action_2=action2_raw,
                    value_1=value, # Centralized value
                    value_2=value, # Centralized value
                    reward=timestep.reward, # Joint reward
                    log_prob_1=log_prob1,
                    log_prob_2=log_prob2,
                    obs=obs_current,
                    prev_actions_1=prev_actions_1,
                    prev_actions_2=prev_actions_2,
                    prev_reward=prev_reward,
                )

                # Update previous actions for both agents
                new_prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=-1).at[..., 0].set(action1_raw)
                new_prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=-1).at[..., 0].set(action2_raw)

                runner_state = (rng_next_step, train_state, timestep, new_prev_actions_1, new_prev_actions_2, timestep.reward)
                return runner_state, transition

            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # COLLECT TRAJECTORIES (new version) - This second scan seems redundant if the first one collects num_steps.
            # If it's intended for a different purpose, ensure runner_state_carry is correctly initialized.
            # For now, assuming the first scan is the primary trajectory collection.
            # If this second scan is indeed needed and _env_step is called again,
            # ensure runner_state_carry is correctly formed.
            # runner_state_carry = runner_state[-1] # This was problematic, runner_state is not a list of states from scan here.
            # runner_state_carry = runner_state # runner_state is the final state from the first scan.
            # runner_state, transitions = jax.lax.scan(
            #     _env_step, runner_state_carry, None, config.num_steps
            # )


            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_actions_1, prev_actions_2, prev_reward = runner_state
            rng, _rng = jax.random.split(rng)
            
            # Get last_value from the centralized critic using the final observation
            # select_action_ppo returns: (actions_tuple, log_probs_tuple), value, dists_tuple
            _, _, last_val_central, _ = select_action_ppo(
                train_state, timestep.observation, prev_actions_1, prev_actions_2, _rng, config
            )
            
            # Calculate GAE. Since critic is centralized, value_1 and value_2 in Transition are the same.
            # calculate_gae uses transition.value_1.
            advantages_central, targets_central = calculate_gae(
                transitions, last_val_central, config.gamma, config.gae_lambda
            )

            # ppo_update_networks expects advantages and targets for each agent.
            # Since we have a centralized critic and joint reward, these will be the same for both.
            advantages_for_update = (advantages_central, advantages_central)
            targets_for_update = (targets_central, targets_central)

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
                    transitions, advantages_tuple, targets_tuple = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        advantages=advantages_tuple,
                        targets=targets_tuple,
                        config=config,
                    )
                    return new_train_state, update_info

                rng, train_state, transitions, advantages_tuple, targets_tuple = update_state

                rng, _rng = jax.random.split(rng)
                
                # Sticking to the original approach of grouping by sequences
                # Swap axes to get shape (num_envs_per_device, seq_len, ...)
                transitions_swapped = jtu.tree_map(lambda x: x.swapaxes(0, 1), transitions)
                advantages_swapped = (advantages_tuple[0].swapaxes(0, 1), 
                                      advantages_tuple[1].swapaxes(0, 1))
                targets_swapped = (targets_tuple[0].swapaxes(0, 1), 
                                   targets_tuple[1].swapaxes(0, 1))

                # Shuffle along the num_envs_per_device dimension
                perm = jax.random.permutation(_rng, config.num_envs_per_device)
                transitions_shuffled = jtu.tree_map(lambda x: jnp.take(x, perm, axis=0), transitions_swapped)
                advantages_shuffled = (jnp.take(advantages_swapped[0], perm, axis=0), 
                                       jnp.take(advantages_swapped[1], perm, axis=0))
                targets_shuffled = (jnp.take(targets_swapped[0], perm, axis=0), 
                                     jnp.take(targets_swapped[1], perm, axis=0))
                
                minibatch_size_envs = config.num_envs_per_device // config.num_minibatches
                if config.num_envs_per_device % config.num_minibatches != 0:
                    raise ValueError(
                        f"num_envs_per_device ({config.num_envs_per_device}) must be divisible by "
                        f"num_minibatches ({config.num_minibatches}). "
                        f"Consider adjusting these TrainConfig values."
                    )
                
                # Reshape for minibatches
                transitions_minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (config.num_minibatches, minibatch_size_envs) + x.shape[1:] 
                    ),
                    transitions_shuffled,
                )
                advantages_minibatches = (
                    jnp.reshape(advantages_shuffled[0], 
                               (config.num_minibatches, minibatch_size_envs) + advantages_shuffled[0].shape[1:]),
                    jnp.reshape(advantages_shuffled[1], 
                               (config.num_minibatches, minibatch_size_envs) + advantages_shuffled[1].shape[1:])
                )
                targets_minibatches = (
                    jnp.reshape(targets_shuffled[0], 
                               (config.num_minibatches, minibatch_size_envs) + targets_shuffled[0].shape[1:]),
                    jnp.reshape(targets_shuffled[1], 
                               (config.num_minibatches, minibatch_size_envs) + targets_shuffled[1].shape[1:])
                )
                
                # Prepare minibatches for _update_minbatch
                minibatches = (transitions_minibatches, advantages_minibatches, targets_minibatches)

                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (rng, train_state, transitions, advantages_tuple, targets_tuple)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            update_state = (rng, train_state, transitions, advantages_for_update, targets_for_update)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)

            runner_state = (rng, train_state, timestep, prev_actions_1, prev_actions_2, prev_reward)
            return runner_state, loss_info

        # Setup runner state for multiple devices

        rng, rng_rollout = jax.random.split(rng)
        rng = jax.random.split(rng, num=config.num_devices)
        train_state = replicate(train_state, jax.local_devices()[: config.num_devices])
        runner_state = (rng, train_state, timestep, prev_actions_1, prev_actions_2, prev_reward)
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
                eval_stats = eval_ppo.rollout(
                    rng_rollout,
                    env,
                    env_params_single,
                    train_state,
                    config,
                )

                # eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
                n = config.num_envs_per_device * eval_stats.length
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
                        / config.num_envs_per_device,
                        "eval/total_terminations": eval_stats.terminations
                        / config.num_envs_per_device,
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
        entity="Terra_MARL1",
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
