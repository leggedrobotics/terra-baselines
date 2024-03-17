import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from flax.training.train_state import TrainState
import optax
import wandb
from utils.utils_ppo import (Transition, calculate_gae, ppo_update_networks, 
                             rollout, select_action_ppo, get_cfgs_init, 
                             wrap_action, save_pkl_object)
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm

# jax.config.update("jax_enable_x64", True)

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)

# TODO curriculum

@dataclass
class TrainConfig:
    name: str
    project: str = "excavator-oss"
    group: str = "default"
    name: str = "foundations-733-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    num_envs: int = 2048
    num_steps: int = 64
    update_epochs: int = 3
    num_minibatches: int = 8
    total_timesteps: int = 3_000_000_000
    lr: float = 0.001
    clip_eps: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    checkpoint_interval: int = 100  # Number of updates between checkpoints
    log_interval: int = 1  # Number of updates between logging to wandb
    # terra 
    num_embeddings_agent_min = 60  # should be at least as big as the biggest map axis
    clip_action_maps = True  # clips the action maps to [-1, 1]
    mask_out_arm_extension = True
    local_map_normalization_bounds = [-16, 16]
    loaded_max = 100
    
    def __post_init__(self):
        num_devices = jax.local_device_count()  # Adjust for device count
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        assert self.num_envs % num_devices == 0, "Number of environments must be divisible by the number of devices."
        self.num_updates = (self.total_timesteps // (self.num_steps * self.num_envs)) // num_devices
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")

    # make object subscriptable
    def __getitem__(self, key):
        return getattr(self, key)


def make_states(config: TrainConfig):
    env = TerraEnvBatch()
    env_params = get_cfgs_init()
    print(f"{env_params.tile_size.shape=}")

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    network, network_params = get_model_ready(_rng, config, env)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.lr, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, train_state



def make_train(
    env: TerraEnvBatch,
    env_params: EnvConfig,
    config: TrainConfig,
):
    # @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        # TERRA: Reset envs
        timestep = env.reset(env_params, reset_rng)  # vmapped inside
        prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        # TRAIN LOOP
        @jax.jit  # TODO remove jit and enable pmap on train function
        def _update_step(runner_state, _):
            """
            Performs a single update step in the training loop.

            This function orchestrates the collection of trajectories from the environment, calculation of advantages, and updating of the network parameters based on the collected data. It involves stepping through the environment to collect data, calculating the advantage estimates for each step, and performing several epochs of updates on the network parameters using the collected data.

            Parameters:
            - runner_state: A tuple containing the current state of the RNG, the training state, the previous timestep, the previous action, and the previous reward.
            - _: Placeholder to match the expected input signature for jax.lax.scan.

            Returns:
            - runner_state: Updated runner state after performing the update step.
            - loss_info: A dictionary containing information about the loss and other metrics for this update step.
            """
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                """
                Executes a step in the environment for all agents.

                This function takes the current state of the runners (agents), selects an action for each agent based on the current observation using the PPO algorithm, and then steps the environment forward using these actions. The environment returns the next state, reward, and whether the episode has ended for each agent. These are then used to create a transition tuple containing the current state, action, reward, and next state, which can be used for training the model.

                Parameters:
                - runner_state: Tuple containing the current rng state, train_state, previous timestep, previous action, and previous reward.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - runner_state: Updated runner state after stepping the environment.
                - transition: A namedtuple containing the transition information (current state, action, reward, next state) for this step.
                """
                rng, train_state, prev_timestep, prev_action, prev_reward = runner_state

                # SELECT ACTION
                rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                print(f"{prev_timestep.observation['agent_state'].shape=}")
                action, log_prob, value, _ = select_action_ppo(train_state, prev_timestep.observation, _rng_model)

                # STEP ENV
                _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                action_env = wrap_action(action, env.batch_cfg.action_type)
                timestep = env.step(env_params, prev_timestep, action_env, _rng_env) # vmapped inside
                transition = Transition(
                    # done=timestep.last(),
                    done=timestep.done,
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                runner_state = (rng, train_state, timestep, action, timestep.reward)
                return runner_state, transition

            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, prev_reward = runner_state
            rng, _rng = jax.random.split(rng)
            print(f"last {timestep.observation['agent_state'].shape=}")
            _, _, last_val, _ = select_action_ppo(train_state, timestep.observation, _rng)
            # advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
            advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):                
                """
                Performs a single epoch of updates on the network parameters.

                This function iterates over minibatches of the collected data and applies updates to the network parameters based on the PPO algorithm. It is called multiple times to perform multiple epochs of updates.

                Parameters:
                - update_state: A tuple containing the current state of the RNG, the training state, and the collected transitions, advantages, and targets.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - update_state: Updated state after performing the epoch of updates.
                - update_info: Information about the updates performed in this epoch.
                """
                def _update_minbatch(train_state, batch_info):
                    """
                    Updates the network parameters based on a single minibatch of data.

                    This function applies the PPO update rule to the network parameters using the data from a single minibatch. It is called for each minibatch in an epoch.

                    Parameters:
                    - train_state: The current training state, including the network parameters.
                    - batch_info: A tuple containing the transitions, advantages, and targets for the minibatch.

                    Returns:
                    - new_train_state: The training state after applying the updates.
                    - update_info: Information about the updates performed on this minibatch.
                    """
                    # TODO randomization here?
                    transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
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

                shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                # [num_minibatches, minibatch_size, seq_len, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                update_state = (rng, train_state, transitions, advantages, targets)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            update_state = (rng, train_state, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)

            eval_stats = rollout(
                _rng,
                env,
                env_params,
                train_state,
                1,
            )
            
            # TODO: pmean
            # eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            n = config.num_envs_per_device * eval_stats.length
            loss_info.update(
                {
                    "eval/rewards": eval_stats.reward / n,
                    "eval/max_reward": eval_stats.max_reward,
                    "eval/min_reward": eval_stats.min_reward,
                    "eval/lengths": eval_stats.length,
                    "lr": config.lr,
                    "eval/FORWARD %": eval_stats.action_0 / n,
                    "eval/BACKWARD %": eval_stats.action_1 / n,
                    "eval/CLOCK %": eval_stats.action_2 / n,
                    "eval/ANTICLOCK %": eval_stats.action_3 / n,
                    "eval/CABIN_CLOCK %": eval_stats.action_4 / n,
                    "eval/CABIN_ANTICLOCK %": eval_stats.action_5 / n,
                    "eval/EXTEND_ARM %": eval_stats.action_6 / n,
                    "eval/RETRACT_ARM %": eval_stats.action_7 / n,
                    "eval/DO": eval_stats.action_8 / n,
                }
            )
            runner_state = (rng, train_state, timestep, prev_action, prev_reward)
            return runner_state, loss_info

        runner_state = (rng, train_state, timestep, prev_action, prev_reward)
        # runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        for i in tqdm(range(config.num_updates)):
            # params_before_update = jax.tree_map(lambda x: x.copy(), train_state.params)
            
            runner_state, loss_info = jax.block_until_ready(_update_step(runner_state, None))

            # Save checkpoint
            if i % config.checkpoint_interval == 0:
                checkpoint = {
                    "train_config": config,
                    "env_config": env_params,
                    "model": train_state.params,
                    "loss_info": loss_info,
                }
                save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")
            
            # # DEBUG
            # _, train_state, _, _, _ = runner_state
            # params_after_update = train_state.params
            # params_changed = jax.tree_map(lambda before, after: jnp.any(before != after), params_before_update, params_after_update)
            # # print(params_changed)
            # any_param_changed = any(jax.tree_leaves(params_changed))
            # print(f"Any parameter changed: {any_param_changed}")
            if i % config.log_interval == 0:
                wandb.log(loss_info)
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


def train(config: TrainConfig):
    run = wandb.init(
        entity="operators",
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    rng, env, env_params, train_state = make_states(config)

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, config)
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

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
        default="somewhere",
    )
    args, _ = parser.parse_known_args()

    name = f"{args.name}-{args.machine}-{DT}"
    train(TrainConfig(name=name))