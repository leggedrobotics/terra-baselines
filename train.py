# import os
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig

import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
# import pyrallis
import wandb
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from utils.utils_ppo import Transition, calculate_gae, ppo_update_networks, rollout, select_action_ppo, get_cfgs_init, wrap_action
from datetime import datetime

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)

# TODO curriculum

@dataclass
class TrainConfig:
    project: str = "excavator-oss"
    group: str = "default"
    name: str = "single-task-ppo-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # env_id: str = "MiniGrid-Empty-6x6"
    # agent
    # action_emb_dim: int = 16
    # rnn_hidden_dim: int = 1024
    # rnn_num_layers: int = 1
    # head_hidden_dim: int = 256
    # training
    num_envs: int = 4096
    num_steps: int = 32
    update_epochs: int = 3
    num_minibatches: int = 256
    total_timesteps: int = 3_000_000_000
    lr: float = 3e-04
    clip_eps: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 80
    seed: int = 42

    # Model
    clip_action_maps = True  # clips the action maps to [-1, 1]
    maps_net_normalization_bounds = [-1, 8]  # automatically set to [-1, 1] if clip_action_maps is True
    local_map_normalization_bounds = [-16, 16]
    loaded_max = 100
    num_embeddings_agent_min = 60  # should be at least as big as the biggest map axis
    mask_out_arm_extension = True

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        assert self.num_envs % num_devices == 0
        self.num_updates = self.total_timesteps_per_device // self.num_steps // self.num_envs_per_device
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")

    # make object subscriptable
    def __getitem__(self, key):
        return getattr(self, key)

def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    env = TerraEnvBatch()
    env_params = get_cfgs_init()
    print(f"{env_params.tile_size.shape=}")

    # setup training state
    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    network, network_params = get_model_ready(_rng, config, env)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, train_state


def make_train(
    env: TerraEnvBatch,
    env_params: EnvConfig,
    config: TrainConfig,
):
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)
        print(f"{reset_rng[33]}")

        # TERRA: Reset envs
        timestep = env.reset(env_params, reset_rng)  # vmapped inside
        prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward = runner_state

                # SELECT ACTION
                rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                # dist, value, hstate = train_state.apply_fn(
                #     train_state.params,
                #     {
                #         # [batch_size, seq_len=1, ...]
                #         "observation": prev_timestep.observation[:, None],
                #         "prev_action": prev_action[:, None],
                #         "prev_reward": prev_reward[:, None],
                #     },
                #     prev_hstate,
                # )
                # action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # # squeeze seq_len where possible
                # action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)
                # TODO squeeze?
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
            # calculate value of the last step for bootstrapping
            # _, last_val, _ = train_state.apply_fn(
            #     train_state.params,
            #     {
            #         "observation": timestep.observation[:, None],
            #         "prev_action": prev_action[:, None],
            #         "prev_reward": prev_reward[:, None],
            #     },
            #     hstate,
            # )
            # TODO squeeze?
            rng, _rng = jax.random.split(rng)
            _, _, last_val, _ = select_action_ppo(train_state, timestep.observation, _rng)
            # advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
            advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):

                def _update_minbatch(init, batch_info):
                    train_state, rng = init

                    # TODO randomization here?
                    rng, _rng_model = jax.random.split(rng)
                    transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                        rng_model=_rng_model,
                    )

                    new_init = (new_train_state, rng)
                    return new_init, update_info

                rng, train_state, transitions, advantages, targets = update_state

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                # [seq_len, batch_size, ...]
                batch = (transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                # [num_minibatches, minibatch_size, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                (train_state, rng), update_info = jax.lax.scan(_update_minbatch, (train_state, rng), minibatches)

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
            # eval_rng = jax.random.split(_rng, num=config.eval_episodes_per_device)
            # # vmap only on rngs
            # eval_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None))(
            #     eval_rng,
            #     env,
            #     env_params,
            #     train_state,
            #     1,
            # )

            eval_stats = rollout(
                _rng,
                env,
                env_params,
                train_state,
                1,
            )
            
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            loss_info.update(
                {
                    # "eval/returns": eval_stats.reward.mean(0),
                    # "eval/lengths": eval_stats.length.mean(0),
                    "eval/returns": eval_stats.reward.mean(),
                    "eval/lengths": eval_stats.length.mean(),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            runner_state = (rng, train_state, timestep, prev_action, prev_reward)
            return runner_state, loss_info

        runner_state = (rng, train_state, timestep, prev_action, prev_reward)
        runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_updates)
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


# @pyrallis.wrap()
def train(config: TrainConfig):
    # logging to wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    rng, env, env_params, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, config)
    print(f"{rng=}")
    train_fn_lowered = train_fn.lower(rng, train_state)
    train_fn = train_fn_lowered.compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s")

    print("Logging...")
    loss_info = unreplicate(train_info["loss_info"])

    total_transitions = 0
    for i in range(config.num_updates):
        # summing total transitions per update from all devices
        total_transitions += config.num_steps * config.num_envs_per_device * jax.local_device_count()
        info = jtu.tree_map(lambda x: x[i].item(), loss_info)
        info["transitions"] = total_transitions
        wandb.log(info)

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (config.total_timesteps_per_device * jax.local_device_count()) / elapsed_time

    print("Final return: ", float(loss_info["eval/returns"][-1]))
    run.finish()

if __name__ == "__main__":
    train(TrainConfig())