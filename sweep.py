import jax
import jax.numpy as jnp
import optax
import time
import wandb
from dataclasses import asdict, dataclass
from datetime import datetime
from flax.training.train_state import TrainState

from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from train import make_train, TrainConfig
from utils.models import get_model_ready

@dataclass
class TrainConfigSweep(TrainConfig):
    existence: float = -0.1
    collision_move: float = -0.1
    move: float = -0.05
    collision_turn: float = -0.1
    cabin_turn: float = -0.01
    wheel_turn: float = -0.01
    dig_wrong: float = -0.3
    dump_wrong: float = -0.3
    dump_no_dump_area: float = -3.0
    dump_close_to_dug_area: float = -0.15
    dig_correct: float = 3.0
    dump_correct: float = 3.0
    terminal: float = 100.0

def make_states(config: TrainConfigSweep):
    env = TerraEnvBatch()
    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    # Replace the rewards witht the sweep values
    env_params = env_params._replace(
        rewards=env_params.rewards._replace(
            existence=config.reward_existence,
            collision_move=config.reward_collision_move,
            move=config.reward_move,
            collision_turn=config.reward_collision_turn,
            cabin_turn=config.reward_cabin_turn,
            wheel_turn=config.reward_wheel_turn,
            dig_wrong=config.reward_dig_wrong,
            dump_wrong=config.reward_dump_wrong,
            dump_no_dump_area=config.reward_dump_no_dump_area,
            dump_close_to_dug_area=config.reward_dump_close_to_dug_area,
            dig_correct=config.reward_dig_correct,
            dump_correct=config.reward_dump_correct,
            terminal=config.reward_terminal,
        )
    )
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

def train(config: TrainConfigSweep):
    run = wandb.init(
        entity="terra-sp-thesis",
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
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
