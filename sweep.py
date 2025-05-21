import jax
import time
import sys
import wandb
from dataclasses import asdict, dataclass

from terra.config import EnvConfig
from train import make_states, make_train, TrainConfig

@dataclass
class TrainConfigSweep(TrainConfig):
    # Training config
    project: str = "sweep"
    total_timesteps: int = 2_000_000_000

    # Rewards
    existence: float = 0.0
    collision_move: float = -0.5
    move: float = -0.2
    move_with_turned_wheels=-0.2
    cabin_turn: float = -0.2
    wheel_turn: float = -0.2
    dig_wrong: float = -1.0
    dump_wrong: float = 0.0
    dig_correct: float = 1.0
    dump_correct: float = 1.0
    terminal: float = 50.0


def train(config: TrainConfigSweep):
    run = wandb.init(
        entity="terra-sp-thesis",
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    # Replace the rewards with the sweep values
    env_params = EnvConfig()
    env_params = env_params._replace(
        rewards=env_params.rewards._replace(
            existence=config.existence,
            collision_move=config.collision_move,
            move=config.move,
            cabin_turn=config.cabin_turn,
            wheel_turn=config.wheel_turn,
            dig_wrong=config.dig_wrong,
            dump_wrong=config.dump_wrong,
            dig_correct=config.dig_correct,
            dump_correct=config.dump_correct,
            terminal=config.terminal,
        )
    )
    rng, env, env_params, train_state = make_states(config, env_params)
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

def sweep_train():
    config = wandb.config
    if "name" not in config:
        config["name"] = f"sweep-{wandb.run.id}"
    # Convert wandb.config to TrainConfigSweep
    train_config = TrainConfigSweep(**dict(config))
    train(train_config)

if __name__ == "__main__":
    # If called with "create" argument, create the sweep and print the ID
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        sweep_config = {
            "program": "train.py",
            "method": "bayes",
            "metric": {
                "name": "eval/positive_terminations",
                "goal": "maximize",
            },
            "parameters": {
                "existence": {"values": [-0.1, -0.05, 0.0]},
                "collision_move": {"values": [-0.2, -0.1, -0.05, 0.0]},
                "move": {"values": [-0.3, -0.2, -0.1, -0.05, 0.0]},
                "move_with_turned_wheels": {"values": [-0.3, -0.2, -0.1, -0.05, 0.0]},
                "cabin_turn": {"values": [-0.3, -0.2, -0.1, -0.05, 0.0]},
                "wheel_turn": {"values": [-0.3, -0.2, -0.1, -0.05, 0.0]},
                "dig_wrong": {"values": [-2.0, -1.0, -0.5, 0.0]},
                "dump_wrong": {"values": [-2.0, -1.0, -0.5, 0.0]},
                "dig_correct": {"values": [0.5, 1.0, 2.0, 3.0]},
                "dump_correct": {"values": [0.5, 1.0, 2.0, 3.0]},
                "terminal": {"values": [10.0, 25.0, 50.0]},
            }
        }
        sweep_id = wandb.sweep(sweep_config, project="sweep")
    else:
        # Called by wandb agent
        sweep_train()
