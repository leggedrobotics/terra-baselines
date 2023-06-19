"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

# import os
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object
from time import gmtime
from time import strftime
from utils.curriculum import Curriculum
from terra.env import TerraEnvBatch

def _init_wandb(run_name, config):
    import wandb
    run = wandb.init(
            project="single-excavator",
            entity=None,
            # sync_tensorboard=True,
            config=config,
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
    return run


def main(config, mle_log, log_ext=""):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    now = strftime("%Y_%m_%d_%H_%M_%S", gmtime())  # note: UTC time
    run_name = config.run_name + "_" + now
    if config["wandb"]:
        run = _init_wandb(run_name, config)

    rng = jax.random.PRNGKey(config.seed_model)
    # Setup the model architecture
    rng, rng_init, rng_maps_buffer = jax.random.split(rng, 3)
    
    curriculum = Curriculum(rl_config=config)
    env = env = TerraEnvBatch()
    config.num_embeddings_agent_min = curriculum.get_num_embeddings_agent_min()
    model, params = get_model_ready(rng_init, config, env)
    del rng_init

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    # elif config.train_type == "ES":
    #     from utils.es import train_es as train_fn
    else:
        raise ValueError("Unknown train_type.")

    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
        rng, config, model, params, mle_log, env, curriculum
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
        "env_config": env.env_cfg._asdict(),
        "batch_config": env.batch_cfg._asdict()
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{run_name}.pkl",
    )
    run.finish()


if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="agents/Terra/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-se",
        "--seed_env",
        type=int,
        default=22,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-sm",
        "--seed_model",
        type=int,
        default=33,
        help="Random seed for the model.",
    )
    parser.add_argument(
        "-lr",
        "--lrate",
        type=float,
        default=5e-04,
        help="Random seed of experiment.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        type=bool,
        default=True,
        help="Use wandb.",
    )
    parser.add_argument(
        "-n",
        "--run_name",
        type=str,
        default="ppo",
        help="Name used to store the run on wandb.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_env, args.seed_model, args.lrate, args.wandb, args.run_name)
    main(
        config.train_config,
        mle_log=None,
        log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
    )
