"""
This is a test bench.
- Purpose: evaluate changes with little compute
- How: Identify easy tasks that the agent should easily be able to solve, 
    and train on this set of tasks. Get scores on the performance on the taskset, 
    and log the tasks that the agent couldn't complete, for visualization.
"""

# import os
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object
from time import gmtime
from time import strftime
from utils.curriculum_testbench import CurriculumTestbench
from utils.reset_manager import ResetManager
from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from terra.config import TestbenchConfig
from utils.helpers import load_pkl_object
from eval_tracked import rollout_episode
from eval_tracked import print_stats
from visualize_train_logs import animate_from_obs_seq
import numpy as np
# import gc

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


def main(config, mle_log, deterministic_eval, animate, gif_name):
    """Run training with ES or PPO. Store logs and agent ckpt."""

    now = strftime("%Y_%m_%d_%H_%M_%S", gmtime())  # note: UTC time
    run_name = config["run_name"] + "_" + now
    if config["wandb"]:
        run = _init_wandb(run_name, config)

    rng = jax.random.PRNGKey(config["seed_model"])
    # Setup the model architecture
    rng, rng_init, rng_maps_buffer = jax.random.split(rng, 3)
    curriculum = CurriculumTestbench(rl_config=config)
    curr_len = curriculum.get_curriculum_len()

    for task_idx in range(curr_len):

        curriculum.set_dofs(task_idx)

        testbench_cfg = TestbenchConfig()
        env = TerraEnvBatch(batch_cfg=testbench_cfg, rendering=True, n_imgs_row=3)
        config["num_embeddings_agent_min"] = curriculum.get_num_embeddings_agent_min()
        model, params = get_model_ready(rng_init, config, env)
        del rng_init

        if config["model_path"] is not None:
            print(f"\nLoading pre-trained model from: {config['model_path']}")
            log = load_pkl_object(config['model_path'])
            params = log['network']
            print("Pre-trained model loaded. Skipping the training stage.\n")
        else:
            # Run the training loop (either evosax ES or PPO)
            if config["train_type"] == "PPO":
                from utils.ppo import train_ppo as train_fn
            else:
                raise ValueError("Unknown train_type.")

            reset_manager = ResetManager(config, env.observation_shapes)

            # Log and store the results.
            log_steps, log_return, params, _ = train_fn(
                rng, config, model, params, mle_log, env, curriculum, reset_manager, run_name
            )

            data_to_store = {
                "log_steps": log_steps,
                "log_return": log_return,
                "network": params,
                "train_config": config,
                "curriculum": curriculum.curriculum_dicts,
                "default_env_cfg": EnvConfig(),
                "batch_config": env.batch_cfg._asdict()
            }

            save_pkl_object(
                data_to_store,
                f"agents/{config['env_name']}/{run_name}.pkl",
            )

        reset_manager = ResetManager(config, env.observation_shapes, eval=True)
        force_resets_dummy = reset_manager.dummy()
        eval_cfgs, _ = curriculum.get_cfgs_eval()
        cum_rewards, stats, obs_seq = rollout_episode(
            env,
            model,
            params,
            eval_cfgs,
            force_resets_dummy,
            config,
            curriculum.get_dof_max_steps(task_idx),
            deterministic_eval,
        )
        print_stats(stats)

        obs_log_filename = "agents/Terra/" + config["run_name"] + "/eval_best.pkl"
        save_pkl_object(
            obs_seq,
            obs_log_filename,
        )

        if animate:
            # gc.collect()
            animate_from_obs_seq(env, obs_seq, f"{gif_name}_{task_idx}")
        
        run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="agents/Terra/testbench_ppo.yaml",
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
        default="testbench",
        help="Name used to store the run on wandb.",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=None,
        help="Pre-trained model.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        type=int,
        default=1,
        help="Deterministic.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_env, args.seed_model, args.lrate, args.wandb, args.run_name, args.model_path)

    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    main(
        config["train_config"],
        mle_log=None,
        deterministic_eval=deterministic,
        animate=False,
        gif_name="testbench",
    )
