"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation

# from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.actions import TrackedAction, WheeledAction


def rollout_episode(
    env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, seed
):
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions_1 = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )
    prev_actions_2 = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    t_counter = 0
    reward_seq = []
    obs_seq = []
    while True:
        obs_seq.append(timestep.observation)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            obs = obs_to_model_input(timestep.observation, prev_actions_1, prev_actions_2, rl_config)
            v, logits_pi = model.apply(model_params, obs)
            
            num_actions = env.batch_cfg.action_type.get_num_actions()
            total_logits = logits_pi.shape[-1]

            if total_logits == 2 * num_actions:
                logits_1 = logits_pi[..., :num_actions]
                logits_2 = logits_pi[..., num_actions:]
                
                pi_1 = tfp.distributions.Categorical(logits=logits_1)
                pi_2 = tfp.distributions.Categorical(logits=logits_2)
                
                rng_act_1, rng_act_2 = jax.random.split(rng_act)
                action_1 = pi_1.sample(seed=rng_act_1)
                action_2 = pi_2.sample(seed=rng_act_2)
            else:
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=rng_act)
                action_1 = action
                action_2 = action

            prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=1)
            prev_actions_1 = prev_actions_1.at[:, 0].set(action_1)
            prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=1)
            prev_actions_2 = prev_actions_2.at[:, 0].set(action_2)
        else:
            raise RuntimeError("Model is None!")
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        action_env_1 = wrap_action(action_1, env.batch_cfg.action_type)
        action_env_2 = wrap_action(action_2, env.batch_cfg.action_type)
        timestep = env.step(
            timestep, action_env_1, action_env_2, rng_step
        )
        reward_seq.append(timestep.reward)
        print(t_counter, timestep.reward, action_1, action_2, timestep.done)
        print(10 * "=")
        t_counter += 1
        # if done or t_counter == max_frames:
        #     break
        # else:
        if jnp.all(timestep.done).item() or t_counter == max_frames:
            break
        # env_state = next_env_state
        # obs = next_obs
    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return obs_seq, np.cumsum(reward_seq)


def update_render(seq, env: TerraEnvBatch, frame):
    obs = {k: v[:, frame] for k, v in seq.items()}
    return env.terra_env.render_obs(obs, mode="gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="ppo_2023_05_09_10_01_23",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    parser.add_argument(
        "-nx",
        "--n_envs_x",
        type=int,
        default=1,
        help="Number of environments on x.",
    )
    parser.add_argument(
        "-ny",
        "--n_envs_y",
        type=int,
        default=1,
        help="Number of environments on y.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=10,
        help="Number of steps.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=".",
        help="Output path.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs_x * args.n_envs_y

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    # curriculum = Curriculum(rl_config=config, n_devices=n_devices)
    # env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    )  # take first config and replicate
    suffle_maps = True
    env = TerraEnvBatch(
        rendering=True,
        n_envs_x_rendering=args.n_envs_x,
        n_envs_y_rendering=args.n_envs_y,
        display=False,
        shuffle_maps=suffle_maps,
    )
    config.num_embeddings_agent_min = 60  # curriculum.get_num_embeddings_agent_min()

    model = load_neural_network(config, env)
    model_params = log["model"]
    # replicated_params = log['network']
    # model_params = jax.tree_map(lambda x: x[0], replicated_params)
    obs_seq, cum_rewards = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        seed=args.seed,
    )

    for o in tqdm(obs_seq, desc="Rendering"):
        env.terra_env.render_obs_pygame(o, generate_gif=True)

    env.terra_env.rendering_engine.create_gif(args.out_path)
