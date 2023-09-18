"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.ppo import obs_to_model_input, wrap_action, clip_action_maps_in_obs, cut_local_map_layers
from terra.state import State
import matplotlib.animation as animation
from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp


def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model

def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
    obs_log = {k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()}
    return obs_log


def rollout_episode(env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames):
    rng = jax.random.PRNGKey(0)


    rng, *rng_reset = jax.random.split(rng, rl_config["num_test_rollouts"] + 1)
    reset_seeds = jnp.array([r[0] for r in rng_reset])
    env_state, obs, maps_buffer_keys = env.reset(reset_seeds, env_cfgs)

    t_counter = 0
    reward_seq = []
    obs_seq = []
    while True:
        obs_seq.append(obs)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            if rl_config["clip_action_maps"]:
                obs = clip_action_maps_in_obs(obs)
            if rl_config["mask_out_arm_extension"]:
                obs = cut_local_map_layers(obs)
            obs = obs_to_model_input(obs)
            action_mask = jnp.ones((8,), dtype=jnp.bool_)  # TODO implement action masking
            v, logits_pi = model.apply(model_params, obs, action_mask)
            pi = tfp.distributions.Categorical(logits=logits_pi)
            action = pi.sample(seed=rng_act)
        else:
            raise RuntimeError("Model is None!")
        next_env_state, (next_obs, reward, done, info), maps_buffer_keys = env.step(env_state, wrap_action(action, env.batch_cfg.action_type), env_cfgs, maps_buffer_keys)
        reward_seq.append(reward)
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        # if done or t_counter == max_frames:
        #     break
        # else:
        if jnp.all(done).item() or t_counter == max_frames:
            break
        env_state = next_env_state
        obs = next_obs
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
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs_x * args.n_envs_y

    log = load_pkl_object(f"{args.run_name}" + ".pkl")
    config = log["train_config"]
    config["num_test_rollouts"] = n_envs

    n_devices = 1

    curriculum = Curriculum(rl_config=config, n_devices=n_devices)
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env = TerraEnvBatch(rendering=True, n_envs_x_rendering=args.n_envs_x, n_envs_y_rendering=args.n_envs_y, display=False, rendering_engine="pygame")
    config["num_embeddings_agent_min"] = curriculum.get_num_embeddings_agent_min()
    

    model = load_neural_network(config, env)
    replicated_params = log['network']
    model_params = jax.tree_map(lambda x: x[0], replicated_params)
    obs_seq, cum_rewards = rollout_episode(
        env, model, model_params, env_cfgs, config, max_frames=args.n_steps
    )

    for o in tqdm(obs_seq, desc="Rendering"):
        env.terra_env.render_obs_pygame(o, generate_gif=True)
    
    env.terra_env.rendering_engine.create_gif()
