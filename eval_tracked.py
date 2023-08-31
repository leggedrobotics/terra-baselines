import numpy as np
import jax
import math
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.ppo import obs_to_model_input, wrap_action
from utils.curriculum import Curriculum
from utils.reset_manager import ResetManager
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

# def _get_path_length_step(obs, next_obs):
#     print(obs["agent_state"][0])
#     agent_pos = obs["agent_state"][:, [0, 1]]
#     agent_pos_next = next_obs["agent_state"][:, [0, 1]]
#     path_len = np.abs(agent_pos - agent_pos_next).sum()
#     return path_len


def rollout_episode(env: TerraEnvBatch, model, model_params, env_cfgs, force_resets, rl_config, max_frames, deterministic):
    """
    NOTE: this function assumes it's a tracked agent in the way it computes the stats.
    """
    rng = jax.random.PRNGKey(0)


    rng, *rng_reset = jax.random.split(rng, rl_config["num_test_rollouts"] + 1)
    reset_seeds = jnp.array([r[0] for r in rng_reset])
    env_state, obs, maps_buffer_keys = env.reset(reset_seeds, env_cfgs)

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    # avg_coverage = None
    # avg_path_length = None
    # avg_workspaces = None
    obs_seq = {}
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            obs_model = obs_to_model_input(obs)
            action_mask = jnp.ones((8,), dtype=jnp.bool_)  # TODO implement action masking
            v, logits_pi = model.apply(model_params, obs_model, action_mask)
            if deterministic:
                action = np.argmax(logits_pi, axis=-1)
            else:
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=rng_act)
        else:
            raise RuntimeError("Model is None!")
        next_env_state, (next_obs, reward, done, info), maps_buffer_keys = env.step(env_state, wrap_action(action, env.batch_cfg.action_type), env_cfgs, maps_buffer_keys, force_resets)
        # path_length_step = _get_path_length_step(obs, next_obs)
        reward_seq.append(reward)
        # print(t_counter, reward, action, done)
        print(t_counter)
        print(10 * "=")
        t_counter += 1
        if jnp.all(done).item() or t_counter == max_frames:
            break
        env_state = next_env_state
        obs = next_obs

        # Log stats
        if episode_done_once is None:
            episode_done_once = done
        if episode_length is None:
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)
        # if avg_path_length is None:
        #     avg_path_length = jnp.zeros_like(done, dtype=jnp.int32)
        # if avg_workspaces is None:
        #     avg_workspaces = jnp.zeros_like(done, dtype=jnp.int32)
        episode_done_once = episode_done_once | done
        episode_length += ~episode_done_once
        # avg_path_length += path_length_step
        # avg_workspaces += 

        stats = {
            "episode_done_once": episode_done_once,
            "episode_length": episode_length,
            # "avg_coverage": avg_coverage,
            # "avg_path_length": avg_path_length,
            # "avg_workspaces": avg_workspaces,
        }
    return np.cumsum(reward_seq), stats, obs_seq


def print_stats(stats,):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    # avg_path_length = stats["avg_path_length"]
    # avg_workspaces = stats["avg_workspaces"]
    # avg_coverage = stats["avg_coverage"]

    print("\nStats:\n")
    print(f"Number of episodes finished at least once: {episode_done_once.sum()} / {len(episode_done_once)} ({100 * episode_done_once.sum()/len(episode_done_once)}%)")
    print(f"First episode length average: {episode_length.mean()}")
    print(f"First episode length min: {episode_length.min()}")
    print(f"First episode length max: {episode_length.max()}")


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
        "-n",
        "--n_envs",
        type=int,
        default=1,
        help="Number of environments.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=10,
        help="Number of steps.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        type=int,
        default=1,
        help="Deterministic.",
    )
    args, _ = parser.parse_known_args()

    log = load_pkl_object(f"{args.run_name}")

    # TODO revert once train_config is available
    # config = log["train_config"]
    from utils.helpers import load_config
    config = load_config("agents/Terra/ppo.yaml", 22, 33, 5e-04, True, "")["train_config"]
    
    config["num_test_rollouts"] = args.n_envs
    
    curriculum = Curriculum(rl_config=config)
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env = TerraEnvBatch(rendering=True, n_imgs_row=int(math.sqrt(args.n_envs)))
    config["num_embeddings_agent_min"] = curriculum.get_num_embeddings_agent_min()
    
    reset_manager = ResetManager(config, env.observation_shapes, eval=True)
    force_resets = reset_manager.dummy()

    model = load_neural_network(config, env)
    model_params = log["network"]
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    cum_rewards, stats, _ = rollout_episode(
        env, model, model_params, env_cfgs, force_resets, config, max_frames=args.n_steps, deterministic=deterministic
    )

    print_stats(stats)
