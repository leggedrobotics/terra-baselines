import numpy as np
import jax
import math
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import WheeledAction, TrackedAction, WheeledActionType, TrackedActionType
import jax.numpy as jnp
from utils.ppo import obs_to_model_input, wrap_action, clip_action_maps_in_obs, cut_local_map_layers
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

# def _get_path_length_step(obs, next_obs):
#     print(obs["agent_state"][0])
#     agent_pos = obs["agent_state"][:, [0, 1]]
#     agent_pos_next = next_obs["agent_state"][:, [0, 1]]
#     path_len = np.abs(agent_pos - agent_pos_next).sum()
#     return path_len


def rollout_episode(env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, deterministic):
    """
    NOTE: this function assumes it's a tracked agent in the way it computes the stats.
    """
    rng = jax.random.PRNGKey(0)


    rng, *rng_reset = jax.random.split(rng, rl_config["num_test_rollouts"] + 1)
    reset_seeds = jnp.array([r[0] for r in rng_reset])
    env_state, obs, maps_buffer_keys = env.reset(reset_seeds, env_cfgs)
    
    tile_size = env_cfgs.tile_size[0].item()
    move_tiles = env_cfgs.agent.move_tiles[0].item()
    
    action_type = env.batch_cfg.action_type
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = ()
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (
            WheeledActionType.CLOCK_FORWARD,
            WheeledActionType.CLOCK_BACKWARD,
            WheeledActionType.ANTICLOCK_FORWARD,
            WheeledActionType.ANTICLOCK_BACKWARD,
        )
        do_action = WheeledActionType.DO
    else:
        raise(ValueError(f"{action_type=}"))
    
    areas = (obs["target_map"] == -1).sum(tuple([i for i in range(len(obs["target_map"].shape))][1:])) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(tuple([i for i in range(len(target_maps_init.shape))][1:]))

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            if rl_config["clip_action_maps"]:
                obs = clip_action_maps_in_obs(obs)
            if rl_config["mask_out_arm_extension"]:
                obs = cut_local_map_layers(obs)
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
        next_env_state, (next_obs, reward, done, info), maps_buffer_keys = env.step(env_state, wrap_action(action, env.batch_cfg.action_type), env_cfgs, maps_buffer_keys)
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
        if move_cumsum is None:
            move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
        if do_cumsum is None:
            do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
        
        episode_done_once = episode_done_once | done
        
        episode_length += ~episode_done_once
        
        move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
        for move_action in move_actions:
            move_mask = (action == move_action) * (~episode_done_once)
            move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
        for l_action in l_actions:
            l_mask = (action == l_action) * (~episode_done_once)
            move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
        move_cumsum += move_cumsum_tmp

        do_cumsum += (action == do_action) * (~episode_done_once)

        

    # Path efficiency -- only include finished envs
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas)).sum() / episode_done_once.sum()
    
    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = reference_workspace_area * ((n_dig_actions * episode_done_once) / areas).sum() / episode_done_once.sum()

    # Coverage scores
    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(tuple([i for i in range(len(obs["action_map"].shape))][1:]))
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score = coverage_scores.mean()

    stats = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "path_efficiency": path_efficiency,
        "workspaces_efficiency": workspaces_efficiency,
        "coverage": coverage_score,
    }
    return np.cumsum(reward_seq), stats, obs_seq


def print_stats(stats,):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    completion_rate = 100 * episode_done_once.sum()/len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(f"Path efficiency: {path_efficiency:.2f}")
    print(f"Workspaces efficiency: {workspaces_efficiency:.2f}")
    print(f"Coverage: {coverage:.2f}")


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
        "-d",
        "--deterministic",
        type=int,
        default=1,
        help="Deterministic.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs_x * args.n_envs_y

    log = load_pkl_object(f"{args.run_name}")

    # TODO revert once train_config is available
    config = log["train_config"]
    # from utils.helpers import load_config
    # config = load_config("agents/Terra/ppo.yaml", 22333, 33222, 5e-04, True, "")["train_config"]
    
    config["num_test_rollouts"] = n_envs

    n_devices = 1
    
    curriculum = Curriculum(rl_config=config, n_devices=n_devices)
    env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env = TerraEnvBatch(rendering=False, n_envs_x_rendering=args.n_envs_x, n_envs_y_rendering=args.n_envs_y)
    config["num_embeddings_agent_min"] = curriculum.get_num_embeddings_agent_min()
    

    model = load_neural_network(config, env)
    replicated_params = log['network']
    model_params = jax.tree_map(lambda x: x[0], replicated_params)
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    cum_rewards, stats, _ = rollout_episode(
        env, model, model_params, env_cfgs, config, max_frames=args.n_steps, deterministic=deterministic
    )

    print_stats(stats)
