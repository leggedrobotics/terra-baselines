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

class Instance:
    def __init__(
            self,
            state,
            obs,
            value,
            action,
            reward,
            ) -> None:
        self.state = state
        self.obs = obs
        self.value = value
        self.action = action
        self.reward = reward

def tree_search(rng, env_state, obs, model, maps_buffer_keys, clip_action_maps, mask_out_arm_extension, deterministic, n_branches = 4, depth = 1, n_rollouts = 1, gamma = 0.995):
    """Naive tree search implementation (can be optimized a lot)"""
    action_mask = jnp.ones((8,), dtype=jnp.bool_)  # TODO implement action masking

    # TODO shouldn't do this copy here, most of the tree will be repeated this way
    # last_instance_level = [Instance(env_state, obs, 0, -1, 0) for _ in range(n_branches)]
    last_instance_level = [Instance(env_state, obs, 0, -1, 0)]
    instance_levels = []
    for depth_level in range(depth):
        # print(f"{depth_level=}")
        new_instance_level = []
        for instance_idx, instance in enumerate(last_instance_level):
            # print(f"{instance_idx=}")

            if clip_action_maps:
                obs = clip_action_maps_in_obs(instance.obs)
            if mask_out_arm_extension:
                obs = cut_local_map_layers(obs)
            obs_model = obs_to_model_input(obs)
            _, logits_pi = model.apply(model_params, obs_model, action_mask)
            actions = np.argsort(logits_pi, axis=-1)[..., -n_branches:]
            actions = [actions[..., i] for i in range(actions.shape[-1])]
            # print(f"{actions=}")
            for action_idx, action in enumerate(actions):
                # print(f"{action_idx=}")
                # TODO take into account 'done' for following env
                original_action = action.copy()
                local_cum_disc_reward = np.zeros((actions[0].shape[0],))
                for rollout_level in range(n_rollouts):
                    rng, rng_act = jax.random.split(rng)
                    next_env_state, (next_obs_dict, reward, done, info), maps_buffer_keys = env.step(instance.state, wrap_action(action, env.batch_cfg.action_type), env_cfgs, maps_buffer_keys)
                    local_cum_disc_reward += (gamma ** ((depth_level * n_rollouts) + rollout_level)) * reward / 200
                    if clip_action_maps:
                        next_obs = clip_action_maps_in_obs(next_obs_dict)
                    if mask_out_arm_extension:
                        next_obs = cut_local_map_layers(next_obs)
                    next_obs_model = obs_to_model_input(next_obs)
                    v, logits_pi = model.apply(model_params, next_obs_model, action_mask)
                    if deterministic:
                        action = np.argmax(logits_pi, axis=-1)
                    else:
                        pi = tfp.distributions.Categorical(logits=logits_pi)
                        action = pi.sample(seed=rng_act)
                new_instance_level.append(Instance(next_env_state, next_obs_dict, v, original_action, local_cum_disc_reward))
        instance_levels.append(new_instance_level)
        last_instance_level = new_instance_level

    # print(f"{instance_levels=}")

    # Backprop
    cum_disc_rewards = np.zeros((len(instance_levels[-1]), action.shape[0]))
    # print(f"{cum_disc_rewards.shape=}")
    for current_depth, instance_level in enumerate(instance_levels):
        # print(f"{current_depth=}")
        values = np.array([el.value.squeeze(-1).tolist() for el in instance_level])
        rewards = np.array([el.reward.tolist() for el in instance_level])
        n_repeats = max(1, n_branches ** (depth - current_depth - 1))
        # print(f"0 {values.shape=}")
        # print(f"0 {rewards.shape=}")
        # print(f"{n_repeats=}")
        values = values.repeat(n_repeats, 0)
        rewards = rewards.repeat(n_repeats, 0)
        # print(f"1 {values.shape=}")
        # print(f"1 {rewards.shape=}")
        if n_repeats == 1:
            cum_disc_rewards += (gamma ** (depth * n_rollouts)) * values
        else:
            cum_disc_rewards += rewards
    
    # print(f"{cum_disc_rewards=}")
    winning_branches = np.argmax(cum_disc_rewards, 0)
    winning_first_branches = winning_branches  // (n_branches ** (depth - 1))
    # print(f"{winning_branches=}")
    print(f"{winning_first_branches=}")

    winning_actions = np.array([instance_levels[0][idx].action.tolist()[action_idx] for action_idx, idx in enumerate(winning_first_branches.tolist())])
    print(f"{winning_actions=}")

    return winning_actions, rng


def rollout_episode(env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, deterministic, seed, search):
    """
    NOTE: this function assumes it's a tracked agent in the way it computes the stats.
    """
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)


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
            if search:
                action, rng = tree_search(rng, env_state, obs, model, maps_buffer_keys, rl_config["clip_action_maps"], rl_config["mask_out_arm_extension"], deterministic)
            elif deterministic:
                if rl_config["clip_action_maps"]:
                    obs = clip_action_maps_in_obs(obs)
                if rl_config["mask_out_arm_extension"]:
                    obs = cut_local_map_layers(obs)
                obs_model = obs_to_model_input(obs)
                action_mask = jnp.ones((8,), dtype=jnp.bool_)  # TODO implement action masking
                v, logits_pi = model.apply(model_params, obs_model, action_mask)
                action = np.argmax(logits_pi, axis=-1)
            else:
                if rl_config["clip_action_maps"]:
                    obs = clip_action_maps_in_obs(obs)
                if rl_config["mask_out_arm_extension"]:
                    obs = cut_local_map_layers(obs)
                obs_model = obs_to_model_input(obs)
                action_mask = jnp.ones((8,), dtype=jnp.bool_)  # TODO implement action masking
                v, logits_pi = model.apply(model_params, obs_model, action_mask)
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
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()
    
    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = reference_workspace_area * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    # Coverage scores
    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(tuple([i for i in range(len(obs["action_map"].shape))][1:]))
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    stats = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "path_efficiency": {
            "mean": path_efficiency_mean,
            "std": path_efficiency_std,
        },
        "workspaces_efficiency": {
            "mean": workspaces_efficiency_mean,
            "std": workspaces_efficiency_std,
        },
        "coverage": {
            "mean": coverage_score_mean,
            "std": coverage_score_std,
        },
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
    print(f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})")
    print(f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})")
    print(f"Coverage: {coverage['mean']:.2f} ({coverage['std']:.2f})")


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
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-search",
        "--tree_search",
        type=int,
        default=0,
        help="Random seed for the environment.",
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
    search = bool(args.tree_search)
    print(f"\nTree Search = {search}\n")

    cum_rewards, stats, _ = rollout_episode(
        env, model, model_params, env_cfgs, config, max_frames=args.n_steps, deterministic=deterministic, seed=args.seed, search=search
    )

    print_stats(stats)
