import numpy as np
import jax
import math
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from train_mixed_agents import MixedAgentTrainConfig

#sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig

# from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints


def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
    obs_log = {
        k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
    }
    return obs_log


def get_excavator_obs(obs):
    """
    Extract only excavator fields from the full observation dict.
    """
    return {
        "agent_states": obs["agent_states"],
        "local_map_action_neg": obs["local_map_action_neg"],
        "local_map_action_pos": obs["local_map_action_pos"],
        "local_map_target_neg": obs["local_map_target_neg"],
        "local_map_target_pos": obs["local_map_target_pos"],
        "local_map_dumpability": obs["local_map_dumpability"],
        "local_map_obstacles": obs["local_map_obstacles"],
        "action_map": obs["action_map"],
        "target_map": obs["target_map"],
        "traversability_mask": obs["traversability_mask"],
        "dumpability_mask": obs["dumpability_mask"],
    }


def rollout_episode(
    env: TerraEnvBatch,
    model,
    model_params,
    env_cfgs,
    rl_config,
    max_frames,
    deterministic,
    seed,
):
    """
    Alternating network: only log stats when the active agent (slot 0) is the excavator.
    Uses simplified single-agent logic but only processes excavator turns.
    """
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    # Initialize reward_components structure to match training (avoid fori_loop pytree mismatch)
    try:
        if hasattr(timestep, 'info') and isinstance(timestep.info, dict):
            batch_shape = timestep.reward.shape
            MAX_AGENTS = 4
            dummy_components = {
                "agent_rewards": jnp.zeros(batch_shape + (MAX_AGENTS,), dtype=jnp.float32),
                "agent_active": jnp.zeros(batch_shape + (MAX_AGENTS,), dtype=jnp.int32),
                "num_agents": jnp.zeros(batch_shape, dtype=jnp.int32),
                "terminal": jnp.zeros_like(timestep.reward),
                "trench": jnp.zeros_like(timestep.reward),
                "existence": jnp.zeros_like(timestep.reward),
            }
            timestep = timestep._replace(info={**timestep.info, "reward_components": dummy_components})
    except Exception:
        pass
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    tile_size = env_cfgs.tile_size[0].item()
    move_tiles = env_cfgs.agent.move_tiles[0].item()

    action_type = env.batch_cfg.action_type
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = (TrackedActionType.CLOCK, TrackedActionType.ANTICLOCK)
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (WheeledActionType.WHEELS_LEFT, WheeledActionType.WHEELS_RIGHT)
        do_action = WheeledActionType.DO
    else:
        raise NotImplementedError(f"Action type {action_type} not supported for eval.")

    obs = timestep.observation
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
    ) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}
    AGENT_TYPE_IDX = 6  # agent_type is at index 6 in per-agent feature
    EXCAVATOR_TYPE = 0  # excavator is type 0, skidsteer should be type 1
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        
        # In alternating network, slot 0 is always the active agent (whose turn it is)
        # Print full per-agent features for first few steps
        if t_counter < 5:
            print("  -> agent_states[0]:")
            for idx, val in enumerate(timestep.observation["agent_states"][0]):
                print(f"     idx {idx}: {val}")
            last_idx = max(0, int(timestep.observation["num_agents"]) - 1)
            print("  -> agent_states[last_active]:")
            for idx, val in enumerate(timestep.observation["agent_states"][last_idx]):
                print(f"     idx {idx}: {val}")
        # Check if the active agent (slot 0) is the excavator
        agent_type = timestep.observation["agent_states"][0, AGENT_TYPE_IDX]
        is_excavator_turn = (agent_type == EXCAVATOR_TYPE)
        print(f"Step {t_counter}: slot 0 agent type = {agent_type}, is_excavator_turn = {is_excavator_turn}")
        
        if model is not None:
            # Always step the environment, but only log stats for excavator
            obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, logits_pi = model.apply(model_params, obs_model)
            if deterministic:
                action = np.argmax(logits_pi, axis=-1)
            else:
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=rng_act)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)
        else:
            raise RuntimeError("Model is None!")
        
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )
        reward = timestep.reward
        next_obs = timestep.observation
        done = timestep.info["task_done"]

        # Only log stats if the active agent is the excavator
        if is_excavator_turn:
            #print(f"  -> Excavator turn detected! Logging stats.")
            #print(f"     reward = {reward}, done = {done}")
            
            # Custom completion check: only consider digging done, not dumping
            # Match the environment's digging completion logic exactly
            dig_requirements = jnp.where(timestep.observation["target_map"] < 0, timestep.observation["target_map"], 0)
            actual_digs = jnp.where(timestep.observation["target_map"] < 0, timestep.observation["action_map"], 0)
            # Apply jnp.all() per environment, not across the entire batch
            digging_complete = jnp.all(actual_digs <= dig_requirements, axis=tuple(range(1, len(actual_digs.shape))))
            
            # Use custom completion instead of environment's done
            custom_done = digging_complete
            
            # Debug: show task progress
            # if t_counter % 20 == 0:  # print every 20 steps to avoid spam
            #     target_count = (timestep.observation["target_map"] < 0).sum()
            #     dug_count = (dig_requirements == actual_digs).sum()
            #     print(f"     Task progress: target_tiles={target_count}, dug_tiles={dug_count}, digging_complete={digging_complete}")
            
            # Debug: compare environment done vs custom done
            if jnp.all(done).item():
                print(f"     Environment says done=True, but custom_done={custom_done}")
                print(f"     target_tiles.sum()={target_tiles.sum()}, dug_tiles.sum()={dug_tiles.sum()}")
            
            # Debug: print final maps when episode ends
            if jnp.all(done).item():
                print("Final target_map:\n", timestep.observation["target_map"])
                print("Final action_map:\n", timestep.observation["action_map"])
                print("Final dig_requirements:\n", dig_requirements)
                print("Final actual_digs:\n", actual_digs)
                print("Final custom_done:", custom_done)
                print("Amount needed (dig_requirements.sum()):", dig_requirements.sum())
                print("Amount dug (actual_digs.sum()):", actual_digs.sum())
                print("All dug correctly (actual_digs <= dig_requirements):", jnp.all(actual_digs <= dig_requirements))
            
            reward_seq.append(reward)
            print(t_counter)
            print(10 * "=")
            t_counter += 1
            if episode_done_once is None:
                episode_done_once = custom_done
            if episode_length is None:
                episode_length = jnp.zeros_like(custom_done, dtype=jnp.int32)
            if move_cumsum is None:
                move_cumsum = jnp.zeros_like(custom_done, dtype=jnp.int32)
            if do_cumsum is None:
                do_cumsum = jnp.zeros_like(custom_done, dtype=jnp.int32)

            episode_done_once = episode_done_once | custom_done
            episode_length += ~episode_done_once

            move_cumsum_tmp = jnp.zeros_like(custom_done, dtype=jnp.int32)
            for move_action in move_actions:
                move_mask = (action == move_action) * (~episode_done_once)
                move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
            for l_action in l_actions:
                l_mask = (action == l_action) * (~episode_done_once)
                move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
            move_cumsum += move_cumsum_tmp

            do_cumsum += (action == do_action) * (~episode_done_once)
            print(f"     episode_done_once = {episode_done_once}, move_cumsum = {move_cumsum}, do_cumsum = {do_cumsum}")
            # Debug: print custom_done and episode_done_once
            print(f"     custom_done = {custom_done}, shape = {getattr(custom_done, 'shape', type(custom_done))}")
            print(f"     episode_done_once = {episode_done_once}, shape = {getattr(episode_done_once, 'shape', type(episode_done_once))}")
            # Now check for break
            if jnp.all(custom_done).item() or t_counter == max_frames:
                print(f"  -> Episode ended: all done = {jnp.all(custom_done).item()}, t_counter = {t_counter}, max_frames = {max_frames}")
                break
        else:
            print(f"  -> Skidsteer turn (agent_type={agent_type}), skipping stats.")

        obs = next_obs

    # Path efficiency -- only include finished envs
    if episode_done_once is None:
        # No excavator steps were taken
        return np.array([]), {}, obs_seq
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    # Coverage scores
    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
        tuple([i for i in range(len(obs["action_map"].shape))][1:])
    )
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


def print_stats(
    stats,
):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    # Handle case where episode_done_once might be None or have unexpected shape
    if episode_done_once is None:
        completion_rate = 0.0
    else:
        try:
            completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)
        except (TypeError, IndexError):
            # Fallback if len() fails
            completion_rate = 100.0 if episode_done_once.sum() > 0 else 0.0

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(
        f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})"
    )
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
        "-n",
        "--n_envs",
        type=int,
        default=128,
        help="Number of environments.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=600,
        help="Number of steps.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        type=int,
        default=0,
        help="Deterministic. 0 for stochastic, 1 for deterministic.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    # from utils.helpers import load_config
    # config = load_config("agents/Terra/ppo.yaml", 22333, 33222, 5e-04, True, "")["train_config"]

    config.num_test_rollouts = n_envs
    config.num_devices = 1

    # curriculum = Curriculum(rl_config=config, n_devices=n_devices)
    # env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    )  # take first config and replicate
    shuffle_maps = True
    env = TerraEnvBatch(rendering=False, shuffle_maps=shuffle_maps)
    config.num_embeddings_agent_min = 60

    model = load_neural_network(config, env)
    model_params = log["model"]
    # model_params = jax.tree_map(lambda x: x[0], replicated_params)
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    cum_rewards, stats, _ = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        deterministic=deterministic,
        seed=args.seed,
    )

    print_stats(stats)
