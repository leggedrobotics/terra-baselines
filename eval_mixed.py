"""
Evaluation script for mixed agent training (Tracked Excavator + Skid Steer).
Based on visualize_mixed.py but with comprehensive evaluation metrics.
"""

import numpy as np
import jax
import math
from tqdm import tqdm
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)
from terra.config import BatchConfig
import jax.numpy as jnp
from utils.utils_ppo import action_type_from_policy_action, obs_to_model_input, policy, wrap_action
from terra.state import State
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
import sys
from pathlib import Path
from train_mixed import MixedAgentTrainConfig
sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig

# Default maps dir matches inference_single_map.py convention.
_DEFAULT_MAPS_DIR = Path(__file__).parent / "inference" / "maps"


def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
    obs_log = {
        k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
    }
    return obs_log


def rollout_episode(
    env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, deterministic, seed
):
    """
    Mixed agent evaluation rollout with comprehensive metrics.
    """
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    
    # Initialize reward_components structure to match training
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
                "move_meters": jnp.zeros_like(timestep.reward),
                "macro_move_count": jnp.zeros_like(timestep.reward),
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
    obs_seq = {}
    
    # Per-agent tracking across the batch (agent_type: 0=excavator, 1=truck, 2=skidsteer)
    per_agent_move_m = {
        0: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.float32),
        1: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.float32),
        2: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.float32),
    }
    per_agent_do_events = {
        0: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.int32),
        1: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.int32),
        2: jnp.zeros(rl_config.num_test_rollouts, dtype=jnp.int32),
    }
    # Last seen position per agent-type per env (None until first seen)
    last_pos_per_agent = {0: None, 1: None, 2: None}
    prev_action_map = None
    AGENT_TYPE_IDX = 6  # agent_type is at index 6 in per-agent feature
    EXCAVATOR_TYPE = 0  # excavator is type 0
    
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        
        # Print agent info for first few steps
        if t_counter < 5:
            print("  -> agent_states[0] (env0):")
            for idx, val in enumerate(timestep.observation["agent_states"][0, 0]):
                print(f"     idx {idx}: {val}")
            last_idx = max(0, int(timestep.observation["num_agents"][0]) - 1)
            print("  -> agent_states[last_active] (env0):")
            for idx, val in enumerate(timestep.observation["agent_states"][0, last_idx]):
                print(f"     idx {idx}: {val}")
        
        # Check if the active agent (slot 0) is the excavator
        agent_type = timestep.observation["agent_states"][0, 0, AGENT_TYPE_IDX]
        is_excavator_turn = (agent_type == EXCAVATOR_TYPE)
        print(f"Step {t_counter}: slot 0 agent type = {agent_type}, is_excavator_turn = {is_excavator_turn}")
        
        if model is not None:
            obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, pi = policy(model.apply, model_params, obs_model)
            if deterministic:
                action = pi.mode()
            else:
                action = pi.sample(seed=rng_act)
            action_type_sample = action_type_from_policy_action(action)
            
            # Debug: print action for first few steps
            if t_counter < 5:
                print(f"     Model action (env0): {action[0]}")
                print(f"     Action logits shape: {pi.action_logits.shape}")
                print(f"     Action logits (env0): {pi.action_logits[0]}")
                # Convert logits to probabilities for better understanding
                import jax.nn as jnn
                probs = jnn.softmax(pi.action_logits[0])
                print(f"     Action probabilities (env0): {probs}")
                print(f"     DO action (6) probability: {probs[6]:.6f}")
                print(f"     DO_NOTHING action (7) probability: {probs[7]:.6f}")
            
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action_type_sample)
        else:
            raise RuntimeError("Model is None!")
        
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )
        
        reward = timestep.reward
        next_obs = timestep.observation
        done = timestep.info["task_done"]

        # Per-agent accumulation for all envs (active agent is index 0 per env).
        # Gate every per-env update on ~episode_done_once so finished envs stop
        # accumulating movement / do events for the remaining rollout steps.
        # Without this, metrics like move_m, do_events, path_efficiency, and
        # workspaces_efficiency get inflated by post-terminal activity.
        try:
            agent_states_batch = obs["agent_states"]  # [B, MAX_AGENTS, feat]
            active_pos_batch = agent_states_batch[:, 0, 0:2]  # [B, 2]
            active_type_batch = agent_states_batch[:, 0, AGENT_TYPE_IDX].astype(jnp.int32)  # [B]

            if episode_done_once is None:
                active_env_mask = jnp.ones(active_type_batch.shape[0], dtype=jnp.bool_)
            else:
                active_env_mask = ~episode_done_once  # [B]
            active_env_f = active_env_mask.astype(jnp.float32)
            active_env_i = active_env_mask.astype(jnp.int32)

            # Movement distance per agent type across batch
            for atype in (0, 1, 2):
                mask = (active_type_batch == atype)  # [B]
                if last_pos_per_agent[atype] is None:
                    last_pos_per_agent[atype] = active_pos_batch
                else:
                    delta = jnp.linalg.norm(active_pos_batch - last_pos_per_agent[atype], axis=1)  # [B]
                    per_agent_move_m[atype] = (
                        per_agent_move_m[atype]
                        + delta * tile_size * mask.astype(jnp.float32) * active_env_f
                    )
                    # Update last pos only for envs where this type acted AND is still running.
                    update_pos_mask = mask & active_env_mask
                    last_pos_per_agent[atype] = jnp.where(
                        update_pos_mask[:, None], active_pos_batch, last_pos_per_agent[atype]
                    )

            # Do events: detect any change in action_map per env, attribute to active agent type
            if prev_action_map is not None:
                changed = (obs["action_map"] != prev_action_map)  # [B, H, W]
                any_change = changed.reshape((changed.shape[0], -1)).any(axis=1).astype(jnp.int32)  # [B]
                any_change = any_change * active_env_i  # zero out finished envs
                for atype in (0, 1, 2):
                    mask = (active_type_batch == atype).astype(jnp.int32)
                    per_agent_do_events[atype] = per_agent_do_events[atype] + any_change * mask
            prev_action_map = obs["action_map"].copy()
        except Exception:
            pass

        # Log every step
        reward_seq.append(reward)
        print(t_counter)
        print(10 * "=")
        t_counter += 1
        if episode_done_once is None:
            episode_done_once = done
        if episode_length is None:
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)

        episode_done_once = episode_done_once | done
        episode_length += ~episode_done_once

        # Per-step concise log for env0
        try:
            active_type0 = int(timestep.observation["agent_states"][0, 0, AGENT_TYPE_IDX])
            any_change0 = 0
            if prev_action_map is not None:
                any_change0 = int(((obs["action_map"] != prev_action_map)[0]).any())
            print(f"     step_info: env0.active_type={active_type0}, do_changed={any_change0}")
        except Exception:
            pass

        # Check for break
        if jnp.all(done).item() or t_counter == max_frames:
            print(f"  -> Episode ended: all done = {jnp.all(done).item()}, t_counter = {t_counter}, max_frames = {max_frames}")
            break

        obs = next_obs

    # Calculate comprehensive metrics
    if episode_done_once is None:
        return np.array([]), {}, obs_seq
    
    # Team movement per env in meters (sum across agent types)
    team_move_m = per_agent_move_m[0] + per_agent_move_m.get(1, 0) + per_agent_move_m.get(2, 0)
    team_path_efficiency = (team_move_m / jnp.sqrt(areas))
    path_efficiency = team_path_efficiency[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    # Excavators and skidsteers: 2 DO events per cycle (dig + dump), trucks: 1 DO event per dump (loading is automatic)
    excavator_ops = per_agent_do_events[0] // 2  # dig-dump cycles
    skidsteer_ops = per_agent_do_events.get(2, 0) // 2  # dig-dump cycles
    truck_ops = per_agent_do_events.get(1, 0)  # dump operations (no division, loading is automatic)
    team_workspace_ops = excavator_ops + skidsteer_ops + truck_ops
    workspaces_efficiency = (reference_workspace_area * (team_workspace_ops / areas))[episode_done_once]
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    # Coverage scores using dirt-flow logic
    reduce_axes = tuple([i for i in range(len(obs["action_map"].shape))][1:])
    # Dig completion: 1 - (undug / total_dig_req)
    dig_req_mask = (target_maps_init < 0)
    undug_mask = dig_req_mask & (obs["action_map"] >= 0)
    undug_count = undug_mask.sum(reduce_axes)
    total_dig_req = dig_req_mask.sum(reduce_axes)
    dig_coverage = jnp.where(total_dig_req > 0, 1.0 - (undug_count / jnp.maximum(total_dig_req, 1)), 1.0)
    # Dump completion: fraction of dumped dirt (units) on dump zones
    dumped_vals = jnp.where(obs["action_map"] > 0, obs["action_map"], 0)
    dump_correct = jnp.where(target_maps_init > 0, dumped_vals, 0).sum(reduce_axes)
    dumped_total = dumped_vals.sum(reduce_axes)
    dump_coverage = jnp.where(dumped_total > 0, dump_correct / jnp.maximum(dumped_total, 1), 1.0)
    # Total completion: dirt in correct zones / (dumped_dirt + undug_dirt + loaded_dirt)
    try:
        loaded_feat_idx = 5  # agent_states feature index for 'loaded' amount
        loaded_amount = jnp.maximum(obs["agent_states"][:, :, loaded_feat_idx], 0)
        loaded_sum = loaded_amount.sum(axis=1)  # per-env total loaded amount
    except Exception:
        loaded_sum = jnp.zeros_like(dumped_total)
    denom_total_dirt = dumped_total + undug_count + loaded_sum
    total_completion = jnp.where(denom_total_dirt > 0, dump_correct / jnp.maximum(denom_total_dirt, 1), 0.0)
    coverage_scores = episode_done_once + (~episode_done_once) * total_completion
    # Also mask sub-components so completed envs count as 1.0
    dig_cov_scores = episode_done_once + (~episode_done_once) * dig_coverage
    dump_cov_scores = episode_done_once + (~episode_done_once) * dump_coverage
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    # Average steps till completion over completed envs
    avg_steps_till_completion = (
        episode_length[episode_done_once].mean() if episode_length is not None else jnp.array(0)
    )

    # Parallel step estimate
    try:
        num_agents_arr = timestep.observation["num_agents"].astype(jnp.float32)
        avg_num_agents_completed = num_agents_arr[episode_done_once].mean() if episode_done_once.any() else jnp.array(1.0)
    except Exception:
        avg_num_agents_completed = jnp.array(1.0)
    parallel_steps_to_goal_mean = float(avg_steps_till_completion) / max(float(avg_num_agents_completed), 1.0) if episode_length is not None else 0.0

    # Goal efficiency metrics
    if episode_length is not None:
        completed_steps = episode_length[episode_done_once]
        goal_steps_mean = completed_steps.mean() if completed_steps.size > 0 else jnp.array(0)
        goal_steps_std = completed_steps.std() if completed_steps.size > 0 else jnp.array(0)
        goal_efficiency_mean = (1.0 / jnp.maximum(goal_steps_mean, 1)) if completed_steps.size > 0 else jnp.array(0.0)
        progress_rate = coverage_scores / jnp.maximum(episode_length, 1)
        goal_progress_rate_mean = progress_rate.mean()
        goal_progress_rate_std = progress_rate.std()
    else:
        goal_steps_mean = jnp.array(0)
        goal_steps_std = jnp.array(0)
        goal_efficiency_mean = jnp.array(0.0)
        goal_progress_rate_mean = jnp.array(0.0)
        goal_progress_rate_std = jnp.array(0.0)

    # Aggregate per-agent metrics over completed envs
    def _agg_per_agent(v_dict):
        out = {}
        for k, v in v_dict.items():
            try:
                out[k] = jnp.where(episode_done_once, v, 0).sum() / jnp.maximum(episode_done_once.sum(), 1)
            except Exception:
                out[k] = jnp.array(0)
        return out

    per_agent_move_m_mean = _agg_per_agent(per_agent_move_m)
    per_agent_do_events_mean = _agg_per_agent(per_agent_do_events)
    # Per-agent path efficiency: distance / sqrt(area)
    sqrt_areas = jnp.sqrt(areas)
    per_agent_path_eff_mean = {}
    for k, v in per_agent_move_m.items():
        try:
            path_eff = jnp.where(episode_done_once, v / sqrt_areas, 0)
            per_agent_path_eff_mean[k] = path_eff.sum() / jnp.maximum(episode_done_once.sum(), 1)
        except Exception:
            per_agent_path_eff_mean[k] = jnp.array(0.0)

    # Per-agent workspace efficiency: reference_area * (operations / area)
    # Excavators/skidsteers: divide by 2 (dig-dump cycles), trucks: no division (dump operations)
    per_agent_workspace_eff_mean = {}
    for k, v in per_agent_do_events.items():
        try:
            if k == 1:  # Truck: each DO event is 1 operation (dump only, loading is automatic)
                n_ops = v
            else:  # Excavator (0) or Skidsteer (2): divide by 2 (dig-dump cycles)
                n_ops = v // 2
            ws_eff = jnp.where(episode_done_once, reference_workspace_area * (n_ops / areas), 0)
            per_agent_workspace_eff_mean[k] = ws_eff.sum() / jnp.maximum(episode_done_once.sum(), 1)
        except Exception:
            per_agent_workspace_eff_mean[k] = jnp.array(0.0)

    # Collaborative stats
    try:
        mean_of_agent_path_eff = (
            per_agent_path_eff_mean.get(0, 0.0)
            + per_agent_path_eff_mean.get(1, 0.0)
            + per_agent_path_eff_mean.get(2, 0.0)
        ) / 3.0
    except Exception:
        mean_of_agent_path_eff = jnp.array(0.0)
    collab_gain_path_eff = path_efficiency_mean - mean_of_agent_path_eff

    # Diversity of work (entropy over agent shares of do events)
    do_means = jnp.array([
        per_agent_do_events_mean.get(0, 0.0),
        per_agent_do_events_mean.get(1, 0.0),
        per_agent_do_events_mean.get(2, 0.0),
    ], dtype=jnp.float32)
    do_total = do_means.sum() + 1e-8
    p = do_means / do_total
    diversity_entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + 1e-8), 0.0))

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
            "total": {
                "mean": coverage_score_mean,
                "std": coverage_score_std,
            },
            "dig": {
                "mean": dig_cov_scores.mean(),
                "std": dig_cov_scores.std(),
            },
            "dump": {
                "mean": dump_cov_scores.mean(),
                "std": dump_cov_scores.std(),
            },
        },
        "avg_steps_till_completion": avg_steps_till_completion,
        "goal_efficiency": {
            "goal_steps_mean": goal_steps_mean,
            "goal_steps_std": goal_steps_std,
            "goal_efficiency_mean": goal_efficiency_mean,
            "goal_progress_rate_mean": goal_progress_rate_mean,
            "goal_progress_rate_std": goal_progress_rate_std,
            "parallel_steps_to_goal_mean": parallel_steps_to_goal_mean,
        },
        "per_agent": {
            "move_m_mean": {
                "excavator": per_agent_move_m_mean.get(0, jnp.array(0.0)),
                "truck": per_agent_move_m_mean.get(1, jnp.array(0.0)),
                "skidsteer": per_agent_move_m_mean.get(2, jnp.array(0.0)),
            },
            "path_efficiency_mean": {
                "excavator": per_agent_path_eff_mean.get(0, jnp.array(0.0)),
                "truck": per_agent_path_eff_mean.get(1, jnp.array(0.0)),
                "skidsteer": per_agent_path_eff_mean.get(2, jnp.array(0.0)),
            },
            "workspace_efficiency_mean": {
                "excavator": per_agent_workspace_eff_mean.get(0, jnp.array(0.0)),
                "truck": per_agent_workspace_eff_mean.get(1, jnp.array(0.0)),
                "skidsteer": per_agent_workspace_eff_mean.get(2, jnp.array(0.0)),
            },
            "do_events_mean": {
                "excavator": per_agent_do_events_mean.get(0, jnp.array(0)),
                "truck": per_agent_do_events_mean.get(1, jnp.array(0)),
                "skidsteer": per_agent_do_events_mean.get(2, jnp.array(0)),
            },
        },
        "collaboration": {
            "team_path_efficiency_mean": path_efficiency_mean,
            "mean_of_agent_path_efficiency_mean": mean_of_agent_path_eff,
            "collab_gain_path_efficiency": collab_gain_path_eff,
            "diversity_entropy_do_events": diversity_entropy,
        },
    }
    return np.cumsum(reward_seq), stats, obs_seq


def print_stats(stats):
    """Print comprehensive evaluation statistics."""
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]
    avg_steps_till_completion = stats.get("avg_steps_till_completion", None)
    per_agent = stats.get("per_agent", {})
    collaboration = stats.get("collaboration", {})
    goal_eff = stats.get("goal_efficiency", {})

    # Handle case where episode_done_once might be None or have unexpected shape
    if episode_done_once is None:
        completion_rate = 0.0
    else:
        try:
            completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)
        except (TypeError, IndexError):
            completion_rate = 100.0 if episode_done_once.sum() > 0 else 0.0

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    print(f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})")
    print(f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})")
    try:
        print(f"Coverage (total): {float(coverage['total']['mean']):.2f} ({float(coverage['total']['std']):.2f})")
        print(f"  Dig coverage:  {float(coverage['dig']['mean']):.2f} ({float(coverage['dig']['std']):.2f})")
        print(f"  Dump coverage: {float(coverage['dump']['mean']):.2f} ({float(coverage['dump']['std']):.2f})")
    except Exception:
        pass
    if avg_steps_till_completion is not None:
        try:
            print(f"Avg steps till completion: {float(avg_steps_till_completion):.2f}")
        except Exception:
            pass
    if per_agent:
        mm = per_agent.get("move_m_mean", {})
        pe = per_agent.get("path_efficiency_mean", {})
        we = per_agent.get("workspace_efficiency_mean", {})
        de = per_agent.get("do_events_mean", {})
        def _fmt(x):
            try:
                return float(x)
            except Exception:
                return 0.0
        print("Per-agent (mean over completed envs):")
        print(f"  Excavator: move_m={_fmt(mm.get('excavator', 0)):.2f}, path_eff={_fmt(pe.get('excavator', 0)):.2f}, workspace_eff={_fmt(we.get('excavator', 0)):.2f}, do_events={int(_fmt(de.get('excavator', 0)))}")
        print(f"  Truck:     move_m={_fmt(mm.get('truck', 0)):.2f}, path_eff={_fmt(pe.get('truck', 0)):.2f}, workspace_eff={_fmt(we.get('truck', 0)):.2f}, do_events={int(_fmt(de.get('truck', 0)))}")
        print(f"  Skidsteer: move_m={_fmt(mm.get('skidsteer', 0)):.2f}, path_eff={_fmt(pe.get('skidsteer', 0)):.2f}, workspace_eff={_fmt(we.get('skidsteer', 0)):.2f}, do_events={int(_fmt(de.get('skidsteer', 0)))}")
    if collaboration:
        try:
            tpe = float(collaboration.get("team_path_efficiency_mean", 0.0))
            mpe = float(collaboration.get("mean_of_agent_path_efficiency_mean", 0.0))
            gain = float(collaboration.get("collab_gain_path_efficiency", 0.0))
            div = float(collaboration.get("diversity_entropy_do_events", 0.0))
            print("Collaboration:")
            print(f"  Team path_eff mean: {tpe:.3f}")
            print(f"  Mean of agent path_eff means: {mpe:.3f}")
            print(f"  Collab gain (team - mean_of_agents): {gain:.3f}")
            print(f"  Do-event diversity (entropy): {div:.3f}")
        except Exception:
            pass
    if goal_eff:
        try:
            gs = float(goal_eff.get("goal_steps_mean", 0.0))
            gsst = float(goal_eff.get("goal_steps_std", 0.0))
            ge = float(goal_eff.get("goal_efficiency_mean", 0.0))
            gprm = float(goal_eff.get("goal_progress_rate_mean", 0.0))
            gprs = float(goal_eff.get("goal_progress_rate_std", 0.0))
            psg = float(goal_eff.get("parallel_steps_to_goal_mean", 0.0))
            print("Goal efficiency:")
            print(f"  Steps to goal (completed envs): mean={gs:.2f} std={gsst:.2f}")
            print(f"  Goal efficiency (1/steps): mean={ge:.4f}")
            print(f"  Progress rate (coverage/steps): mean={gprm:.4f} std={gprs:.4f}")
            print(f"  Parallel steps to goal (mean): {psg:.2f}")
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="mixed_agents_checkpoint.pkl",
        help="Path to mixed agent trained checkpoint.",
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
        default=100,
        help="Number of environments.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=800,
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
    parser.add_argument(
        "--map_name",
        type=str,
        default=None,
        help=(
            "If set, evaluate only on the single map `inference/maps/<map_name>`. "
            "All --n_envs rollouts use this map with different seeds."
        ),
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help=(
            "Explicit path to a single-map folder (overrides --map_name). "
            "Same convention as inference_single_map.py."
        ),
    )
    args, _ = parser.parse_known_args()

    single_map_path = None
    if args.map_path:
        single_map_path = args.map_path
    elif args.map_name:
        single_map_path = str(_DEFAULT_MAPS_DIR / args.map_name)
    if single_map_path is not None:
        if not Path(single_map_path).exists():
            raise FileNotFoundError(f"--map_path/--map_name resolves to missing folder: {single_map_path}")
        print(f"Single-map eval: evaluating {args.n_envs} rollouts on '{single_map_path}'")
    n_envs = args.n_envs

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    env_cfgs = log["env_config"]
    
    # Debug: print configuration
    print("=== Eval configuration (from checkpoint) ===")
    agent_types_ckpt = getattr(env_cfgs, "agent_types", None)
    action_types_ckpt = getattr(env_cfgs, "action_types", None)
    print(f"agent_types: {agent_types_ckpt}")
    print(f"action_types: {action_types_ckpt}")
    
    # Custom handling for different field types (same as visualize_mixed.py)
    def replicate_field(x):
        if x is None:
            return None
        # Handle tuples generically (e.g., agent_types of length 1–4)
        if isinstance(x, tuple):
            return jnp.array(x)[None, ...].repeat(n_envs, 0)
        # Handle scalars (int, float, bool) - just replicate the value
        elif isinstance(x, (int, float, bool)):
            return jnp.array([x] * n_envs)
        # Handle arrays - take first element and replicate
        else:
            return x[0][None, ...].repeat(n_envs, 0)
    
    env_cfgs = jax.tree_map(replicate_field, env_cfgs)
    
    # Choose env batch action type from checkpoint action_types
    action_types_ckpt = getattr(env_cfgs, "action_types", 0)
    def _normalize_types_list(x):
        try:
            if isinstance(x, tuple):
                return [int(v) for v in x]
            if hasattr(x, "ndim"):
                import numpy as _np
                if getattr(x, "ndim", 0) == 0:
                    return [int(_np.array(x).item())]
                if x.ndim >= 2:
                    x = x[0]
                return [int(v) for v in _np.array(x).tolist()]
            if isinstance(x, (int,)):
                return [int(x)]
            if isinstance(x, (list, tuple)):
                return [int(v) for v in x]
        except Exception:
            pass
        return [0]
    action_types_list = _normalize_types_list(action_types_ckpt)
    all_wheeled = len(action_types_list) > 0 and all(t == 1 for t in action_types_list)
    batch_cfg = BatchConfig(action_type=WheeledAction if all_wheeled else TrackedAction)
    print(f"selected batch action_type: {'WheeledAction' if batch_cfg.action_type is WheeledAction else 'TrackedAction'}")
    
    shuffle_maps = single_map_path is None
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=False,
        shuffle_maps=shuffle_maps,
        single_map_path=single_map_path,
    )
    config.num_embeddings_agent_min = 60

    model = load_neural_network(config, env)
    model_params = log["model"]
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    # Run evaluation
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
