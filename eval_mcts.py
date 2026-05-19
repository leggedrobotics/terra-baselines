"""
Evaluation script with optional MCTS (gumbel_muzero_policy) planning on top
of a trained PPO policy. Mirrors eval_mixed.py behavior for multi-agent
evaluation and single-map support, and adds MCTS as an OPT-IN add-on.

By default (no --use-mcts) this behaves like eval_mixed.py (PPO greedy or
stochastic). Pass --use-mcts to plan actions with MCTS at each step.

Ported from:
  - TerraSingleAgentOfficial/terra-baselines/eval.py  (MCTS step fn)
  - TerraProject/terra-baselines/eval_mixed.py        (multi-agent + single map)
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from utils.utils_ppo import action_type_from_policy_action, obs_to_model_input, policy, wrap_action
from terra.env import TerraEnvBatch
from terra.config import BatchConfig
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)

from train import TrainConfig  # needed for unpickling checkpoints
from train_mixed import MixedAgentTrainConfig
sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig

try:
    import mctx
except ImportError as e:
    mctx = None
    _MCTX_IMPORT_ERR = e

_DEFAULT_MAPS_DIR = Path(__file__).parent / "inference" / "maps"


# ----------------------------------------------------------------------------
# MCTS helpers (adapted from TerraSingleAgentOfficial/terra-baselines/eval.py)
# ----------------------------------------------------------------------------

def fix_env_cfg_dtypes(env_cfgs):
    """Fix the dtypes in env_cfgs to prevent JAX type promotion issues during MCTS.

    Best-effort: only applies fixes on fields that exist (multi-agent configs may
    differ from single-agent ones).
    """
    try:
        agent = env_cfgs.agent
        kwargs = {}
        for name in (
            "angles_base",
            "angles_cabin",
            "max_wheel_angle",
            "move_tiles",
            "dig_depth",
            "height",
            "width",
        ):
            if hasattr(agent, name):
                kwargs[name] = jnp.int8(getattr(agent, name))
        if kwargs:
            fixed_agent = agent._replace(**kwargs)
            env_cfgs = env_cfgs._replace(agent=fixed_agent)
    except Exception as e:
        print(f"[eval_mcts] fix_env_cfg_dtypes: skipping ({e})")
    return env_cfgs


def make_mcts_step_fn(model, env, config):
    """Create a JIT-compiled MCTS step function using gumbel_muzero_policy."""
    if mctx is None:
        raise ImportError(
            f"mctx is required for MCTS eval but is not installed: {_MCTX_IMPORT_ERR}"
        )

    num_envs = config.num_test_rollouts

    def apply_model(params, inp):
        val, policy_out = model.apply(params, inp)
        logits_pi = policy_out[0] if isinstance(policy_out, tuple) else policy_out
        if logits_pi.shape[-1] > 8:
            logits_pi = logits_pi.at[..., 8].set(-1e9)
        if logits_pi.shape[-1] > 9:
            logits_pi = logits_pi.at[..., 9].set(-1e9)
        return val, logits_pi

    def recurrent_fn(params, rng, actions, embedding):
        timestep, prev_actions = embedding
        rng, rng_env = jrandom.split(rng)
        rng_envs = jrandom.split(rng_env, num_envs)

        actions = actions.astype(jnp.int32)
        terra_actions = wrap_action(actions, env.batch_cfg.action_type)
        next_timestep = env.step(timestep, terra_actions, rng_envs)
        next_obs = next_timestep.observation

        # Strip reward_components from info so the embedding pytree structure
        # stays consistent with the root (env.reset returns no reward_components,
        # but env.step does).  Dict key filtering is safe at JAX trace time.
        if isinstance(next_timestep.info, dict) and "reward_components" in next_timestep.info:
            next_timestep = next_timestep._replace(
                info={k: v for k, v in next_timestep.info.items() if k != "reward_components"}
            )

        next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        next_prev_actions = next_prev_actions.at[:, 0].set(actions)

        inp = obs_to_model_input(next_obs, next_prev_actions, config)
        value, logits_pi = apply_model(params, inp)

        reward = next_timestep.reward
        done = next_timestep.done
        discount = (1.0 - done) * config.gamma

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits_pi,
            value=value[:, 0],
        ), (next_timestep, next_prev_actions)

    @jax.jit
    def mcts_step(params, rng, timestep, prev_actions):
        obs = timestep.observation
        inp = obs_to_model_input(obs, prev_actions, config)
        value, logits_pi = apply_model(params, inp)

        ppo_action = jnp.argmax(logits_pi, axis=-1)

        # Normalize root timestep's info to match what recurrent_fn returns
        # (strip reward_components so the embedding pytree structure is stable
        # across root and all expansions).
        root_timestep = timestep
        if isinstance(root_timestep.info, dict) and "reward_components" in root_timestep.info:
            root_timestep = root_timestep._replace(
                info={k: v for k, v in root_timestep.info.items() if k != "reward_components"}
            )

        root = mctx.RootFnOutput(
            prior_logits=logits_pi,
            value=value[:, 0],
            embedding=(root_timestep, prev_actions),
        )

        rng, rng_mcts = jrandom.split(rng)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_mcts,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
        )

        mcts_action = policy_output.action.astype(jnp.int32)
        actions = mcts_action

        rng, rng_step = jrandom.split(rng)
        rng_steps = jrandom.split(rng_step, num_envs)
        action_type = env.batch_cfg.action_type
        next_timestep = env.step(timestep, wrap_action(actions, action_type), rng_steps)

        next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        next_prev_actions = next_prev_actions.at[:, 0].set(actions)

        return rng, next_timestep, next_prev_actions, actions, ppo_action, mcts_action

    return mcts_step


# ----------------------------------------------------------------------------
# Rollout (multi-agent metrics, same as eval_mixed.py) with optional MCTS
# ----------------------------------------------------------------------------

def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
    obs_log = {
        k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
    }
    return obs_log


def rollout_episode(
    env: TerraEnvBatch,
    model,
    model_params,
    env_cfgs,
    rl_config,
    max_frames,
    deterministic,
    seed,
    use_mcts=False,
):
    mode_str = "MCTS" if use_mcts else ("PPO (greedy)" if deterministic else "PPO (stochastic)")
    print(f"[eval_mcts] mode: {mode_str}, seed={seed}")

    rng = jrandom.PRNGKey(seed)
    rng, _rng = jrandom.split(rng)
    rng_reset = jrandom.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)

    # Keep pytree structure stable for PPO path (eval_mixed.py parity).
    # MCTS path must NOT inject reward_components here: the root embedding is
    # derived from env.reset (no reward_components), and recurrent_fn strips it
    # from env.step results, so both sides of the tree stay consistent.
    if not use_mcts:
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
                    "dig_completion_edge": jnp.zeros_like(timestep.reward),
                    "dig_completion_inner": jnp.zeros_like(timestep.reward),
                    "dig_completion_total": jnp.zeros_like(timestep.reward),
                    "dig_completion_min_edge_inner": jnp.zeros_like(timestep.reward),
                    "remaining_edge_dig_tiles": jnp.zeros_like(timestep.reward),
                    "remaining_inner_dig_tiles": jnp.zeros_like(timestep.reward),
                }
                timestep = timestep._replace(
                    info={**timestep.info, "reward_components": dummy_components}
                )
        except Exception:
            pass

    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32,
    )

    tile_size = env_cfgs.tile_size[0].item()
    move_tiles = env_cfgs.agent.move_tiles[0].item()

    action_type = env.batch_cfg.action_type
    if action_type == TrackedAction:
        pass  # stats below don't rely on move_actions in mixed version
    elif action_type == WheeledAction:
        pass
    else:
        raise NotImplementedError(f"Action type {action_type} not supported for eval.")

    obs = timestep.observation
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
    ) * (tile_size ** 2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )

    # Build MCTS step function if requested
    mcts_step = None
    if use_mcts:
        mcts_step = make_mcts_step_fn(model, env, rl_config)
        print("[eval_mcts] Warming up MCTS JIT compilation...")
        t0 = time.time()
        _rng, _ts, _pa, _act, _ppo, _mc = mcts_step(model_params, rng, timestep, prev_actions)
        jax.block_until_ready(_act)
        print(f"[eval_mcts] JIT warmup complete in {time.time() - t0:.2f}s")
        # Reset for actual run
        rng = jrandom.PRNGKey(seed)
        rng, _rng = jrandom.split(rng)
        rng_reset = jrandom.split(_rng, rl_config.num_test_rollouts)
        timestep = env.reset(env_cfgs, rng_reset)
        prev_actions = jnp.zeros(
            (rl_config.num_test_rollouts, rl_config.num_prev_actions),
            dtype=jnp.int32,
        )
        obs = timestep.observation

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    obs_seq = {}

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
    last_pos_per_agent = {0: None, 1: None, 2: None}
    prev_action_map = None
    AGENT_TYPE_IDX = 6
    EXCAVATOR_TYPE = 0

    mcts_ppo_diff_count = 0

    start_time = time.time()
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)

        if use_mcts:
            rng, timestep, prev_actions, action, ppo_act, mcts_act = mcts_step(
                model_params, rng, timestep, prev_actions
            )
            mcts_ppo_diff_count += int((np.array(ppo_act) != np.array(mcts_act)).sum())
        else:
            rng, rng_act, rng_step = jrandom.split(rng, 3)
            obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, pi = policy(model.apply, model_params, obs_model)
            if deterministic:
                action = pi.mode()
            else:
                action = pi.sample(seed=rng_act)
            action_type_sample = action_type_from_policy_action(action)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action_type_sample)
            rng_step = jrandom.split(rng_step, rl_config.num_test_rollouts)
            timestep = env.step(
                timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
            )

        reward = timestep.reward
        next_obs = timestep.observation
        done = timestep.info["task_done"]

        # Per-agent accumulation (eval_mixed.py parity)
        try:
            agent_states_batch = obs["agent_states"]
            active_pos_batch = agent_states_batch[:, 0, 0:2]
            active_type_batch = agent_states_batch[:, 0, AGENT_TYPE_IDX].astype(jnp.int32)

            if episode_done_once is None:
                active_env_mask = jnp.ones(active_type_batch.shape[0], dtype=jnp.bool_)
            else:
                active_env_mask = ~episode_done_once
            active_env_f = active_env_mask.astype(jnp.float32)
            active_env_i = active_env_mask.astype(jnp.int32)

            for atype in (0, 1, 2):
                mask = (active_type_batch == atype)
                if last_pos_per_agent[atype] is None:
                    last_pos_per_agent[atype] = active_pos_batch
                else:
                    delta = jnp.linalg.norm(active_pos_batch - last_pos_per_agent[atype], axis=1)
                    per_agent_move_m[atype] = (
                        per_agent_move_m[atype]
                        + delta * tile_size * mask.astype(jnp.float32) * active_env_f
                    )
                    update_pos_mask = mask & active_env_mask
                    last_pos_per_agent[atype] = jnp.where(
                        update_pos_mask[:, None], active_pos_batch, last_pos_per_agent[atype]
                    )

            if prev_action_map is not None:
                changed = (obs["action_map"] != prev_action_map)
                any_change = changed.reshape((changed.shape[0], -1)).any(axis=1).astype(jnp.int32)
                any_change = any_change * active_env_i
                for atype in (0, 1, 2):
                    mask = (active_type_batch == atype).astype(jnp.int32)
                    per_agent_do_events[atype] = per_agent_do_events[atype] + any_change * mask
            prev_action_map = obs["action_map"].copy()
        except Exception:
            pass

        reward_seq.append(reward)
        t_counter += 1
        if episode_done_once is None:
            episode_done_once = done
        if episode_length is None:
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)
        episode_done_once = episode_done_once | done
        episode_length += ~episode_done_once

        if t_counter % 25 == 0:
            elapsed = time.time() - start_time
            sps = t_counter / max(elapsed, 1e-6)
            print(
                f"[eval_mcts] step {t_counter}/{max_frames} | {sps:.1f} sps | "
                f"done {int(episode_done_once.sum())}/{rl_config.num_test_rollouts}"
            )

        if jnp.all(done).item() or t_counter == max_frames:
            print(f"[eval_mcts] episode ended: all_done={bool(jnp.all(done).item())}, t={t_counter}")
            break

        obs = next_obs

    if use_mcts:
        total = t_counter * rl_config.num_test_rollouts
        pct = 100.0 * mcts_ppo_diff_count / max(total, 1)
        print(f"[eval_mcts] MCTS != PPO: {mcts_ppo_diff_count}/{total} ({pct:.1f}%)")

    if episode_done_once is None:
        return np.array([]), {}, obs_seq

    # ---- Metrics (same as eval_mixed.py) ----
    team_move_m = per_agent_move_m[0] + per_agent_move_m.get(1, 0) + per_agent_move_m.get(2, 0)
    team_path_efficiency = (team_move_m / jnp.sqrt(areas))
    path_efficiency = team_path_efficiency[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    reference_workspace_area = 0.5 * np.pi * (8 ** 2)
    excavator_ops = per_agent_do_events[0] // 2
    skidsteer_ops = per_agent_do_events.get(2, 0) // 2
    truck_ops = per_agent_do_events.get(1, 0)
    team_workspace_ops = excavator_ops + skidsteer_ops + truck_ops
    workspaces_efficiency = (reference_workspace_area * (team_workspace_ops / areas))[episode_done_once]
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    reduce_axes = tuple([i for i in range(len(obs["action_map"].shape))][1:])
    dig_req_mask = (target_maps_init < 0)
    undug_mask = dig_req_mask & (obs["action_map"] >= 0)
    undug_count = undug_mask.sum(reduce_axes)
    total_dig_req = dig_req_mask.sum(reduce_axes)
    dig_coverage = jnp.where(total_dig_req > 0, 1.0 - (undug_count / jnp.maximum(total_dig_req, 1)), 1.0)
    dumped_vals = jnp.where(obs["action_map"] > 0, obs["action_map"], 0)
    dump_correct = jnp.where(target_maps_init > 0, dumped_vals, 0).sum(reduce_axes)
    dumped_total = dumped_vals.sum(reduce_axes)
    dump_coverage = jnp.where(dumped_total > 0, dump_correct / jnp.maximum(dumped_total, 1), 1.0)
    try:
        loaded_feat_idx = 5
        loaded_amount = jnp.maximum(obs["agent_states"][:, :, loaded_feat_idx], 0)
        loaded_sum = loaded_amount.sum(axis=1)
    except Exception:
        loaded_sum = jnp.zeros_like(dumped_total)
    denom_total_dirt = dumped_total + undug_count + loaded_sum
    total_completion = jnp.where(denom_total_dirt > 0, dump_correct / jnp.maximum(denom_total_dirt, 1), 0.0)
    coverage_scores = episode_done_once + (~episode_done_once) * total_completion
    dig_cov_scores = episode_done_once + (~episode_done_once) * dig_coverage
    dump_cov_scores = episode_done_once + (~episode_done_once) * dump_coverage
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    avg_steps_till_completion = (
        episode_length[episode_done_once].mean() if episode_length is not None else jnp.array(0)
    )
    try:
        num_agents_arr = timestep.observation["num_agents"].astype(jnp.float32)
        avg_num_agents_completed = (
            num_agents_arr[episode_done_once].mean() if episode_done_once.any() else jnp.array(1.0)
        )
    except Exception:
        avg_num_agents_completed = jnp.array(1.0)
    parallel_steps_to_goal_mean = (
        float(avg_steps_till_completion) / max(float(avg_num_agents_completed), 1.0)
        if episode_length is not None
        else 0.0
    )

    if episode_length is not None:
        completed_steps = episode_length[episode_done_once]
        goal_steps_mean = completed_steps.mean() if completed_steps.size > 0 else jnp.array(0)
        goal_steps_std = completed_steps.std() if completed_steps.size > 0 else jnp.array(0)
        goal_efficiency_mean = (
            (1.0 / jnp.maximum(goal_steps_mean, 1)) if completed_steps.size > 0 else jnp.array(0.0)
        )
        progress_rate = coverage_scores / jnp.maximum(episode_length, 1)
        goal_progress_rate_mean = progress_rate.mean()
        goal_progress_rate_std = progress_rate.std()
    else:
        goal_steps_mean = jnp.array(0)
        goal_steps_std = jnp.array(0)
        goal_efficiency_mean = jnp.array(0.0)
        goal_progress_rate_mean = jnp.array(0.0)
        goal_progress_rate_std = jnp.array(0.0)

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

    sqrt_areas = jnp.sqrt(areas)
    per_agent_path_eff_mean = {}
    for k, v in per_agent_move_m.items():
        try:
            path_eff = jnp.where(episode_done_once, v / sqrt_areas, 0)
            per_agent_path_eff_mean[k] = path_eff.sum() / jnp.maximum(episode_done_once.sum(), 1)
        except Exception:
            per_agent_path_eff_mean[k] = jnp.array(0.0)

    per_agent_workspace_eff_mean = {}
    for k, v in per_agent_do_events.items():
        try:
            n_ops = v if k == 1 else v // 2
            ws_eff = jnp.where(episode_done_once, reference_workspace_area * (n_ops / areas), 0)
            per_agent_workspace_eff_mean[k] = ws_eff.sum() / jnp.maximum(episode_done_once.sum(), 1)
        except Exception:
            per_agent_workspace_eff_mean[k] = jnp.array(0.0)

    try:
        mean_of_agent_path_eff = (
            per_agent_path_eff_mean.get(0, 0.0)
            + per_agent_path_eff_mean.get(1, 0.0)
            + per_agent_path_eff_mean.get(2, 0.0)
        ) / 3.0
    except Exception:
        mean_of_agent_path_eff = jnp.array(0.0)
    collab_gain_path_eff = path_efficiency_mean - mean_of_agent_path_eff

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
        "path_efficiency": {"mean": path_efficiency_mean, "std": path_efficiency_std},
        "workspaces_efficiency": {
            "mean": workspaces_efficiency_mean,
            "std": workspaces_efficiency_std,
        },
        "coverage": {
            "total": {"mean": coverage_score_mean, "std": coverage_score_std},
            "dig": {"mean": dig_cov_scores.mean(), "std": dig_cov_scores.std()},
            "dump": {"mean": dump_cov_scores.mean(), "std": dump_cov_scores.std()},
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
        "mcts": {
            "used": bool(use_mcts),
            "mcts_ppo_disagreements": int(mcts_ppo_diff_count),
        },
    }
    return np.cumsum(reward_seq), stats, obs_seq


def print_stats(stats):
    episode_done_once = stats["episode_done_once"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]
    avg_steps_till_completion = stats.get("avg_steps_till_completion", None)
    per_agent = stats.get("per_agent", {})
    collaboration = stats.get("collaboration", {})
    goal_eff = stats.get("goal_efficiency", {})
    mcts_info = stats.get("mcts", {})

    if episode_done_once is None:
        completion_rate = 0.0
    else:
        try:
            completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)
        except (TypeError, IndexError):
            completion_rate = 100.0 if episode_done_once.sum() > 0 else 0.0

    print("\nStats:\n")
    if mcts_info:
        print(f"MCTS used: {mcts_info.get('used', False)}")
        if mcts_info.get("used"):
            print(f"  MCTS != PPO disagreements: {mcts_info.get('mcts_ppo_disagreements', 0)}")
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
    parser.add_argument("-run", "--run_name", type=str, default="mixed_agents_checkpoint.pkl",
                        help="Path to mixed agent trained checkpoint.")
    parser.add_argument("-env", "--env_name", type=str, default="Terra", help="Environment name.")
    parser.add_argument("-n", "--n_envs", type=int, default=100, help="Number of environments.")
    parser.add_argument("-steps", "--n_steps", type=int, default=800, help="Number of steps.")
    parser.add_argument("-d", "--deterministic", type=int, default=0,
                        help="Deterministic PPO (0 stochastic, 1 greedy). Ignored when --use-mcts is set.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--map_name", type=str, default=None,
                        help="If set, evaluate only on `inference/maps/<map_name>`.")
    parser.add_argument("--map_path", type=str, default=None,
                        help="Explicit path to a single-map folder (overrides --map_name).")
    # --- MCTS add-on flags (OFF by default) ---
    parser.add_argument("--use-mcts", dest="use_mcts", action="store_true",
                        help="Enable MCTS (gumbel_muzero_policy) planning at each step.")
    parser.add_argument("-sim", "--num_simulations", type=int, default=32,
                        help="Number of MCTS simulations per step (only used with --use-mcts).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor for MCTS recurrent_fn (default: config.gamma or 0.99).")
    args, _ = parser.parse_known_args()

    # Single-map resolution (eval_mixed.py parity)
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
    # MCTS params (only used when --use-mcts)
    config.num_simulations = args.num_simulations
    if args.gamma is not None:
        config.gamma = args.gamma
    if not hasattr(config, "gamma"):
        config.gamma = 0.99

    env_cfgs = log["env_config"]

    print("=== Eval configuration (from checkpoint) ===")
    agent_types_ckpt = getattr(env_cfgs, "agent_types", None)
    action_types_ckpt = getattr(env_cfgs, "action_types", None)
    print(f"agent_types: {agent_types_ckpt}")
    print(f"action_types: {action_types_ckpt}")

    def replicate_field(x):
        if x is None:
            return None
        if isinstance(x, tuple):
            return jnp.array(x)[None, ...].repeat(n_envs, 0)
        elif isinstance(x, (int, float, bool)):
            return jnp.array([x] * n_envs)
        else:
            return x[0][None, ...].repeat(n_envs, 0)

    env_cfgs = jax.tree_map(replicate_field, env_cfgs)

    # Fix dtypes only when MCTS is enabled (avoids needless changes in default path)
    if args.use_mcts:
        env_cfgs = fix_env_cfg_dtypes(env_cfgs)

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

    if args.use_mcts:
        print(f"\n[eval_mcts] Mode: MCTS (num_simulations={config.num_simulations}, gamma={config.gamma})")
    else:
        print(f"\n[eval_mcts] Mode: PPO, deterministic={deterministic}")

    cum_rewards, stats, _ = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        deterministic=deterministic,
        seed=args.seed,
        use_mcts=args.use_mcts,
    )
    print_stats(stats)
