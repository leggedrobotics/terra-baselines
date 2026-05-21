#!/usr/bin/env python3
"""Fast local rollout probe for downloaded Terra checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.benchmark_checkpoint_masked import (  # noqa: E402
    _build_env_config,
    _terrain_scores,
    _unbatch_env_config,
)
from terra.env import TerraEnvBatch  # noqa: E402
from train import TrainConfig  # noqa: F401,E402 - needed for unpickling old checkpoints
from train_mixed import MixedAgentTrainConfig  # noqa: E402
from utils.action_masking import apply_action_mask  # noqa: E402
from utils.helpers import load_pkl_object  # noqa: E402
from utils.models import load_neural_network, restore_checkpoint_model_config  # noqa: E402
from utils.utils_ppo import obs_to_model_input, wrap_action  # noqa: E402

sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

ACTION_NAMES = (
    "FORWARD",
    "BACKWARD",
    "CLOCK",
    "ANTICLOCK",
    "CABIN_CLOCK",
    "CABIN_ANTICLOCK",
    "DO",
    "DO_NOTHING",
)
AGENT_STATE_LOADED_INDEX = 5


def _failure_summary(final_action_map, target_map, agent_states, agent_active, task_done):
    reduce_axes = tuple(range(1, final_action_map.ndim))
    dig_req = target_map < 0
    total_dig_req = np.maximum(np.asarray(dig_req.sum(reduce_axes)), 1)
    undug_tiles = np.asarray((dig_req & (final_action_map >= 0)).sum(reduce_axes))
    undug_fraction = undug_tiles / total_dig_req

    dirt = np.clip(np.asarray(final_action_map), 0, None)
    dump_req = np.asarray(target_map > 0)
    dirt_total = dirt.sum(reduce_axes)
    dirt_on_dump = np.where(dump_req, dirt, 0).sum(reduce_axes)
    dirt_off_dump = dirt_total - dirt_on_dump
    dirt_off_dump_fraction = dirt_off_dump / np.maximum(dirt_total, 1)

    loaded = np.asarray(agent_states[:, :, AGENT_STATE_LOADED_INDEX]) * np.asarray(agent_active)
    loaded_total = loaded.sum(axis=1)
    failed = ~np.asarray(task_done, dtype=bool)
    positive_dirt = dirt_total > 0
    failed_positive_dirt = failed & positive_dirt

    def failure_mean(values):
        if not failed.any():
            return None
        return float(np.asarray(values)[failed].mean())

    def failed_positive_dirt_mean(values):
        if not failed_positive_dirt.any():
            return None
        return float(np.asarray(values)[failed_positive_dirt].mean())

    return {
        "final_undug_tiles_mean": float(undug_tiles.mean()),
        "final_undug_fraction_mean": float(undug_fraction.mean()),
        "final_dirt_total_mean": float(dirt_total.mean()),
        "final_dirt_on_dump_mean": float(dirt_on_dump.mean()),
        "final_dirt_off_dump_mean": float(dirt_off_dump.mean()),
        "final_dirt_off_dump_fraction_mean": float(dirt_off_dump_fraction.mean()),
        "final_positive_dirt_env_count": int(positive_dirt.sum()),
        "final_loaded_total_mean": float(loaded_total.mean()),
        "final_loaded_env_count": int((loaded_total > 0).sum()),
        "failure_undug_fraction_mean": failure_mean(undug_fraction),
        "failure_dirt_off_dump_fraction_mean": failure_mean(dirt_off_dump_fraction),
        "failure_dirt_off_dump_fraction_positive_mean": failed_positive_dirt_mean(
            dirt_off_dump_fraction
        ),
        "failure_positive_dirt_env_count": int(failed_positive_dirt.sum()),
        "failure_loaded_env_count": int(((loaded_total > 0) & failed).sum()),
        "failure_mostly_dug_count": int(((undug_fraction <= 0.2) & failed).sum()),
        "failure_moved_dirt_mostly_on_dump_count": int(
            ((dirt_off_dump_fraction <= 0.2) & failed_positive_dirt).sum()
        ),
        "failure_near_complete_count": int(
            (
                (undug_fraction <= 0.2)
                & (dirt_off_dump_fraction <= 0.2)
                & failed_positive_dirt
            ).sum()
        ),
    }


def _safe_mean(values, mask):
    count = jnp.maximum(mask.astype(jnp.float32).sum(), 1.0)
    return (values * mask.astype(jnp.float32)).sum() / count


def run_rollout(
    env,
    env_cfgs,
    model,
    params,
    config,
    num_envs,
    num_steps,
    seed,
    use_mask,
    stochastic,
):
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    timestep = env.reset(env_cfgs, jax.random.split(reset_key, num_envs))
    target_maps_init = timestep.observation["target_map"].copy()
    prev_actions = jnp.zeros((num_envs, config.num_prev_actions), dtype=jnp.int32)

    def step(carry, _):
        (
            timestep,
            prev_actions,
            rng,
            done_once,
            task_done_once,
            terrain_changed_once,
            final_action_map,
            final_agent_states,
            final_agent_active,
            returns,
            lengths,
            first_task_done_step,
            first_terrain_change_step,
            step_idx,
            invalid_selected,
            action_counts,
            do_valid_count,
            do_selected_count,
            do_selected_when_valid_count,
            terrain_change_count,
            entropy_sum,
            max_prob_sum,
            alive_count_sum,
        ) = carry

        rng, rng_env, rng_action = jax.random.split(rng, 3)
        alive = ~done_once
        obs_model = obs_to_model_input(timestep.observation, prev_actions, config)
        _, logits = model.apply(params, obs_model)
        action_mask = obs_model[22]
        logits_for_action = apply_action_mask(logits, action_mask) if use_mask else logits
        probs = jax.nn.softmax(logits_for_action, axis=-1)
        if stochastic:
            action = jax.random.categorical(rng_action, logits_for_action, axis=-1)
        else:
            action = jnp.argmax(logits_for_action, axis=-1)
        action = action.astype(jnp.int32)
        selected_allowed = jnp.take_along_axis(action_mask, action[:, None], axis=-1)[:, 0]

        before_action_map = timestep.observation["action_map"]
        next_timestep = env.step(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            jax.random.split(rng_env, num_envs),
        )
        next_done = next_timestep.done.astype(jnp.bool_)
        action_map_after = jnp.where(
            next_done[:, None, None],
            next_timestep.info["final_observation"]["action_map"],
            next_timestep.observation["action_map"],
        )
        map_delta = action_map_after - before_action_map
        terrain_changed = (
            jnp.sum(jnp.abs(map_delta), axis=tuple(range(1, map_delta.ndim))) > 0
        ) & alive

        task_done = next_timestep.info["task_done"].astype(jnp.bool_) & alive
        newly_done = next_done & alive
        newly_task_done = task_done & ~task_done_once
        newly_terrain_changed = terrain_changed & ~terrain_changed_once
        action_is_do = action == 6
        do_valid = action_mask[:, 6]
        alive_f = alive.astype(jnp.float32)

        returns = returns + next_timestep.reward * alive_f
        lengths = lengths + alive.astype(jnp.int32)
        final_action_map = jnp.where(
            newly_done[:, None, None],
            next_timestep.info["final_observation"]["action_map"],
            final_action_map,
        )
        final_agent_states = jnp.where(
            newly_done[:, None, None],
            next_timestep.info["final_observation"]["agent_states"],
            final_agent_states,
        )
        final_agent_active = jnp.where(
            newly_done[:, None],
            next_timestep.info["final_observation"]["agent_active"],
            final_agent_active,
        )
        first_task_done_step = jnp.where(newly_task_done, step_idx + 1, first_task_done_step)
        first_terrain_change_step = jnp.where(
            newly_terrain_changed,
            step_idx + 1,
            first_terrain_change_step,
        )
        done_once = done_once | next_timestep.done.astype(jnp.bool_)
        task_done_once = task_done_once | task_done
        terrain_changed_once = terrain_changed_once | terrain_changed
        prev_actions = jnp.concatenate([action[:, None], prev_actions[:, :-1]], axis=-1)

        action_counts = action_counts + jnp.bincount(
            action,
            weights=alive_f,
            length=len(ACTION_NAMES),
        )
        invalid_selected = invalid_selected + jnp.sum((~selected_allowed & alive).astype(jnp.int32))
        do_valid_count = do_valid_count + jnp.sum((do_valid & alive).astype(jnp.int32))
        do_selected_count = do_selected_count + jnp.sum((action_is_do & alive).astype(jnp.int32))
        do_selected_when_valid_count = do_selected_when_valid_count + jnp.sum(
            (action_is_do & do_valid & alive).astype(jnp.int32)
        )
        terrain_change_count = terrain_change_count + jnp.sum((terrain_changed & alive).astype(jnp.int32))
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        entropy_sum = entropy_sum + jnp.sum(entropy * alive_f)
        max_prob_sum = max_prob_sum + jnp.sum(jnp.max(probs, axis=-1) * alive_f)
        alive_count_sum = alive_count_sum + jnp.sum(alive_f)

        carry = (
            next_timestep,
            prev_actions,
            rng,
            done_once,
            task_done_once,
            terrain_changed_once,
            final_action_map,
            final_agent_states,
            final_agent_active,
            returns,
            lengths,
            first_task_done_step,
            first_terrain_change_step,
            step_idx + 1,
            invalid_selected,
            action_counts,
            do_valid_count,
            do_selected_count,
            do_selected_when_valid_count,
            terrain_change_count,
            entropy_sum,
            max_prob_sum,
            alive_count_sum,
        )
        return carry, None

    init = (
        timestep,
        prev_actions,
        rng,
        jnp.zeros((num_envs,), dtype=jnp.bool_),
        jnp.zeros((num_envs,), dtype=jnp.bool_),
        jnp.zeros((num_envs,), dtype=jnp.bool_),
        timestep.observation["action_map"],
        timestep.observation["agent_states"],
        timestep.observation["agent_active"],
        jnp.zeros((num_envs,), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.int32),
        jnp.full((num_envs,), -1, dtype=jnp.int32),
        jnp.full((num_envs,), -1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((len(ACTION_NAMES),), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    (
        timestep,
        _,
        _,
        done_once,
        task_done_once,
        terrain_changed_once,
        final_action_map,
        final_agent_states,
        final_agent_active,
        returns,
        lengths,
        first_task_done_step,
        first_terrain_change_step,
        _,
        invalid_selected,
        action_counts,
        do_valid_count,
        do_selected_count,
        do_selected_when_valid_count,
        terrain_change_count,
        entropy_sum,
        max_prob_sum,
        alive_count_sum,
    ), _ = jax.jit(lambda state: jax.lax.scan(step, state, None, length=num_steps))(init)

    action_counts_np = np.asarray(action_counts)
    total_actions = max(1.0, float(action_counts_np.sum()))
    task_done_np = np.asarray(task_done_once)
    terrain_changed_np = np.asarray(terrain_changed_once)
    result = {
        "mode": "masked" if use_mask else "unmasked",
        "action_selection": "stochastic" if stochastic else "argmax",
        "seed": seed,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "success_count": int(task_done_np.sum()),
        "success_rate": float(task_done_np.mean()),
        "done_count": int(np.asarray(done_once).sum()),
        "done_rate": float(np.asarray(done_once).mean()),
        "terrain_changed_count": int(terrain_changed_np.sum()),
        "terrain_changed_rate": float(terrain_changed_np.mean()),
        "avg_return": float(np.asarray(returns.mean())),
        "max_return": float(np.asarray(returns.max())),
        "avg_length": float(np.asarray(lengths.mean())),
        "avg_success_step": float(
            np.asarray(
                _safe_mean(
                    first_task_done_step.astype(jnp.float32),
                    first_task_done_step >= 0,
                )
            )
        )
        if task_done_np.any()
        else None,
        "avg_first_terrain_change_step": float(
            np.asarray(
                _safe_mean(
                    first_terrain_change_step.astype(jnp.float32),
                    first_terrain_change_step >= 0,
                )
            )
        )
        if terrain_changed_np.any()
        else None,
        "invalid_selected": int(np.asarray(invalid_selected)),
        "invalid_selected_rate": float(np.asarray(invalid_selected)) / total_actions,
        "do_valid_rate": float(np.asarray(do_valid_count)) / total_actions,
        "do_selected_rate": float(np.asarray(do_selected_count)) / total_actions,
        "do_selected_when_valid_rate": float(np.asarray(do_selected_when_valid_count)) / total_actions,
        "terrain_change_per_action_rate": float(np.asarray(terrain_change_count)) / total_actions,
        "mean_policy_entropy": float(np.asarray(entropy_sum / jnp.maximum(alive_count_sum, 1.0))),
        "mean_max_prob": float(np.asarray(max_prob_sum / jnp.maximum(alive_count_sum, 1.0))),
        "action_counts": dict(zip(ACTION_NAMES, action_counts_np.astype(int).tolist())),
        "action_percent": dict(zip(ACTION_NAMES, (action_counts_np / total_actions).tolist())),
    }
    final_action_map = jnp.where(
        done_once[:, None, None],
        final_action_map,
        timestep.observation["action_map"],
    )
    final_agent_states = jnp.where(
        done_once[:, None, None],
        final_agent_states,
        timestep.observation["agent_states"],
    )
    final_agent_active = jnp.where(
        done_once[:, None],
        final_agent_active,
        timestep.observation["agent_active"],
    )
    result.update(_terrain_scores({"action_map": final_action_map}, target_maps_init))
    result.update(
        _failure_summary(
            np.asarray(final_action_map),
            np.asarray(target_maps_init),
            np.asarray(final_agent_states),
            np.asarray(final_agent_active),
            task_done_np,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", default="solo_excavator")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=550)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", choices=("unmasked", "masked"), default=["unmasked"])
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    log = load_pkl_object(str(args.checkpoint))
    config = log["train_config"]
    restore_checkpoint_model_config(config, log["model"])
    config.num_test_rollouts = args.num_envs
    config.num_devices = 1
    config.benchmark_stochastic = False

    env_config = _unbatch_env_config(log["env_config"])
    batch_cfg, env_cfgs = _build_env_config(
        env_config,
        args.config,
        args.num_envs,
        enforce_foundation_border_alignment=True,
    )
    env = TerraEnvBatch(batch_cfg=batch_cfg, rendering=False, shuffle_maps=True)
    model = load_neural_network(config, env)

    results = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seeds": args.seeds,
        "rollouts": [],
    }
    for seed in args.seeds:
        for mode in args.modes:
            results["rollouts"].append(
                run_rollout(
                    env,
                    env_cfgs,
                    model,
                    log["model"],
                    config,
                    args.num_envs,
                    args.num_steps,
                    seed,
                    mode == "masked",
                    args.stochastic,
                )
            )

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
