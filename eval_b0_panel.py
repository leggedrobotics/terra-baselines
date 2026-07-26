#!/usr/bin/env python3
"""Evaluate one B0 feasibility panel on every frozen development identity."""

from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from eval_fixed_bank import (
    environment_completion_contract,
    exact_reset_keys,
    grouped_results,
    load_manifest,
    sha256_file,
    verify_exact_reset,
)
from eval_mcts import rollout_episode
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.helpers import load_pkl_object

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

HORIZON = 450
CHECKPOINT_CADENCE = 100
INITIAL_UPDATES = 500
TRANSITIONS_PER_UPDATE = 131_072
REQUIRED_CELL_SUCCESSES = 6
NUM_TRACKED_ACTIONS = 8
COMPLETION_PROGRESS = 0.01
PLATEAU_EVALUATIONS = 5
PANEL_SPECS = {
    "foundation_geometry": {
        "config_name": "b0_foundation_geometry_v1",
        "family": "foundation",
        "cells": ("f_osm_all", "f_procedural_all"),
        "seed": 2026072701,
        "train_count": 16,
        "reward_contract": "corrected_dense_v1",
    },
    "foundation_distance": {
        "config_name": "b0_foundation_distance_v1",
        "family": "foundation",
        "cells": (
            "f_apron_d02",
            "f_apron_d04",
            "f_apron_d06",
            "f_apron_d08",
        ),
        "seed": 2026072702,
        "train_count": 32,
        "reward_contract": "corrected_dense_v1",
    },
    "trench_distance": {
        "config_name": "b0_trench_distance_v1",
        "family": "trench",
        "cells": (
            "t_straight_both_d02",
            "t_straight_both_d04",
            "t_straight_both_d06",
            "t_straight_both_d08",
        ),
        "seed": 2026072703,
        "train_count": 32,
        "reward_contract": "corrected_dense_v1_trench_absolute_off",
    },
    "trench_side": {
        "config_name": "b0_trench_side_v1",
        "family": "trench",
        "cells": (
            "t_straight_both_d02",
            "t_straight_one_d02",
        ),
        "seed": 2026072704,
        "train_count": 16,
        "reward_contract": "corrected_dense_v1_trench_absolute_off",
    },
    "trench_topology": {
        "config_name": "b0_trench_topology_v1",
        "family": "trench",
        "cells": (
            "t_straight_both_d02",
            "t_segmented2_both_d02",
            "t_segmented3_both_d02",
            "t_T_both_d02",
            "t_X_both_d02",
            "t_disconnected_both_d02",
        ),
        "seed": 2026072705,
        "train_count": 48,
        "reward_contract": "corrected_dense_v1_trench_absolute_off",
    },
}


def config_value(config, name):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def panel_spec(panel: str) -> dict:
    try:
        return PANEL_SPECS[panel]
    except KeyError as error:
        raise ValueError(f"unknown B0 panel {panel}") from error


def verify_b0_checkpoint(
    checkpoint: dict,
    panel: str,
    expected_next_update: int,
    *,
    planned_updates: int = INITIAL_UPDATES,
) -> dict:
    spec = panel_spec(panel)
    if int(checkpoint["next_update"]) != expected_next_update:
        raise RuntimeError(
            "B0 checkpoint update mismatch: "
            f"{checkpoint['next_update']} != {expected_next_update}"
        )
    if "optimizer_state" not in checkpoint:
        raise RuntimeError("B0 checkpoint lacks optimizer state")
    expected = {
        "config_name": spec["config_name"],
        "seed": spec["seed"],
        "num_devices": 4,
        "num_envs_per_device": 1024,
        "num_envs": 4096,
        "num_steps": 32,
        "env_steps_per_update": TRANSITIONS_PER_UPDATE,
        "num_updates": planned_updates,
        "total_timesteps": planned_updates * TRANSITIONS_PER_UPDATE,
        "actual_total_timesteps": (planned_updates * TRANSITIONS_PER_UPDATE),
        "lr": 3e-4,
        "clip_eps": 0.2,
        "gamma": 0.9984,
        "gae_lambda": 0.95,
        "update_epochs": 2,
        "num_minibatches": 32,
        "vf_coef": 2.0,
        "max_grad_norm": 0.5,
        "log_train_interval": 1,
        "log_eval_interval": 0,
        "eval_episodes": 32,
        "checkpoint_interval": 1 if planned_updates == 1 else 100,
        "keep_checkpoint_history": True,
        "cache_clear_interval": 0 if planned_updates == 1 else 1000,
        "ent_schedule_start": 0.15,
        "ent_schedule_end": 0.005,
        "ent_schedule_steps": 950,
        "agent_types_override": (0,),
        "action_types_override": (0,),
        "dump_bonus_mult": 0.5,
        "excavator_relocate_dumped_mult": 1.5,
        "excavator_relocate_dug_dirt_mult": 1.5,
        "transport_relocate_mult": 1.5,
        "curriculum_increase_level_threshold": 20,
        "curriculum_decrease_level_threshold": 80,
        "curriculum_last_level_type": "none",
        "single_map_path": None,
        "replay_map_count": 0,
        "target_map_repeat": 0,
        "model_size": "base",
        "model_core": "mlp",
        "map_encoder": "resnet_spatial_8x8",
        "encoder_compute_dtype": "float32",
        "attention_compute_dtype": "encoder",
        "token_mixer_residual_init_scale": 0.0,
        "critic_hidden_dims": None,
        "use_value_clip": False,
        "flat_minibatch_shuffle": True,
        "fail_on_nonfinite": True,
        "finite_check_interval": 1,
        "resume_from": None,
        "warm_start_from": None,
        # The current CLI resolves the unused scratch-run flag to False.
        # It is inert while resume_from is None, but freeze the serialized
        # value so the receipt matches the checkpoint exactly.
        "load_env_from_checkpoint": False,
        "teacher_checkpoint": None,
        "teacher_obs_downsample": 1,
    }
    config = checkpoint["train_config"]
    observed = {name: config_value(config, name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            "B0 checkpoint config mismatch: "
            f"observed={observed}, expected={expected}"
        )
    levels = config_value(config, "curriculum_levels_override")
    expected_path = f"panels/train/{panel}"
    if len(levels) != 1:
        raise RuntimeError("B0 checkpoint must have exactly one map level")
    level = levels[0]
    if (
        level["maps_path"] != expected_path
        or int(level["max_steps_in_episode"]) != HORIZON
        or int(level["rewards_type"]) != 0
        or bool(level["apply_trench_rewards"])
    ):
        raise RuntimeError(f"B0 checkpoint map level mismatch: {level}")
    transition_integrity = checkpoint.get("transition_integrity")
    if transition_integrity is None or any(
        int(value) != 0 for value in transition_integrity.values()
    ):
        raise RuntimeError(
            "B0 checkpoint transition integrity failed: " f"{transition_integrity}"
        )
    return {
        "passed": True,
        "checked_config": observed,
        "transition_integrity": transition_integrity,
    }


def configure_for_panel(train_config, panel: str, count: int):
    config = copy.deepcopy(train_config)
    config.num_devices = 1
    config.num_envs_per_device = count
    config.num_envs = count
    config.num_test_rollouts = count
    config.eval_episodes = count
    config.eval_episodes_per_device = count
    config.num_minibatches = math.gcd(
        int(getattr(config, "num_minibatches", 32)),
        count,
    )
    if config.num_minibatches <= 0:
        raise RuntimeError("B0 evaluator requires a positive minibatch divisor")
    config.agent_types_override = (0,)
    config.action_types_override = (0,)
    config.curriculum_levels_override = [
        {
            "maps_path": f"panels/development/{panel}",
            "max_steps_in_episode": HORIZON,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
    ]
    config.curriculum_increase_level_threshold = 20
    config.curriculum_decrease_level_threshold = 80
    config.curriculum_last_level_type = "none"
    config.single_map_path = None
    config.replay_map_count = 0
    config.target_map_repeat = 0
    config.teacher_checkpoint = None
    config.teacher_obs_downsample = 1
    config.resume_from = None
    config.warm_start_from = None
    config.resume_update = None
    config.load_env_from_checkpoint = False
    return config


def successful_trajectories(
    rows: list[dict],
    per_map: list[dict],
    stats: dict,
) -> dict[str, dict | None]:
    actions = np.asarray(stats["action_sequence"], dtype=np.int32)
    effects = np.asarray(
        stats["action_had_effect_sequence"],
        dtype=bool,
    )
    if actions.shape != effects.shape or actions.shape[1] != len(rows):
        raise RuntimeError("B0 action traces have invalid shapes")
    if np.any(actions < 0) or np.any(actions >= NUM_TRACKED_ACTIONS):
        raise RuntimeError("B0 action trace contains an invalid action")
    result: dict[str, dict | None] = {}
    for cell in sorted({row["primary_cell"] for row in rows}):
        candidates = [
            index
            for index, row in enumerate(per_map)
            if row["primary_cell"] == cell and row["success"]
        ]
        if not candidates:
            result[cell] = None
            continue
        index = candidates[0]
        length = int(per_map[index]["steps"])
        trajectory_actions = actions[:length, index]
        trajectory_effects = effects[:length, index]
        result[cell] = {
            "map_id": rows[index]["map_id"],
            "slot_index": int(rows[index]["slot_index"]),
            "steps": length,
            "actions": trajectory_actions.tolist(),
            "action_had_effect": trajectory_effects.tolist(),
            "all_actions_in_discrete_range": bool(
                np.all(
                    (trajectory_actions >= 0)
                    & (trajectory_actions < NUM_TRACKED_ACTIONS)
                )
            ),
            "final_success": True,
        }
    return result


def summarize_checkpoint(
    rows: list[dict],
    stats: dict,
) -> dict:
    count = len(rows)
    successes = np.asarray(stats["episode_done_once"], dtype=bool)
    terminations = np.asarray(
        stats["episode_terminated_once"],
        dtype=bool,
    )
    lengths = np.asarray(stats["episode_length"], dtype=np.int32)
    terminal_completion = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in stats["terminal_completion"].items()
    }
    raw_integrity = stats["integrity"]
    if not bool(raw_integrity.get("supported", False)):
        raise RuntimeError("B0 evaluator lacks state-integrity metrics")
    integrity_metrics = {
        key: np.asarray(value)
        for key, value in raw_integrity.items()
        if key != "supported"
    }
    for name, values in {
        "successes": successes,
        "terminations": terminations,
        "lengths": lengths,
        **terminal_completion,
        **integrity_metrics,
    }.items():
        if np.asarray(values).shape != (count,):
            raise RuntimeError(
                f"B0 field {name} has shape {np.asarray(values).shape}, "
                f"expected {(count,)}"
            )
    absolute = terminal_completion["absolute"]
    if not np.array_equal(
        successes,
        np.isclose(absolute, 1.0, atol=1e-6),
    ):
        raise RuntimeError("B0 task_done does not match absolute completion")
    expected_termination = successes | (lengths >= HORIZON)
    integrity_metrics["termination_disagreement"] = terminations != expected_termination
    expected_slots = np.arange(count, dtype=np.int32)
    observed_slots = np.asarray(
        integrity_metrics["slot_index_zero_based"],
        dtype=np.int32,
    )
    integrity_metrics["slot_index_disagreement"] = observed_slots != expected_slots
    integrity_metrics["integrity_unavailable"] = np.zeros(
        count,
        dtype=bool,
    )
    per_map, grouped = grouped_results(
        rows,
        successes,
        terminations,
        lengths,
        horizon=HORIZON,
        completion_metrics={
            f"terminal_{key}": value for key, value in terminal_completion.items()
        },
        integrity_metrics=integrity_metrics,
    )
    trajectories = successful_trajectories(rows, per_map, stats)
    cells = {}
    for cell in sorted({row["primary_cell"] for row in rows}):
        selected = [row for row in per_map if row["primary_cell"] == cell]
        success_count = sum(int(row["success"]) for row in selected)
        integrity_failures = sum(int(row["integrity_failure"]) for row in selected)
        completion = np.asarray(
            [row["terminal_absolute"] for row in selected],
            dtype=np.float64,
        )
        trajectory = trajectories[cell]
        cells[cell] = {
            "successes": success_count,
            "episodes": len(selected),
            "median_terminal_absolute_completion": float(np.median(completion)),
            "integrity_failure_count": integrity_failures,
            "successful_action_trajectory": trajectory,
            "performance_passed": (success_count >= REQUIRED_CELL_SUCCESSES),
            "integrity_passed": integrity_failures == 0,
            "trajectory_saved": trajectory is not None,
            "passed": (
                success_count >= REQUIRED_CELL_SUCCESSES
                and integrity_failures == 0
                and trajectory is not None
            ),
        }
    return {
        "overall": grouped["overall"],
        "grouped": grouped,
        "cells": cells,
        "integrity_failure_count": grouped["integrity"]["failure_count"],
        "per_map": per_map,
    }


def consecutive_cell_witnesses(records: list[dict]) -> dict:
    cells = sorted(records[0]["summary"]["cells"])
    result = {}
    for cell in cells:
        pairs = []
        for previous, current in itertools.pairwise(records):
            if (
                current["checkpoint_update"] - previous["checkpoint_update"]
                == CHECKPOINT_CADENCE
                and previous["summary"]["cells"][cell]["passed"]
                and current["summary"]["cells"][cell]["passed"]
            ):
                pairs.append(
                    [
                        previous["checkpoint_update"],
                        current["checkpoint_update"],
                    ]
                )
        trajectory_saved = any(
            record["summary"]["cells"][cell]["trajectory_saved"]
            for record in records
            if record["summary"]["cells"][cell]["passed"]
        )
        result[cell] = {
            "required_successes": REQUIRED_CELL_SUCCESSES,
            "episodes": 8,
            "required_consecutive_evaluations": 2,
            "passing_update_pairs": pairs,
            "trajectory_saved_from_passing_checkpoint": (trajectory_saved),
            "passed": bool(pairs) and trajectory_saved,
        }
    return {
        "cells": result,
        "unwitnessed_cells": [
            cell for cell, gate in result.items() if not gate["passed"]
        ],
        "passed": all(gate["passed"] for gate in result.values()),
    }


def slight_improvement(records: list[dict], cells: list[str]) -> dict:
    if len(records) < 2:
        return {
            "passed": False,
            "reason": "at_least_two_evaluations_required",
            "cells": {},
        }
    window_start = max(1, len(records) - PLATEAU_EVALUATIONS)
    window_indices = range(window_start, len(records))
    cell_results = {}
    for cell in cells:
        events = []
        for index in window_indices:
            previous = records[:index]
            current = records[index]
            previous_success = max(
                record["summary"]["cells"][cell]["successes"] for record in previous
            )
            previous_completion = max(
                record["summary"]["cells"][cell]["median_terminal_absolute_completion"]
                for record in previous
            )
            current_cell = current["summary"]["cells"][cell]
            success_gain = current_cell["successes"] - previous_success
            completion_gain = (
                current_cell["median_terminal_absolute_completion"]
                - previous_completion
            )
            integrity_clean = current_cell["integrity_failure_count"] == 0
            event_passed = integrity_clean and (
                success_gain >= 1 or completion_gain + 1e-12 >= COMPLETION_PROGRESS
            )
            events.append(
                {
                    "checkpoint_update": current["checkpoint_update"],
                    "previous_best_successes": previous_success,
                    "current_successes": current_cell["successes"],
                    "success_gain": success_gain,
                    "previous_best_median_completion": (previous_completion),
                    "current_median_completion": current_cell[
                        "median_terminal_absolute_completion"
                    ],
                    "completion_gain": completion_gain,
                    "integrity_clean": integrity_clean,
                    "passed": event_passed,
                }
            )
        passing_events = [event for event in events if event["passed"]]
        cell_results[cell] = {
            "window_evaluations": PLATEAU_EVALUATIONS,
            "required_completion_gain": COMPLETION_PROGRESS,
            "events": events,
            "last_improvement_update": (
                passing_events[-1]["checkpoint_update"] if passing_events else None
            ),
            "passed": bool(passing_events),
        }
    return {
        "plateau_evaluations": PLATEAU_EVALUATIONS,
        "window_updates": [
            records[index]["checkpoint_update"] for index in window_indices
        ],
        "cells": cell_results,
        "passed": any(item["passed"] for item in cell_results.values()),
    }


def decision(records: list[dict]) -> dict:
    witness = consecutive_cell_witnesses(records)
    if witness["passed"]:
        return {
            "decision": "panel_witness_passed",
            "continue_same_panel": False,
            "conditional_cell_isolates": [],
            "witness": witness,
            "slight_improvement": {
                "passed": False,
                "reason": "panel_already_passed",
            },
        }
    unwitnessed = witness["unwitnessed_cells"]
    progress = slight_improvement(records, unwitnessed)
    if progress["passed"]:
        outcome = "continue_same_panel"
        isolates: list[str] = []
    else:
        witnessed = [cell for cell, gate in witness["cells"].items() if gate["passed"]]
        if witnessed:
            outcome = "conditional_cell_isolates"
            isolates = unwitnessed
        else:
            outcome = "stop_and_diagnose_panel"
            isolates = []
    return {
        "decision": outcome,
        "continue_same_panel": outcome == "continue_same_panel",
        "conditional_cell_isolates": isolates,
        "witness": witness,
        "slight_improvement": progress,
    }


def checkpoint_paths(pattern: str) -> list[Path]:
    paths = sorted({Path(path).resolve() for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {pattern}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--panel",
        choices=tuple(PANEL_SPECS),
        required=True,
    )
    parser.add_argument("--expected-checkpoints", type=int, default=5)
    parser.add_argument("--planned-updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026072700)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if environment_completion_contract() != "exact_visible_dump_v1":
        raise RuntimeError("B0 requires exact_visible_dump_v1")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = panel_spec(args.panel)
    bank_root = args.bank_root.resolve()
    relative_path = f"panels/development/{args.panel}"
    directory = bank_root / relative_path
    rows = load_manifest(directory)
    if len(rows) != spec["train_count"]:
        raise RuntimeError(
            f"B0 {args.panel} development count {len(rows)} does not "
            f"match declared {spec['train_count']}"
        )
    observed_cells = tuple(sorted({row["primary_cell"] for row in rows}))
    if observed_cells != tuple(sorted(spec["cells"])):
        raise RuntimeError(f"B0 {args.panel} cell mismatch: {observed_cells}")

    paths = checkpoint_paths(args.checkpoint_glob)
    if len(paths) != args.expected_checkpoints:
        raise RuntimeError(
            f"expected {args.expected_checkpoints} checkpoints, " f"got {len(paths)}"
        )
    checkpoints = [(path, load_pkl_object(str(path))) for path in paths]
    checkpoints.sort(key=lambda item: int(item[1].get("next_update", 0)))
    expected_updates = list(
        range(
            CHECKPOINT_CADENCE,
            CHECKPOINT_CADENCE * len(checkpoints) + 1,
            CHECKPOINT_CADENCE,
        )
    )
    observed_updates = [
        int(checkpoint.get("next_update", 0)) for _, checkpoint in checkpoints
    ]
    if observed_updates != expected_updates:
        raise RuntimeError(f"B0 checkpoint cadence mismatch: {observed_updates}")
    reference_config = checkpoints[0][1]["train_config"]
    checkpoint_gates = {}
    for path, checkpoint in checkpoints:
        _validate_checkpoint_architecture(
            checkpoint,
            reference_config,
        )
        checkpoint_gates[str(path)] = verify_b0_checkpoint(
            checkpoint,
            args.panel,
            int(checkpoint["next_update"]),
            planned_updates=args.planned_updates,
        )

    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(len(rows))
    config = configure_for_panel(
        reference_config,
        args.panel,
        len(rows),
    )
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(
        lambda value: value[0],
        env_params,
    )
    reset_keys = exact_reset_keys(len(rows))
    reset_verification = verify_exact_reset(
        env,
        env_params,
        reset_keys,
        directory,
        len(rows),
    )
    model = SimpleNamespace(apply=initialized_state.apply_fn)
    records = []
    for checkpoint_path, checkpoint in checkpoints:
        _, stats, _ = rollout_episode(
            env,
            model,
            checkpoint["model"],
            env_params,
            config,
            max_frames=HORIZON,
            deterministic=True,
            seed=args.seed,
            use_mcts=False,
            reset_keys=reset_keys,
            record_observations=False,
            record_actions=True,
            preserve_terminal_states=True,
            expected_slot_indices=np.arange(
                len(rows),
                dtype=np.int32,
            ),
        )
        summary = summarize_checkpoint(rows, stats)
        record = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_update": int(checkpoint["next_update"]),
            "summary": summary,
        }
        records.append(record)
        with output.open("w") as stream:
            json.dump(
                {
                    "schema": "terra_b0_panel_eval_v1",
                    "panel": args.panel,
                    "records": records,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        print(
            f"{args.panel} update {record['checkpoint_update']}: "
            f"{summary['overall']['successes']}/"
            f"{summary['overall']['episodes']}"
        )

    payload = {
        "schema": "terra_b0_panel_eval_v1",
        "completion_contract": "exact_visible_dump_v1",
        "reward_contract": spec["reward_contract"],
        "panel": args.panel,
        "family": spec["family"],
        "cells": list(spec["cells"]),
        "bank_root": str(bank_root),
        "relative_path": relative_path,
        "manifest": str(directory / "manifest.jsonl"),
        "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
        "deterministic": True,
        "horizon": HORIZON,
        "reset_seed": args.seed,
        "exact_manifest_enumeration": True,
        "reset_verification": reset_verification,
        "checkpoint_gates": checkpoint_gates,
        "records": records,
        "adjudication": decision(records),
    }
    with output.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"B0_PANEL_DECISION={payload['adjudication']['decision']}")


if __name__ == "__main__":
    main()
