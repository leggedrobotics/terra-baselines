#!/usr/bin/env python3
"""Diagnose B0 policies that regress from digging to movement-only behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from eval_b0_panel import (
    HORIZON,
    configure_for_panel,
    panel_spec,
    summarize_checkpoint,
    verify_b0_checkpoint,
)
from eval_fixed_bank import (
    environment_completion_contract,
    exact_reset_keys,
    load_manifest,
    sha256_file,
    verify_exact_reset,
)
from eval_mcts import rollout_episode
from train_mixed import _validate_checkpoint_architecture, make_mixed_agent_states
from utils.helpers import load_pkl_object

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
DO_ACTION = 6


def run_length_encode(actions: np.ndarray) -> list[dict]:
    actions = np.asarray(actions, dtype=np.int32)
    if actions.ndim != 1:
        raise ValueError("actions must be one-dimensional")
    if not len(actions):
        return []
    starts = np.r_[0, np.flatnonzero(actions[1:] != actions[:-1]) + 1]
    ends = np.r_[starts[1:], len(actions)]
    return [
        {
            "action": int(actions[start]),
            "name": ACTION_NAMES[int(actions[start])],
            "count": int(end - start),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def action_diagnostics(
    actions: np.ndarray,
    effects: np.ndarray,
    lengths: np.ndarray,
    rows: list[dict],
    task_per_map: list[dict],
) -> dict:
    actions = np.asarray(actions, dtype=np.int32)
    effects = np.asarray(effects, dtype=bool)
    lengths = np.asarray(lengths, dtype=np.int32)
    if actions.shape != effects.shape or actions.shape[1] != len(rows):
        raise ValueError("action/effect traces do not match manifest rows")
    if lengths.shape != (len(rows),):
        raise ValueError("episode lengths do not match manifest rows")
    if np.any(actions < 0) or np.any(actions >= len(ACTION_NAMES)):
        raise ValueError("action trace contains an invalid tracked action")

    task_by_slot = {int(task["slot_index_zero_based"]): task for task in task_per_map}
    per_map = []
    for index, row in enumerate(rows):
        length = int(lengths[index])
        sequence = actions[:length, index]
        sequence_effects = effects[:length, index]
        counts = np.bincount(sequence, minlength=len(ACTION_NAMES))
        effect_counts = np.bincount(
            sequence[sequence_effects],
            minlength=len(ACTION_NAMES),
        )
        do_steps = np.flatnonzero(sequence == DO_ACTION)
        effective_do_steps = np.flatnonzero((sequence == DO_ACTION) & sequence_effects)
        task = task_by_slot[index]
        per_map.append(
            {
                "slot_index_zero_based": index,
                "map_id": row["map_id"],
                "primary_cell": row["primary_cell"],
                "steps": length,
                "action_counts": {
                    name: int(counts[action])
                    for action, name in enumerate(ACTION_NAMES)
                },
                "effective_action_counts": {
                    name: int(effect_counts[action])
                    for action, name in enumerate(ACTION_NAMES)
                },
                "first_do_step": int(do_steps[0]) if len(do_steps) else None,
                "first_effective_do_step": (
                    int(effective_do_steps[0]) if len(effective_do_steps) else None
                ),
                "action_switches": int(np.count_nonzero(sequence[1:] != sequence[:-1])),
                "trace_rle": run_length_encode(sequence),
                "terminal_absolute": float(task["terminal_absolute"]),
                "terminal_dig": float(task["terminal_dig"]),
                "terminal_dump_volume": float(task["terminal_dump_volume"]),
                "success": bool(task["success"]),
            }
        )

    cells = {}
    for cell in sorted({row["primary_cell"] for row in per_map}):
        selected = [row for row in per_map if row["primary_cell"] == cell]
        counts = {
            name: sum(row["action_counts"][name] for row in selected)
            for name in ACTION_NAMES
        }
        effect_counts = {
            name: sum(row["effective_action_counts"][name] for row in selected)
            for name in ACTION_NAMES
        }
        first_do = [
            row["first_do_step"] for row in selected if row["first_do_step"] is not None
        ]
        cells[cell] = {
            "maps": len(selected),
            "maps_with_do": len(first_do),
            "maps_with_effective_do": sum(
                row["first_effective_do_step"] is not None for row in selected
            ),
            "median_first_do_step": (float(np.median(first_do)) if first_do else None),
            "action_counts": counts,
            "effective_action_counts": effect_counts,
            "dominant_action": max(counts, key=counts.get),
            "median_terminal_absolute": float(
                np.median([row["terminal_absolute"] for row in selected])
            ),
            "median_terminal_dig": float(
                np.median([row["terminal_dig"] for row in selected])
            ),
            "median_terminal_dump_volume": float(
                np.median([row["terminal_dump_volume"] for row in selected])
            ),
        }
    return {"cells": cells, "per_map": per_map}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--panel",
        choices=("trench_distance", "trench_side"),
        required=True,
    )
    parser.add_argument("--planned-updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026072700)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if environment_completion_contract() != "exact_visible_dump_v1":
        raise RuntimeError("B0 diagnosis requires exact_visible_dump_v1")
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
        raise RuntimeError("B0 diagnostic manifest size mismatch")

    checkpoints = [
        (path.resolve(), load_pkl_object(str(path.resolve())))
        for path in args.checkpoint
    ]
    checkpoints.sort(key=lambda item: int(item[1]["next_update"]))
    reference_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        _validate_checkpoint_architecture(checkpoint, reference_config)
        verify_b0_checkpoint(
            checkpoint,
            args.panel,
            int(checkpoint["next_update"]),
            planned_updates=args.planned_updates,
        )

    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(len(rows))
    config = configure_for_panel(reference_config, args.panel, len(rows))
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
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
            expected_slot_indices=np.arange(len(rows), dtype=np.int32),
        )
        task = summarize_checkpoint(rows, stats)
        if task["integrity_failure_count"]:
            raise RuntimeError("B0 collapse diagnostic rollout failed integrity")
        records.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_update": int(checkpoint["next_update"]),
                "task_cells": task["cells"],
                "action_diagnostics": action_diagnostics(
                    stats["action_sequence"],
                    stats["action_had_effect_sequence"],
                    stats["episode_length"],
                    rows,
                    task["per_map"],
                ),
            }
        )
        print(
            f"{args.panel} collapse diagnosis update "
            f"{checkpoint['next_update']} complete"
        )

    payload = {
        "schema": "terra_b0_collapse_diagnosis_v1",
        "completion_contract": "exact_visible_dump_v1",
        "panel": args.panel,
        "bank_root": str(bank_root),
        "relative_path": relative_path,
        "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
        "deterministic": True,
        "horizon": HORIZON,
        "reset_seed": args.seed,
        "reset_verification": reset_verification,
        "records": records,
    }
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("B0_COLLAPSE_DIAGNOSIS_COMPLETE")


if __name__ == "__main__":
    main()
