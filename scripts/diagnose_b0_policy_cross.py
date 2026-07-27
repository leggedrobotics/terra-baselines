#!/usr/bin/env python3
"""Cross B0 train/development identities with greedy/sampled policy modes."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from eval_b0_panel import (
    HORIZON,
    PANEL_SPECS,
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


def aggregate_cross(records: list[dict]) -> dict:
    result = {}
    for split in ("train", "development"):
        selected = [record for record in records if record["split"] == split]
        deterministic = [
            record for record in selected if record["policy_mode"] == "deterministic"
        ]
        sampled = [record for record in selected if record["policy_mode"] == "sampled"]
        if len(deterministic) != 1 or not sampled:
            raise ValueError(f"{split} requires one deterministic and sampled records")

        def totals(items: list[dict]) -> dict:
            successes = sum(
                int(item["summary"]["overall"]["successes"]) for item in items
            )
            episodes = sum(
                int(item["summary"]["overall"]["episodes"]) for item in items
            )
            return {
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
            }

        cells = {}
        for cell in sorted(deterministic[0]["summary"]["cells"]):
            cell_successes = sum(
                int(item["summary"]["cells"][cell]["successes"]) for item in sampled
            )
            cell_episodes = sum(
                int(item["summary"]["cells"][cell]["episodes"]) for item in sampled
            )
            cells[cell] = {
                "successes": cell_successes,
                "episodes": cell_episodes,
                "success_rate": cell_successes / cell_episodes,
            }
        result[split] = {
            "deterministic": totals(deterministic),
            "sampled": {
                **totals(sampled),
                "seeds": [int(item["seed"]) for item in sampled],
                "per_cell": cells,
            },
        }
    result["rate_differences"] = {
        "train_minus_development_deterministic": (
            result["train"]["deterministic"]["success_rate"]
            - result["development"]["deterministic"]["success_rate"]
        ),
        "train_minus_development_sampled": (
            result["train"]["sampled"]["success_rate"]
            - result["development"]["sampled"]["success_rate"]
        ),
        "sampled_minus_deterministic_train": (
            result["train"]["sampled"]["success_rate"]
            - result["train"]["deterministic"]["success_rate"]
        ),
        "sampled_minus_deterministic_development": (
            result["development"]["sampled"]["success_rate"]
            - result["development"]["deterministic"]["success_rate"]
        ),
    }
    return result


def compact_summary(summary: dict) -> dict:
    cells = {}
    for cell, values in summary["cells"].items():
        cells[cell] = {
            "successes": int(values["successes"]),
            "episodes": int(values["episodes"]),
            "median_terminal_absolute_completion": float(
                values["median_terminal_absolute_completion"]
            ),
            "integrity_failure_count": int(values["integrity_failure_count"]),
        }
    return {
        "overall": summary["overall"],
        "cells": cells,
        "integrity_failure_count": int(summary["integrity_failure_count"]),
        "per_map": summary["per_map"],
    }


TRACE_COMPONENTS = (
    "absolute",
    "dig",
    "dump_purity",
    "dump_volume",
    "unloaded",
    "accepted_dump_volume",
    "illegal_dump_volume",
)


def selected_failed_traces(rows: list[dict], summary: dict, stats: dict) -> dict:
    actions = np.asarray(stats["action_sequence"], dtype=np.int32)
    effects = np.asarray(stats["action_had_effect_sequence"], dtype=bool)
    completion = {
        name: np.asarray(stats["completion_sequence"][name], dtype=np.float32)
        for name in TRACE_COMPONENTS
    }
    if (
        actions.ndim != 2
        or actions.shape[1] != len(rows)
        or effects.shape != actions.shape
        or any(values.shape != actions.shape for values in completion.values())
        or any(not np.all(np.isfinite(values)) for values in completion.values())
        or np.any(actions < 0)
        or np.any(actions >= 8)
    ):
        raise RuntimeError("B0 policy-cross failed traces have invalid shapes/actions")

    result = {}
    for cell in sorted({row["primary_cell"] for row in rows}):
        failed = [
            index
            for index, row in enumerate(summary["per_map"])
            if row["primary_cell"] == cell and not row["success"]
        ]

        def peak(index):
            length = int(summary["per_map"][index]["steps"])
            return float(completion["absolute"][:length, index].max())

        selected = {}
        if failed:
            high = max(
                failed,
                key=lambda index: (
                    peak(index),
                    summary["per_map"][index]["terminal_absolute"],
                    -index,
                ),
            )
            selected.setdefault(high, []).append("high_completion")
            zero = next((index for index in failed if peak(index) <= 1e-6), None)
            if zero is not None:
                selected.setdefault(zero, []).append("zero_progress")
        traces = []
        for index, labels in selected.items():
            length = int(summary["per_map"][index]["steps"])
            traces.append(
                {
                    "selection": labels,
                    "map_id": rows[index]["map_id"],
                    "source_id": rows[index]["source_id"],
                    "primary_cell": cell,
                    "slot_index": int(rows[index]["slot_index"]),
                    "steps": length,
                    "actions": actions[:length, index].tolist(),
                    "action_had_effect": effects[:length, index].tolist(),
                    "completion": {
                        name: completion[name][:length, index].tolist()
                        for name in TRACE_COMPONENTS
                    },
                    "peak_absolute_completion": peak(index),
                    "terminal_absolute_completion": float(
                        summary["per_map"][index]["terminal_absolute"]
                    ),
                    "final_success": False,
                }
            )
        result[cell] = {"failed_identities": len(failed), "traces": traces}
    return result


def write_partial(output: Path, payload: dict) -> Path:
    partial = Path(f"{output}.partial")
    staging = Path(f"{partial}.tmp")
    with staging.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(staging, partial)
    return partial


def payload_for(
    args,
    bank_root: Path,
    checkpoints: list[tuple[Path, dict]],
    reset_verifications: dict,
    records: list[dict],
    *,
    status: str,
) -> dict:
    completed_updates = sorted({int(record["checkpoint_update"]) for record in records})
    checkpoint_rows = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "update": int(checkpoint["next_update"]),
        }
        for path, checkpoint in checkpoints
    ]
    payload = {
        "schema": "terra_b0_policy_cross_v1",
        "status": status,
        "completion_contract": "exact_visible_dump_v1",
        "panel": args.panel,
        "bank_root": str(bank_root),
        "horizon": HORIZON,
        "checkpoints": checkpoint_rows,
        "completed_checkpoint_updates": completed_updates,
        "reset_verifications": reset_verifications,
        "records": records,
        "cross_summary_by_checkpoint": {
            str(update): aggregate_cross(
                [
                    record
                    for record in records
                    if int(record["checkpoint_update"]) == update
                ]
            )
            for update in completed_updates
        },
    }
    if len(checkpoint_rows) == 1:
        payload.update(
            {
                "checkpoint": checkpoint_rows[0]["path"],
                "checkpoint_sha256": checkpoint_rows[0]["sha256"],
                "checkpoint_update": checkpoint_rows[0]["update"],
                "cross_summary": payload["cross_summary_by_checkpoint"][
                    str(checkpoint_rows[0]["update"])
                ],
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--panel",
        choices=tuple(PANEL_SPECS),
        required=True,
    )
    parser.add_argument("--planned-updates", type=int, default=1000)
    parser.add_argument("--deterministic-seed", type=int, default=2026072800)
    parser.add_argument(
        "--sampled-seed",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sampled_seeds = args.sampled_seed or [
        2026072801,
        2026072802,
        2026072803,
        2026072804,
    ]
    if len(sampled_seeds) < 2 or len(sampled_seeds) != len(set(sampled_seeds)):
        raise ValueError("provide at least two unique sampled seeds")
    if args.deterministic_seed in sampled_seeds:
        raise ValueError("deterministic and sampled seeds must be disjoint")
    if environment_completion_contract() != "exact_visible_dump_v1":
        raise RuntimeError("B0 policy cross requires exact_visible_dump_v1")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank_root = args.bank_root.resolve()
    checkpoints = [
        (path.resolve(), load_pkl_object(str(path.resolve())))
        for path in args.checkpoint
    ]
    checkpoints.sort(key=lambda item: int(item[1]["next_update"]))
    updates = [int(checkpoint["next_update"]) for _, checkpoint in checkpoints]
    if len(updates) != len(set(updates)):
        raise ValueError("B0 policy cross requires unique checkpoint updates")
    reference_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        _validate_checkpoint_architecture(checkpoint, reference_config)
        verify_b0_checkpoint(
            checkpoint,
            args.panel,
            int(checkpoint["next_update"]),
            planned_updates=args.planned_updates,
        )
    spec = panel_spec(args.panel)
    records = []
    reset_verifications = {}
    contexts = {}
    for split in ("train", "development"):
        relative_path = f"panels/{split}/{args.panel}"
        directory = bank_root / relative_path
        rows = load_manifest(directory)
        if len(rows) != spec["train_count"]:
            raise RuntimeError(f"{split} B0 policy-cross manifest size mismatch")
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(len(rows))
        config = configure_for_panel(
            copy.deepcopy(reference_config),
            args.panel,
            len(rows),
        )
        config.curriculum_levels_override[0]["maps_path"] = relative_path
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        reset_keys = exact_reset_keys(len(rows))
        reset_verifications[split] = verify_exact_reset(
            env,
            env_params,
            reset_keys,
            directory,
            len(rows),
        )
        contexts[split] = (
            rows,
            relative_path,
            directory,
            config,
            env,
            env_params,
            reset_keys,
            SimpleNamespace(apply=initialized_state.apply_fn),
        )

    modes = [
        ("deterministic", args.deterministic_seed, True),
        *[("sampled", seed, False) for seed in sampled_seeds],
    ]
    for checkpoint_path, checkpoint in checkpoints:
        update = int(checkpoint["next_update"])
        checkpoint_sha256 = sha256_file(checkpoint_path)
        for split, context in contexts.items():
            (
                rows,
                relative_path,
                directory,
                config,
                env,
                env_params,
                reset_keys,
                model,
            ) = context
            for policy_mode, seed, deterministic in modes:
                _, stats, _ = rollout_episode(
                    env,
                    model,
                    checkpoint["model"],
                    env_params,
                    config,
                    max_frames=HORIZON,
                    deterministic=deterministic,
                    seed=seed,
                    use_mcts=False,
                    reset_keys=reset_keys,
                    record_observations=False,
                    record_actions=True,
                    preserve_terminal_states=True,
                    expected_slot_indices=np.arange(len(rows), dtype=np.int32),
                    record_completion=True,
                )
                summary = compact_summary(summarize_checkpoint(rows, stats))
                if summary["integrity_failure_count"]:
                    raise RuntimeError("B0 policy-cross rollout failed integrity")
                records.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_update": update,
                        "split": split,
                        "relative_path": relative_path,
                        "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                        "policy_mode": policy_mode,
                        "deterministic": deterministic,
                        "seed": seed,
                        "summary": summary,
                        "failed_traces": selected_failed_traces(rows, summary, stats),
                    }
                )
                print(
                    f"{args.panel} update {update} {split} {policy_mode} seed {seed}: "
                    f"{summary['overall']['successes']}/"
                    f"{summary['overall']['episodes']}"
                )
        write_partial(
            output,
            payload_for(
                args,
                bank_root,
                checkpoints,
                reset_verifications,
                records,
                status="incomplete",
            ),
        )

    partial = write_partial(
        output,
        payload_for(
            args,
            bank_root,
            checkpoints,
            reset_verifications,
            records,
            status="complete",
        ),
    )
    os.replace(partial, output)
    print("B0_POLICY_CROSS_COMPLETE")


if __name__ == "__main__":
    main()
