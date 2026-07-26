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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_pkl_object(str(checkpoint_path))
    _validate_checkpoint_architecture(checkpoint, checkpoint["train_config"])
    verify_b0_checkpoint(
        checkpoint,
        args.panel,
        int(checkpoint["next_update"]),
        planned_updates=args.planned_updates,
    )
    spec = panel_spec(args.panel)
    records = []
    reset_verifications = {}

    for split in ("train", "development"):
        relative_path = f"panels/{split}/{args.panel}"
        directory = bank_root / relative_path
        rows = load_manifest(directory)
        if len(rows) != spec["train_count"]:
            raise RuntimeError(f"{split} B0 policy-cross manifest size mismatch")
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(len(rows))
        config = configure_for_panel(
            copy.deepcopy(checkpoint["train_config"]),
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
        model = SimpleNamespace(apply=initialized_state.apply_fn)
        modes = [
            ("deterministic", args.deterministic_seed, True),
            *[("sampled", seed, False) for seed in sampled_seeds],
        ]
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
            )
            summary = compact_summary(summarize_checkpoint(rows, stats))
            if summary["integrity_failure_count"]:
                raise RuntimeError("B0 policy-cross rollout failed integrity")
            records.append(
                {
                    "split": split,
                    "relative_path": relative_path,
                    "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                    "policy_mode": policy_mode,
                    "deterministic": deterministic,
                    "seed": seed,
                    "summary": summary,
                }
            )
            print(
                f"{args.panel} {split} {policy_mode} seed {seed}: "
                f"{summary['overall']['successes']}/"
                f"{summary['overall']['episodes']}"
            )

    payload = {
        "schema": "terra_b0_policy_cross_v1",
        "completion_contract": "exact_visible_dump_v1",
        "panel": args.panel,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_update": int(checkpoint["next_update"]),
        "bank_root": str(bank_root),
        "horizon": HORIZON,
        "reset_verifications": reset_verifications,
        "records": records,
        "cross_summary": aggregate_cross(records),
    }
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("B0_POLICY_CROSS_COMPLETE")


if __name__ == "__main__":
    main()
