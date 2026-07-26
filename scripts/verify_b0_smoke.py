#!/usr/bin/env python3
"""Verify one exact saved update-1 B0 panel checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from eval_b0_panel import (
    PANEL_SPECS,
    panel_spec,
    sha256_file,
    verify_b0_checkpoint,
)
from terra.maps_buffer import validate_exact_dataset_contract
from utils import helpers

HARD_ZERO_FIELDS = (
    "mass_residual_violation_count",
    "target_mutation_count",
    "obstacle_mutation_count",
    "step_reward_residual_violation_count",
)


def config_value(config, name):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def assert_finite_tree(tree, label: str) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise RuntimeError(f"{label} has no leaves")
    for index, leaf in enumerate(leaves):
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise RuntimeError(f"{label} leaf {index} is non-finite")
    return len(leaves)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--panel",
        choices=tuple(PANEL_SPECS),
        required=True,
    )
    parser.add_argument("--expected-dataset-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.checkpoint, args.aggregate):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = panel_spec(args.panel)
    expected_dataset_count = (
        spec["train_count"]
        if args.expected_dataset_count is None
        else args.expected_dataset_count
    )
    if expected_dataset_count <= 0:
        raise ValueError("--expected-dataset-count must be positive")
    validate_exact_dataset_contract(
        args.dataset,
        expected_dataset_count,
    )

    helpers.register_checkpoint_config_classes()
    checkpoint = helpers.load_pkl_object(str(args.checkpoint))
    checkpoint_gate = verify_b0_checkpoint(
        checkpoint,
        args.panel,
        1,
        planned_updates=1,
    )
    config = checkpoint["train_config"]
    extra_expected = {
        "checkpoint_interval": 1,
        "keep_checkpoint_history": True,
        "log_eval_interval": 0,
    }
    extra_observed = {name: config_value(config, name) for name in extra_expected}
    if extra_observed != extra_expected:
        raise RuntimeError(
            "B0 smoke checkpoint settings mismatch: " f"{extra_observed}"
        )

    aggregate = json.loads(args.aggregate.read_text())
    if (
        aggregate.get("schema") != "terra_training_episode_aggregate_v2"
        or aggregate.get("contract") != "exact_visible_dump_v1"
        or int(aggregate.get("update", -1)) != 1
    ):
        raise RuntimeError("B0 smoke aggregate contract mismatch")
    totals = aggregate["totals"]
    for field in HARD_ZERO_FIELDS:
        if int(totals[field]) != 0:
            raise RuntimeError(f"B0 smoke aggregate {field} is nonzero")
    if spec["family"] not in aggregate["family_names"]:
        raise RuntimeError("B0 smoke aggregate lacks family provenance")
    missing_cells = set(spec["cells"]) - set(aggregate["primary_cell_names"])
    if missing_cells:
        raise RuntimeError(f"B0 smoke aggregate lacks cells {sorted(missing_cells)}")

    receipt = {
        "schema": "terra_b0_update1_smoke_v1",
        "status": "passed",
        "panel": args.panel,
        "reward_contract": spec["reward_contract"],
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "aggregate": str(args.aggregate.resolve()),
        "aggregate_sha256": sha256_file(args.aggregate),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.jsonl"),
        "expected_dataset_count": expected_dataset_count,
        "model_leaf_count": assert_finite_tree(
            checkpoint["model"],
            "B0 model",
        ),
        "optimizer_leaf_count": assert_finite_tree(
            checkpoint["optimizer_state"],
            "B0 optimizer",
        ),
        "checkpoint_gate": checkpoint_gate,
        "extra_checked_config": extra_observed,
    }
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"B0_{args.panel.upper()}_UPDATE1_SMOKE_PASSED")


if __name__ == "__main__":
    main()
