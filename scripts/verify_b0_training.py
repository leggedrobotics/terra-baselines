#!/usr/bin/env python3
"""Verify one complete bounded B0 panel training milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from eval_b0_panel import (
    CHECKPOINT_CADENCE,
    INITIAL_UPDATES,
    PANEL_SPECS,
    panel_spec,
    verify_b0_checkpoint,
)
from terra.maps_buffer import validate_exact_dataset_contract
from train_mixed import _validate_checkpoint_architecture
from utils.helpers import load_pkl_object

HARD_ZERO_FIELDS = (
    "mass_residual_violation_count",
    "target_mutation_count",
    "obstacle_mutation_count",
    "step_reward_residual_violation_count",
)
SUM_FIELDS = (
    "episode_count",
    "task_done_count",
    "timeout_count",
    "step_count",
    "productive_workspace_cycles",
    "explicit_noop_count",
    "no_effect_action_count",
    "reward_residual_violation_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def assert_equal_trees(first: Any, second: Any, label: str) -> int:
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        raise RuntimeError(f"{label} tree definitions differ")
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    for index, (first_leaf, second_leaf) in enumerate(
        zip(first_leaves, second_leaves, strict=True)
    ):
        if not np.array_equal(
            np.asarray(jax.device_get(first_leaf)),
            np.asarray(jax.device_get(second_leaf)),
        ):
            raise RuntimeError(f"{label} leaf {index} differs")
    return len(first_leaves)


def numbered_checkpoints(
    directory: Path,
    expected_updates: int,
) -> list[tuple[int, Path, dict]]:
    records = []
    for path in directory.glob("*_update_*.pkl"):
        checkpoint = load_pkl_object(str(path))
        records.append(
            (
                int(checkpoint["next_update"]),
                path.resolve(),
                checkpoint,
            )
        )
    records.sort(key=lambda item: item[0])
    expected = list(
        range(
            CHECKPOINT_CADENCE,
            expected_updates + 1,
            CHECKPOINT_CADENCE,
        )
    )
    if [update for update, _, _ in records] != expected:
        raise RuntimeError("B0 numbered checkpoint cadence is incomplete")
    return records


def aggregate_receipts(
    directory: Path,
    panel: str,
    expected_updates: int,
) -> tuple[list[Path], dict[str, Any]]:
    spec = panel_spec(panel)
    rows = []
    for path in directory.glob("*_update_*.json"):
        payload = json.loads(path.read_text())
        rows.append((int(payload["update"]), path.resolve(), payload))
    rows.sort(key=lambda item: item[0])
    if [update for update, _, _ in rows] != list(range(1, expected_updates + 1)):
        raise RuntimeError("B0 aggregate update sequence is incomplete")

    totals = {field: 0 for field in SUM_FIELDS}
    maximum_mass_residual = 0
    maximum_step_reward_residual = 0.0
    for update, path, payload in rows:
        if (
            payload.get("schema") != "terra_training_episode_aggregate_v2"
            or payload.get("contract") != "exact_visible_dump_v1"
        ):
            raise RuntimeError(f"B0 aggregate contract mismatch at {update}: {path}")
        if spec["family"] not in payload["family_names"]:
            raise RuntimeError(f"B0 aggregate lacks family at update {update}")
        missing_cells = set(spec["cells"]) - set(payload["primary_cell_names"])
        if missing_cells:
            raise RuntimeError(
                f"B0 aggregate lacks {sorted(missing_cells)} " f"at update {update}"
            )
        row_totals = payload["totals"]
        for field in HARD_ZERO_FIELDS:
            if int(row_totals[field]) != 0:
                raise RuntimeError(f"B0 hard gate {field} failed at update {update}")
        maximum_mass_residual = max(
            maximum_mass_residual,
            int(row_totals["maximum_mass_residual"]),
        )
        maximum_step_reward_residual = max(
            maximum_step_reward_residual,
            float(row_totals["maximum_step_reward_residual"]),
        )
        for field in SUM_FIELDS:
            totals[field] += int(row_totals[field])
    if maximum_mass_residual != 0:
        raise RuntimeError("B0 maximum mass residual is nonzero")
    return [path for _, path, _ in rows], {
        "receipts": len(rows),
        "hard_zero_fields": list(HARD_ZERO_FIELDS),
        "maximum_mass_residual": maximum_mass_residual,
        "maximum_step_reward_residual": (maximum_step_reward_residual),
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--panel",
        choices=tuple(PANEL_SPECS),
        required=True,
    )
    parser.add_argument(
        "--expected-updates",
        type=int,
        default=INITIAL_UPDATES,
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_updates % CHECKPOINT_CADENCE:
        raise ValueError("--expected-updates must be divisible by checkpoint cadence")
    directory = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.expected_dataset_count is not None and args.dataset is None:
        raise ValueError("--expected-dataset-count requires --dataset")
    dataset_gate = None
    if args.dataset is not None:
        expected_dataset_count = (
            panel_spec(args.panel)["train_count"]
            if args.expected_dataset_count is None
            else args.expected_dataset_count
        )
        if expected_dataset_count <= 0:
            raise ValueError("--expected-dataset-count must be positive")
        dataset = args.dataset.resolve()
        validate_exact_dataset_contract(dataset, expected_dataset_count)
        dataset_gate = {
            "path": str(dataset),
            "manifest_sha256": sha256_file(dataset / "manifest.jsonl"),
            "expected_count": expected_dataset_count,
        }

    numbered = numbered_checkpoints(
        directory,
        args.expected_updates,
    )
    reference_config = numbered[0][2]["train_config"]
    checkpoint_gates = {}
    for update, _, checkpoint in numbered:
        _validate_checkpoint_architecture(checkpoint, reference_config)
        checkpoint_gates[update] = verify_b0_checkpoint(
            checkpoint,
            args.panel,
            update,
            planned_updates=args.expected_updates,
        )

    finals = sorted(directory.glob("*_FINAL.pkl"))
    if len(finals) != 1:
        raise RuntimeError(f"expected one B0 FINAL checkpoint, got {finals}")
    final_path = finals[0].resolve()
    final = load_pkl_object(str(final_path))
    _validate_checkpoint_architecture(final, reference_config)
    final_gate = verify_b0_checkpoint(
        final,
        args.panel,
        args.expected_updates,
        planned_updates=args.expected_updates,
    )
    last_update, last_path, last = numbered[-1]
    if last_update != args.expected_updates:
        raise RuntimeError("B0 terminal checkpoint update mismatch")
    equality = {
        "model_leaves": assert_equal_trees(
            final["model"],
            last["model"],
            "B0 FINAL/last model",
        ),
        "optimizer_leaves": assert_equal_trees(
            final["optimizer_state"],
            last["optimizer_state"],
            "B0 FINAL/last optimizer",
        ),
    }

    aggregate_paths, aggregate_gate = aggregate_receipts(
        directory / "episode_aggregates",
        args.panel,
        args.expected_updates,
    )
    checkpoint_paths = [path for _, path, _ in numbered]
    receipt = {
        "schema": "terra_b0_training_gate_v1",
        "passed": True,
        "panel": args.panel,
        "reward_contract": panel_spec(args.panel)["reward_contract"],
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint_dir": str(directory),
        "dataset_gate": dataset_gate,
        "numbered_checkpoint_gate": {
            "count": len(numbered),
            "updates": [update for update, _, _ in numbered],
            "manifest_sha256": manifest_digest(checkpoint_paths),
            "gates": checkpoint_gates,
        },
        "aggregate_gate": {
            **aggregate_gate,
            "manifest_sha256": manifest_digest(aggregate_paths),
        },
        "final_gate": {
            **final_gate,
            "path": str(final_path),
            "sha256": sha256_file(final_path),
        },
        "last_update": {
            "path": str(last_path),
            "sha256": sha256_file(last_path),
        },
        "final_equals_last_update": equality,
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("B0_TRAINING_GATE=PASSED")


if __name__ == "__main__":
    main()
