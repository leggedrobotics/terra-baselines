#!/usr/bin/env python3
"""Verify a complete 1,000-update F0/F0R training arm before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from eval_f0_identity import (
    F0_TREATMENTS,
    IDENTITIES,
    verify_production_checkpoint,
)
from train_mixed import _validate_checkpoint_architecture
from utils.helpers import load_pkl_object

SCHEMA = "terra_f0_training_gate_v1"
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def assert_equal_trees(first: Any, second: Any, label: str) -> int:
    first_definition = jax.tree_util.tree_structure(first)
    second_definition = jax.tree_util.tree_structure(second)
    if first_definition != second_definition:
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


def numbered_checkpoints(directory: Path) -> list[tuple[int, Path, dict]]:
    records = []
    for path in directory.glob("*_update_*.pkl"):
        checkpoint = load_pkl_object(str(path))
        records.append((int(checkpoint["next_update"]), path.resolve(), checkpoint))
    records.sort(key=lambda item: item[0])
    expected = list(range(100, 1001, 100))
    if [update for update, _, _ in records] != expected:
        raise RuntimeError("F0 numbered checkpoint cadence is incomplete")
    return records


def aggregate_receipts(
    directory: Path,
    identity: str,
) -> tuple[list[Path], dict[str, Any]]:
    identity_spec = IDENTITIES[identity]
    rows = []
    for path in directory.glob("*_update_*.json"):
        payload = json.loads(path.read_text())
        rows.append((int(payload["update"]), path.resolve(), payload))
    rows.sort(key=lambda item: item[0])
    if [update for update, _, _ in rows] != list(range(1, 1001)):
        raise RuntimeError("F0 aggregate update sequence is incomplete")

    totals = {field: 0 for field in SUM_FIELDS}
    maximum_mass_residual = 0
    maximum_step_reward_residual = 0.0
    for update, path, payload in rows:
        if (
            payload.get("schema") != "terra_training_episode_aggregate_v2"
            or payload.get("contract") != "exact_visible_dump_v1"
        ):
            raise RuntimeError(
                f"aggregate contract mismatch at update {update}: {path}"
            )
        if identity_spec["family"] not in payload["family_names"]:
            raise RuntimeError(f"aggregate lacks family provenance at update {update}")
        if identity_spec["primary_cell"] not in payload["primary_cell_names"]:
            raise RuntimeError(f"aggregate lacks cell provenance at update {update}")
        row_totals = payload["totals"]
        for field in HARD_ZERO_FIELDS:
            if int(row_totals[field]) != 0:
                raise RuntimeError(
                    f"aggregate hard gate {field} failed at update {update}"
                )
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
        raise RuntimeError("aggregate maximum mass residual is nonzero")
    return [path for _, path, _ in rows], {
        "receipts": len(rows),
        "hard_zero_fields": list(HARD_ZERO_FIELDS),
        "maximum_mass_residual": maximum_mass_residual,
        "maximum_step_reward_residual": maximum_step_reward_residual,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--identity", choices=tuple(IDENTITIES), required=True)
    parser.add_argument(
        "--treatment",
        choices=tuple(F0_TREATMENTS),
        default="corrected_dense_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    numbered = numbered_checkpoints(directory)
    reference_config = numbered[0][2]["train_config"]
    checkpoint_gates = {}
    for update, _, checkpoint in numbered:
        _validate_checkpoint_architecture(checkpoint, reference_config)
        checkpoint_gates[update] = verify_production_checkpoint(
            checkpoint,
            args.identity,
            update,
            args.treatment,
        )

    finals = sorted(directory.glob("*_FINAL.pkl"))
    if len(finals) != 1:
        raise RuntimeError(f"expected one FINAL checkpoint, got {finals}")
    final_path = finals[0].resolve()
    final = load_pkl_object(str(final_path))
    _validate_checkpoint_architecture(final, reference_config)
    final_gate = verify_production_checkpoint(
        final,
        args.identity,
        1000,
        args.treatment,
    )
    update_1000_path = numbered[-1][1]
    update_1000 = numbered[-1][2]
    equality = {
        "model_leaves": assert_equal_trees(
            final["model"],
            update_1000["model"],
            "FINAL/update-1000 model",
        ),
        "optimizer_leaves": assert_equal_trees(
            final["optimizer_state"],
            update_1000["optimizer_state"],
            "FINAL/update-1000 optimizer",
        ),
    }

    aggregate_paths, aggregate_gate = aggregate_receipts(
        directory / "episode_aggregates",
        args.identity,
    )
    checkpoint_paths = [path for _, path, _ in numbered]
    receipt = {
        "schema": SCHEMA,
        "passed": True,
        "identity": args.identity,
        "reward_contract": args.treatment,
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint_dir": str(directory),
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
        "update_1000": {
            "path": str(update_1000_path),
            "sha256": sha256_file(update_1000_path),
        },
        "final_equals_update_1000": equality,
    }
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("F0_TRAINING_GATE=PASSED")


if __name__ == "__main__":
    main()
