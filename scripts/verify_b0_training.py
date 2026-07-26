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


def assert_equal_aggregate_prefix(
    reference_paths: list[Path],
    candidate_paths: list[Path],
) -> int:
    if len(reference_paths) > len(candidate_paths):
        raise RuntimeError("candidate aggregate history is shorter than reference")
    for reference_path, candidate_path in zip(
        reference_paths,
        candidate_paths,
        strict=False,
    ):
        reference = json.loads(reference_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        reference.pop("run_name", None)
        candidate.pop("run_name", None)
        if reference != candidate:
            raise RuntimeError(
                "B0 aggregate prefix differs at update "
                f"{reference.get('update')}: {reference_path} vs {candidate_path}"
            )
    return len(reference_paths)


def verify_reproducible_prefix(
    reference_directory: Path,
    candidate_numbered: list[tuple[int, Path, dict]],
    candidate_aggregate_paths: list[Path],
    panel: str,
    reference_updates: int,
) -> dict[str, Any]:
    reference_numbered = numbered_checkpoints(
        reference_directory,
        reference_updates,
    )
    candidate_by_update = {
        update: (path, checkpoint) for update, path, checkpoint in candidate_numbered
    }
    checkpoint_records = []
    for update, reference_path, reference in reference_numbered:
        if update not in candidate_by_update:
            raise RuntimeError(f"candidate lacks prefix checkpoint {update}")
        candidate_path, candidate = candidate_by_update[update]
        if int(reference["train_config"].seed) != int(candidate["train_config"].seed):
            raise RuntimeError(f"B0 prefix seed differs at update {update}")
        checkpoint_records.append(
            {
                "update": update,
                "model_leaves": assert_equal_trees(
                    reference["model"],
                    candidate["model"],
                    f"B0 prefix model at update {update}",
                ),
                "optimizer_leaves": assert_equal_trees(
                    reference["optimizer_state"],
                    candidate["optimizer_state"],
                    f"B0 prefix optimizer at update {update}",
                ),
                "train_state_step_leaves": assert_equal_trees(
                    reference["train_state_step"],
                    candidate["train_state_step"],
                    f"B0 prefix train step at update {update}",
                ),
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "candidate_path": str(candidate_path),
                "candidate_sha256": sha256_file(candidate_path),
            }
        )

    reference_aggregate_paths, reference_aggregate_gate = aggregate_receipts(
        reference_directory / "episode_aggregates",
        panel=panel,
        expected_updates=reference_updates,
    )
    compared_aggregates = assert_equal_aggregate_prefix(
        reference_aggregate_paths,
        candidate_aggregate_paths,
    )
    return {
        "passed": True,
        "reference_checkpoint_dir": str(reference_directory),
        "reference_updates": reference_updates,
        "checkpoint_records": checkpoint_records,
        "aggregate_receipts_compared": compared_aggregates,
        "reference_aggregate_manifest_sha256": manifest_digest(
            reference_aggregate_paths
        ),
        "candidate_prefix_aggregate_manifest_sha256": manifest_digest(
            candidate_aggregate_paths[:reference_updates]
        ),
        "reference_aggregate_gate": reference_aggregate_gate,
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
    parser.add_argument("--reference-checkpoint-dir", type=Path)
    parser.add_argument("--reference-updates", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_updates % CHECKPOINT_CADENCE:
        raise ValueError("--expected-updates must be divisible by checkpoint cadence")
    if (args.reference_checkpoint_dir is None) != (args.reference_updates is None):
        raise ValueError(
            "--reference-checkpoint-dir and --reference-updates must be used together"
        )
    if args.reference_updates is not None:
        if args.reference_updates % CHECKPOINT_CADENCE:
            raise ValueError(
                "--reference-updates must be divisible by checkpoint cadence"
            )
        if args.reference_updates >= args.expected_updates:
            raise ValueError("--reference-updates must be below --expected-updates")
    directory = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

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
    reproducible_prefix = None
    if args.reference_checkpoint_dir is not None:
        reproducible_prefix = verify_reproducible_prefix(
            args.reference_checkpoint_dir.resolve(),
            numbered,
            aggregate_paths,
            args.panel,
            args.reference_updates,
        )
    receipt = {
        "schema": "terra_b0_training_gate_v1",
        "passed": True,
        "panel": args.panel,
        "reward_contract": panel_spec(args.panel)["reward_contract"],
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
        "last_update": {
            "path": str(last_path),
            "sha256": sha256_file(last_path),
        },
        "final_equals_last_update": equality,
        "reproducible_prefix": reproducible_prefix,
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("B0_TRAINING_GATE=PASSED")


if __name__ == "__main__":
    main()
