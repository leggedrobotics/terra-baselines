#!/usr/bin/env python3
"""Audit partial trench resets against the exact axis-v2 owner sidecars."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from terra.env_generation.partial_completion import _load_source_layers
from terra.maps_buffer import (
    PARTIAL_COMPLETION_MANIFEST,
    PARTIAL_RESET_BANK_INDEX,
    PARTIAL_RESET_BANK_SCHEMA,
    _trench_records_from_metadata,
    partial_reset_bank_sha256,
    trench_axis_contract_sanity_check,
    trench_axis_owners_sanity_check,
)
from terra.state import State
from tools.audit_trench_alignment_feasibility import (
    clear_positions,
    env_config,
    pose_axis_bits,
    terra_geometry,
    translated,
)


_WORKER: dict = {}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def init_worker(cfg, cones, footprints) -> None:
    _WORKER.update(cfg=cfg, cones=cones, footprints=footprints)


def audit_triplet(task: dict) -> list[dict]:
    source = Path(task["source"])
    partial = Path(task["partial"])
    source_index = int(task["source_index"])
    target, occupancy, _ = _load_source_layers(source, source_index)
    metadata = json.loads(
        (source / "metadata" / f"trench_{source_index}.json").read_text()
    )
    owners = np.load(
        source / "trench_axis_owners" / f"img_{source_index}.npy",
        allow_pickle=False,
    )
    records, axis_count = _trench_records_from_metadata(metadata, 4)
    if axis_count <= 0:
        raise RuntimeError(f"{task['maps_path']} is labelled trench but has no axes")
    records = np.asarray(records[:axis_count], dtype=np.float64)

    results = []
    for row in task["rows"]:
        action = np.load(
            partial / "actions" / f"img_{row['sidecar_index']}.npy",
            allow_pickle=False,
        )
        remaining = (target < 0) & (action >= 0)
        partial_target = np.array(target, copy=True)
        partial_target[(target < 0) & ~remaining] = 0
        partial_owners = np.where(remaining, owners, 0).astype(np.uint8)
        trench_axis_owners_sanity_check(
            partial_target, partial_owners, axis_count, max_trench_type=4
        )
        trench_axis_contract_sanity_check(metadata, owners, axis_count)

        runtime_blocked = np.asarray(
            State._build_traversability_mask(
                jnp.asarray(action), jnp.asarray(occupancy)
            )
        ).astype(bool)
        # Conservative re-approach stations stay footprint-clear even after
        # every currently fresh trench cell has become a hole.
        persistent_blocked = runtime_blocked | remaining
        a1 = np.zeros_like(remaining)
        a2 = np.zeros_like(remaining)
        pose_count = 0
        action_count = 0
        for heading, footprint in enumerate(_WORKER["footprints"]):
            for position in clear_positions(persistent_blocked, footprint):
                bits = pose_axis_bits(
                    *position, heading, records, _WORKER["cfg"]
                )
                if bits == 0:
                    continue
                pose_count += 1
                for cone_offsets in _WORKER["cones"][heading]:
                    cells = translated(position, cone_offsets)
                    inside = (
                        (cells[:, 0] >= 0)
                        & (cells[:, 0] < target.shape[0])
                        & (cells[:, 1] >= 0)
                        & (cells[:, 1] < target.shape[1])
                    )
                    cells = cells[inside]
                    if np.any(occupancy[cells[:, 0], cells[:, 1]]):
                        continue
                    fresh = cells[remaining[cells[:, 0], cells[:, 1]]]
                    if fresh.size == 0:
                        continue
                    compatible = (
                        partial_owners[fresh[:, 0], fresh[:, 1]] & np.uint8(bits)
                    ) != 0
                    valid = fresh[compatible]
                    a1[valid[:, 0], valid[:, 1]] = True
                    if np.all(compatible):
                        action_count += 1
                        a2[fresh[:, 0], fresh[:, 1]] = True

        remaining_cells = int(remaining.sum())
        a1_cells = int((a1 & remaining).sum())
        a2_cells = int((a2 & remaining).sum())
        results.append(
            {
                "maps_path": task["maps_path"],
                "condition_id": task["condition_id"],
                "source_index": source_index,
                "source_map_id": row["source_map_id"],
                "source_scenario_id": row["source_scenario_id"],
                "reset_tier": int(row["reset_tier"]),
                "pile_mode": row["pile_mode"],
                "completion_fraction": row["achieved_completion_fraction"],
                "remaining_cells": remaining_cells,
                "persistent_aligned_pose_count": pose_count,
                "candidate_atomic_action_count": action_count,
                "a1_remaining_cells_covered": a1_cells,
                "remaining_cells_covered": a2_cells,
                "a1_complete": a1_cells == remaining_cells,
                "alignment_chain_complete": a2_cells == remaining_cells,
            }
        )
    return results


def tasks(canonical: Path, partial: Path) -> list[dict]:
    index = json.loads((partial / PARTIAL_RESET_BANK_INDEX).read_text())
    if index.get("schema") != PARTIAL_RESET_BANK_SCHEMA:
        raise RuntimeError("partial-reset bank has the wrong schema")
    result = []
    for maps_path in index["supported_maps_paths"]:
        source = canonical / maps_path
        canonical_rows = jsonl(source / "manifest.jsonl")
        partial_leaf = partial / maps_path
        by_source: dict[int, list[dict]] = {}
        for row in jsonl(partial_leaf / PARTIAL_COMPLETION_MANIFEST):
            by_source.setdefault(int(row["source_index"]), []).append(row)
        for source_index, rows in sorted(by_source.items()):
            canonical_row = canonical_rows[source_index - 1]
            if canonical_row["family"] != "trench":
                continue
            rows.sort(key=lambda row: int(row["reset_tier"]))
            if [int(row["reset_tier"]) for row in rows] != [1, 2, 3]:
                raise RuntimeError(
                    f"{maps_path}:{source_index} is not one complete reset triplet"
                )
            result.append(
                {
                    "maps_path": maps_path,
                    "condition_id": canonical_row["primary_cell"],
                    "source": str(source),
                    "partial": str(partial_leaf),
                    "source_index": source_index,
                    "rows": rows,
                }
            )
    return result


def audit(canonical: Path, partial: Path, workers: int) -> dict:
    digest = partial_reset_bank_sha256(partial)
    index = json.loads((partial / PARTIAL_RESET_BANK_INDEX).read_text())
    if digest != index.get("bank_sha256"):
        raise RuntimeError("partial-reset bank digest mismatch")
    if index.get("canonical_loader_registry_sha256") != __import__(
        "hashlib"
    ).sha256((canonical / "dataset.json").read_bytes()).hexdigest():
        raise RuntimeError("partial-reset bank is bound to a different full bank")

    cfg = env_config()
    cones, footprints = terra_geometry(cfg)
    work = tasks(canonical, partial)
    if workers == 1:
        init_worker(cfg, cones, footprints)
        groups = [audit_triplet(task) for task in work]
    else:
        with multiprocessing.get_context("spawn").Pool(
            workers,
            initializer=init_worker,
            initargs=(cfg, cones, footprints),
        ) as pool:
            groups = pool.map(audit_triplet, work)
    rows = [row for group in groups for row in group]
    failures = [row for row in rows if not row["alignment_chain_complete"]]
    return {
        "schema": "terra_partial_trench_alignment_audit_v2",
        "owner_contract": "generator_owner_bits_v1",
        "partial_reset_bank_sha256": digest,
        "canonical_loader_registry_sha256": index[
            "canonical_loader_registry_sha256"
        ],
        "persistent_pose_semantics": (
            "footprint clear with staged soil, existing holes, and all remaining "
            "fresh trench cells treated as blocked"
        ),
        "claim_limit": (
            "Conservative persistent local aligned-action coverage; not an exact "
            "navigation, spoil-ordering, or full-episode plan."
        ),
        "trench_source_triplets": len(work),
        "audited_sidecars": len(rows),
        "passed_sidecars": len(rows) - len(failures),
        "failed_sidecars": len(failures),
        "accepted": not failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    receipt = audit(
        args.canonical_root.resolve(), args.partial_root.resolve(), args.workers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )
    if not receipt["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
