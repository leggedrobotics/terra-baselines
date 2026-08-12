#!/usr/bin/env python3
"""Build the static V8 R2 reward-admission receipts and distance sidecar.

This is intentionally one direct, frozen-bank analysis path.  It does not
train, evaluate a policy, or rewrite the accepted V8 bank.  The derived
distance sidecar contains only distance arrays and identity receipts; all
physical reset arrays continue to come from the accepted base bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from skimage import __version__ as skimage_version
from skimage.graph import MCP_Geometric

SCHEMA = "terra_v8_r2_admission_receipt_v1"
DISTANCE_PROTOCOL_ID = "obstacle_geodesic_8_physical_global_v1"
DISTANCE_SIDECAR_SCHEMA = "terra_r2_distance_sidecar_v1"
BASE_DATASET_SCHEMA = "terra_curriculum_loader_bank_v1"
EXACT_DATASET_SCHEMA = "terra_exact_map_dataset_v1"
IDENTITY_CONTRACT = "terra_reset_arrays_sha256_v1"
ACCEPTED_DUMP_CONTRACT = "exact_visible_dump_v1"
EXPECTED_BANK_FILE_SHA256 = (
    "715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798"
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "8b49bd848c542e30b9e4d45639e4678905244b09a50482ff1c5c2f1d979dff19"
)
EXPECTED_ENVIRONMENT_PROTOCOL_SHA256 = (
    "9917b9238e9e6e844377e6d4a8ca18d1f0defbbacf887642743e579243109367"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834"
)
EXPECTED_EVAL_SHA256 = {
    "development": "dd8c3b381e57889827462222c81f29003a8b19f6285abd87247db5e60a2fea26",
    "promotion": "d717f8ac1009932ebb27fe7d257e5511b6e939d7cd1921db00df88f102527e56",
    "capability_development": (
        "8b9733b6d542f851d141803ec8bdaef7e4ef2db939a0e4bcb3fcad663e6df6be"
    ),
    "capability_promotion": (
        "ca7faa37d3e4fe6c675d6bc111ebde5674d22d39a4f3c231937d17e82cf12c6c"
    ),
}
EXPECTED_FINAL_EXACT = {
    "development": 546,
    "promotion": 549,
    "capability_development": 31,
    "capability_promotion": 31,
}
RESET_ARRAY_FOLDERS = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
    "distance",
)
PHYSICAL_ARRAY_FOLDERS = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("w") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def array_set_sha256(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


@dataclass(frozen=True)
class DatasetEntry:
    group: str
    split: str
    condition_id: str | None
    relative_path: str
    slot_count: int


def enumerate_datasets(index: dict[str, Any]) -> list[DatasetEntry]:
    entries = [
        DatasetEntry(
            group="train",
            split="train",
            condition_id=row["condition_id"],
            relative_path=row["maps_path"],
            slot_count=int(row["map_count"]),
        )
        for row in index["train"]
    ]
    for split, row in sorted(index["evaluation_panels"].items()):
        entries.append(
            DatasetEntry(
                group="main_evaluation",
                split=split,
                condition_id=None,
                relative_path=row["maps_path"],
                slot_count=int(row["slot_count"]),
            )
        )
    for split, row in sorted(index["capability_floor_evaluation_panels"].items()):
        entries.append(
            DatasetEntry(
                group="capability_evaluation",
                split=split,
                condition_id=None,
                relative_path=row["maps_path"],
                slot_count=int(row["slot_count"]),
            )
        )
    return entries


def canonical_distance_tiles(target: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    if target.shape != occupancy.shape or target.ndim != 2:
        raise ValueError("target and occupancy must be matching 2-D arrays")
    accepted = np.logical_and(target > 0, np.logical_not(occupancy))
    if not np.any(accepted):
        raise ValueError("map has no accepted dump cell")
    costs = np.ones(target.shape, dtype=np.float64)
    costs[np.asarray(occupancy, dtype=bool)] = np.inf
    starts = [tuple(int(v) for v in cell) for cell in np.argwhere(accepted)]
    distance, _ = MCP_Geometric(costs, fully_connected=True).find_costs(starts)
    traversable = np.logical_not(occupancy)
    if not np.all(np.isfinite(distance[traversable])):
        raise ValueError("not every traversable cell reaches the accepted dump mask")
    distance = np.asarray(distance, dtype=np.float64)
    distance[np.asarray(occupancy, dtype=bool)] = 0.0
    return distance


def load_arrays(dataset: Path, slot: int) -> dict[str, np.ndarray]:
    arrays = {
        name: np.load(dataset / name / f"img_{slot}.npy", allow_pickle=False)
        for name in RESET_ARRAY_FOLDERS
    }
    shapes = {array.shape for array in arrays.values()}
    if shapes != {(64, 64)}:
        raise ValueError(f"{dataset} slot {slot}: unexpected shapes {shapes}")
    return arrays


def build_d0(eval_paths: dict[str, Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    panels: dict[str, Any] = {}
    for panel, path in sorted(eval_paths.items()):
        observed_sha = sha256_file(path)
        if observed_sha != EXPECTED_EVAL_SHA256[panel]:
            raise ValueError(
                f"{panel}: frozen eval hash {observed_sha} != {EXPECTED_EVAL_SHA256[panel]}"
            )
        history = json.loads(path.read_text())
        if len(history) != 20:
            raise ValueError(f"{panel}: expected 20 checkpoints, got {len(history)}")
        final = history[-1]
        checks = {
            "checkpoint_update": final.get("checkpoint_update") == 20_000,
            "checkpoint_sha256": (
                final.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
            ),
            "deterministic": final.get("deterministic") is True,
            "exact_manifest_enumeration": (
                final.get("exact_manifest_enumeration") is True
            ),
            "horizon": final.get("horizon") == 450,
            "completion_contract": (
                final.get("completion_contract") == ACCEPTED_DUMP_CONTRACT
            ),
            "reset_verification": final.get("reset_verification", {}).get("passed")
            is True,
            "integrity": final.get("summary", {}).get("integrity", {}).get("passed")
            is True,
        }
        if not all(checks.values()):
            raise ValueError(f"{panel}: frozen-eval contract failed: {checks}")
        per_map = final["per_map"]
        exact = sum(bool(row["success"]) for row in per_map)
        if exact != EXPECTED_FINAL_EXACT[panel]:
            raise ValueError(
                f"{panel}: exact={exact}, expected {EXPECTED_FINAL_EXACT[panel]}"
            )
        for row in per_map:
            rows.append(
                {
                    "panel": panel,
                    "slot_index": int(row["slot_index"]),
                    "scenario_id": row["scenario_id"],
                    "map_id": row["map_id"],
                    "source_id": row["source_id"],
                    "family": row["family"],
                    "condition_id": row["primary_cell"],
                    "reset_seed": int(row["reset_seed"]),
                    "success": bool(row["success"]),
                    "steps": int(row["steps"]),
                    "terminal_absolute": float(row["terminal_absolute"]),
                    "terminal_dig": float(row["terminal_dig"]),
                    "terminal_dump_purity": float(row["terminal_dump_purity"]),
                    "terminal_illegal_dump_volume": float(
                        row["terminal_illegal_dump_volume"]
                    ),
                    "terminal_unloaded": bool(row["terminal_unloaded"]),
                    "termination_reason": row["termination_reason"],
                    "integrity_failure": bool(row["integrity_failure"]),
                }
            )
        summary = final["summary"]
        panels[panel] = {
            "input_path": str(path),
            "input_sha256": observed_sha,
            "manifest_sha256": final["manifest_sha256"],
            "episodes": len(per_map),
            "exact_successes": exact,
            "exact_rate": exact / len(per_map),
            "foundation_exact": summary["by_family"]["foundation"]["successes"],
            "trench_exact": summary["by_family"]["trench"]["successes"],
            "macro_graded_completion": summary["graded"]["macro_completion"],
            "micro_graded_completion": summary["graded"]["micro"]["mean"],
            "checks": checks,
        }
    rows.sort(key=lambda row: (row["panel"], row["slot_index"]))
    rows_sha = write_jsonl(output / "d0_per_identity.jsonl", rows)
    receipt = {
        "schema": SCHEMA,
        "receipt": "D0_fixed_eval_integrity",
        "status": "passed",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "panels": panels,
        "per_identity_rows": "d0_per_identity.jsonl",
        "per_identity_rows_sha256": rows_sha,
    }
    write_json(output / "d0_receipt.json", receipt)
    return receipt


def terminal_return(
    *,
    success: bool,
    steps: int,
    gamma: float,
    success_bonus: float,
    failure_penalty: float,
    step_cost_total: float,
    phi_reset: float,
    phi_terminal: float,
    horizon: int,
) -> float:
    if not 1 <= steps <= horizon:
        raise ValueError("steps must lie within the frozen horizon")
    per_step = step_cost_total / horizon
    step_sum = per_step * (1.0 - gamma**steps) / (1.0 - gamma)
    terminal = (
        gamma ** (steps - 1) * success_bonus
        if success
        else -(gamma ** (steps - 1)) * failure_penalty
    )
    return terminal - step_sum - phi_reset + gamma**steps * phi_terminal


def build_dominance(
    output: Path,
    *,
    distance_bound: float,
    gamma: float,
    success_bonus: float,
    failure_penalty: float,
    alpha: float,
    beta: float,
    step_cost_total: float,
    horizon: int,
) -> dict[str, Any]:
    phi_reset = beta * distance_bound
    phi_success_min = alpha + beta * 0.0
    # Phi = alpha Q + beta(P + D_bound): exact success has Q=1 and P>=0.
    phi_success_min += beta * distance_bound
    phi_failure_max = alpha + beta * (distance_bound + distance_bound)
    successes = [
        terminal_return(
            success=True,
            steps=steps,
            gamma=gamma,
            success_bonus=success_bonus,
            failure_penalty=failure_penalty,
            step_cost_total=step_cost_total,
            phi_reset=phi_reset,
            phi_terminal=phi_success_min,
            horizon=horizon,
        )
        for steps in range(1, horizon + 1)
    ]
    failure = terminal_return(
        success=False,
        steps=horizon,
        gamma=gamma,
        success_bonus=success_bonus,
        failure_penalty=failure_penalty,
        step_cost_total=step_cost_total,
        phi_reset=phi_reset,
        phi_terminal=phi_failure_max,
        horizon=horizon,
    )
    min_success = min(successes)
    min_success_step = successes.index(min_success) + 1
    margin = min_success - failure

    # Solve the exact affine inequality at the worst success step.
    base_without_bonus = terminal_return(
        success=True,
        steps=min_success_step,
        gamma=gamma,
        success_bonus=0.0,
        failure_penalty=failure_penalty,
        step_cost_total=step_cost_total,
        phi_reset=phi_reset,
        phi_terminal=phi_success_min,
        horizon=horizon,
    )
    bonus_coefficient = gamma ** (min_success_step - 1)
    minimum_bonus_strict_threshold = (failure - base_without_bonus) / bonus_coefficient

    dwell_rows = []
    for q in (0.0, 0.5, 1.0):
        for p in (-distance_bound, 0.0, distance_bound):
            phi = alpha * q + beta * (p + distance_bound)
            dwell_rows.append(
                {
                    "Q": q,
                    "P": p,
                    "Phi": phi,
                    "implicit_dwell_cost_per_step": (1.0 - gamma) * phi,
                    "explicit_step_cost_per_step": step_cost_total / horizon,
                }
            )
    receipt = {
        "schema": SCHEMA,
        "receipt": "analytic_terminal_dominance",
        "status": "passed" if margin > 0.0 else "failed",
        "constants": {
            "distance_bound": distance_bound,
            "potential_gamma": gamma,
            "success_bonus": success_bonus,
            "failure_penalty": failure_penalty,
            "alpha": alpha,
            "beta": beta,
            "step_cost_total_over_horizon": step_cost_total,
            "horizon": horizon,
        },
        "bounds": {
            "Q": [0.0, 1.0],
            "P": [-distance_bound, distance_bound],
            "phi_reset": phi_reset,
            "phi_success_min": phi_success_min,
            "phi_failure_max": phi_failure_max,
        },
        "enumeration": {
            "success_steps": [1, horizon],
            "minimum_success_return": min_success,
            "minimum_success_step": min_success_step,
            "success_return_at_step_1": successes[0],
            "maximum_failure_return": failure,
            "failure_step": horizon,
            "strict_margin": margin,
            "minimum_success_bonus_strict_threshold": minimum_bonus_strict_threshold,
        },
        "inequality": (
            "min_{T=1..450}[gamma^(T-1) B - c/450 sum_{t=0}^{T-1} "
            "gamma^t - Phi_reset + gamma^T Phi_success_min] > "
            "[-gamma^449 F - c/450 sum_{t=0}^{449} gamma^t - Phi_reset "
            "+ gamma^450 Phi_failure_max]"
        ),
        "dwell_grid": dwell_rows,
    }
    if receipt["status"] != "passed":
        raise ValueError(f"terminal dominance failed: margin={margin}")
    write_json(output / "dominance_receipt.json", receipt)
    return receipt


def build_distance_and_d4b(
    bank_root: Path,
    output: Path,
    *,
    tile_size_m: float,
    distance_ref_m: float,
    distance_bound: float,
    relocation_progress_mult: float,
    dump_correct: float,
    reward_normalizer: float,
    success_bonus_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bank_index_path = bank_root / "dataset.json"
    if sha256_file(bank_index_path) != EXPECTED_BANK_FILE_SHA256:
        raise ValueError("accepted bank dataset.json hash does not match frozen V8")
    index = json.loads(bank_index_path.read_text())
    if index.get("schema") != BASE_DATASET_SCHEMA:
        raise ValueError("unexpected accepted-bank index schema")
    if index.get("source_registry_sha256") != EXPECTED_SOURCE_REGISTRY_SHA256:
        raise ValueError("source-registry hash mismatch")
    if index.get("environment_protocol_sha256") != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256:
        raise ValueError("environment-protocol hash mismatch")

    sidecar = output / "canonical_distance_v1"
    sidecar.mkdir()
    distance_protocol = {
        "schema": "terra_distance_protocol_v1",
        "distance_protocol_id": DISTANCE_PROTOCOL_ID,
        "distance_metric": "obstacle_geodesic_8_physical_metres",
        "neighbor_costs_tiles": {"cardinal": 1.0, "diagonal": math.sqrt(2.0)},
        "diagonal_corner_crossing": "permitted_when_destination_is_traversable",
        "accepted_sources": "(target > 0) AND (NOT occupancy)",
        "traversable": "NOT occupancy",
        "obstacle_output_value": 0.0,
        "distance_normalization": "global_reference_metres",
        "tile_size_m": tile_size_m,
        "distance_ref_m": distance_ref_m,
        "distance_bound": distance_bound,
        "clipping": False,
        "out_of_bound_policy": "reject_dataset_release",
        "implementation": {
            "library": "scikit-image",
            "version": skimage_version,
            "routine": "skimage.graph.MCP_Geometric(costs, fully_connected=True)",
            "input_costs": "1.0 traversable, +inf occupancy",
            "output_dtype": "float32",
        },
    }
    write_json(sidecar / "distance_protocol.json", distance_protocol)

    all_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    all_max_tiles: list[float] = []
    dig_max_tiles: list[float] = []
    dig_mean_tiles: list[float] = []
    dig_median_tiles: list[float] = []
    global_max: dict[str, Any] | None = None
    dig_global_max: dict[str, Any] | None = None
    dig_mean_global_max: dict[str, Any] | None = None

    dataset_entries = enumerate_datasets(index)
    for entry in dataset_entries:
        dataset = bank_root / entry.relative_path
        metadata = json.loads((dataset / "dataset.json").read_text())
        if metadata.get("schema") != EXACT_DATASET_SCHEMA:
            raise ValueError(f"{dataset}: unexpected exact-dataset schema")
        if metadata.get("slot_count") != entry.slot_count:
            raise ValueError(f"{dataset}: slot-count mismatch")
        if metadata.get("scenario_identity_contract") != IDENTITY_CONTRACT:
            raise ValueError(f"{dataset}: unexpected identity contract")
        manifest = read_jsonl(dataset / "manifest.jsonl")
        if len(manifest) != entry.slot_count:
            raise ValueError(f"{dataset}: manifest count mismatch")
        if [row["slot_index"] for row in manifest] != list(
            range(1, entry.slot_count + 1)
        ):
            raise ValueError(f"{dataset}: non-contiguous manifest slots")

        derived_dataset = sidecar / entry.relative_path
        distance_folder = derived_dataset / "distance"
        distance_folder.mkdir(parents=True)
        derived_manifest: list[dict[str, Any]] = []
        for manifest_row in manifest:
            slot = int(manifest_row["slot_index"])
            arrays = load_arrays(dataset, slot)
            legacy_reset_sha = array_set_sha256(arrays, RESET_ARRAY_FOLDERS)
            if legacy_reset_sha != manifest_row["scenario_id"]:
                raise ValueError(
                    f"{dataset} slot {slot}: reset arrays do not match scenario_id"
                )
            physical_sha = array_set_sha256(arrays, PHYSICAL_ARRAY_FOLDERS)
            target = np.squeeze(arrays["images"])
            occupancy = np.asarray(np.squeeze(arrays["occupancy"]), dtype=bool)
            distance_tiles = canonical_distance_tiles(target, occupancy)
            distance_m = distance_tiles * tile_size_m
            normalized = np.asarray(distance_m / distance_ref_m, dtype=np.float32)
            traversable = np.logical_not(occupancy)
            observed_bound = float(np.max(normalized[traversable]))
            if observed_bound > distance_bound:
                raise ValueError(
                    f"{dataset} slot {slot}: distance {observed_bound} exceeds "
                    f"bound {distance_bound}"
                )
            output_array = distance_folder / f"img_{slot}.npy"
            np.save(output_array, normalized, allow_pickle=False)
            derived_arrays = dict(arrays)
            derived_arrays["distance"] = normalized
            derived_reset_sha = array_set_sha256(derived_arrays, RESET_ARRAY_FOLDERS)
            accepted = np.logical_and(target > 0, traversable)
            dig = np.logical_and(target < 0, traversable)
            dig_values_tiles = distance_tiles[dig]
            if dig_values_tiles.size == 0:
                raise ValueError(f"{dataset} slot {slot}: no required dig cells")
            stats = {
                "max_traversable_distance_tiles": float(
                    distance_tiles[traversable].max()
                ),
                "max_traversable_distance_m": float(distance_m[traversable].max()),
                "max_normalized_distance": observed_bound,
                "max_initial_dig_distance_tiles": float(dig_values_tiles.max()),
                "max_initial_dig_distance_m": float(
                    dig_values_tiles.max() * tile_size_m
                ),
                "mean_initial_dig_distance_tiles": float(dig_values_tiles.mean()),
                "mean_initial_dig_distance_m": float(
                    dig_values_tiles.mean() * tile_size_m
                ),
                "median_initial_dig_distance_tiles": float(np.median(dig_values_tiles)),
                "accepted_dump_cells": int(accepted.sum()),
                "required_dig_volume": int((-target[dig]).sum()),
            }
            row = {
                "group": entry.group,
                "split": entry.split,
                "dataset_relative_path": entry.relative_path,
                "slot_index": slot,
                "condition_id": manifest_row["primary_cell"],
                "family": manifest_row["family"],
                "map_id": manifest_row["map_id"],
                "source_id": manifest_row["source_id"],
                "legacy_scenario_id": manifest_row["scenario_id"],
                "episode_id": manifest_row.get("episode_id"),
                "reset_seed": manifest_row.get("reset_seed"),
                "legacy_reset_arrays_sha256": legacy_reset_sha,
                "physical_reset_arrays_sha256": physical_sha,
                "canonical_distance_npy": str(output_array.relative_to(sidecar)),
                "canonical_distance_npy_sha256": sha256_file(output_array),
                "derived_reset_arrays_sha256": derived_reset_sha,
                **stats,
            }
            all_rows.append(row)
            derived_manifest.append(row)
            all_max_tiles.append(stats["max_traversable_distance_tiles"])
            dig_max_tiles.append(stats["max_initial_dig_distance_tiles"])
            dig_mean_tiles.append(stats["mean_initial_dig_distance_tiles"])
            dig_median_tiles.append(stats["median_initial_dig_distance_tiles"])
            if (
                global_max is None
                or stats["max_traversable_distance_tiles"]
                > global_max["max_traversable_distance_tiles"]
            ):
                global_max = row
            if (
                dig_global_max is None
                or stats["max_initial_dig_distance_tiles"]
                > dig_global_max["max_initial_dig_distance_tiles"]
            ):
                dig_global_max = row
            if (
                dig_mean_global_max is None
                or stats["mean_initial_dig_distance_tiles"]
                > dig_mean_global_max["mean_initial_dig_distance_tiles"]
            ):
                dig_mean_global_max = row

            if entry.group == "train":
                legacy_distance = np.asarray(arrays["distance"], dtype=np.float64)
                v0 = int((-target[dig]).sum())
                h_reset = float(np.sum((-target[dig]) * legacy_distance[dig]))
                target_scale = float(np.clip(170.0 / max(v0, 1), 2.0, 5.0) / 2.0)
                projected = (
                    h_reset
                    * relocation_progress_mult
                    * target_scale
                    * dump_correct
                    / reward_normalizer
                )
                train_rows.append(
                    {
                        "condition_id": manifest_row["primary_cell"],
                        "family": manifest_row["family"],
                        "map_id": manifest_row["map_id"],
                        "source_id": manifest_row["source_id"],
                        "scenario_id": manifest_row["scenario_id"],
                        "dataset_relative_path": entry.relative_path,
                        "slot_index": slot,
                        "required_dig_volume": v0,
                        "legacy_h_reset": h_reset,
                        "legacy_target_scale": target_scale,
                        "legacy_projected_relocation_return": projected,
                        "canonical_h_reset_over_v0": float(
                            np.sum((-target[dig]) * normalized[dig]) / v0
                        ),
                        **stats,
                    }
                )

        write_jsonl(derived_dataset / "manifest.jsonl", derived_manifest)
        write_json(
            derived_dataset / "dataset.json",
            {
                "schema": DISTANCE_SIDECAR_SCHEMA,
                "distance_protocol_id": DISTANCE_PROTOCOL_ID,
                "distance_protocol_path_from_sidecar_root": "distance_protocol.json",
                "base_bank_dataset_relative_path": entry.relative_path,
                "base_dataset_json_sha256": sha256_file(dataset / "dataset.json"),
                "base_manifest_sha256": sha256_file(dataset / "manifest.jsonl"),
                "slot_count": entry.slot_count,
                "shape": [64, 64],
                "distance_metric": distance_protocol["distance_metric"],
                "distance_normalization": distance_protocol["distance_normalization"],
                "tile_size_m": tile_size_m,
                "distance_ref_m": distance_ref_m,
                "distance_bound": distance_bound,
                "clipping": False,
                "identity_semantics": {
                    "legacy_scenario_id_preserved_as_logical_map_identity": True,
                    "physical_reset_arrays_must_match_base": True,
                    "derived_reset_arrays_sha256_changes_only_with_distance": True,
                },
            },
        )

    all_rows.sort(
        key=lambda row: (
            row["group"],
            row["dataset_relative_path"],
            row["slot_index"],
        )
    )
    train_rows.sort(
        key=lambda row: (row["condition_id"], row["slot_index"], row["map_id"])
    )
    rows_sha = write_jsonl(sidecar / "rows.jsonl", all_rows)
    train_rows_sha = write_jsonl(output / "d4b_train_scale_rows.jsonl", train_rows)

    high = [
        row
        for row in train_rows
        if row["legacy_projected_relocation_return"] > success_bonus_threshold
    ]
    high_ids = [
        {
            "condition_id": row["condition_id"],
            "map_id": row["map_id"],
            "scenario_id": row["scenario_id"],
            "legacy_projected_relocation_return": row[
                "legacy_projected_relocation_return"
            ],
            "required_dig_volume": row["required_dig_volume"],
        }
        for row in high
    ]
    high_ids.sort(key=lambda row: (row["condition_id"], row["map_id"]))
    high_ids_sha = write_jsonl(output / "d4b_high_budget_34.jsonl", high_ids)
    high_counts = Counter(row["condition_id"] for row in high)
    budgets = [row["legacy_projected_relocation_return"] for row in train_rows]
    foundation_budgets = [
        row["legacy_projected_relocation_return"]
        for row in train_rows
        if row["family"] == "foundation"
    ]
    trench_budgets = [
        row["legacy_projected_relocation_return"]
        for row in train_rows
        if row["family"] == "trench"
    ]
    if len(high_ids) != 34:
        raise ValueError(f"expected 34 high-budget maps, got {len(high_ids)}")
    expected_counts = {
        "v7-fnd-bearing-walls-adjacent": 9,
        "v7-fnd-courtyard-adjacent": 1,
        "v7-fnd-courtyard-pads-adjacent": 2,
        "v7-fnd-irregular-adjacent": 3,
        "v7-fnd-slab-adjacent": 19,
    }
    if dict(sorted(high_counts.items())) != expected_counts:
        raise ValueError(f"unexpected high-budget condition counts: {high_counts}")

    dataset_counts = Counter(row["group"] for row in all_rows)
    if dataset_counts != {
        "train": 4512,
        "main_evaluation": 2880,
        "capability_evaluation": 128,
    }:
        raise ValueError(f"unexpected dataset counts: {dataset_counts}")

    distance_receipt = {
        "schema": DISTANCE_SIDECAR_SCHEMA,
        "status": "passed",
        "base_bank_root": str(bank_root),
        "base_bank_dataset_json_sha256": EXPECTED_BANK_FILE_SHA256,
        "base_bank_source_registry_sha256": EXPECTED_SOURCE_REGISTRY_SHA256,
        "base_environment_protocol_sha256": EXPECTED_ENVIRONMENT_PROTOCOL_SHA256,
        "distance_protocol": "distance_protocol.json",
        "distance_protocol_sha256": sha256_file(sidecar / "distance_protocol.json"),
        "rows": "rows.jsonl",
        "rows_sha256": rows_sha,
        "datasets": len(dataset_entries),
        "scenarios": len(all_rows),
        "scenario_counts": dict(sorted(dataset_counts.items())),
        "physical_identity_contract": (
            "map_id/source_id/logical legacy_scenario_id and images/occupancy/"
            "dumpability/actions are byte-identical to the base bank; only distance "
            "and therefore derived_reset_arrays_sha256 differ"
        ),
        "distance_statistics_tiles": {
            "max_traversable": quantiles(all_max_tiles),
            "max_initial_dig": quantiles(dig_max_tiles),
            "mean_initial_dig": quantiles(dig_mean_tiles),
            "median_initial_dig": quantiles(dig_median_tiles),
        },
        "observed_global_max": global_max,
        "observed_initial_dig_max": dig_global_max,
        "observed_initial_dig_mean_max": dig_mean_global_max,
        "observed_max_normalized": max(
            row["max_normalized_distance"] for row in all_rows
        ),
        "bound_headroom_fraction": 1.0
        - max(row["max_normalized_distance"] for row in all_rows) / distance_bound,
    }
    write_json(sidecar / "dataset.json", distance_receipt)

    d4b = {
        "schema": SCHEMA,
        "receipt": "D4b_static_scale_overlap_and_distance_bound",
        "status": "passed",
        "legacy_formula": ("H_reset * 1.5 * clip(170 / max(V0,1),2,5) / 2 / 70"),
        "legacy_success_bonus_threshold": success_bonus_threshold,
        "train_scenarios": len(train_rows),
        "train_scale_rows": "d4b_train_scale_rows.jsonl",
        "train_scale_rows_sha256": train_rows_sha,
        "legacy_projected_budget": {
            "min": min(budgets),
            "max": max(budgets),
            "spread_ratio": max(budgets) / min(budgets),
            "foundation_median": float(np.median(foundation_budgets)),
            "trench_median": float(np.median(trench_budgets)),
        },
        "above_success_bonus": {
            "count": len(high_ids),
            "condition_counts": dict(sorted(high_counts.items())),
            "rows": "d4b_high_budget_34.jsonl",
            "rows_sha256": high_ids_sha,
        },
        "canonical_distance_sidecar": "canonical_distance_v1/dataset.json",
        "canonical_distance_sidecar_sha256": sha256_file(sidecar / "dataset.json"),
    }
    write_json(output / "d4b_receipt.json", d4b)
    return distance_receipt, d4b


def receipt_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    tree = hashlib.sha256()
    for row in files:
        tree.update(canonical_json(row).encode())
        tree.update(b"\n")
    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "files_excluding_this_manifest": files,
        "tree_sha256": tree.hexdigest(),
    }
    write_json(output / "receipt_manifest.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--development-eval", type=Path, required=True)
    parser.add_argument("--promotion-eval", type=Path, required=True)
    parser.add_argument("--capability-development-eval", type=Path, required=True)
    parser.add_argument("--capability-promotion-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size-m", type=float, default=0.571428571428125)
    parser.add_argument("--distance-ref-m", type=float, default=16.0)
    parser.add_argument("--distance-bound", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=0.9984)
    parser.add_argument("--success-bonus", type=float, default=6.0)
    parser.add_argument("--failure-penalty", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--step-cost-total", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=450)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    eval_paths = {
        "development": args.development_eval.resolve(),
        "promotion": args.promotion_eval.resolve(),
        "capability_development": args.capability_development_eval.resolve(),
        "capability_promotion": args.capability_promotion_eval.resolve(),
    }
    build_d0(eval_paths, args.output_dir)
    build_distance_and_d4b(
        args.bank_root.resolve(),
        args.output_dir,
        tile_size_m=args.tile_size_m,
        distance_ref_m=args.distance_ref_m,
        distance_bound=args.distance_bound,
        relocation_progress_mult=1.5,
        dump_correct=1.0,
        reward_normalizer=70.0,
        success_bonus_threshold=200.0 * 1.2 * 2.0 / 70.0,
    )
    build_dominance(
        args.output_dir,
        distance_bound=args.distance_bound,
        gamma=args.gamma,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        alpha=args.alpha,
        beta=args.beta,
        step_cost_total=args.step_cost_total,
        horizon=args.horizon,
    )
    receipt_manifest(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
