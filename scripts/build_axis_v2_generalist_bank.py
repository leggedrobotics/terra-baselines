#!/usr/bin/env python3
"""Build the one axis-v2 generalist bank used by the 8-GPU experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import terra

MAP_GENERATION_ROOT = Path(terra.__file__).resolve().parents[1] / "tools/map_generation"
if str(MAP_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MAP_GENERATION_ROOT))

from terra.benchmark_protocol import frozen_environment_protocol
from terra.config import REWARD_V2_DISTANCE_BOUND, REWARD_V2_DISTANCE_REF_M
from terra.env_generation.distance import (
    REWARD_V2_DISTANCE_METRIC,
    REWARD_V2_DISTANCE_NORMALIZATION,
    REWARD_V2_DISTANCE_PROTOCOL_ID,
    compute_reward_v2_distance_map,
)
from terra.maps_buffer import (
    EXACT_DATASET_SCHEMA,
    RESET_ARRAY_FOLDERS,
    RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
    reset_array_scenario_sha256,
    trench_axis_contract_sanity_check,
    trench_axis_owners_sanity_check,
    validate_exact_dataset_contract,
)
from tools.audit_trench_alignment_feasibility import (
    analyze_case,
    env_config,
    terra_geometry,
)
from tools.map_generation.generate_prototypes_v9 import (
    TRENCH_AXIS_CONTRACT,
    sha256_mask,
    trench_axis_owners,
)
from tools.map_generation.materialize_loader_bank import (
    _exact_reset_seeds,
    episode_id,
    materialize_loader_bank,
)
from tools.map_generation.materialize_splits import materialize_splits

from utils.accepted_bank import AXIS_V2_RELEASE_ID


ARRAYS = ("images", "occupancy", "dumpability", "actions", "distance")
PHYSICAL_ARRAYS = ("images", "occupancy", "dumpability", "actions")
SPLITS = ("development", "promotion", "sealed")
SPLIT_COUNTS = {"train": 96, "promotion": 16, "development": 16, "sealed": 32}
CAPABILITY_IDS = ("fnd-slab-allfree", "trn-straight-allfree")
CORE_IDS = (
    "v7-fnd-slab-adjacent",
    "v7-fnd-irregular-adjacent",
    "v7-fnd-courtyard-adjacent",
    "v7-fnd-bearing-walls-adjacent",
    "v7-fnd-pads-adjacent",
    "v7-fnd-courtyard-pads-adjacent",
)
SCHEMA = "terra_axis_v2_generalist_bank_build_v1"
AUDIT_SCHEMA = "terra_axis_v2_generalist_bank_audit_v1"
REVIEW_SCHEMA = "terra_axis_v2_review_admission_v1"
MIXTURE_SCHEMA = "terra_axis_v2_training_mixture_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def validate_terra_checkout(root: Path, revision: str) -> None:
    root = root.resolve()
    imported = Path(terra.__file__).resolve().parents[1]
    if imported != root:
        raise RuntimeError(f"imported Terra root {imported} does not match {root}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if head != revision or status:
        raise RuntimeError(
            f"Terra must be clean at {revision}; observed head={head}, dirty={bool(status)}"
        )


def clean_git_revision(root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError(f"builder checkout must be committed and clean: {root}")
    return revision


def source_levels(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index = json.loads((root / "dataset.json").read_text())
    levels = {row["condition_id"]: row for row in index["train"]}
    if len(levels) != len(index["train"]):
        raise RuntimeError(f"{root}: duplicate training condition")
    return levels, index


def panel_entries(
    root: Path,
    index: dict[str, Any],
    panel_key: str,
    split: str,
    conditions: set[str],
) -> list[tuple[Path, dict[str, Any]]]:
    descriptor = index[panel_key][split]
    directory = root / descriptor["maps_path"]
    rows = [
        row
        for row in read_jsonl(directory / "manifest.jsonl")
        if row["primary_cell"] in conditions
    ]
    observed = {row["primary_cell"] for row in rows}
    if observed != conditions:
        raise RuntimeError(
            f"{directory}: panel support mismatch; expected={sorted(conditions)}, "
            f"observed={sorted(observed)}"
        )
    return [(directory, row) for row in rows]


def branch_depths(graph_path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    graph = json.loads(graph_path.read_text())
    if graph.get("release_id") != AXIS_V2_RELEASE_ID:
        raise RuntimeError(f"{graph_path}: wrong release")
    names = ("Anchor", "Nearby core", "Composed")
    depths: dict[str, str] = {}
    for depth, name in enumerate(names):
        for family in ("foundation", "trench"):
            for condition in graph["depths"][str(depth)][family]:
                if condition in depths:
                    raise RuntimeError(f"{graph_path}: duplicate {condition}")
                depths[condition] = name
    constraints = tuple(
        sorted(
            graph["depths"]["2"]["foundation"]
            + graph["depths"]["2"]["trench"]
        )
    )
    if len(depths) != 40 or len(constraints) != 32:
        raise RuntimeError(f"{graph_path}: expected 40 total and 32 constraints")
    return depths, constraints


def source_item(
    source_root: Path,
    source_row: dict[str, Any],
    condition: str,
    family: str,
) -> dict[str, Any]:
    return {
        "directory": source_root,
        "row": source_row,
        "condition": condition,
        "family": family,
    }


def copy_item(
    item: dict[str, Any],
    destination: Path,
    slot: int,
    *,
    split: str,
    protocol_sha256: str,
    reset_seed: int | None,
    tile_size_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = item["directory"]
    source_row = item["row"]
    source_slot = int(source_row["slot_index"])
    arrays = {
        name: np.ascontiguousarray(
            np.squeeze(
                np.load(source / name / f"img_{source_slot}.npy", allow_pickle=False)
            )
        )
        for name in PHYSICAL_ARRAYS
    }
    shapes = {array.shape for array in arrays.values()}
    if shapes != {(64, 64)}:
        raise RuntimeError(f"{source}:{source_slot}: unexpected shapes {shapes}")

    metadata = json.loads(
        (source / "metadata" / f"trench_{source_slot}.json").read_text()
    )
    owner_path = source / "trench_axis_owners" / f"img_{source_slot}.npy"
    if owner_path.is_file():
        owners = np.ascontiguousarray(np.squeeze(np.load(owner_path, allow_pickle=False)))
    elif item["family"] == "foundation":
        owners = np.zeros((64, 64), dtype=np.uint8)
    elif item["condition"] == "trn-straight-allfree":
        metadata["trench_arms"] = metadata.pop("trench_segments_yx")
        owners = trench_axis_owners(arrays["images"], metadata)
        metadata["trench_axis_contract"] = TRENCH_AXIS_CONTRACT
        metadata["trench_axis_owners_sha256"] = sha256_mask(owners)
    else:
        raise RuntimeError(
            f"{item['condition']} has no generator-owned trench-axis sidecar"
        )

    owners = np.ascontiguousarray(owners, dtype=np.uint8)
    axis_count = len(metadata.get("axes_ABC", []) or [])
    trench_type = axis_count if axis_count else -1
    trench_axis_owners_sanity_check(
        arrays["images"], owners, trench_type, max_trench_type=4
    )
    trench_axis_contract_sanity_check(metadata, owners, trench_type)

    distance = compute_reward_v2_distance_map(
        arrays["images"],
        arrays["occupancy"],
        tile_size_m=tile_size_m,
        distance_ref_m=REWARD_V2_DISTANCE_REF_M,
        distance_bound=REWARD_V2_DISTANCE_BOUND,
    )
    arrays["distance"] = np.ascontiguousarray(distance, dtype=np.float32)
    scenario_id = reset_array_scenario_sha256(arrays)

    for name in (*ARRAYS, "trench_axis_owners", "metadata"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    for name in PHYSICAL_ARRAYS:
        shutil.copy2(
            source / name / f"img_{source_slot}.npy",
            destination / name / f"img_{slot}.npy",
        )
    np.save(destination / "distance" / f"img_{slot}.npy", arrays["distance"])
    np.save(destination / "trench_axis_owners" / f"img_{slot}.npy", owners)
    write_json(destination / "metadata" / f"trench_{slot}.json", metadata)

    output_row = dict(source_row)
    output_row.update(
        {
            "slot_index": slot,
            "scenario_id": scenario_id,
            "split": split,
            "family": item["family"],
            "primary_cell": item["condition"],
        }
    )
    for field in ("reset_seed", "episode_id", "environment_protocol_sha256"):
        output_row.pop(field, None)
    if reset_seed is not None:
        output_row.update(
            {
                "reset_seed": reset_seed,
                "episode_id": episode_id(scenario_id, reset_seed, protocol_sha256),
                "environment_protocol_sha256": protocol_sha256,
            }
        )
    identity = {
        "maps_path": None,
        "slot_index": slot,
        "map_id": output_row["map_id"],
        "scenario_id": scenario_id,
        "source_id": output_row["source_id"],
        "split": split,
        "family": item["family"],
        "primary_cell": item["condition"],
        "owner_sha256": sha256_mask(owners),
        "distance_npy_sha256": sha256_file(
            destination / "distance" / f"img_{slot}.npy"
        ),
        "axis_count": axis_count,
        "multi_axis_target_cells": int(
            np.count_nonzero((owners != 0) & ((owners & (owners - 1)) != 0))
        ),
    }
    return output_row, identity


def materialize_dataset(
    root: Path,
    relative: str,
    items: list[dict[str, Any]],
    *,
    split: str,
    protocol_sha256: str,
    tile_size_m: float,
    evaluation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    destination = root / relative
    items = sorted(items, key=lambda item: (item["condition"], item["row"]["map_id"]))
    seeds = _exact_reset_seeds(len(items)) if evaluation else [None] * len(items)
    rows = []
    identities = []
    for slot, (item, seed) in enumerate(zip(items, seeds), start=1):
        row, identity = copy_item(
            item,
            destination,
            slot,
            split=split,
            protocol_sha256=protocol_sha256,
            reset_seed=seed,
            tile_size_m=tile_size_m,
        )
        identity["maps_path"] = relative
        rows.append(row)
        identities.append(identity)
    write_jsonl(destination / "manifest.jsonl", rows)
    return rows, identities


def dataset_metadata(
    root: Path,
    relative: str,
    rows: list[dict[str, Any]],
    registry_sha256: str,
    tile_size_m: float,
) -> None:
    directory = root / relative
    write_json(
        directory / "dataset.json",
        {
            "schema": EXACT_DATASET_SCHEMA,
            "slot_count": len(rows),
            "unique_identity_count": len({row["map_id"] for row in rows}),
            "shape": [64, 64],
            "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
            "distance_metric": REWARD_V2_DISTANCE_METRIC,
            "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
            "tile_size_m": tile_size_m,
            "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
            "distance_bound": REWARD_V2_DISTANCE_BOUND,
            "accepted_dump_contract": "exact_visible_dump_v1",
            "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
            "source_registry": os.path.relpath(root / "source_registry.jsonl", directory),
            "source_registry_sha256": registry_sha256,
        },
    )
    validate_exact_dataset_contract(directory, len(rows))


def build_bank(
    candidates: Path,
    legacy: Path,
    reviewed_v6: Path,
    output: Path,
    terra_revision: str,
    graph_path: Path,
) -> dict[str, Any]:
    for path in (candidates, legacy, reviewed_v6, graph_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)

    depths, constraint_ids = branch_depths(graph_path)
    builder_revision = clean_git_revision(Path(__file__).resolve().parents[1])
    protocol = frozen_environment_protocol(terra_revision)
    protocol_sha256 = protocol["environment_protocol_sha256"]
    tile_size_m = float(protocol["map"]["tile_size_m_derived_float64"])
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.building-", dir=output.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        split_root = temporary / "split"
        loader_root = temporary / "loader32"
        root = temporary / "bank"
        materialize_splits(
            candidates / "manifest.csv",
            candidates / "dataset",
            split_root,
            SPLIT_COUNTS,
        )
        materialize_loader_bank(
            split_root,
            loader_root,
            terra_revision,
            reviewed_v6,
        )

        new_levels, new_index = source_levels(loader_root)
        legacy_levels, legacy_index = source_levels(legacy)
        if set(new_levels) != set(constraint_ids):
            raise RuntimeError("materialized V6 support does not match the axis-v2 graph")
        legacy_needed = set(CAPABILITY_IDS) | set(CORE_IDS)
        if not legacy_needed <= set(legacy_levels):
            raise RuntimeError("legacy bank is missing axis-v2 anchor/foundation support")
        if float(legacy_index["tile_size_m"]) != tile_size_m:
            raise RuntimeError("legacy and axis-v2 tile sizes differ")

        all_conditions = tuple(sorted(set(constraint_ids) | legacy_needed))
        train_rows: dict[str, list[dict[str, Any]]] = {}
        all_rows: dict[str, list[dict[str, Any]]] = {}
        identities: list[dict[str, Any]] = []
        training_descriptors = []
        for level_index, condition in enumerate(all_conditions):
            if condition in new_levels:
                source_bank, descriptor = loader_root, new_levels[condition]
            else:
                source_bank, descriptor = legacy, legacy_levels[condition]
            family = descriptor["family"]
            source_directory = source_bank / descriptor["maps_path"]
            items = [
                source_item(source_directory, row, condition, family)
                for row in read_jsonl(source_directory / "manifest.jsonl")
            ]
            if len(items) != 96:
                raise RuntimeError(f"{condition}: expected 96 training maps")
            relative = f"train/{level_index:03d}__{condition}"
            rows, item_identities = materialize_dataset(
                root,
                relative,
                items,
                split="train",
                protocol_sha256=protocol_sha256,
                tile_size_m=tile_size_m,
                evaluation=False,
            )
            train_rows[relative] = rows
            all_rows[relative] = rows
            identities.extend(item_identities)
            training_descriptors.append(
                {
                    "level_index": level_index,
                    "condition_id": condition,
                    "family": family,
                    "branch_depth": depths[condition],
                    "maps_path": relative,
                    "map_count": len(rows),
                }
            )

        evaluation_panels = {}
        capability_panels = {}
        main_conditions = set(constraint_ids) | set(CORE_IDS)
        for split in SPLITS:
            new_items = [
                source_item(directory, row, row["primary_cell"], row["family"])
                for directory, row in panel_entries(
                    loader_root,
                    new_index,
                    "evaluation_panels",
                    split,
                    set(constraint_ids),
                )
            ]
            core_items = [
                source_item(directory, row, row["primary_cell"], row["family"])
                for directory, row in panel_entries(
                    legacy,
                    legacy_index,
                    "evaluation_panels",
                    split,
                    set(CORE_IDS),
                )
            ]
            relative = f"evaluation/main/{split}"
            rows, item_identities = materialize_dataset(
                root,
                relative,
                new_items + core_items,
                split=split,
                protocol_sha256=protocol_sha256,
                tile_size_m=tile_size_m,
                evaluation=True,
            )
            all_rows[relative] = rows
            identities.extend(item_identities)
            evaluation_panels[split] = {
                "maps_path": relative,
                "slot_count": len(rows),
                "conditions": len(main_conditions),
            }

            capability_items = [
                source_item(directory, row, row["primary_cell"], row["family"])
                for directory, row in panel_entries(
                    legacy,
                    legacy_index,
                    "capability_floor_evaluation_panels",
                    split,
                    set(CAPABILITY_IDS),
                )
            ]
            relative = f"evaluation/capability_floor/{split}"
            rows, item_identities = materialize_dataset(
                root,
                relative,
                capability_items,
                split=split,
                protocol_sha256=protocol_sha256,
                tile_size_m=tile_size_m,
                evaluation=True,
            )
            all_rows[relative] = rows
            identities.extend(item_identities)
            capability_panels[split] = {
                "maps_path": relative,
                "slot_count": len(rows),
                "conditions": len(CAPABILITY_IDS),
            }

        map_ids = [row["map_id"] for rows in all_rows.values() for row in rows]
        scenario_ids = [
            row["scenario_id"] for rows in all_rows.values() for row in rows
        ]
        if len(map_ids) != len(set(map_ids)) or len(scenario_ids) != len(
            set(scenario_ids)
        ):
            raise RuntimeError("combined bank has duplicate map or scenario identity")
        source_splits: dict[str, set[str]] = {}
        for identity in identities:
            source_splits.setdefault(identity["source_id"], set()).add(identity["split"])
        leaked_sources = {
            source: sorted(splits)
            for source, splits in source_splits.items()
            if len(splits) > 1
        }
        if leaked_sources:
            first = sorted(leaked_sources)[0]
            raise RuntimeError(f"source leakage across splits: {first} -> {leaked_sources[first]}")

        registry = [
            {
                key: identity[key]
                for key in (
                    "map_id",
                    "scenario_id",
                    "source_id",
                    "split",
                    "family",
                    "primary_cell",
                )
            }
            for identity in sorted(
                identities,
                key=lambda row: (row["split"], row["primary_cell"], row["map_id"]),
            )
        ]
        write_jsonl(root / "source_registry.jsonl", registry)
        registry_sha256 = sha256_file(root / "source_registry.jsonl")
        for relative, rows in all_rows.items():
            dataset_metadata(root, relative, rows, registry_sha256, tile_size_m)

        representative_cases = []
        for descriptor in training_descriptors:
            if descriptor["family"] != "trench":
                continue
            directory = root / descriptor["maps_path"]
            row = read_jsonl(directory / "manifest.jsonl")[0]
            representative_cases.append(
                {
                    "label": f"{descriptor['maps_path']}:1:{row['map_id']}",
                    "dataset": descriptor["maps_path"],
                    "map_id": row["map_id"],
                    "condition": descriptor["condition_id"],
                    "target": directory / "images" / "img_1.npy",
                    "occupancy": directory / "occupancy" / "img_1.npy",
                    "owners": directory / "trench_axis_owners" / "img_1.npy",
                    "metadata": directory / "metadata" / "trench_1.json",
                }
            )
        cfg = env_config()
        cones, footprints = terra_geometry(cfg)
        feasibility = [
            analyze_case(case, cfg, cones, footprints) for case in representative_cases
        ]
        feasibility_failures = [
            row for row in feasibility if not row["a1_pass"] or not row["a2_pass"]
        ]
        if feasibility_failures:
            raise RuntimeError(
                "representative axis feasibility failed for "
                + ", ".join(row["condition"] for row in feasibility_failures)
            )

        foundation_conditions = {
            descriptor["condition_id"]
            for descriptor in training_descriptors
            if descriptor["family"] == "foundation"
        }
        trench_conditions = {
            descriptor["condition_id"]
            for descriptor in training_descriptors
            if descriptor["family"] == "trench"
        }
        net4_multibit = {
            condition: sum(
                identity["multi_axis_target_cells"]
                for identity in identities
                if identity["primary_cell"] == condition
            )
            for condition in sorted(trench_conditions)
            if "trn-net4" in condition
        }
        if len(net4_multibit) != 3 or any(count <= 0 for count in net4_multibit.values()):
            raise RuntimeError(f"net4 owner intersections are missing: {net4_multibit}")

        write_json(root / "environment_protocol.json", protocol)
        mixture = {
            "schema": MIXTURE_SCHEMA,
            "family_balance": {"foundation": 0.5, "trench": 0.5},
            "fixed_protocol": {
                "accepted_dump_contract": "exact_visible_dump_v1",
                "apply_trench_rewards": False,
                "full_resets": True,
                "max_steps_in_episode": 450,
                "rewards_type": "DENSE",
            },
            "stages": [
                {"name": "capability_anchors", "new_conditions": list(CAPABILITY_IDS)},
                {"name": "nearby_geometry_core", "new_conditions": list(CORE_IDS)},
                {"name": "constraint_branches", "new_conditions": list(constraint_ids)},
            ],
            "note": (
                "Only generator-provenanced trench axes are trained. Seven legacy V7 "
                "trench geometries are excluded; all six legacy foundation geometries remain."
            ),
        }
        write_json(root / "training_mixture.json", mixture)

        payload_sha256 = canonical_sha256(
            {
                "environment_protocol_sha256": protocol_sha256,
                "identities": identities,
            }
        )
        distance_artifact_sha256 = canonical_sha256(
            [
                {
                    "maps_path": identity["maps_path"],
                    "slot_index": identity["slot_index"],
                    "distance_npy_sha256": identity["distance_npy_sha256"],
                }
                for identity in identities
            ]
        )
        reviewed_receipt = json.loads(reviewed_v6.read_text())
        legacy_review_path = legacy / "review_admission.json"
        review = {
            "schema": REVIEW_SCHEMA,
            "release_id": AXIS_V2_RELEASE_ID,
            "decision": "accept",
            "decision_source": "explicit_user_instruction",
            "decision_date": "2026-08-25",
            "accepted_conditions": list(all_conditions),
            "candidate_dataset_sha256": payload_sha256,
            "reviewed_v6_release": reviewed_receipt["release"],
            "reviewed_v6_receipt_sha256": sha256_file(reviewed_v6),
            "legacy_v8_review_receipt_sha256": sha256_file(legacy_review_path),
            "builder_baselines_revision": builder_revision,
            "note": (
                "Physical V6 generators and legacy foundation/control sources retain their "
                "accepted review provenance. Axis owner bits, canonical distance arrays, "
                "identity rebinding, and representative local feasibility are regenerated "
                "and audited for this release."
            ),
        }
        write_json(root / "review_admission.json", review)

        audit = {
            "schema": AUDIT_SCHEMA,
            "accepted": True,
            "owner_contract": TRENCH_AXIS_CONTRACT,
            "failed_maps": 0,
            "maps_audited_a0_and_distance": len(identities),
            "canonical_distance_artifact_sha256": distance_artifact_sha256,
            "foundation_conditions": len(foundation_conditions),
            "trench_conditions": len(trench_conditions),
            "foundation_maps_with_nonzero_owners": 0,
            "net4_multi_axis_target_cells": net4_multibit,
            "representative_local_feasibility_maps": len(feasibility),
            "representative_a1_failures": 0,
            "representative_a2_failures": 0,
            "representative_results": feasibility,
            "claim_limit": (
                "A0 and canonical distance are exhaustive. A1/A2 are one regenerated "
                "training map per trench condition and do not prove navigation, spoil "
                "ordering, or episode-horizon success."
            ),
        }
        write_json(root / "audit_receipt.json", audit)

        root_index = {
            "schema": "terra_curriculum_loader_bank_v1",
            "release_id": AXIS_V2_RELEASE_ID,
            "release_name": "axis-v2 V6 constraints plus V7 foundations train-96",
            "status": "accepted",
            "shape": [64, 64],
            "train": training_descriptors,
            "train_maps_per_condition": 96,
            "v6_constraint_condition_ids": list(constraint_ids),
            "v6_capability_floor_condition_ids": list(CAPABILITY_IDS),
            "v7_core_condition_ids": list(CORE_IDS),
            "included_in_main_macro": sorted(main_conditions),
            "evaluation_panels": evaluation_panels,
            "capability_floor_evaluation_panels": capability_panels,
            "source_registry": "source_registry.jsonl",
            "source_registry_sha256": registry_sha256,
            "environment_protocol": "environment_protocol.json",
            "environment_protocol_sha256": protocol_sha256,
            "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
            "review_admission": "review_admission.json",
            "review_admission_sha256": sha256_file(root / "review_admission.json"),
            "audit_receipt": "audit_receipt.json",
            "audit_receipt_sha256": sha256_file(root / "audit_receipt.json"),
            "training_mixture": "training_mixture.json",
            "training_mixture_sha256": sha256_file(root / "training_mixture.json"),
            "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
            "distance_metric": REWARD_V2_DISTANCE_METRIC,
            "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
            "tile_size_m": tile_size_m,
            "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
            "distance_bound": REWARD_V2_DISTANCE_BOUND,
            "canonical_distance_artifact_sha256": distance_artifact_sha256,
        }
        write_json(root / "dataset.json", root_index)
        receipt = {
            "schema": SCHEMA,
            "status": "passed",
            "terra_revision": terra_revision,
            "builder_baselines_revision": builder_revision,
            "environment_protocol_sha256": protocol_sha256,
            "output_dataset_sha256": sha256_file(root / "dataset.json"),
            "logical_payload_sha256": payload_sha256,
            "canonical_distance_artifact_sha256": distance_artifact_sha256,
            "source_registry_sha256": registry_sha256,
            "train_conditions": len(training_descriptors),
            "foundation_conditions": len(foundation_conditions),
            "trench_conditions": len(trench_conditions),
            "maps": len(identities),
            "main_evaluation_conditions": len(main_conditions),
            "representative_local_feasibility_maps": len(feasibility),
        }
        write_json(root / "build_receipt.json", receipt)
        root.rename(output)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--legacy-bank", type=Path, required=True)
    parser.add_argument("--reviewed-v6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terra-root", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args()
    validate_terra_checkout(args.terra_root, args.terra_revision)
    receipt = build_bank(
        args.candidates.resolve(),
        args.legacy_bank.resolve(),
        args.reviewed_v6.resolve(),
        args.output.resolve(),
        args.terra_revision,
        args.graph.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
