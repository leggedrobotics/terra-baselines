#!/usr/bin/env python3
"""Materialize the one canonical-distance V8 bank used by R2 reward-v2.

The accepted V8 bank remains immutable.  This script hard-links its payload
into a new directory, atomically replaces only distance arrays and the
identity/protocol metadata that those arrays necessarily change, and verifies
every replacement against the frozen D4 admission sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from terra.config import (
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
)
from terra.env_generation.distance import (
    REWARD_V2_DISTANCE_METRIC,
    REWARD_V2_DISTANCE_NORMALIZATION,
    REWARD_V2_DISTANCE_PROTOCOL_ID,
    compute_reward_v2_distance_map,
)
from terra.maps_buffer import (
    RESET_ARRAY_FOLDERS,
    reset_array_scenario_sha256,
)

SCHEMA = "terra_v8_r2_materialized_distance_bank_v1"
SIDECAR_SCHEMA = "terra_r2_distance_sidecar_v1"
EXPECTED_BASE_DATASET_SHA256 = (
    "715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798"
)
EXPECTED_BASE_LOGICAL_BANK_SHA256 = (
    "f2e451ca33c8902f70305b89e56025392a6cb0469f44265697e6f4e9a0b72e21"
)
EXPECTED_SIDECAR_DATASET_SHA256 = (
    "f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980"
)
EXPECTED_SIDECAR_ROWS_SHA256 = (
    "b6bcae37a1750f7d78c1645af408320c50b0fa28d38098ff91e9cecbfec251a8"
)
EXPECTED_DISTANCE_PROTOCOL_SHA256 = (
    "ea7bf132f4d4f11265c30c443754619f1fb3ed0c6a07db229a72eb29c4b12ca3"
)
EXPECTED_SCENARIOS = 7_520
PHYSICAL_ARRAY_FOLDERS = ("images", "occupancy", "dumpability", "actions")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Accepted-bank directories are intentionally read-only. copytree creates
    # distinct directory inodes, so making only the derived parent writable
    # cannot affect the immutable base or its hard-linked file payloads.
    path.parent.chmod(path.parent.stat().st_mode | 0o200)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode()
    atomic_write_bytes(path, payload)


def atomic_copy(source: Path, destination: Path) -> None:
    atomic_write_bytes(destination, source.read_bytes())


def array_set_sha256(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = str(path.relative_to(root))
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate_rows_sha256(rows: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_sha256 in sorted(rows):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_sha256.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_entries(index: dict[str, Any]) -> list[tuple[str, int]]:
    entries = [(row["maps_path"], int(row["map_count"])) for row in index["train"]]
    for panels_name in (
        "evaluation_panels",
        "capability_floor_evaluation_panels",
    ):
        panels = index[panels_name]
        entries.extend(
            (row["maps_path"], int(row["slot_count"]))
            for _, row in sorted(panels.items())
        )
    if len(entries) != 53 or len({path for path, _ in entries}) != len(entries):
        raise ValueError("the frozen V8 bank must contain exactly 53 datasets")
    return entries


def episode_id(
    scenario_id: str, reset_seed: int, environment_protocol_sha256: str
) -> str:
    return canonical_json_sha256(
        {
            "schema": "terra_episode_id_v1",
            "scenario_id": scenario_id,
            "reset_seed": reset_seed,
            "environment_protocol_sha256": environment_protocol_sha256,
        }
    )


def validate_sidecar(sidecar: Path, base: Path) -> tuple[dict[str, Any], list[dict]]:
    dataset_path = sidecar / "dataset.json"
    if sha256_file(dataset_path) != EXPECTED_SIDECAR_DATASET_SHA256:
        raise ValueError(
            "canonical sidecar dataset.json is not the admitted D4 artifact"
        )
    receipt = json.loads(dataset_path.read_text())
    if receipt.get("schema") != SIDECAR_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("canonical sidecar did not pass admission")
    if receipt.get("base_bank_dataset_json_sha256") != EXPECTED_BASE_DATASET_SHA256:
        raise ValueError("canonical sidecar names a different base bank")
    if receipt.get("rows_sha256") != EXPECTED_SIDECAR_ROWS_SHA256:
        raise ValueError("canonical sidecar row identity changed")
    if receipt.get("distance_protocol_sha256") != EXPECTED_DISTANCE_PROTOCOL_SHA256:
        raise ValueError("canonical distance protocol changed")
    if sha256_file(sidecar / "rows.jsonl") != EXPECTED_SIDECAR_ROWS_SHA256:
        raise ValueError("canonical sidecar rows.jsonl bytes changed")
    if sha256_file(sidecar / "distance_protocol.json") != (
        EXPECTED_DISTANCE_PROTOCOL_SHA256
    ):
        raise ValueError("canonical distance protocol bytes changed")
    if sha256_file(base / "dataset.json") != EXPECTED_BASE_DATASET_SHA256:
        raise ValueError("base bank dataset.json is not the admitted V8 bank")
    rows = read_jsonl(sidecar / "rows.jsonl")
    if len(rows) != EXPECTED_SCENARIOS:
        raise ValueError(f"canonical sidecar must contain {EXPECTED_SCENARIOS} rows")
    return receipt, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--canonical-sidecar", type=Path, required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_bank.expanduser().resolve()
    sidecar = args.canonical_sidecar.expanduser().resolve()
    output = args.output_bank.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    if not base.is_dir() or not sidecar.is_dir():
        raise FileNotFoundError("base bank and canonical sidecar must exist")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("output bank and receipt must both be new paths")
    if output == base or base in output.parents or output in base.parents:
        raise ValueError("output bank must be a separate sibling of the immutable base")

    sidecar_receipt, sidecar_rows = validate_sidecar(sidecar, base)
    index = json.loads((base / "dataset.json").read_text())
    review = json.loads((base / "review_admission.json").read_text())
    if review.get("candidate_dataset_sha256") != EXPECTED_BASE_LOGICAL_BANK_SHA256:
        raise ValueError("base bank logical payload identity changed")
    entries = dataset_entries(index)
    expected_slots = {
        (relative_path, slot)
        for relative_path, count in entries
        for slot in range(1, count + 1)
    }
    rows_by_slot = {
        (row["dataset_relative_path"], int(row["slot_index"])): row
        for row in sidecar_rows
    }
    if set(rows_by_slot) != expected_slots or len(rows_by_slot) != len(sidecar_rows):
        raise ValueError("canonical sidecar does not cover the exact V8 slot set")

    base_tree_before = tree_sha256(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        shutil.copytree(base, temporary, copy_function=os.link, dirs_exist_ok=True)
        scenario_by_map: dict[str, str] = {}
        distance_files: list[tuple[str, str]] = []
        physical_files: list[tuple[str, str]] = []
        metadata_files: list[tuple[str, str]] = []
        changed_scenarios = 0

        protocol = json.loads((sidecar / "distance_protocol.json").read_text())
        if protocol != {
            **protocol,
            "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
            "distance_metric": REWARD_V2_DISTANCE_METRIC,
            "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
            "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
            "distance_bound": REWARD_V2_DISTANCE_BOUND,
        }:
            raise ValueError(
                "admission protocol does not match imported Terra constants"
            )
        tile_size_m = float(protocol["tile_size_m"])
        environment_protocol_sha256 = index["environment_protocol_sha256"]

        for relative_path, slot_count in entries:
            source_dataset = base / relative_path
            target_dataset = temporary / relative_path
            manifest = read_jsonl(source_dataset / "manifest.jsonl")
            if len(manifest) != slot_count:
                raise ValueError(f"{relative_path}: manifest count changed")
            derived_manifest: list[dict[str, Any]] = []
            for base_row in manifest:
                slot = int(base_row["slot_index"])
                key = (relative_path, slot)
                admitted = rows_by_slot[key]
                if base_row["map_id"] != admitted["map_id"] or (
                    base_row["source_id"] != admitted["source_id"]
                ):
                    raise ValueError(
                        f"{relative_path} slot {slot}: logical identity changed"
                    )
                expected_reset_seed = base_row.get("reset_seed")
                expected_episode_id = base_row.get("episode_id")
                if admitted.get("reset_seed") != expected_reset_seed or (
                    admitted.get("episode_id") != expected_episode_id
                ):
                    raise ValueError(
                        f"{relative_path} slot {slot}: reset pose identity changed"
                    )
                arrays = {
                    name: np.load(
                        source_dataset / name / f"img_{slot}.npy",
                        allow_pickle=False,
                    )
                    for name in RESET_ARRAY_FOLDERS
                }
                legacy_scenario = reset_array_scenario_sha256(arrays)
                if legacy_scenario != base_row["scenario_id"] or legacy_scenario != (
                    admitted["legacy_scenario_id"]
                ):
                    raise ValueError(
                        f"{relative_path} slot {slot}: legacy reset hash changed"
                    )
                physical_sha = array_set_sha256(arrays, PHYSICAL_ARRAY_FOLDERS)
                if physical_sha != admitted["physical_reset_arrays_sha256"]:
                    raise ValueError(
                        f"{relative_path} slot {slot}: physical arrays changed"
                    )

                canonical_relative = admitted["canonical_distance_npy"]
                expected_relative = f"{relative_path}/distance/img_{slot}.npy"
                if canonical_relative != expected_relative:
                    raise ValueError(
                        f"{relative_path} slot {slot}: sidecar path changed"
                    )
                canonical_path = sidecar / canonical_relative
                canonical_sha = sha256_file(canonical_path)
                if canonical_sha != admitted["canonical_distance_npy_sha256"]:
                    raise ValueError(
                        f"{relative_path} slot {slot}: sidecar bytes changed"
                    )
                canonical = np.load(canonical_path, allow_pickle=False)
                recomputed = compute_reward_v2_distance_map(
                    np.squeeze(arrays["images"]),
                    np.squeeze(arrays["occupancy"]),
                    tile_size_m=tile_size_m,
                    distance_ref_m=REWARD_V2_DISTANCE_REF_M,
                    distance_bound=REWARD_V2_DISTANCE_BOUND,
                )
                if not np.array_equal(canonical, recomputed):
                    delta = float(np.max(np.abs(canonical - recomputed)))
                    raise ValueError(
                        f"{relative_path} slot {slot}: canonical primitive mismatch {delta}"
                    )
                derived_arrays = dict(arrays)
                derived_arrays["distance"] = canonical
                derived_scenario = reset_array_scenario_sha256(derived_arrays)
                if derived_scenario != admitted["derived_reset_arrays_sha256"]:
                    raise ValueError(
                        f"{relative_path} slot {slot}: derived reset hash changed"
                    )

                destination = target_dataset / "distance" / f"img_{slot}.npy"
                atomic_copy(canonical_path, destination)
                if sha256_file(destination) != canonical_sha:
                    raise ValueError(
                        f"{relative_path} slot {slot}: distance copy failed"
                    )
                distance_files.append((expected_relative, canonical_sha))
                for folder in PHYSICAL_ARRAY_FOLDERS:
                    path = source_dataset / folder / f"img_{slot}.npy"
                    physical_files.append(
                        (f"{relative_path}/{folder}/img_{slot}.npy", sha256_file(path))
                    )
                metadata_path = source_dataset / "metadata" / f"trench_{slot}.json"
                metadata_files.append(
                    (
                        f"{relative_path}/metadata/trench_{slot}.json",
                        sha256_file(metadata_path),
                    )
                )

                derived_row = dict(base_row)
                derived_row["scenario_id"] = derived_scenario
                if "episode_id" in derived_row:
                    derived_row["episode_id"] = episode_id(
                        derived_scenario,
                        int(derived_row["reset_seed"]),
                        environment_protocol_sha256,
                    )
                derived_manifest.append(derived_row)
                previous = scenario_by_map.setdefault(
                    base_row["map_id"], derived_scenario
                )
                if previous != derived_scenario:
                    raise ValueError(
                        f"map_id {base_row['map_id']} has multiple scenarios"
                    )
                changed_scenarios += derived_scenario != legacy_scenario

            atomic_write_jsonl(target_dataset / "manifest.jsonl", derived_manifest)

        if len(scenario_by_map) != EXPECTED_SCENARIOS:
            raise ValueError("V8 map IDs are not one-to-one with admitted scenarios")
        registry = read_jsonl(base / "source_registry.jsonl")
        if len(registry) != EXPECTED_SCENARIOS:
            raise ValueError("base source registry count changed")
        derived_registry = []
        for row in registry:
            derived = dict(row)
            derived["scenario_id"] = scenario_by_map[row["map_id"]]
            derived_registry.append(derived)
        atomic_write_jsonl(temporary / "source_registry.jsonl", derived_registry)
        derived_registry_sha = sha256_file(temporary / "source_registry.jsonl")

        for relative_path, _ in entries:
            metadata_path = temporary / relative_path / "dataset.json"
            metadata = json.loads((base / relative_path / "dataset.json").read_text())
            metadata.update(
                {
                    "source_registry_sha256": derived_registry_sha,
                    "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
                    "distance_metric": REWARD_V2_DISTANCE_METRIC,
                    "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
                    "tile_size_m": tile_size_m,
                    "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
                    "distance_bound": REWARD_V2_DISTANCE_BOUND,
                }
            )
            atomic_write_json(metadata_path, metadata)

        derived_index = dict(index)
        derived_index.update(
            {
                "source_registry_sha256": derived_registry_sha,
                "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
                "distance_metric": REWARD_V2_DISTANCE_METRIC,
                "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
                "tile_size_m": tile_size_m,
                "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
                "distance_bound": REWARD_V2_DISTANCE_BOUND,
                "derived_from_bank_dataset_sha256": EXPECTED_BASE_DATASET_SHA256,
                "canonical_distance_sidecar_dataset_sha256": (
                    EXPECTED_SIDECAR_DATASET_SHA256
                ),
            }
        )
        atomic_write_json(temporary / "dataset.json", derived_index)

        if changed_scenarios != EXPECTED_SCENARIOS:
            raise ValueError(
                "every R2 scenario identity must change with its distance array"
            )
        base_tree_after = tree_sha256(base)
        if base_tree_after != base_tree_before:
            raise RuntimeError("immutable base bank changed during materialization")
        output_tree = tree_sha256(temporary)
        treatment_dataset_sha = sha256_file(temporary / "dataset.json")
        receipt = {
            "schema": SCHEMA,
            "status": "passed",
            "base_bank": {
                "root": str(base),
                "dataset_json_sha256": EXPECTED_BASE_DATASET_SHA256,
                "logical_bank_sha256": EXPECTED_BASE_LOGICAL_BANK_SHA256,
                "tree_sha256_before": base_tree_before,
                "tree_sha256_after": base_tree_after,
                "unchanged": True,
            },
            "canonical_sidecar": {
                "root": str(sidecar),
                "dataset_json_sha256": EXPECTED_SIDECAR_DATASET_SHA256,
                "rows_sha256": EXPECTED_SIDECAR_ROWS_SHA256,
                "distance_protocol_sha256": EXPECTED_DISTANCE_PROTOCOL_SHA256,
                "distance_arrays_aggregate_sha256": aggregate_rows_sha256(
                    distance_files
                ),
            },
            "treatment_bank": {
                "root": str(output),
                "dataset_json_sha256": treatment_dataset_sha,
                "source_registry_sha256": derived_registry_sha,
                "tree_sha256": output_tree,
            },
            "pair_equivalence": {
                "scenarios": EXPECTED_SCENARIOS,
                "datasets": len(entries),
                "map_id_source_id_preserved": True,
                "physical_arrays_preserved": True,
                "metadata_and_pose_sidecars_preserved": True,
                "only_reset_array_changed": "distance",
                "physical_arrays_aggregate_sha256": aggregate_rows_sha256(
                    physical_files
                ),
                "metadata_aggregate_sha256": aggregate_rows_sha256(metadata_files),
                "legacy_scenario_ids_changed": changed_scenarios,
                "derived_scenario_ids_and_episode_ids_recomputed": True,
            },
            "distance_protocol": {
                "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
                "distance_metric": REWARD_V2_DISTANCE_METRIC,
                "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
                "tile_size_m": tile_size_m,
                "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
                "distance_bound": REWARD_V2_DISTANCE_BOUND,
                "canonical_primitive_recomputed_for_every_scenario": True,
            },
        }
        os.replace(temporary, output)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(receipt_path, receipt)
        print(
            canonical_json(
                {
                    "status": "passed",
                    "scenarios": EXPECTED_SCENARIOS,
                    "treatment_bank_dataset_sha256": treatment_dataset_sha,
                    "treatment_bank_tree_sha256": output_tree,
                    "distance_sidecar_sha256": EXPECTED_SIDECAR_DATASET_SHA256,
                }
            )
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
