#!/usr/bin/env python3
"""Derive the strict-gate V8 runtime bank without changing reset arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import terra
from terra.benchmark_protocol import frozen_environment_protocol

from utils.accepted_bank import V8_TRENCH_ALIGNED_EVALUATION_FAMILY
from utils.accepted_bank import V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
from utils.accepted_bank import _canonical_json_sha256


ARRAY_FOLDERS = ("images", "occupancy", "dumpability", "actions", "distance")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_terra_checkout(terra_root: Path, terra_revision: str) -> None:
    """Bind the requested protocol revision to the code actually imported."""
    terra_root = terra_root.resolve()
    imported_root = Path(terra.__file__).resolve().parents[1]
    if imported_root != terra_root:
        raise RuntimeError(
            f"imported Terra root {imported_root} does not match {terra_root}"
        )
    head = subprocess.check_output(
        ["git", "-C", str(terra_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != terra_revision:
        raise RuntimeError(
            f"Terra revision mismatch: checkout is {head}, requested {terra_revision}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(terra_root), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise RuntimeError("Terra checkout must be committed and clean")


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json_lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _rebind_episode(row: dict, protocol_sha256: str) -> dict:
    rebound = dict(row)
    rebound["environment_protocol_sha256"] = protocol_sha256
    rebound["episode_id"] = _canonical_json_sha256(
        {
            "schema": "terra_episode_id_v1",
            "scenario_id": rebound["scenario_id"],
            "reset_seed": rebound["reset_seed"],
            "environment_protocol_sha256": protocol_sha256,
        }
    )
    return rebound


def _copy_filtered_panel(
    source: Path,
    destination: Path,
    protocol_sha256: str,
) -> dict:
    source_rows = _json_lines(source / "manifest.jsonl")
    kept = [
        row
        for row in source_rows
        if row.get("primary_cell")
        not in V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
    ]
    if not kept:
        raise RuntimeError(f"{source} has no rows after condition filtering")
    destination.mkdir(parents=True)
    for folder in (*ARRAY_FOLDERS, "metadata"):
        (destination / folder).mkdir()

    rows = []
    for new_slot, row in enumerate(kept, start=1):
        source_slot = int(row["slot_index"])
        for folder in ARRAY_FOLDERS:
            shutil.copy2(
                source / folder / f"img_{source_slot}.npy",
                destination / folder / f"img_{new_slot}.npy",
            )
        shutil.copy2(
            source / "metadata" / f"trench_{source_slot}.json",
            destination / "metadata" / f"trench_{new_slot}.json",
        )
        rebound = _rebind_episode(row, protocol_sha256)
        rebound["slot_index"] = new_slot
        rows.append(rebound)

    metadata = json.loads((source / "dataset.json").read_text())
    metadata["slot_count"] = len(rows)
    metadata["unique_identity_count"] = len({row["map_id"] for row in rows})
    metadata["condition_profile"] = "trench_aligned_37_v1"
    metadata["excluded_conditions"] = list(
        V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
    )
    metadata["derived_from_panel"] = f"evaluation/gate_main/{source.name}"
    (destination / "dataset.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    _write_json_lines(destination / "manifest.jsonl", rows)
    return {
        "source_slots": len(source_rows),
        "selected_slots": len(rows),
        "selected_conditions": len({row["primary_cell"] for row in rows}),
    }


def _rebind_existing_panel(directory: Path, protocol_sha256: str) -> dict:
    rows = [
        _rebind_episode(row, protocol_sha256)
        for row in _json_lines(directory / "manifest.jsonl")
    ]
    _write_json_lines(directory / "manifest.jsonl", rows)
    return {
        "slots": len(rows),
        "conditions": len({row["primary_cell"] for row in rows}),
    }


def derive_bank(
    input_root: Path,
    output_root: Path,
    terra_revision: str,
) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    root_index_path = input_root / "dataset.json"
    old_protocol_path = input_root / "environment_protocol.json"
    root_index = json.loads(root_index_path.read_text())
    old_protocol = json.loads(old_protocol_path.read_text())
    new_protocol = frozen_environment_protocol(terra_revision)
    ignored = {"terra_revision", "environment_protocol_sha256"}
    old_payload = {key: value for key, value in old_protocol.items() if key not in ignored}
    new_payload = {key: value for key, value in new_protocol.items() if key not in ignored}
    if old_payload != new_payload:
        raise RuntimeError(
            "The requested Terra revision changes environment semantics beyond "
            "the revision/hash fields; regenerate the bank instead of rebinding it."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        shutil.copytree(input_root, temporary / "bank", dirs_exist_ok=True)
        derived = temporary / "bank"
        protocol_sha256 = new_protocol["environment_protocol_sha256"]
        (derived / "environment_protocol.json").write_text(
            json.dumps(new_protocol, indent=2, sort_keys=True) + "\n"
        )

        panel_receipts = {}
        for panel in ("development", "promotion", "sealed"):
            panel_receipts[panel] = _copy_filtered_panel(
                derived / "evaluation" / "gate_main" / panel,
                derived
                / "evaluation"
                / V8_TRENCH_ALIGNED_EVALUATION_FAMILY
                / panel,
                protocol_sha256,
            )
        capability_receipts = {
            panel: _rebind_existing_panel(
                derived / "evaluation" / "capability_floor" / panel,
                protocol_sha256,
            )
            for panel in ("development", "promotion", "sealed")
        }

        receipt = {
            "schema": "terra_trench_aligned_runtime_bank_derivation_v1",
            "condition_profile": "trench_aligned_37_v1",
            "source_root": str(input_root),
            "source_dataset_sha256": _sha256(root_index_path),
            "source_environment_protocol_sha256": old_protocol[
                "environment_protocol_sha256"
            ],
            "runtime_terra_revision": terra_revision,
            "runtime_environment_protocol_sha256": protocol_sha256,
            "semantic_payload_unchanged_except_revision": True,
            "excluded_conditions": list(
                V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
            ),
            "main_evaluation_family": V8_TRENCH_ALIGNED_EVALUATION_FAMILY,
            "main_evaluation_panels": panel_receipts,
            "capability_floor_evaluation_panels": capability_receipts,
            "reset_arrays": "copied byte-for-byte",
        }
        receipt_path = derived / "trench_aligned_runtime_derivation.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        root_index["environment_protocol_sha256"] = protocol_sha256
        root_index["trench_aligned_runtime_derivation"] = receipt_path.name
        root_index["trench_aligned_runtime_derivation_sha256"] = _sha256(receipt_path)
        (derived / "dataset.json").write_text(
            json.dumps(root_index, indent=2, sort_keys=True) + "\n"
        )
        derived.rename(output_root)
        temporary.rmdir()
        return receipt
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--terra-root", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    args = parser.parse_args()
    _validate_terra_checkout(args.terra_root, args.terra_revision)
    receipt = derive_bank(args.input_root, args.output_root, args.terra_revision)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
