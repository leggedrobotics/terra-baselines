#!/usr/bin/env python3
"""Derive a family-only partial-reset bank without changing sidecar bytes."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from terra.maps_buffer import (
    PARTIAL_COMPLETION_REJECTIONS,
    PARTIAL_RESET_BANK_INDEX,
    partial_reset_bank_sha256,
)


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def subset_bank(
    canonical_root: Path,
    input_root: Path,
    output_root: Path,
    family: str,
) -> dict:
    canonical_root = canonical_root.resolve()
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    canonical = json.loads((canonical_root / "dataset.json").read_text())
    selected_paths = [
        row["maps_path"] for row in canonical["train"] if row.get("family") == family
    ]
    if not selected_paths:
        raise RuntimeError(f"canonical bank contains no {family!r} conditions")

    index = json.loads((input_root / PARTIAL_RESET_BANK_INDEX).read_text())
    source_digest = partial_reset_bank_sha256(input_root)
    if index.get("bank_sha256") != source_digest:
        raise RuntimeError("input partial-reset bank digest mismatch")
    source_supported = set(index["supported_maps_paths"])
    supported_paths = [path for path in selected_paths if path in source_supported]
    if not supported_paths:
        raise RuntimeError(f"partial-reset bank supports no {family!r} conditions")

    source_audit_path = input_root / "trench_alignment_audit.json"
    source_audit = json.loads(source_audit_path.read_text())
    if source_audit.get("partial_reset_bank_sha256") != source_digest:
        raise RuntimeError("alignment audit does not describe the input bank")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for maps_path in supported_paths:
            shutil.copytree(input_root / maps_path, temporary / maps_path)

        selected_set = set(selected_paths)
        root_rejections = [
            row
            for row in _json_lines(input_root / PARTIAL_COMPLETION_REJECTIONS)
            if row.get("maps_path") in selected_set
        ]
        _write_json_lines(temporary / PARTIAL_COMPLETION_REJECTIONS, root_rejections)

        derived_index = {
            **index,
            "eligible_maps_paths": selected_paths,
            "supported_maps_paths": supported_paths,
            "rejected_condition_count": len(selected_paths) - len(supported_paths),
            "excluded_declared_condition_count": 0,
            "derived_from_partial_reset_bank_sha256": source_digest,
            "subset_family": family,
        }
        derived_index["bank_sha256"] = partial_reset_bank_sha256(temporary)
        (temporary / PARTIAL_RESET_BANK_INDEX).write_text(
            json.dumps(derived_index, indent=2, sort_keys=True) + "\n"
        )

        audit_rows = [
            row
            for row in source_audit.get("rows", [])
            if row.get("maps_path") in set(supported_paths)
        ]
        failed_rows = [
            row for row in audit_rows if not row["alignment_chain_complete"]
        ]
        audit_triplets = {
            (row["maps_path"], int(row["source_index"])) for row in audit_rows
        }
        derived_audit = {
            **source_audit,
            "partial_reset_bank_sha256": derived_index["bank_sha256"],
            "derived_from_partial_reset_bank_sha256": source_digest,
            "trench_source_triplets": len(audit_triplets),
            "audited_sidecars": len(audit_rows),
            "passed_sidecars": len(audit_rows) - len(failed_rows),
            "failed_sidecars": len(failed_rows),
            "accepted": not failed_rows,
            "rows": audit_rows,
        }
        (temporary / "trench_alignment_audit.json").write_text(
            json.dumps(derived_audit, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "schema": "terra_partial_reset_family_subset_receipt_v1",
        "family": family,
        "source_bank_sha256": source_digest,
        "bank_sha256": derived_index["bank_sha256"],
        "eligible_conditions": len(selected_paths),
        "supported_conditions": len(supported_paths),
        "source_triplets": len(audit_triplets),
        "sidecars": len(audit_rows),
        "alignment_accepted": derived_audit["accepted"],
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    receipt = subset_bank(
        args.canonical_root, args.input_root, args.output_root, args.family
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["alignment_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
