#!/usr/bin/env python3
"""Derive an alignment-admitted partial bank by dropping failed triplets."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from terra.config import PARTIAL_RESET_FRACTIONS
from terra.maps_buffer import PARTIAL_COMPLETION_CONFIG
from terra.maps_buffer import PARTIAL_COMPLETION_MANIFEST
from terra.maps_buffer import PARTIAL_COMPLETION_REJECTIONS
from terra.maps_buffer import PARTIAL_RESET_BANK_INDEX
from terra.maps_buffer import partial_reset_bank_sha256


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json_lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def filter_bank(
    input_root: Path,
    output_root: Path,
    audit_path: Path,
) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    index = json.loads((input_root / PARTIAL_RESET_BANK_INDEX).read_text())
    source_digest = partial_reset_bank_sha256(input_root)
    if source_digest != index.get("bank_sha256"):
        raise RuntimeError("input partial-reset bank digest mismatch")
    audit = json.loads(audit_path.read_text())
    if audit.get("partial_reset_bank_sha256") != source_digest:
        raise RuntimeError("alignment audit does not describe the input bank")
    failed_rows = [
        row for row in audit.get("rows", []) if not row["alignment_chain_complete"]
    ]
    failed_triplets = {
        (row["maps_path"], int(row["source_index"])) for row in failed_rows
    }
    if not failed_triplets:
        raise RuntimeError("alignment audit has no failed source triplets to filter")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    supported_paths = []
    root_rejections = _json_lines(input_root / PARTIAL_COMPLETION_REJECTIONS)
    try:
        for maps_path in index["supported_maps_paths"]:
            source_leaf = input_root / maps_path
            destination_leaf = temporary / maps_path
            rows = _json_lines(source_leaf / PARTIAL_COMPLETION_MANIFEST)
            kept = [
                row
                for row in rows
                if (maps_path, int(row["source_index"])) not in failed_triplets
            ]
            removed_sources = sorted(
                {
                    int(row["source_index"])
                    for row in rows
                    if (maps_path, int(row["source_index"])) in failed_triplets
                }
            )
            if not kept:
                root_rejections.append(
                    {
                        "maps_path": maps_path,
                        "error": "all partial source triplets failed alignment admission",
                        "alignment_rejected_sources": removed_sources,
                    }
                )
                continue

            actions = destination_leaf / "actions"
            actions.mkdir(parents=True)
            rewritten = []
            for sidecar_index, row in enumerate(kept, start=1):
                old_index = int(row["sidecar_index"])
                shutil.copy2(
                    source_leaf / "actions" / f"img_{old_index}.npy",
                    actions / f"img_{sidecar_index}.npy",
                )
                rewritten.append({**row, "sidecar_index": sidecar_index})

            rejection_rows = _json_lines(
                source_leaf / PARTIAL_COMPLETION_REJECTIONS
            )
            for source_index in removed_sources:
                failures = [
                    row
                    for row in failed_rows
                    if row["maps_path"] == maps_path
                    and int(row["source_index"]) == source_index
                ]
                rejection_rows.append(
                    {
                        "maps_path": maps_path,
                        "source_index": source_index,
                        "error": "source triplet rejected by strict-alignment audit",
                        "failed_tiers": sorted(row["reset_tier"] for row in failures),
                        "minimum_remaining_cells_covered": min(
                            row["remaining_cells_covered"] for row in failures
                        ),
                    }
                )
            config = json.loads(
                (source_leaf / PARTIAL_COMPLETION_CONFIG).read_text()
            )
            tier_counts = [
                sum(int(row["reset_tier"]) == tier for row in rewritten)
                for tier in range(1, len(PARTIAL_RESET_FRACTIONS) + 1)
            ]
            if len(set(tier_counts)) != 1 or not tier_counts[0]:
                raise RuntimeError(
                    f"alignment filtering broke triplet support for {maps_path}"
                )
            config.update(
                successful_variant_count=len(rewritten),
                tier_success_counts=tier_counts,
                accepted_source_count=tier_counts[0],
                rejected_variant_count=len(rejection_rows),
                rejected_source_count=int(config.get("rejected_source_count", 0))
                + len(removed_sources),
                alignment_rejected_source_count=len(removed_sources),
                alignment_admission_schema="terra_partial_trench_alignment_audit_v1",
                alignment_source_bank_sha256=source_digest,
                selected_pile_mode_counts={
                    mode: sum(row["pile_mode"] == mode for row in rewritten)
                    // len(PARTIAL_RESET_FRACTIONS)
                    for mode in config["pile_mode_policy"]
                },
            )
            (destination_leaf / PARTIAL_COMPLETION_CONFIG).write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
            _write_json_lines(
                destination_leaf / PARTIAL_COMPLETION_MANIFEST,
                rewritten,
            )
            _write_json_lines(
                destination_leaf / PARTIAL_COMPLETION_REJECTIONS,
                rejection_rows,
            )
            supported_paths.append(maps_path)

        _write_json_lines(
            temporary / PARTIAL_COMPLETION_REJECTIONS,
            root_rejections,
        )
        dropped_conditions = len(index["supported_maps_paths"]) - len(supported_paths)
        index.update(
            supported_maps_paths=supported_paths,
            rejected_condition_count=int(index["rejected_condition_count"])
            + dropped_conditions,
            derived_from_partial_reset_bank_sha256=source_digest,
            alignment_admission_schema="terra_partial_trench_alignment_audit_v1",
            alignment_rejected_source_triplet_count=len(failed_triplets),
            alignment_rejected_sidecar_count=len(failed_rows),
        )
        index["bank_sha256"] = partial_reset_bank_sha256(temporary)
        (temporary / PARTIAL_RESET_BANK_INDEX).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output_root)
        return index
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    receipt = filter_bank(args.input_root, args.output_root, args.audit)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
