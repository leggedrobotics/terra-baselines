#!/usr/bin/env python3
"""Merge the exact D2 sampled-action audit shards into one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

EXPECTED_LABELS = ("flat_u1000", "flat_u4000")
EXPECTED_SEEDS = tuple(range(2026072500, 2026072508))
EXPECTED_KEYS = {(label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS}
STATIC_FIELDS = (
    "schema",
    "completion_contract",
    "observer_only",
    "source_revisions",
    "bank_root",
    "mode",
    "horizon",
    "numerical_tolerances",
    "reset_integrity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_payloads(payloads: list[dict], input_paths: list[Path]) -> dict:
    """Validate and merge the preregistered two-checkpoint, eight-seed grid."""
    if not payloads or len(payloads) != len(input_paths):
        raise ValueError("sampled merge requires matching non-empty payloads and paths")
    reference = payloads[0]
    if (
        reference["schema"] != "terra_historical_curriculum_audit_v1"
        or reference["mode"] != "sampled"
        or reference["horizon"] != 450
    ):
        raise RuntimeError("unexpected sampled-audit reference contract")

    records_by_key = {}
    input_receipts = []
    reward_atol = reference["numerical_tolerances"][
        "terminal_reward_reconstruction_atol"
    ]
    for path, payload in zip(input_paths, payloads, strict=True):
        for field in STATIC_FIELDS:
            if payload[field] != reference[field]:
                raise RuntimeError(f"sampled shard disagrees on {field}: {path}")
        for record in payload["records"]:
            key = (record["checkpoint_label"], int(record["seed"]))
            if key in records_by_key:
                raise RuntimeError(f"duplicate sampled record {key}")
            if (
                record["dataset"] != "development/M0"
                or record["mode"] != "sampled"
                or record["horizon"] != 450
            ):
                raise RuntimeError(f"sampled record identity mismatch: {key}")
            for row in record["per_map"]:
                if (
                    row["target_mutation"]
                    or row["obstacle_mutation"]
                    or row["nonfinite_state"]
                ):
                    raise RuntimeError(f"sampled rollout integrity failed: {key}")
                reward_error = float(row["terminal_reward_reconstruction_error"])
                if not math.isfinite(reward_error) or reward_error > reward_atol:
                    raise RuntimeError(f"sampled reward reconstruction failed: {key}")
            records_by_key[key] = record
        input_receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": len(payload["records"]),
            }
        )

    observed_keys = set(records_by_key)
    if observed_keys != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - observed_keys)
        extra = sorted(observed_keys - EXPECTED_KEYS)
        raise RuntimeError(f"sampled grid mismatch: missing={missing}, extra={extra}")

    merged = deepcopy(reference)
    merged["seeds"] = list(EXPECTED_SEEDS)
    merged["records"] = [
        records_by_key[(label, seed)]
        for label in EXPECTED_LABELS
        for seed in EXPECTED_SEEDS
    ]
    merged["execution_shards"] = input_receipts
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
    payloads = [json.loads(path.read_text()) for path in args.input]
    merged = merge_payloads(payloads, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print("HISTORICAL_SAMPLED_AUDIT_MERGE_PASSED")


if __name__ == "__main__":
    main()
