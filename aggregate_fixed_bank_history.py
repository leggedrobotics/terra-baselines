#!/usr/bin/env python3
"""Reduce fixed-bank JSON receipts into consecutive mastery and retention."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or any(
            not isinstance(record, dict) for record in payload
        ):
            raise RuntimeError(f"{path} must contain a JSON record or list.")
        records.extend(payload)
    return records


def aggregate_history(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["split"],
            record["stratum"],
            record.get(
                "policy_mode",
                "deterministic" if record.get("deterministic") else "sampled",
            ),
        )
        grouped[key].append(record)

    histories = []
    for (split, stratum, policy_mode), group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda record: (
                int(record["checkpoint_update"]),
                record["checkpoint_sha256"],
            ),
        )
        updates = [int(record["checkpoint_update"]) for record in ordered]
        if len(updates) != len(set(updates)):
            raise RuntimeError(
                f"duplicate checkpoint updates for {split}/{stratum}/{policy_mode}"
            )
        passed = [
            bool(record["summary"]["mastery_gate"]["passed"]) for record in ordered
        ]
        first_mastery_index = next(
            (
                index
                for index in range(1, len(passed))
                if passed[index - 1] and passed[index]
            ),
            None,
        )
        two_consecutive = first_mastery_index is not None
        first_mastery_update = updates[first_mastery_index] if two_consecutive else None
        retention_passed = None
        retention_checks = []
        if two_consecutive:
            mastery_rates = {
                family: float(values["success_rate"])
                for family, values in ordered[first_mastery_index]["summary"][
                    "by_family"
                ].items()
            }
            retention_passed = True
            for record in ordered[first_mastery_index + 1 :]:
                family_regressions = {
                    family: (
                        float(record["summary"]["by_family"][family]["success_rate"])
                        < mastery_rate - 0.05 - 1e-12
                    )
                    for family, mastery_rate in mastery_rates.items()
                }
                integrity_passed = bool(record["summary"]["integrity"]["passed"])
                check_passed = not any(family_regressions.values()) and integrity_passed
                retention_checks.append(
                    {
                        "checkpoint_update": int(record["checkpoint_update"]),
                        "family_regressions": family_regressions,
                        "integrity_passed": integrity_passed,
                        "passed": check_passed,
                    }
                )
                retention_passed &= check_passed

        histories.append(
            {
                "split": split,
                "stratum": stratum,
                "policy_mode": policy_mode,
                "checkpoint_updates": updates,
                "checkpoint_passed": passed,
                "two_consecutive_mastery": two_consecutive,
                "first_mastery_update": first_mastery_update,
                "retention_passed": retention_passed,
                "retention_checks": retention_checks,
            }
        )

    return {
        "schema": "terra_fixed_bank_history_v1",
        "history_count": len(histories),
        "histories": histories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [path.resolve() for path in args.input]
    result = aggregate_history(load_records(paths))
    result["inputs"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path in paths
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
