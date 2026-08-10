#!/usr/bin/env python3
"""Select one reward-v2 checkpoint using only frozen promotion panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

UPDATES = tuple(range(1_000, 40_001, 1_000))
PROTOCOL_ID = "material_potential_v2"
DISTANCE_PROTOCOL_ID = "obstacle_geodesic_8_physical_global_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(
    path: Path, *, stratum: str, episodes: int, conditions: int
) -> list[dict]:
    records = json.loads(path.read_text())
    if (
        not isinstance(records, list)
        or tuple(record.get("checkpoint_update") for record in records) != UPDATES
    ):
        raise ValueError(f"{path}: expected promotion checkpoints {UPDATES}")
    for record in records:
        expected = {
            "schema": "terra_fixed_bank_eval_v4",
            "completion_contract": "exact_visible_dump_v1",
            "split": "promotion",
            "stratum": stratum,
            "horizon": 450,
            "deterministic": True,
            "policy_mode": "deterministic",
            "exact_manifest_enumeration": True,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path}: {key} changed")
        summary = record.get("summary", {})
        if summary.get("integrity", {}).get("passed") is not True:
            raise ValueError(f"{path}: integrity failed")
        if summary.get("overall", {}).get("episodes") != episodes:
            raise ValueError(f"{path}: episode count changed")
        if summary.get("graded", {}).get("condition_count") != conditions:
            raise ValueError(f"{path}: condition count changed")
        protocol = record.get("r2_protocol_receipt", {})
        if (
            protocol.get("reward_protocol_id") != PROTOCOL_ID
            or protocol.get("distance_protocol_id") != DISTANCE_PROTOCOL_ID
        ):
            raise ValueError(f"{path}: reward-v2 protocol receipt changed")
    return records


def checkpoint_identity(record: dict) -> tuple[str, str, int]:
    return (
        str(record.get("checkpoint")),
        str(record.get("checkpoint_sha256")),
        int(record.get("checkpoint_update", -1)),
    )


def select_records(main: list[dict], capability: list[dict]) -> dict:
    if len(main) != len(capability) or not main:
        raise ValueError("promotion panels must contain the same checkpoints")
    for main_record, capability_record in zip(main, capability):
        if checkpoint_identity(main_record) != checkpoint_identity(capability_record):
            raise ValueError("promotion panels name different checkpoints")

    candidates = []
    for main_record, capability_record in zip(main, capability):
        main_summary = main_record["summary"]
        capability_summary = capability_record["summary"]
        main_exact = int(main_summary["overall"]["successes"])
        capability_exact = int(capability_summary["overall"]["successes"])
        main_macro = float(main_summary["graded"]["macro_completion"])
        capability_macro = float(capability_summary["graded"]["macro_completion"])
        main_worst = float(main_summary["graded"]["worst_condition_completion"])
        capability_worst = float(
            capability_summary["graded"]["worst_condition_completion"]
        )
        if not (
            0 <= main_exact <= 720
            and 0 <= capability_exact <= 32
            and all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (
                    main_macro,
                    capability_macro,
                    main_worst,
                    capability_worst,
                )
            )
        ):
            raise ValueError("promotion result is outside its valid range")
        combined_macro = (45.0 * main_macro + 2.0 * capability_macro) / 47.0
        worst = min(main_worst, capability_worst)
        path, sha256, update = checkpoint_identity(main_record)
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("promotion checkpoint lacks a SHA-256")
        candidates.append(
            {
                "checkpoint": path,
                "checkpoint_sha256": sha256,
                "checkpoint_update": update,
                "exact_successes": main_exact + capability_exact,
                "episodes": 752,
                "macro_completion": combined_macro,
                "worst_condition_completion": worst,
                "main": {
                    "exact_successes": main_exact,
                    "episodes": 720,
                    "macro_completion": main_macro,
                },
                "capability": {
                    "exact_successes": capability_exact,
                    "episodes": 32,
                    "macro_completion": capability_macro,
                },
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["exact_successes"],
            row["macro_completion"],
            row["worst_condition_completion"],
            -row["checkpoint_update"],
        ),
    )
    checkpoint_path = Path(selected["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint bytes do not match promotion receipt")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--capability-promotion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_records = load_panel(
        args.promotion, stratum="all", episodes=720, conditions=45
    )
    capability_records = load_panel(
        args.capability_promotion,
        stratum="capability",
        episodes=32,
        conditions=2,
    )
    selected = select_records(main_records, capability_records)
    receipt = {
        "schema": "terra_v8_r2_scratch_promotion_selection_v1",
        "passed": True,
        "selection_rule": (
            "max combined promotion exact, then 47-condition macro completion, "
            "then worst condition, then earliest update"
        ),
        "candidate_updates": list(UPDATES),
        "inputs": {
            "promotion": {
                "path": str(args.promotion.resolve()),
                "sha256": sha256_file(args.promotion),
            },
            "capability_promotion": {
                "path": str(args.capability_promotion.resolve()),
                "sha256": sha256_file(args.capability_promotion),
            },
        },
        "selected": selected,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
