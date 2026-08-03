#!/usr/bin/env python3
"""Check the small set of claims every Legacy-Easy evaluation must preserve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--panel", choices=("promotion", "development"), required=True)
    parser.add_argument(
        "--mode", choices=("deterministic", "sampled"), required=True
    )
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--episode-bank-sha256", required=True)
    args = parser.parse_args()

    records = json.loads(args.result.read_text())
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("expected exactly one fixed-bank checkpoint record")
    record = records[0]
    explicit_bank = record["explicit_episode_bank"]
    summary = record["summary"]

    expected = {
        "schema": "terra_fixed_bank_eval_v4",
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint_sha256": args.checkpoint_sha256,
        "split": args.panel,
        "stratum": "legacy_easy_capability_floor",
        "horizon": 450,
        "policy_mode": args.mode,
        "deterministic": args.mode == "deterministic",
        "exact_manifest_enumeration": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"{args.result}: expected {key}={value!r}, got {record.get(key)!r}"
            )
    if explicit_bank["episode_bank_sha256"] != args.episode_bank_sha256:
        raise ValueError("result binds a different episode bank")
    if explicit_bank["panel"] != args.panel:
        raise ValueError("result binds a different explicit episode panel")
    if explicit_bank["diagnostic_only"] is not True:
        raise ValueError("Legacy-Easy result is not marked diagnostic-only")
    if explicit_bank["included_in_constrained_macro"] is not False:
        raise ValueError("Legacy-Easy result entered the constrained macro")
    if explicit_bank["slot_count"] != 48:
        raise ValueError("Legacy-Easy promotion/development panels require 48 episodes")
    if summary["overall"]["episodes"] != 48:
        raise ValueError("result does not account for all 48 episodes")
    if summary["integrity"] != {
        "failure_count": 0,
        "mass_residual_failures": 0,
        "nonfinite_states": 0,
        "obstacle_mutations": 0,
        "passed": True,
        "slot_index_disagreements": 0,
        "target_mutations": 0,
        "termination_disagreements": 0,
        "unavailable": 0,
    }:
        raise ValueError(f"result integrity failed: {summary['integrity']!r}")


if __name__ == "__main__":
    main()
