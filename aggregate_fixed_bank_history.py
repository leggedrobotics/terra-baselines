#!/usr/bin/env python3
"""Reduce fixed-bank JSON receipts into consecutive mastery and retention."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CONDITION_SIZE = 8
FAMILY_SIZE = 32


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


def _panel_counts(summary: dict, key: str, expected_size: int) -> dict:
    panels = {}
    for name, values in summary[key].items():
        successes = int(values["successes"])
        episodes = int(values["episodes"])
        if successes < 0 or successes > episodes:
            raise RuntimeError(
                f"{key}/{name} has invalid successes {successes}/{episodes}"
            )
        panels[name] = {
            "successes": successes,
            "episodes": episodes,
            "expected_episodes": expected_size,
        }
    return panels


def _evaluation(record: dict) -> dict:
    summary = record["summary"]
    families = _panel_counts(summary, "by_family", FAMILY_SIZE)
    conditions = _panel_counts(
        summary,
        "by_primary_cell",
        CONDITION_SIZE,
    )
    overall_episodes = int(summary["overall"]["episodes"])
    complete = (
        bool(record.get("exact_manifest_enumeration", False))
        and bool(record.get("reset_verification", {}).get("passed", False))
        and bool(families)
        and len(conditions) == 4 * len(families)
        and all(panel["episodes"] == FAMILY_SIZE for panel in families.values())
        and all(panel["episodes"] == CONDITION_SIZE for panel in conditions.values())
        and overall_episodes == FAMILY_SIZE * len(families)
        and overall_episodes == CONDITION_SIZE * len(conditions)
    )
    integrity_passed = bool(summary["integrity"]["passed"])
    eligible = complete and integrity_passed
    performance_passed = all(
        panel["successes"] >= 26 for panel in families.values()
    ) and all(panel["successes"] >= 6 for panel in conditions.values())
    claimed_performance = bool(summary["mastery_gate"]["performance_passed"])
    if claimed_performance != performance_passed:
        raise RuntimeError(
            "mastery_gate.performance_passed disagrees with panel counts"
        )

    return {
        "families": families,
        "conditions": conditions,
        "complete": complete,
        "integrity_passed": integrity_passed,
        "eligible": eligible,
        "performance_passed": performance_passed,
        "passed": eligible and performance_passed,
    }


def _freeze_references(first: dict, second: dict) -> dict:
    references = {}
    for panel_type, expected_size, floor in (
        ("families", FAMILY_SIZE, 26),
        ("conditions", CONDITION_SIZE, 6),
    ):
        if first[panel_type].keys() != second[panel_type].keys():
            raise RuntimeError(f"passing evaluations have different {panel_type}")
        references[panel_type] = {}
        for name in first[panel_type]:
            reference = min(
                first[panel_type][name]["successes"],
                second[panel_type][name]["successes"],
            )
            references[panel_type][name] = {
                "successes": reference,
                "episodes": expected_size,
                "retention_threshold": max(floor, reference - 1),
            }
    return references


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

        evaluations = [_evaluation(record) for record in ordered]
        mastery_checkpoint_states = [
            (
                "pass"
                if evaluation["passed"]
                else "fail" if evaluation["eligible"] else "invalid"
            )
            for evaluation in evaluations
        ]
        first_mastery_index = None
        prior_passing_index = None
        for index, evaluation in enumerate(evaluations):
            if not evaluation["eligible"]:
                continue
            if not evaluation["performance_passed"]:
                prior_passing_index = None
                continue
            if prior_passing_index is not None:
                first_mastery_index = index
                break
            prior_passing_index = index

        two_consecutive = first_mastery_index is not None
        first_mastery_update = updates[first_mastery_index] if two_consecutive else None
        mastery_references = None
        retention_checks = []
        latest_valid_retention_passed = None
        rollback_triggered = False
        rollback_trigger = None
        failure_streaks = {"families": {}, "conditions": {}}
        if two_consecutive:
            mastery_references = _freeze_references(
                evaluations[prior_passing_index],
                evaluations[first_mastery_index],
            )
            failure_streaks = {
                panel_type: {name: 0 for name in mastery_references[panel_type]}
                for panel_type in ("families", "conditions")
            }
            for index in range(first_mastery_index + 1, len(ordered)):
                evaluation = evaluations[index]
                check = {
                    "checkpoint_update": updates[index],
                    "complete": evaluation["complete"],
                    "integrity_passed": evaluation["integrity_passed"],
                    "eligible": evaluation["eligible"],
                    "passed": None,
                    "panels": None,
                    "rollback_triggered": False,
                }
                if not evaluation["eligible"]:
                    retention_checks.append(check)
                    continue

                panel_results = {}
                for panel_type in ("families", "conditions"):
                    if (
                        evaluation[panel_type].keys()
                        != mastery_references[panel_type].keys()
                    ):
                        raise RuntimeError(
                            f"retention evaluation has different {panel_type}"
                        )
                    panel_results[panel_type] = {}
                    for name, reference in mastery_references[panel_type].items():
                        successes = evaluation[panel_type][name]["successes"]
                        passed = successes >= reference["retention_threshold"]
                        failure_streaks[panel_type][name] = (
                            0 if passed else failure_streaks[panel_type][name] + 1
                        )
                        panel_results[panel_type][name] = {
                            "successes": successes,
                            "threshold": reference["retention_threshold"],
                            "passed": passed,
                            "failure_streak": failure_streaks[panel_type][name],
                        }
                        if (
                            not rollback_triggered
                            and failure_streaks[panel_type][name] >= 2
                        ):
                            rollback_triggered = True
                            rollback_trigger = {
                                "checkpoint_update": updates[index],
                                "panel_type": panel_type,
                                "panel": name,
                            }
                            check["rollback_triggered"] = True

                check_passed = all(
                    panel["passed"]
                    for panels in panel_results.values()
                    for panel in panels.values()
                )
                latest_valid_retention_passed = check_passed
                check["passed"] = check_passed
                check["panels"] = panel_results
                retention_checks.append(check)

        histories.append(
            {
                "split": split,
                "stratum": stratum,
                "policy_mode": policy_mode,
                "checkpoint_updates": updates,
                "mastery_checkpoint_states": mastery_checkpoint_states,
                "two_consecutive_mastery": two_consecutive,
                "first_mastery_update": first_mastery_update,
                "mastery_references": mastery_references,
                "latest_valid_retention_passed": (latest_valid_retention_passed),
                "retention_failure_streaks": failure_streaks,
                "rollback_triggered": rollback_triggered,
                "rollback_trigger": rollback_trigger,
                "retention_checks": retention_checks,
            }
        )

    return {
        "schema": "terra_fixed_bank_history_v2",
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
