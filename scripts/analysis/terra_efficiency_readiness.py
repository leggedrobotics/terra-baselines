#!/usr/bin/env python3
"""Report numerical broad-efficiency readiness from two fixed greedy450 panels.

This checks proposed engineering thresholds, not statistical significance or
permission to change the objective. It cannot establish a teacher-free hold or
explain lost maps; those remain separate review requirements.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundation_teacher_release.compare import load_report


COHORTS = {
    "foundation": (384, 365, lambda row: row["family"] == "foundation"),
    "trench": (224, 213, lambda row: row["family"] == "trench"),
    "road_trench": (32, 31, lambda row: row["family"] == "trench" and "road" in row["primary_cell"]),
}


def analyze(previous, current, transitions_per_update):
    if transitions_per_update <= 0:
        raise ValueError("transitions_per_update must be positive")
    update_gap = current["checkpoint_update"] - previous["checkpoint_update"]
    if update_gap <= 0:
        raise ValueError("Current checkpoint must be later than previous checkpoint")
    for key in ("manifest_sha256", "reset_verification", "seed", "horizon",
                "completion_contract", "material_progress_contract"):
        if previous[key] != current[key]:
            raise ValueError(f"Unmatched fixed-panel {key}")
    if not all(report["reset_verification"]["passed"] for report in (previous, current)):
        raise ValueError("Both panels must have verified complete resets")
    a = {row["episode_id"]: row for row in previous["per_map"]}
    b = {row["episode_id"]: row for row in current["per_map"]}
    if a.keys() != b.keys():
        raise ValueError("Episode sets differ")
    identity = ("scenario_id", "source_id", "reset_seed", "slot_index", "family", "primary_cell")
    if any([a[key][field] for field in identity] != [b[key][field] for field in identity] for key in a):
        raise ValueError("Paired initial episode identities differ")

    cohorts = {}
    for name, (expected, floor, select) in COHORTS.items():
        pairs = [(key, a[key], b[key]) for key in a if select(a[key])]
        if len(pairs) != expected:
            raise ValueError(f"Unexpected {name} cohort count: {len(pairs)}")
        accepted = [
            [old["terminal_soil_fraction"] for _, old, new in pairs],
            [new["terminal_soil_fraction"] for _, old, new in pairs],
        ]
        if not all(value is not None and math.isfinite(value) for values in accepted for value in values):
            raise ValueError(f"Unavailable accepted-material progress in {name}")
        before = sum(old["success"] for _, old, new in pairs)
        after = sum(new["success"] for _, old, new in pairs)
        gained = [key for key, old, new in pairs if not old["success"] and new["success"]]
        lost = [key for key, old, new in pairs if old["success"] and not new["success"]]
        cohorts[name] = {
            "episodes": expected, "success_floor": floor,
            "previous_successes": before, "current_successes": after,
            "both_completion_floors_pass": min(before, after) >= floor,
            "gained": len(gained), "lost": len(lost),
            "gained_episode_ids": gained, "lost_episode_ids": lost,
            "previous_accepted_material_mean": mean(accepted[0]),
            "current_accepted_material_mean": mean(accepted[1]),
            "accepted_material_nonregression": mean(accepted[1]) >= mean(accepted[0]) - 1e-8,
        }

    conditions = {}
    for condition in sorted({row["primary_cell"] for row in a.values()}):
        pairs = [(a[key], b[key]) for key in a if a[key]["primary_cell"] == condition]
        old_count = sum(old["success"] for old, new in pairs)
        new_count = sum(new["success"] for old, new in pairs)
        conditions[condition] = {
            "episodes": len(pairs), "previous_successes": old_count, "current_successes": new_count,
            "previous_completion": old_count / len(pairs), "current_completion": new_count / len(pairs),
            "both_80_percent_floor_pass": min(old_count, new_count) / len(pairs) >= 0.8,
        }
    transition_gap = update_gap * transitions_per_update
    checks = {
        "at_least_40960000_transitions_apart": transition_gap >= 40_960_000,
        "both_cohort_completion_floors": all(c["both_completion_floors_pass"] for c in cohorts.values()),
        "both_condition_completion_floors": all(c["both_80_percent_floor_pass"] for c in conditions.values()),
        "accepted_material_nonregression": all(c["accepted_material_nonregression"] for c in cohorts.values()),
        "complete_panels_and_integrity": True,  # Enforced by load_report.
    }
    return {
        "previous_update": previous["checkpoint_update"], "current_update": current["checkpoint_update"],
        "transitions_per_update": transitions_per_update, "transition_gap": transition_gap,
        "checks": checks, "numerical_readiness": all(checks.values()),
        "cohorts": cohorts, "conditions": conditions,
        "unverified_requirements": [
            "A teacher-free hold must be established from the training record.",
            "Review every remaining failure and previously reliable map lost; numerical readiness is not promotion.",
            "Verify retained-pose behavior and material integrity before choosing efficiency coefficients.",
        ],
        "scope": "Proposed engineering thresholds on one fixed panel; not a significance test or an automatic promotion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--transitions-per-update", type=int, required=True,
                        help="Explicit global environments times rollout length; same layout between checkpoints.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(load_report(args.previous), load_report(args.current), args.transitions_per_update)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("previous_update", "current_update", "transition_gap",
                                                 "checks", "numerical_readiness")}))


if __name__ == "__main__":
    main()
