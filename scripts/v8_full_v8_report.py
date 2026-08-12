#!/usr/bin/env python3
"""Build the primary whole-V8 report for the compact and 10M policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.v8_10m_compare import panel_snapshot

POLICIES = ("compact", "10m")
PANELS = ("promotion", "development")


def _read_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty fixed-evaluation list")
    return records


def build_report(paths: dict[tuple[str, str], Path]) -> dict:
    records = {key: _read_records(path) for key, path in paths.items()}
    schedules = {
        key: tuple(int(row["checkpoint_update"]) for row in values)
        for key, values in records.items()
    }
    reference = schedules[(POLICIES[0], PANELS[0])]
    if any(schedule != reference for schedule in schedules.values()):
        raise ValueError(f"fixed-evaluation checkpoint schedules differ: {schedules}")

    policy_rows = []
    condition_rows = []
    for policy in POLICIES:
        for panel in PANELS:
            for record in records[(policy, panel)]:
                snapshot = panel_snapshot(record)
                update = int(record["checkpoint_update"])
                family = snapshot["by_family"]
                policy_rows.append(
                    {
                        "policy": policy,
                        "panel": panel,
                        "update": update,
                        "exact_successes": snapshot["exact_successes"],
                        "episodes": snapshot["episodes"],
                        "exact_success_rate": (
                            snapshot["exact_successes"] / snapshot["episodes"]
                        ),
                        "macro_completion": snapshot["macro_completion"],
                        "foundation_exact_successes": family["foundation"]["successes"],
                        "foundation_episodes": family["foundation"]["episodes"],
                        "foundation_macro_completion": family["foundation"][
                            "macro_completion"
                        ],
                        "trench_exact_successes": family["trench"]["successes"],
                        "trench_episodes": family["trench"]["episodes"],
                        "trench_macro_completion": family["trench"]["macro_completion"],
                        "micro_p10": snapshot["micro_p10"],
                        "worst_condition": snapshot["worst_condition"],
                        "worst_condition_completion": snapshot[
                            "worst_condition_completion"
                        ],
                    }
                )
                for condition_id, result in snapshot["by_condition"].items():
                    condition_rows.append(
                        {
                            "policy": policy,
                            "panel": panel,
                            "update": update,
                            "condition_id": condition_id,
                            "curriculum_stage": (
                                "nearby" if condition_id.startswith("v7-") else "full"
                            ),
                            "family": result["family"],
                            "exact_successes": result["successes"],
                            "episodes": result["episodes"],
                            "exact_success_rate": (
                                result["successes"] / result["episodes"]
                            ),
                            "mean_completion": result["mean_completion"],
                        }
                    )
    return {
        "schema": "terra_v8_whole_distribution_report_v1",
        "primary_metric": "fixed/development/exact_success_rate",
        "secondary_metric": "fixed/development/macro_completion",
        "training_online_metric_role": "active_stage_diagnostic_only",
        "checkpoint_updates": list(reference),
        "policy_rows": policy_rows,
        "condition_rows": condition_rows,
    }


def write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "leaderboard.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, rows in (
        ("policy_leaderboard.csv", report["policy_rows"]),
        ("per_condition.csv", report["condition_rows"]),
    ):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Whole-V8 fixed benchmark",
        "",
        "Primary: deterministic exact completion on the source-disjoint 720-map "
        "development panel at horizon 450. Macro completion is condition-balanced "
        "graded progress. `train/episode_success_rate` is an active-stage diagnostic, "
        "not whole-V8 performance.",
        "",
        "| Policy | Panel | Update | Exact | Exact rate | Macro | Foundation macro | Trench macro | Worst condition | Worst |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in report["policy_rows"]:
        lines.append(
            f"| {row['policy']} | {row['panel']} | {row['update']} | "
            f"{row['exact_successes']}/{row['episodes']} | "
            f"{row['exact_success_rate']:.3f} | {row['macro_completion']:.3f} | "
            f"{row['foundation_macro_completion']:.3f} | "
            f"{row['trench_macro_completion']:.3f} | "
            f"{row['worst_condition']} | {row['worst_condition_completion']:.3f} |"
        )
    lines.extend(
        [
            "",
            "See `per_condition.csv` for every family and condition. Promotion and "
            "development remain separate; development never promotes a checkpoint.",
        ]
    )
    (output_dir / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for policy in POLICIES:
        for panel in PANELS:
            parser.add_argument(f"--{policy}-{panel}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        (policy, panel): getattr(args, f"{policy}_{panel}")
        for policy in POLICIES
        for panel in PANELS
    }
    write_report(build_report(paths), args.output_dir)
    print(args.output_dir / "LEADERBOARD.md")


if __name__ == "__main__":
    main()
