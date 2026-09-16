#!/usr/bin/env python3
"""Compare two complete fixed panels and decide the bounded teacher-free hold."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_report(path):
    report = json.loads(Path(path).read_text())
    if isinstance(report, list):
        if len(report) != 1:
            raise ValueError("Expected exactly one complete panel")
        report = report[0]
    rows = report["per_map"]
    if len(rows) != 608 or len({row["episode_id"] for row in rows}) != 608:
        raise ValueError("Expected all 608 distinct development episodes")
    if any(row["integrity_failure"] or row["integrity_unavailable"] or not row["terminated"]
           for row in rows):
        raise ValueError("Incomplete episode or failed/unavailable integrity check")
    if report["horizon"] != 450 or not report["deterministic"]:
        raise ValueError("Expected greedy450 evaluation")
    return report


def compare(control, treatment):
    for key in ("manifest_sha256", "reset_verification", "seed", "horizon", "checkpoint_update"):
        if control[key] != treatment[key]:
            raise ValueError(f"Unmatched {key}")
    if control["checkpoint_update"] not in (6250, 7500):
        raise ValueError("Expected the u6250 or u7500 decision milestone")
    if control.get("foundation_teacher_release_state") is not None:
        raise ValueError("Control unexpectedly released its foundation teacher")
    release = treatment.get("foundation_teacher_release_state")
    if (not isinstance(release, dict) or release.get("origin_update") != 5000
            or release.get("duration_updates") != 1250
            or not math.isclose(release.get("start_coefficient", -1),
                                .5 * (1 + math.cos(math.pi * 5000 / 20000)))):
        raise ValueError("Treatment must contain the foundation-only u5000 release")
    if control["checkpoint_sha256"] == treatment["checkpoint_sha256"]:
        raise ValueError("Control and treatment cannot be the same checkpoint")
    contracts = []
    for report, name in ((control, "control"), (treatment, "foundation_release")):
        contract = json.loads(json.dumps(report["treatment_fingerprint"]["contract"]))
        if contract["run"].pop("name") != f"foundation-teacher-{name}":
            raise ValueError("Report does not belong to the expected experiment arm")
        saved_release = contract.pop("foundation_teacher_release", None)
        if saved_release != report.get("foundation_teacher_release_state"):
            raise ValueError("Report release state differs from its treatment description")
        contracts.append(contract)
    if contracts[0] != contracts[1]:
        raise ValueError("Arms differ beyond the declared foundation teacher release")
    identity = ("episode_id", "scenario_id", "source_id", "reset_seed", "slot_index")
    a = control["per_map"]
    b = treatment["per_map"]
    if [[row[k] for k in identity] for row in a] != [[row[k] for k in identity] for row in b]:
        raise ValueError("Episode identities or ordering do not match")
    metrics = ("dig_fraction", "terminal_soil_fraction", "off_zone_staged_soil_fraction",
               "loaded_soil_fraction", "longest_material_stall_steps", "longest_task_progress_stall_steps",
               "unique_area_per_retained_productive_setup_m2", "retained_work_inter_setup_straight_line_m",
               "fresh_union_edge_adjacency_fraction", "mean_dig_lateral_score", "base_travel_m")
    cohorts = {}
    selectors = {
        "foundation": lambda r: r["family"] == "foundation",
        "trench": lambda r: r["family"] == "trench",
        "road_trench": lambda r: r["family"] == "trench" and "road" in r["primary_cell"],
    }
    for name, select in selectors.items():
        pairs = [(x, y) for x, y in zip(a, b) if select(x)]
        common = [(x, y) for x, y in pairs if x["success"] and y["success"]]
        stats = dict(episodes=len(pairs), control_successes=sum(x["success"] for x, y in pairs),
                     treatment_successes=sum(y["success"] for x, y in pairs),
                     gained=sum(not x["success"] and y["success"] for x, y in pairs),
                     lost=sum(x["success"] and not y["success"] for x, y in pairs),
                     common_successes=len(common))
        for label, rows in (("all", pairs), ("common_success", common)):
            stats[label] = {}
            for key in metrics:
                values = [(x[key], y[key]) for x, y in rows
                          if x.get(key) is not None and y.get(key) is not None
                          and math.isfinite(x[key]) and math.isfinite(y[key])]
                stats[label][key] = dict(paired_n=len(values),
                    control=mean(x for x, y in values) if values else None,
                    treatment=mean(y for x, y in values) if values else None)
        cohorts[name] = stats
    if [cohorts[n]["episodes"] for n in selectors] != [384, 224, 32]:
        raise ValueError("Unexpected family panel counts")
    conditions = {}
    for name in sorted({row["primary_cell"] for row in a}):
        pairs = [(x, y) for x, y in zip(a, b) if x["primary_cell"] == name]
        conditions[name] = dict(episodes=len(pairs), control=sum(x["success"] for x, y in pairs),
                                treatment=sum(y["success"] for x, y in pairs))
    delta = {name: stats["treatment_successes"] - stats["control_successes"]
             for name, stats in cohorts.items()}
    foundation = cohorts["foundation"]["all"]
    # Explicit engineering stop rules for this first screen, not significance tests.
    checks = dict(
        foundation_completion=delta["foundation"] >= 0,
        trench_retention=cohorts["trench"]["lost"] <= 2,
        road_retention=cohorts["road_trench"]["lost"] <= 1,
        foundation_excavation=(foundation["dig_fraction"]["treatment"]
                               >= foundation["dig_fraction"]["control"] - .005),
        foundation_disposal=(foundation["terminal_soil_fraction"]["treatment"]
                             >= foundation["terminal_soil_fraction"]["control"] - .005),
        condition_retention=all(v["treatment"] >= v["control"] - 2 for v in conditions.values()),
    )
    return dict(update=control["checkpoint_update"], cohorts=cohorts, conditions=conditions,
                hold_checks=checks, allow_bounded_hold=all(checks.values()),
                useful_engineering_signal=all(checks.values()) and delta["foundation"] >= 8,
                scope="One matched seed; no significance or broad efficiency promotion claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load_report(args.control), load_report(args.treatment))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                     ("update", "hold_checks", "allow_bounded_hold", "useful_engineering_signal")}))


if __name__ == "__main__":
    main()
