#!/usr/bin/env python3
"""Summarize one milestone evaluation, optionally paired against a reference.

usage: summarize_eval.py EVAL_DIR [REFERENCE_EVAL_DIR]
"""
import collections
import json
import sys
from pathlib import Path


def load(directory):
    directory = Path(directory)
    (report,) = json.loads((directory / "full608.json").read_text())
    starts = json.loads((directory / "known_starts" / "summary.json").read_text())
    return report, starts


def panel(report):
    rows = report["per_map"]
    count = collections.Counter()
    for row in rows:
        groups = [row["family"]]
        if row["family"] == "trench" and "road" in row["primary_cell"]:
            groups.append("road")
        for group in groups:
            count[group, "n"] += 1
            count[group, "ok"] += bool(row["success"])
    stalls = sum(row["longest_task_progress_stall_steps"] > 100 for row in rows)
    return dict(
        update=report["checkpoint_update"],
        exact=f"{sum(bool(r['success']) for r in rows)}/{len(rows)}",
        foundation=f"{count['foundation', 'ok']}/{count['foundation', 'n']}",
        trench=f"{count['trench', 'ok']}/{count['trench', 'n']}",
        road=f"{count['road', 'ok']}/{count['road', 'n']}",
        episodes_with_stall_over_100=stalls,
        failures={r["map_id"]: r["primary_cell"] for r in rows if not r["success"]},
    )


def main():
    report, starts = load(sys.argv[1])
    summary = panel(report)
    summary["known_starts"] = {
        name: dict(greedy=case["greedy_success"], stalls=case["starts_with_stall_over_100_steps"],
                   start_search=case["start_search_success"])
        for name, case in starts["cases"].items()}
    if len(sys.argv) > 2:
        reference, _ = load(sys.argv[2])
        mine = {r["map_id"]: bool(r["success"]) for r in report["per_map"]}
        theirs = {r["map_id"]: bool(r["success"]) for r in reference["per_map"]}
        assert mine.keys() == theirs.keys()
        summary["versus_reference"] = dict(
            reference_update=reference["checkpoint_update"],
            gained=sum(mine[k] and not theirs[k] for k in mine),
            lost=sum(theirs[k] and not mine[k] for k in mine),
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
