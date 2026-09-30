#!/usr/bin/env python3
"""Collect the GRU generalist evaluations into one CSV and optionally log them to W&B.

usage: collect_results.py OUTPUT_CSV [--wandb]

Every row is one policy evaluated on one panel (greedy, 450 actions). The
dump reach is 6.5 m (the dig reach, frozen v1 benchmark) or 5.5 m (the machine
rule, used for training from u19500 on).
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path("/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923")
LOCAL, EULER = ROOT / "evaluations_local", ROOT / "evaluations_euler"
GRU_UPDATES_55 = (15000, 22500, 25000, 30000, 35000, 40000, 45000, 50000,
                  60000, 70000, 80000, 90000, 100000)
PROMOTION_UPDATES = (40000, 45000, 70000, 80000, 100000)


def evaluations():
    """(policy, update, panel, dump reach m, evaluation directory, host)."""
    yield "ff", 110000, "development", 6.5, EULER / "reference_u110000", "euler_4090"
    for u in (2500, 5000, 7500, 10000, 12500, 15000):
        yield "gru", u, "development", 6.5, LOCAL / f"gru_u{u:06d}", "local_4090"
    yield "ff", 110000, "development", 5.5, LOCAL / "dump55_ff_u110000", "local_4090"
    for u in GRU_UPDATES_55:
        yield "gru", u, "development", 5.5, LOCAL / f"dump55_gru_u{u:06d}", "local_4090"
    yield "ff", 110000, "promotion", 5.5, LOCAL / "promotion55_ff_u110000", "local_4090"
    for u in PROMOTION_UPDATES:
        yield ("gru", u, "promotion", 5.5,
               LOCAL / f"promotion55_gru_gen512_s20260923_update_{u:06d}", "local_4090")


def row_for(policy, update, panel, reach, directory, host):
    (report,) = json.loads((directory / "full608.json").read_text())
    assert report["checkpoint_update"] == update, (directory, report["checkpoint_update"])
    line = [l for l in (directory / "full608.log").read_text().splitlines() if "exact=" in l][-1]
    grades = dict(re.findall(r"(macro|micro_p10|worst)=([0-9.]+)", line))
    maps = report["per_map"]
    row = dict(policy=policy, update=update, panel=panel, dump_reach_m=reach, host=host,
               maps=len(maps), exact=sum(bool(m["success"]) for m in maps),
               macro=float(grades["macro"]), worst_cell=float(grades["worst"]),
               stalled_over_100=sum(m["longest_task_progress_stall_steps"] > 100 for m in maps))
    for family in ("foundation", "trench"):
        members = [m for m in maps if m["family"] == family]
        row[f"{family}_exact"] = sum(bool(m["success"]) for m in members)
        row[f"{family}_maps"] = len(members)
    roads = [m for m in maps if m["family"] == "trench" and "road" in m["primary_cell"]]
    row["road_exact"], row["road_maps"] = sum(bool(m["success"]) for m in roads), len(roads)
    starts = directory / "known_starts" / "summary.json"
    cases = json.loads(starts.read_text())["cases"] if starts.exists() else {}
    for key, case in (("road32", "trn-net4-side1-road"), ("straight32", "trn-straight-side1"),
                      ("tee32", "trn-tee-side2")):
        row[key] = int(cases[case]["greedy_success"].split("/")[0]) if case in cases else None
    return row


def log_to_wandb(rows):
    import wandb

    run = wandb.init(entity="aless-weber-eth", project="mixed-agents",
                     id="gru_gen512_s20260923_evals", name="gru_gen512_s20260923_evals",
                     group="gru_gen512_s20260923", job_type="evaluation", resume="allow",
                     config=dict(training_run="gru_gen512_s20260923",
                                 release_checkpoint="u100000",
                                 dump_reach_switch_update=19500, horizon=450, greedy=True))
    run.log({"evaluations": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    by_update = {}
    for r in rows:
        if r["policy"] != "gru":
            continue
        prefix = f"eval_{r['panel']}_dump{str(r['dump_reach_m']).replace('.', 'p')}"
        metrics = by_update.setdefault(r["update"], {})
        for key in ("exact", "foundation_exact", "trench_exact", "stalled_over_100",
                    "worst_cell", "road32", "straight32", "tee32"):
            if r[key] is not None:
                metrics[f"{prefix}/{key}"] = r[key]
    for update in sorted(by_update):
        run.log(by_update[update], step=update)
    for r in rows:
        if r["policy"] == "ff":
            run.summary[f"teacher_u110000/{r['panel']}_dump{r['dump_reach_m']}_exact"] = r["exact"]
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    rows = [row_for(*spec) for spec in evaluations()]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")
    if args.wandb:
        log_to_wandb(rows)


if __name__ == "__main__":
    main()
