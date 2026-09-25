#!/usr/bin/env python3
"""Greedy 32-start panels on known held-out geometries for one checkpoint.

Reuses the fixed-start evaluator of the September 21 test-time study (same
maps, starts and execution layout as the u110000 road32/straight32 results).
Reports completion, stall length and start search: the best greedy plan over
the 32 starts, since initial positioning costs almost nothing on the machine.

PYTHONPATH must contain the Terra runtime, this repository and the directory
of the fixed-start helper (it imports ``run.write_json``).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

ADAPTATION = Path("/home/lorenzo/moleworks/.artifacts/terra_test_time_compute_20260921/adaptation")
CASES = ("trn-net4-side1-road", "trn-straight-side1", "trn-tee-side2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, default=ADAPTATION / "maps")
    parser.add_argument("--helper", type=Path, default=ADAPTATION / "evaluate.py")
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--dump-max-radius-m", type=float, default=None,
                        help="excavator dump reach in metres; default: the checkpoint's")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("EVAL_FORWARD_CHUNK", "32")
    os.environ["WANDB_MODE"] = "disabled"

    from utils.helpers import load_pkl_object, register_checkpoint_config_classes

    spec = importlib.util.spec_from_file_location("fixed_start_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(str(args.checkpoint))
    if args.dump_max_radius_m is not None:
        checkpoint["train_config"].dump_max_radius_m = args.dump_max_radius_m
    cases = {c["case_id"]: c for c in json.loads((args.maps_root / "maps.json").read_text())["cases"]}
    summary = dict(checkpoint=str(args.checkpoint), update=int(checkpoint["next_update"]), cases={})
    for name in args.cases:
        started = time.monotonic()
        evaluator = helper.FixedEvaluation(checkpoint, cases[name], args.maps_root,
                                           args.output / name, time.monotonic() + 7200)
        rows = evaluator.rollout(checkpoint, "greedy", 1, True, 710000)
        (args.output / name / "greedy.json").write_text(json.dumps(rows, indent=1))
        successes = [row for row in rows if row["success"]]
        stalls = [row["behavior"]["longest_task_progress_stall_steps"] for row in rows]
        best = min(successes, key=lambda row: (
            row["behavior"]["retained_work_straight_line_distance_m"],
            row["behavior"]["retained_work_setups"])) if successes else None
        summary["cases"][name] = dict(
            greedy_success=f"{len(successes)}/{len(rows)}",
            mean_dug_fraction=sum(row["dug_fraction"] for row in rows) / len(rows),
            starts_with_stall_over_100_steps=sum(stall > 100 for stall in stalls),
            start_search_success=best is not None,
            start_search_best=None if best is None else dict(
                reset_seed=best["reset_seed"], steps=best["steps"],
                retained_distance_m=best["behavior"]["retained_work_straight_line_distance_m"],
                retained_setups=best["behavior"]["retained_work_setups"]),
            seconds=time.monotonic() - started,
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(name, summary["cases"][name], flush=True)


if __name__ == "__main__":
    main()
