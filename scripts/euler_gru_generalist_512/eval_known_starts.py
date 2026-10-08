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
    # Terra's machine working rules (metres; 0 = off); default: the checkpoint's.
    for flag in ("--dig-min-radius-m", "--dump-min-radius-m", "--dug-clearance-m",
                 "--dump-min-dug-distance-m"):
        parser.add_argument(flag, type=float, default=None)
    parser.add_argument("--centre-chassis-on-base", action=argparse.BooleanOptionalAction,
                        default=None, help="chassis raster centred on the base cell")
    parser.add_argument("--pull-direction-alignment", action=argparse.BooleanOptionalAction,
                        default=None, help="per-cell edge/trench pull alignment; default: the checkpoint's")
    for flag in ("--edge-band-width-m", "--edge-pull-tolerance-rad", "--trench-pull-tolerance-rad", "--dig-pull-min-length-m"):
        parser.add_argument(flag, type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("EVAL_FORWARD_CHUNK", "32")
    os.environ["WANDB_MODE"] = "disabled"

    from utils.helpers import (
        checkpoint_evaluation_config, checkpoint_pull_direction_rules,
        load_pkl_object, register_checkpoint_config_classes,
    )

    spec = importlib.util.spec_from_file_location("fixed_start_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(str(args.checkpoint))
    checkpoint["train_config"] = checkpoint_evaluation_config(checkpoint)
    if args.dump_max_radius_m is not None:
        checkpoint["train_config"].dump_max_radius_m = args.dump_max_radius_m
    from train_mixed import MACHINE_RULE_FIELDS, PULL_DIRECTION_RULE_FIELDS, apply_pull_direction_rules
    for name, cast in MACHINE_RULE_FIELDS.items():
        if getattr(args, name) is not None:
            setattr(checkpoint["train_config"], name, cast(getattr(args, name)))
    for name, cast in PULL_DIRECTION_RULE_FIELDS.items():
        if getattr(args, name) is not None:
            setattr(checkpoint["train_config"], name, cast(getattr(args, name)))
    # The helper resolves its own evaluation config from both saved copies.
    # Apply deliberate CLI overrides to its in-memory env copy as well.
    if checkpoint.get("env_config") is not None:
        checkpoint["env_config"] = apply_pull_direction_rules(
            checkpoint["env_config"], checkpoint["train_config"]
        )
    cases = {c["case_id"]: c for c in json.loads((args.maps_root / "maps.json").read_text())["cases"]}
    summary = dict(checkpoint=str(args.checkpoint), update=int(checkpoint["next_update"]), cases={})
    rules = {name: getattr(checkpoint["train_config"], name, None)
             for name in ("dump_max_radius_m", *MACHINE_RULE_FIELDS)}
    if any(rules[name] for name in MACHINE_RULE_FIELDS):
        summary["agent_rules"] = rules
    pull_rules = checkpoint_pull_direction_rules(checkpoint)
    if pull_rules["pull_direction_alignment"] or any(
        getattr(args, name) is not None for name in PULL_DIRECTION_RULE_FIELDS
    ):
        summary["pull_direction_rules"] = pull_rules
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
