#!/usr/bin/env python3
"""Paired comparison of evaluate.py results on the same reset maps.

  python scripts/team/compare.py single.json team.json [more.json ...]

The first file is the reference. For each other file it reports success,
successes gained/lost, and on maps both solve:

- the round ratio reference/other (the simulator's decision count), and
- the executed-plan time ratio. The robot keeps only the base poses of
  effective DO actions; its navigation stack drives between them. A machine's
  time is straight-line travel between successive work poses at --nav-speed
  plus scoop cycles for every scooped unit (digs and relifts): --scoop-s per
  --scoop-m3, with a unit of tile_size² × --unit-depth-m (default: a cubic
  cell). Machines of a team work in parallel, so the team time is its slowest
  machine's; waiting for each other is not modeled (optimistic for teams).
"""
import argparse
import json

import numpy as np


def load(path):
    result = json.load(open(path))
    per_env = result["per_env"]
    return result, {key: np.asarray(value) for key, value in per_env.items()}


def machine_seconds(result, per_env, args):
    """[envs, agents] executed-plan seconds, or None for older result files."""
    if "travel_m" not in per_env:
        return None
    cell = result["tile_size_m"]
    depth = cell if args.unit_depth_m is None else args.unit_depth_m
    unit_m3 = cell * cell * depth
    travel = per_env["travel_m"] / args.nav_speed
    scoops = per_env["scooped_units"] * unit_m3 / args.scoop_m3 * args.scoop_s
    return travel + scoops, travel, scoops, unit_m3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--nav-speed", type=float, default=0.5, help="m/s")
    parser.add_argument("--scoop-m3", type=float, default=0.3)
    parser.add_argument("--scoop-s", type=float, default=30.0)
    parser.add_argument("--unit-depth-m", type=float, default=None,
                        help="metres of depth per material unit (default: cell size)")
    args = parser.parse_args()

    ref, ref_env = load(args.results[0])
    ref_time = machine_seconds(ref, ref_env, args)
    print(f"reference {args.results[0]}: agents={ref['agents']} greedy={ref['greedy']} "
          f"success={ref_env['success'].mean():.3f} "
          f"median_steps={np.median(ref_env['steps'][ref_env['success']]):.0f}")
    for path in args.results[1:]:
        other, env = load(path)
        if len(env["success"]) != len(ref_env["success"]) or other["seed"] != ref["seed"]:
            raise ValueError(f"{path} is not paired with the reference (envs/seed differ)")
        both = ref_env["success"] & env["success"]
        rounds = ref_env["steps"][both] / env["steps"][both]
        print(f"{path}: agents={other['agents']} greedy={other['greedy']} "
              f"success={env['success'].mean():.3f} "
              f"(gained {int((env['success'] & ~ref_env['success']).sum())}, "
              f"lost {int((ref_env['success'] & ~env['success']).sum())}) n_both={int(both.sum())}")
        print(f"  rounds: median ref/other {np.median(rounds):.2f}, mean {rounds.mean():.2f}, "
              f"other faster on {(rounds > 1).mean():.2f}")
        other_time = machine_seconds(other, env, args)
        if ref_time is None or other_time is None:
            print("  executed-plan time: not recorded in one of the files")
            continue
        ref_total, _, _, unit_m3 = ref_time
        total, travel, scoops, _ = other_time
        ref_span = ref_total.max(axis=1)[both]
        span = total.max(axis=1)[both]
        speedup = ref_span / span
        volume = ref_env["required_units"][both] * unit_m3
        print(f"  executed-plan time ({unit_m3:.3f} m³/unit): median speedup {np.median(speedup):.2f}, "
              f"mean {speedup.mean():.2f}, faster on {(speedup > 1).mean():.2f}")
        print(f"  reference: {np.median(ref_span) / 60:.1f} min median, "
              f"{np.median(volume / ref_span) * 3600:.1f} m³/h; other: {np.median(span) / 60:.1f} min, "
              f"{np.median(volume / span) * 3600:.1f} m³/h")
        print(f"  other per machine (median over maps): travel {np.median(travel[both], axis=0).round(0).tolist()} s, "
              f"scooping {np.median(scoops[both], axis=0).round(0).tolist()} s, "
              f"scooped units total/reference {np.median(env['scooped_units'][both].sum(1) / ref_env['scooped_units'][both].sum(1)):.2f}")


if __name__ == "__main__":
    main()
