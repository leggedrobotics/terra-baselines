#!/usr/bin/env python3
"""Which held-out maps are byte-identical between the v5m and v6m banks?

M2 runs the wave arm on `curriculum_v5m` (116 held-out maps, 29 conditions) and
the two dose arms on `curriculum_v6m` (128, 32). Prediction 1 compares them on
deep-rung held-out, so the comparison is only sound if the maps of the partitions
it is decided on are the SAME maps. REVIEW_V6 D-6 measured that 40 of the 128 v6m
held-out maps differ from v5m's; this script says exactly which, per condition
and per layer, so the pre-registration can name the not-like-for-like rows
instead of asserting comparability.

    PYTHONPATH=$TERRA:$PWD python scripts/check_m2_holdout_comparability.py \\
        --v5m terra_data/curriculum_v5m --v6m terra_data/curriculum_v6m

Exits non-zero if either prediction-1 partition is not byte-identical across the
two banks, i.e. if prediction 1 has silently become a cross-bank comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LAYERS = ("images", "distance", "occupancy", "dumpability", "actions")

# spec v6 section 3's deep set
DEEP_TAXONOMY = (
    "fnd-slab-apron-c1p2",
    "trn-net4-side2",
    "fnd-slab-side1-obj",
    "fnd-proc-side1-road",
    "trn-net3-side1-road",
    "trn-net4-side1-road",
)
# REVIEW_V6 prediction 1(b): the six worst conditions by M1-C held-out graded
DEEP_MEASURED = (
    "fnd-slab-apron-d16",
    "trn-straight-side1",
    "trn-net3-side1-road",
    "fnd-slab-side1-obj",
    "fnd-slab-side1",
    "trn-straight-side1-tight",
)


def fingerprint(bank_root: Path) -> dict[str, dict[int, dict[str, str]]]:
    directory = bank_root / "held_out" / "all"
    rows = [
        json.loads(line)
        for line in (directory / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    out: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        slot = row["slot_index"]
        layers = {
            layer: hashlib.sha256(
                np.load(directory / layer / f"img_{slot}.npy").tobytes()
            ).hexdigest()
            for layer in LAYERS
        }
        out.setdefault(row["condition_id"], {})[row["map_index"]] = layers
    return out


def partition_report(name, conditions, v5m, v6m) -> bool:
    print(f"\n{name}")
    identical = True
    for condition in conditions:
        if condition not in v5m or condition not in v6m:
            print(f"  {condition:28s} ABSENT (v5m={condition in v5m} v6m={condition in v6m})")
            identical = False
            continue
        maps = v5m[condition]
        same = sum(
            1
            for index, layers in maps.items()
            if index in v6m[condition] and v6m[condition][index] == layers
        )
        verdict = "identical" if same == len(maps) == len(v6m[condition]) else "DIFFERS"
        identical &= verdict == "identical"
        print(f"  {condition:28s} {same}/{len(maps)} maps identical on all 5 layers  {verdict}")
    print(f"  => {'ALL BYTE-IDENTICAL' if identical else 'NOT COMPARABLE'}")
    return identical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5m", type=Path, default=Path("terra_data/curriculum_v5m"))
    parser.add_argument("--v6m", type=Path, default=Path("terra_data/curriculum_v6m"))
    args = parser.parse_args()

    v5m = fingerprint(args.v5m)
    v6m = fingerprint(args.v6m)
    print(f"v5m held-out: {len(v5m)} conditions, {sum(len(m) for m in v5m.values())} maps")
    print(f"v6m held-out: {len(v6m)} conditions, {sum(len(m) for m in v6m.values())} maps")

    ok_taxonomy = partition_report(
        "prediction 1(a) taxonomy-deep 6", DEEP_TAXONOMY, v5m, v6m
    )
    ok_measured = partition_report(
        "prediction 1(b) measurement-hard 6", DEEP_MEASURED, v5m, v6m
    )

    shared = sorted(set(v5m) & set(v6m))
    changed = {
        condition: sum(
            1
            for index, layers in v5m[condition].items()
            if index not in v6m[condition] or v6m[condition][index] != layers
        )
        for condition in shared
    }
    changed = {c: n for c, n in changed.items() if n}
    print(
        f"\nof {len(shared)} shared conditions, {len(changed)} carry a changed "
        f"held-out map ({sum(changed.values())} maps); v6m-only conditions: "
        f"{sorted(set(v6m) - set(v5m))}"
    )
    for condition, count in changed.items():
        print(f"  {condition:28s} {count}/{len(v5m[condition])} changed")
    print("\nNOT like-for-like in any M1<->M2 or wave<->dose table: the rows above.")

    return 0 if (ok_taxonomy and ok_measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
