#!/usr/bin/env python3
"""Materialize the v6-main review bank as Terra-loadable DOSE levels.

`CURRICULUM_SPEC_V6.md` §3 replaces M1's admission waves: a level is no longer
"which conditions exist" but "how much mass each condition gets". All 32 v6
conditions are present in EVERY level — support is constant from update zero —
and only the multiplicity-weighted mixture anneals:

    level | easy set | mid rungs | deep rungs + T2
    L0'   |   ~80%   |   ~18%    |      ~2%
    L1'   |   ~55%   |   ~35%    |     ~10%
    L2'   |   ~35%   |   ~40%    |     ~25%
    L3'   |   ~20%   |   ~35%    |     ~45%

Everything else is M1's contract, unchanged and deliberately so: the loader's
`terra_exact_map_dataset_v1` sidecar, contiguous `img_1..img_N.npy` under the
five array kinds plus `metadata/trench_N.json`, one slot count shared by every
level (the buffer is a dense `[level, n_maps, W, H]` array), hardlinked repeats,
the `manifest.jsonl` rows `eval_fixed_bank.py` groups by, and the split-disjoint
`source_registry.jsonl` the loader re-validates on every load.

Slot duplication IS the sampling weight — the buffer samples slots uniformly and
ignores `slot_weight` — so the dose is expressed purely as how many slots each
condition owns.

Allocation, in three deterministic steps (documented per level in `dose.json`):

1. each family gets exactly `level_size / 2` slots (exact 50/50, as in M1);
2. inside a family, the level's target shares split that budget across the three
   sets by largest-remainder rounding, so the realised masses are the closest
   integers to the table and their sum is exact;
3. inside a (family, set) cell, `round_robin_slots` deals condition-major /
   map-minor, so condition slot counts differ by at most one and map
   multiplicities inside a condition differ by at most one.

Every condition holds >= 1 slot at every level; that is asserted, not assumed.

Outputs (under --out):
    train/L0p train/L1p train/L2p train/L3p   dose levels, --level-size slots each
    train/*/dose.json                          exact multiplicity vector for that level
    held_out/all                               32 conditions x 4 held-out maps = 128
    source_registry.jsonl                      map_id -> (dig source_id, split)
    curriculum_v6m_provenance.json             bank hashes + the full dose table

M2-wave-long is NOT built here: it reuses M1-C's existing `terra_data/curriculum_v5m`
level directories verbatim. See the module note at the bottom of this docstring
and docs/EXPERIMENTS.md for why.

    python3 scripts/build_v6m_dataset.py \\
        --bank /home/lorenzo/moleworks/.artifacts/terra_map_distribution_review_v6/main/review_bank \\
        --out terra_data/curriculum_v6m --holdout 3,7,11,14 --level-size 864
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v5m_dataset import (  # noqa: F401  (shared contract, imported not copied)
    ARRAY_KINDS,
    materialize,
    read_conditions,
    round_robin_slots,
    sha256_file,
)

DEFAULT_BANK = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_map_distribution_review_v6/main/review_bank"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "terra_data" / "curriculum_v6m"

# Spec v6 §3, verbatim set membership.
#
# EASY = the 8 T0 anchors + the 3 mini-junction rungs + obj1 + c3x + d12 +
#        altsides + seg2 + tee-side2. "gapped-ring variants (they ARE the T0
#        rings now)" is a clarification, not an addition: the three capped rings
#        already sit in the T0 anchor set and D1 changed their masks, not their
#        membership.
EASY = [
    # T0 anchors
    "fnd-slab-ring3x",
    "fnd-proc-ring3x",
    "fnd-slab-lg-ring3x",
    "fnd-slab-apron-near",
    "fnd-slab-apron-c3x",
    "trn-straight-side2",
    "trn-straight-side1",
    "trn-tee-side2",
    # D2 mini-junction rungs
    "trn-tee-side2-s",
    "trn-net3-side2-s",
    "trn-net4-side2-s",
    # shallowest rung of every remaining axis
    "fnd-slab-ring3x-obj1",
    "fnd-slab-apron-d12",
    "trn-straight-altsides",
    "trn-seg2-side2",
]
MID = [
    "fnd-slab-apron-c2x",
    "fnd-slab-apron-c1p6",
    "fnd-slab-apron-d16",
    "fnd-slab-side1",
    "fnd-slab-split",
    "fnd-strips-ring3x",
    "fnd-slab-ring3x-obj",
    "fnd-slab-ring3x-road",
    "trn-seg3-side2",
    "trn-net3-side2",
    "trn-straight-side1-tight",
]
DEEP = [
    "fnd-slab-apron-c1p2",
    "fnd-slab-side1-obj",
    "fnd-proc-side1-road",
    "trn-net4-side2",
    "trn-net3-side1-road",
    "trn-net4-side1-road",
]
SETS = {"easy": EASY, "mid": MID, "deep": DEEP}
SET_ORDER = ("easy", "mid", "deep")

# Spec v6 §3 mass table.
DOSE_LEVELS: dict[str, dict[str, float]] = {
    "L0p": {"easy": 0.80, "mid": 0.18, "deep": 0.02},
    "L1p": {"easy": 0.55, "mid": 0.35, "deep": 0.10},
    "L2p": {"easy": 0.35, "mid": 0.40, "deep": 0.25},
    "L3p": {"easy": 0.20, "mid": 0.35, "deep": 0.45},
}
LEVEL_ORDER = ("L0p", "L1p", "L2p", "L3p")


def largest_remainder(shares: dict[str, float], total: int) -> dict[str, int]:
    """Integer split of `total` by `shares`, summing to `total` exactly."""
    raw = {key: shares[key] * total for key in shares}
    floor = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(floor.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - floor[key]), key))
    for key in order[:remaining]:
        floor[key] += 1
    return floor


def build_dose_slots(
    shares: dict[str, float],
    meta: dict[str, dict],
    map_indices: list[int],
    level_size: int,
) -> tuple[list[tuple[str, int]], dict]:
    if level_size % 2:
        raise SystemExit("--level-size must be even for an exact 50/50 family split")
    per_family = level_size // 2
    by_family_set: dict[str, dict[str, list[str]]] = {
        family: {name: [] for name in SET_ORDER}
        for family in ("foundation", "trench")
    }
    for name in SET_ORDER:
        for condition_id in SETS[name]:
            by_family_set[meta[condition_id]["family"]][name].append(condition_id)

    per_family_slots: dict[str, list[tuple[str, int]]] = {}
    budgets: dict[str, dict[str, int]] = {}
    for family in ("foundation", "trench"):
        budget = largest_remainder(shares, per_family)
        # A set with no member in this family cannot hold slots; its mass moves
        # to the next-heavier set in the same family so the family total stays
        # exact. (v6 has at least one member of every set in both families, so
        # this never fires — it is here so a future table edit fails loudly in
        # the numbers rather than silently under-filling a level.)
        for name in SET_ORDER:
            if not by_family_set[family][name] and budget[name]:
                raise SystemExit(
                    f"{family} has no {name} condition but was given "
                    f"{budget[name]} slots"
                )
        budgets[family] = budget
        slots: list[tuple[str, int]] = []
        for name in SET_ORDER:
            slots.extend(
                round_robin_slots(
                    by_family_set[family][name], map_indices, budget[name]
                )
            )
        per_family_slots[family] = slots

    interleaved: list[tuple[str, int]] = []
    for a, b in zip(per_family_slots["foundation"], per_family_slots["trench"]):
        interleaved.append(a)
        interleaved.append(b)
    return interleaved, budgets


def dose_report(
    slots: list[tuple[str, int]],
    budgets: dict[str, dict[str, int]],
    shares: dict[str, float],
    meta: dict[str, dict],
) -> dict:
    counts: dict[str, int] = {}
    for condition_id, _ in slots:
        counts[condition_id] = counts.get(condition_id, 0) + 1
    set_of = {c: name for name in SET_ORDER for c in SETS[name]}
    realised = {name: 0 for name in SET_ORDER}
    for condition_id, count in counts.items():
        realised[set_of[condition_id]] += count
    total = len(slots)
    missing = sorted(set(meta) - set(counts))
    if missing:
        raise SystemExit(f"constant support violated: {missing} hold no slot")
    return {
        "slots": total,
        "target_mass": dict(shares),
        "realised_mass": {
            name: round(realised[name] / total, 5) for name in SET_ORDER
        },
        "set_slots": realised,
        "family_set_budgets": budgets,
        "slots_per_condition": dict(sorted(counts.items())),
        "slots_per_condition_by_set": {
            name: {c: counts[c] for c in SETS[name]} for name in SET_ORDER
        },
        "family_slots": {
            family: sum(
                counts[c] for c in counts if meta[c]["family"] == family
            )
            for family in ("foundation", "trench")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout", type=str, default="3,7,11,14")
    parser.add_argument("--level-size", type=int, default=864)
    args = parser.parse_args()

    import numpy as np

    holdout = sorted(int(token) for token in args.holdout.split(","))
    meta = read_conditions(args.bank)
    n_maps = {entry["n_maps"] for entry in meta.values()}
    if n_maps != {16}:
        raise SystemExit(f"expected 16 maps per condition, got {sorted(n_maps)}")
    train_indices = [i for i in range(16) if i not in holdout]

    declared = [c for name in SET_ORDER for c in SETS[name]]
    if sorted(declared) != sorted(meta):
        missing = sorted(set(meta) - set(declared))
        extra = sorted(set(declared) - set(meta))
        raise SystemExit(f"dose set table mismatch; missing={missing} extra={extra}")
    if len(set(declared)) != len(declared):
        raise SystemExit("a condition appears in more than one dose set")
    if len(meta) != 32:
        raise SystemExit(f"expected the 32 v6 conditions, got {len(meta)}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Identical to M1: source_id is the bank's dig sha256, the held-out indices
    # are common to every condition, and a source that reached both splits is a
    # hard failure here AND again inside terra/maps_buffer.py on every load.
    registry_rows = []
    source_splits: dict[str, set[str]] = {}
    for condition_id in sorted(meta):
        for map_index, source in sorted(meta[condition_id]["maps"].items()):
            split = "held_out" if map_index in holdout else "train"
            registry_rows.append(
                {
                    "map_id": source["map_id"],
                    "source_id": source["dig_sha256"],
                    "split": split,
                    "condition_id": condition_id,
                    "map_index": map_index,
                }
            )
            source_splits.setdefault(source["dig_sha256"], set()).add(split)
    leaks = {k: sorted(v) for k, v in source_splits.items() if len(v) > 1}
    if leaks:
        raise SystemExit(
            f"{len(leaks)} dig sources appear in both splits; holdout {holdout} leaks"
        )
    registry_path = args.out / "source_registry.jsonl"
    with registry_path.open("w") as handle:
        for row in registry_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    registry_sha256 = sha256_file(registry_path)

    probe = next(iter(next(iter(meta.values()))["maps"].values()))
    shape = tuple(int(v) for v in np.load(probe["arrays"]["images"]).shape)
    if len(shape) != 2:
        raise SystemExit(f"expected 2-D maps, got {shape}")

    print(f"bank        : {args.bank}")
    print(f"out         : {args.out}")
    print(f"holdout     : {holdout} (train indices {train_indices})")
    print(f"level size  : {args.level_size}")
    print(f"map shape   : {shape}")
    print(
        f"registry    : {len(registry_rows)} identities, "
        f"{len(source_splits)} dig sources, split-disjoint"
    )
    print(f"sets        : easy={len(EASY)} mid={len(MID)} deep={len(DEEP)}")
    print("dose levels :")

    levels = {}
    for level in LEVEL_ORDER:
        shares = DOSE_LEVELS[level]
        slots, budgets = build_dose_slots(shares, meta, train_indices, args.level_size)
        report = dose_report(slots, budgets, shares, meta)
        levels[level] = materialize(
            args.out / "train" / level,
            slots,
            meta,
            f"train/{level}",
            split="train",
            registry_relative="../../source_registry.jsonl",
            registry_sha256=registry_sha256,
            shape=shape,
        )
        levels[level].update(report)
        (args.out / "train" / level / "dose.json").write_text(
            json.dumps(
                {
                    "schema": "terra_dose_level_v1",
                    "level": level,
                    "spec": "CURRICULUM_SPEC_V6.md section 3",
                    "sets": {name: SETS[name] for name in SET_ORDER},
                    **report,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        realised = report["realised_mass"]
        print(
            f"    {level}: easy {realised['easy']:.3f} mid {realised['mid']:.3f} "
            f"deep {realised['deep']:.3f} "
            f"(target {shares['easy']:.2f}/{shares['mid']:.2f}/{shares['deep']:.2f})"
        )

    print("held-out eval:")
    eval_slots = [
        (condition_id, index) for condition_id in sorted(meta) for index in holdout
    ]
    held_out = materialize(
        args.out / "held_out" / "all",
        eval_slots,
        meta,
        "held_out/all",
        split="held_out",
        registry_relative="../../source_registry.jsonl",
        registry_sha256=registry_sha256,
        shape=shape,
    )

    provenance = {
        "schema": "terra_curriculum_v6m_training_dataset_v1",
        "spec": "CURRICULUM_SPEC_V6.md sections 3 and 4",
        "bank": str(args.bank),
        "bank_manifest_csv_sha256": sha256_file(args.bank / "manifest.csv"),
        "bank_conditions_csv_sha256": sha256_file(args.bank / "conditions.csv"),
        "holdout_map_indices": holdout,
        "train_map_indices": train_indices,
        "level_size": args.level_size,
        "dose_sets": {name: SETS[name] for name in SET_ORDER},
        "dose_mass_table": DOSE_LEVELS,
        "level_order": list(LEVEL_ORDER),
        "levels": levels,
        "held_out": held_out,
        "source_registry_sha256": registry_sha256,
        "arm_levels": {
            # Both dose arms read the SAME directories; they differ only in the
            # promotion rule, which is a training-config knob, not a dataset one.
            "M2-dose": [f"train/{level}" for level in LEVEL_ORDER],
            "M2-dose-fast": [f"train/{level}" for level in LEVEL_ORDER],
            # spec v6 section 4: the budget control is M1-C's EXACT recipe, so it
            # reuses M1-C's own materialization rather than a v6 rebuild. The
            # deep rungs -- which is where prediction 1 is decided -- are
            # byte-identical between the two banks, so the bank difference does
            # not touch the headline comparison. Pre-registered as a caveat.
            "M2-wave-long": [
                "../curriculum_v5m/train/L0",
                "../curriculum_v5m/train/L1",
                "../curriculum_v5m/train/L2",
                "../curriculum_v5m/train/L3",
            ],
        },
    }
    provenance_path = args.out / "curriculum_v6m_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {provenance_path}")
    print(f"bank manifest.csv sha256 = {provenance['bank_manifest_csv_sha256']}")


if __name__ == "__main__":
    main()
