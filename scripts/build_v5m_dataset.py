#!/usr/bin/env python3
"""Materialize the v5-main review bank as Terra-loadable training/eval datasets.

The review bank stores one folder per condition with a ``manifest.json`` that
points into a shared ``dataset/`` directory holding non-contiguous
``img_<global>.npy`` files. Terra's loader (``terra/maps_buffer.py``) instead
needs, per curriculum level, a directory with contiguous ``img_1..img_N.npy``
under ``images/occupancy/dumpability/distance/actions`` plus
``metadata/trench_N.json``, and every level of one run must hold the SAME slot
count (the buffer is a dense ``[level, n_maps, W, H]`` array).

``eval_fixed_bank.py`` additionally needs a ``manifest.jsonl`` whose rows carry
``slot_index`` (1-based, contiguous), ``family`` and ``primary_cell``; it groups
its per-map results by those two fields, which is what gives per-condition
held-out numbers for free.

Outputs (under --out):
    train/L0 train/L1 train/L2 train/L3   admission waves, --level-size slots each
    train/*/dataset.json                  per-level terra_exact_map_dataset_v1 sidecar
    held_out/all                          the held-out indices, one slot per map
    source_registry.jsonl                 map_id -> (dig source_id, split), split-disjoint
    curriculum_v5m_provenance.json        bank hashes + wave composition

Run A trains on [train/L3] restricted to nothing (uses L0), run B on [train/L3],
run C on [train/L0, train/L1, train/L2, train/L3].

Slot allocation inside a level: families are dealt alternately (exact 50/50
foundation/trench balance), each family cycles its admitted conditions, each
condition cycles its train map indices. Condition slot counts inside a family
are therefore equal up to one slot, and map multiplicities inside a condition
equal up to one. The loader has no per-map sampling weights, so duplication IS
the weighting mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

DEFAULT_BANK = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_map_distribution_review_v5/main/review_bank"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "terra_data" / "curriculum_v5m"

ARRAY_KINDS = ("images", "occupancy", "dumpability", "distance", "actions")

# Admission waves. L0 = every T0 anchor. L1 = the shallowest T1 rung of each
# factor axis. L2 = the next rung of every axis that has one. L3 = the deepest
# T1 rung plus all T2 compositions. Cumulative: a level directory holds every
# condition admitted up to and including that wave.
WAVE_ADDITIONS: dict[str, list[str]] = {
    "L0": [
        "fnd-slab-ring3x",
        "fnd-proc-ring3x",
        "fnd-slab-lg-ring3x",
        "fnd-slab-apron-near",
        "fnd-slab-apron-c3x",
        "trn-straight-side2",
        "trn-straight-side1",
        "trn-tee-side2",
    ],
    "L1": [
        "fnd-slab-apron-c2x",  # capacity rung 1
        "fnd-slab-apron-d12",  # distance rung 1
        "fnd-slab-side1",  # dump layout rung 1
        "fnd-strips-ring3x",  # foundation source rung 1
        "fnd-slab-ring3x-obj1",  # site rung 1 (1-2 objects)
        "trn-straight-altsides",  # trench dump rung 1
        "trn-seg2-side2",  # trench geometry rung 1
        "trn-straight-side1-tight",  # trench capacity rung 1
    ],
    "L2": [
        "fnd-slab-apron-c1p6",
        "fnd-slab-apron-d16",
        "fnd-slab-split",
        "fnd-slab-ring3x-obj",
        "fnd-slab-ring3x-road",
        "trn-seg3-side2",
        "trn-net3-side2",
    ],
    "L3": [
        "fnd-slab-apron-c1p2",
        "trn-net4-side2",
        "fnd-proc-side1-road",
        "fnd-slab-side1-obj",
        "trn-net3-side1-road",
        "trn-net4-side1-road",
    ],
}
WAVE_ORDER = ["L0", "L1", "L2", "L3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_conditions(bank: Path) -> dict[str, dict]:
    """condition_id -> {family, tier, maps: {map_index: {kind: Path}}}."""
    import csv

    meta: dict[str, dict] = {}
    with (bank / "conditions.csv").open() as handle:
        for row in csv.DictReader(handle):
            meta[row["condition_id"]] = {
                "family": row["family"],
                "tier": int(row["tier"]),
                "n_maps": int(row["n_maps"]),
            }
    for condition_id, entry in meta.items():
        manifest = json.loads((bank / condition_id / "manifest.json").read_text())
        maps = {}
        for item in manifest["maps"]:
            index = int(item["mapIndex"])
            arrays = {kind: bank / item["arrays"][kind] for kind in ARRAY_KINDS}
            global_stem = Path(item["arrays"]["images"]).stem  # img_<global>
            global_index = int(global_stem.split("_")[1])
            maps[index] = {
                "arrays": arrays,
                "metadata": bank / "dataset" / "metadata" / f"trench_{global_index}.json",
                "map_id": item["id"],
                "global_index": global_index,
                "dig_sha256": item.get("digSha256", ""),
            }
        missing = [k for k in maps.values() if not k["metadata"].exists()]
        if missing:
            raise SystemExit(f"{condition_id}: missing axis metadata for {len(missing)} maps")
        entry["maps"] = maps
    return meta


def round_robin_slots(
    conditions: list[str], map_indices: list[int], count: int
) -> list[tuple[str, int]]:
    """Deal ``count`` (condition, map_index) slots condition-major, map-minor."""
    if not conditions:
        raise ValueError("no conditions to deal from")
    cycle = [(c, m) for m in map_indices for c in conditions]
    return list(itertools.islice(itertools.cycle(cycle), count))


def build_level_slots(
    admitted: list[str], meta: dict[str, dict], map_indices: list[int], level_size: int
) -> list[tuple[str, int]]:
    if level_size % 2:
        raise SystemExit("--level-size must be even for an exact 50/50 family split")
    per_family = level_size // 2
    by_family: dict[str, list[str]] = {"foundation": [], "trench": []}
    for condition_id in admitted:
        by_family[meta[condition_id]["family"]].append(condition_id)
    foundation = round_robin_slots(by_family["foundation"], map_indices, per_family)
    trench = round_robin_slots(by_family["trench"], map_indices, per_family)
    interleaved: list[tuple[str, int]] = []
    for a, b in zip(foundation, trench):
        interleaved.append(a)
        interleaved.append(b)
    return interleaved


def materialize(
    out_dir: Path,
    slots: list[tuple[str, int]],
    meta: dict[str, dict],
    label: str,
    *,
    split: str,
    registry_relative: str,
    registry_sha256: str,
    shape: tuple[int, int],
) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for kind in (*ARRAY_KINDS, "metadata"):
        (out_dir / kind).mkdir(parents=True, exist_ok=True)

    rows = []
    first_slot: dict[tuple[str, int], int] = {}
    for slot, (condition_id, map_index) in enumerate(slots, start=1):
        source = meta[condition_id]["maps"][map_index]
        previous = first_slot.get((condition_id, map_index))
        if previous is None:
            first_slot[(condition_id, map_index)] = slot
            for kind in ARRAY_KINDS:
                shutil.copy2(source["arrays"][kind], out_dir / kind / f"img_{slot}.npy")
            shutil.copy2(source["metadata"], out_dir / "metadata" / f"trench_{slot}.json")
        else:
            # Repeated identities are hardlinks to this level's first copy: the
            # duplication is the sampling weight, and Euler's scratch file quota
            # counts inodes, not directory entries. Nothing ever writes here.
            for kind, prefix, suffix in (
                *[(k, "img_", ".npy") for k in ARRAY_KINDS],
                ("metadata", "trench_", ".json"),
            ):
                target = out_dir / kind / f"{prefix}{slot}{suffix}"
                existing = out_dir / kind / f"{prefix}{previous}{suffix}"
                try:
                    target.hardlink_to(existing)
                except OSError:
                    shutil.copy2(existing, target)
        rows.append(
            {
                "slot_index": slot,
                "map_id": source["map_id"],
                "source_id": source["dig_sha256"],
                "split": split,
                "family": meta[condition_id]["family"],
                "stratum": f"T{meta[condition_id]['tier']}",
                "primary_cell": condition_id,
                # The buffer samples slots uniformly and ignores slot_weight, so
                # the effective per-identity weight is identity_slot_multiplicity.
                "slot_weight": 1.0,
                "condition_id": condition_id,
                "tier": meta[condition_id]["tier"],
                "map_index": map_index,
                "bank_global_index": source["global_index"],
            }
        )
    multiplicity: dict[str, int] = {}
    for row in rows:
        multiplicity[row["map_id"]] = multiplicity.get(row["map_id"], 0) + 1
    for row in rows:
        row["identity_slot_multiplicity"] = multiplicity[row["map_id"]]
    with (out_dir / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    # terra/maps_buffer.validate_exact_dataset_contract requires this sidecar in
    # EVERY level directory before any array reaches JAX.
    (out_dir / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "terra_exact_map_dataset_v1",
                "slot_count": len(rows),
                "shape": [shape[0], shape[1]],
                "distance_metric": "geodesic_traversable_tiles",
                "distance_normalization": "max_finite_distance_unit_scaled",
                "accepted_dump_contract": "exact_visible_dump_v1",
                "unique_identity_count": len(multiplicity),
                "source_registry": registry_relative,
                "source_registry_sha256": registry_sha256,
                "label": label,
                "split": split,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["condition_id"]] = counts.get(row["condition_id"], 0) + 1
    families: dict[str, int] = {}
    for row in rows:
        families[row["family"]] = families.get(row["family"], 0) + 1
    distinct = len({(row["condition_id"], row["map_index"]) for row in rows})
    print(
        f"  {label:<14} slots={len(rows):5d} conditions={len(counts):2d} "
        f"distinct_maps={distinct:3d} family={families} "
        f"slots/condition={min(counts.values())}-{max(counts.values())}"
    )
    return {
        "slots": len(rows),
        "conditions": sorted(counts),
        "slots_per_condition": counts,
        "family_slots": families,
        "distinct_maps": distinct,
        "manifest_sha256": sha256_file(out_dir / "manifest.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--holdout",
        type=str,
        default="3,7,11,14",
        help="map indices held out of training in EVERY condition",
    )
    parser.add_argument(
        "--level-size",
        type=int,
        default=864,
        help="slots per curriculum level (equal across levels; loader requirement)",
    )
    args = parser.parse_args()

    holdout = sorted(int(token) for token in args.holdout.split(","))
    meta = read_conditions(args.bank)
    n_maps = {entry["n_maps"] for entry in meta.values()}
    if n_maps != {16}:
        raise SystemExit(f"expected 16 maps per condition, got {sorted(n_maps)}")
    train_indices = [i for i in range(16) if i not in holdout]

    declared = [c for wave in WAVE_ORDER for c in WAVE_ADDITIONS[wave]]
    if sorted(declared) != sorted(meta):
        missing = sorted(set(meta) - set(declared))
        extra = sorted(set(declared) - set(meta))
        raise SystemExit(f"wave table mismatch; missing={missing} extra={extra}")
    if len(set(declared)) != len(declared):
        raise SystemExit("wave table repeats a condition")
    l0 = WAVE_ADDITIONS["L0"]
    tier0 = sorted(c for c, e in meta.items() if e["tier"] == 0)
    if sorted(l0) != tier0:
        raise SystemExit(f"L0 must be exactly the T0 conditions; T0={tier0}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Source registry. source_id is the bank's dig sha256; it is split-disjoint
    # exactly because the held-out map indices are common across ALL conditions
    # and the bank shares one dig per (map_index, layout group). The loader
    # rejects any source_id that appears in more than one split, so this file is
    # the machine-checked version of the "leak-free by shared-dig" claim.
    import numpy as np

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
    print("train levels:")

    levels = {}
    admitted: list[str] = []
    for wave in WAVE_ORDER:
        admitted = admitted + WAVE_ADDITIONS[wave]
        slots = build_level_slots(admitted, meta, train_indices, args.level_size)
        levels[wave] = materialize(
            args.out / "train" / wave,
            slots,
            meta,
            f"train/{wave}",
            split="train",
            registry_relative="../../source_registry.jsonl",
            registry_sha256=registry_sha256,
            shape=shape,
        )
        levels[wave]["admitted"] = list(admitted)
        levels[wave]["added"] = list(WAVE_ADDITIONS[wave])

    print("held-out eval:")
    eval_slots = [
        (condition_id, index)
        for condition_id in sorted(meta)
        for index in holdout
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
        "schema": "terra_curriculum_v5m_training_dataset_v1",
        "bank": str(args.bank),
        "bank_manifest_csv_sha256": sha256_file(args.bank / "manifest.csv"),
        "bank_conditions_csv_sha256": sha256_file(args.bank / "conditions.csv"),
        "bank_validation_json_sha256": sha256_file(args.bank / "VALIDATION.json"),
        "holdout_map_indices": holdout,
        "train_map_indices": train_indices,
        "level_size": args.level_size,
        "wave_additions": WAVE_ADDITIONS,
        "wave_order": WAVE_ORDER,
        "levels": levels,
        "held_out": held_out,
        "arm_levels": {
            "A_t0-baseline": ["train/L0"],
            "B_uniform-full": ["train/L3"],
            "C_curriculum": ["train/L0", "train/L1", "train/L2", "train/L3"],
        },
    }
    provenance["source_registry_sha256"] = registry_sha256
    provenance_path = args.out / "curriculum_v5m_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {provenance_path}")
    print(f"bank manifest.csv sha256 = {provenance['bank_manifest_csv_sha256']}")


if __name__ == "__main__":
    main()
