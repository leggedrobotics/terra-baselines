"""Split the 95 screened slabs by source AND translation/dihedral-normalized shape."""
import argparse
import json
from pathlib import Path

import numpy as np


def geometry_key(target):
    """Ignore placement, orientation and surrounding dump layout; retain dig depth."""
    y, x = np.where(target < 0)
    if not len(x):
        raise ValueError("precision target has no excavation")
    crop = np.minimum(target[y.min():y.max() + 1, x.min():x.max() + 1], 0)
    variants = []
    for turns in range(4):
        rotated = np.rot90(crop, turns)
        for array in (rotated, np.fliplr(rotated)):
            variants.append((array.shape, array.astype("<f8").tobytes()))
    return min(variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-level", required=True, type=Path)
    parser.add_argument("--screened-slots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in (args.dataset_level / "manifest.jsonl").read_text().splitlines()]
    slots = sorted(json.loads(args.screened_slots.read_text())["precision_slots"])
    if len(slots) != 95 or len(set(slots)) != 95:
        raise ValueError("this experiment requires the original 95 distinct screened slots")
    parents = {slot: slot for slot in slots}

    def root(slot):
        while parents[slot] != slot:
            slot = parents[slot]
        return slot

    identities, shapes = {}, {}
    for slot in slots:
        row = manifest[slot]
        if row["slot_index"] != slot + 1 or row["primary_cell"] != "v7-fnd-slab-adjacent":
            raise ValueError(f"unexpected bank row at zero-based slot {slot}")
        shape = geometry_key(np.load(args.dataset_level / "images" / f"img_{slot + 1}.npy"))
        shapes[slot] = shape
        keys = [(field, row[field]) for field in ("source_id", "dig_sha256", "source_scenario_id")]
        keys.append(("canonical_geometry", shape))
        for key in keys:
            if key in identities:
                parents[root(slot)] = root(identities[key])
            else:
                identities[key] = slot
    groups = {}
    for slot in slots:
        groups.setdefault(root(slot), []).append(slot)
    groups = sorted(groups.values(), key=lambda group: min(group))
    rng = np.random.default_rng(20261013)
    heldout = set(int(i) for i in rng.permutation(len(groups))[:12])
    records = []
    for index, group in enumerate(groups):
        split = "eval" if index in heldout else "train"
        records.append(dict(group=f"slab_{min(group)}", split=split, slots=group,
                            representative_slot=min(group),
                            canonical_shapes=sorted({str(shapes[slot][0]) for slot in group})))
    train = sorted(slot for group in records if group["split"] == "train" for slot in group["slots"])
    evaluation = sorted(slot for group in records if group["split"] == "eval" for slot in group["slots"])
    assert set(train).isdisjoint(evaluation) and sorted(train + evaluation) == slots
    assert {shapes[slot] for slot in train}.isdisjoint(shapes[slot] for slot in evaluation)
    for field in ("source_id", "dig_sha256", "source_scenario_id"):
        assert {manifest[slot][field] for slot in train}.isdisjoint(manifest[slot][field] for slot in evaluation)
    representatives = [group["representative_slot"] for group in records if group["split"] == "eval"]
    summary = dict(train_slots=len(train), eval_slots=len(evaluation), train_groups=len(groups) - len(heldout),
                   eval_groups=len(heldout), eval_starts=4 * len(representatives))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train_slots.json", train), ("eval_slots.json", evaluation)):
        (args.output / name).write_text(json.dumps(dict(precision_slots=subset), indent=2) + "\n")
    fields = ("map_id", "source_id", "dig_sha256", "source_scenario_id", "primary_cell", "required_dig_volume")
    split_manifest = dict(
        seed=20261013, dataset_level=args.dataset_level.name, screened_slots_input=str(args.screened_slots),
        qualification="static per-cell reachability screening; native episode finishability is not established",
        grouping="connected components sharing source_id, dig_sha256, source_scenario_id, or cropped excavation shape under translation, rotation or reflection",
        panel="one representative per held-out geometry group, four fixed reset seeds; no train geometry appears in evaluation",
        summary=summary, groups=records, eval_representative_slots=representatives,
        slots=[dict(slot=slot, **{field: manifest[slot][field] for field in fields}) for slot in slots])
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
