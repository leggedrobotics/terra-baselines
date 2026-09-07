#!/usr/bin/env python3
"""Build the small, obstacle-free R2 foundation bank for the reward sweep.

All free cells outside the depth-one foundation are explicit accepted dump
targets. Geometry, reach, completion, and reset behavior come from Terra.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import label

from eval_fixed_bank import exact_reset_keys
from terra.config import (
    ImmutableMapsConfig,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
)
from terra.env_generation.distance import (
    REWARD_V2_DISTANCE_METRIC,
    REWARD_V2_DISTANCE_NORMALIZATION,
    REWARD_V2_DISTANCE_PROTOCOL_ID,
    compute_reward_v2_distance_map,
)
from terra.env_generation.foundation_border_metadata import (
    build_foundation_border_metadata,
)
from terra.maps_buffer import (
    RESET_ARRAY_FOLDERS,
    RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
    contained_dump_capacity_sanity_check,
    load_maps_from_disk,
    reset_array_scenario_sha256,
)


MAP_SIZE = 64
TILE_SIZE_M = ImmutableMapsConfig().edge_length_m / MAP_SIZE
SPLITS = {"train": 256, "validation": 64, "test": 64}
SHAPES = ("square", "rectangle", "l")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def foundation_mask(rng: np.random.Generator, shape: str) -> tuple[np.ndarray, dict]:
    """Rasterize analytic shapes at cell centers; x is column, y is row."""
    center_x, center_y = (int(value) for value in rng.integers(24, 41, size=2))
    angle_deg = int(rng.integers(0, 12)) * 30
    angle = np.deg2rad(angle_deg)
    rows, cols = np.indices((MAP_SIZE, MAP_SIZE), dtype=np.float64)
    # Exact right-angle rotations must not move boundary cells because of
    # floating cos(pi/2) residuals in the half-open box tests below.
    x = np.round((cols - center_x) * np.cos(angle) + (rows - center_y) * np.sin(angle), 12)
    y = np.round(-(cols - center_x) * np.sin(angle) + (rows - center_y) * np.cos(angle), 12)
    if shape == "square":
        width = height = int(rng.integers(8, 14))
        arm = None
    elif shape == "rectangle":
        width, height = int(rng.integers(6, 11)), int(rng.integers(12, 19))
        arm = None
    elif shape == "l":
        width, height = (int(value) for value in rng.integers(12, 19, size=2))
        arm = int(rng.integers(5, 9))
    else:
        raise ValueError(shape)
    mask = (x >= -width / 2) & (x < width / 2) & (y >= -height / 2) & (y < height / 2)
    if arm is not None:
        mask &= (x < -width / 2 + arm) | (y < -height / 2 + arm)
    geometry = {
        "shape": shape,
        "center_xy_tiles": [center_x, center_y],
        "angle_degrees": angle_deg,
        "width_tiles": width,
        "height_tiles": height,
        "arm_width_tiles": arm,
    }
    return mask, geometry


def render_gallery(root: Path, records: dict[str, list[dict]]) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    palette = ListedColormap(["#875c3d", "#dbe8cb"])
    for row_index, (split, rows) in enumerate(records.items()):
        for column_index, shape in enumerate(SHAPES):
            candidates = [row for row in rows if row["geometry"]["shape"] == shape]
            candidates.sort(key=lambda row: row["dig_cells"])
            row = candidates[len(candidates) // 2]
            target = np.load(root / split / "all" / "images" / f"img_{row['slot_index']}.npy")
            axis = axes[row_index, column_index]
            axis.imshow(target > 0, cmap=palette, vmin=0, vmax=1, origin="lower")
            axis.set_title(
                f"{split} · {shape.upper() if shape == 'l' else shape}\n"
                f"{row['dig_cells']} dig cells · {row['geometry']['angle_degrees']}°"
            )
            axis.set_xticks([0, 16, 32, 48, 63])
            axis.set_yticks([0, 16, 32, 48, 63])
            axis.set_xlabel(f"slot {row['slot_index']} · 0.5714 m/cell", fontsize=9)
    figure.suptitle("Easy foundations: 64 × 64 cells, depth 1, no obstacles", fontsize=16)
    figure.legend(
        handles=[Patch(color="#875c3d", label="Required excavation"),
                 Patch(color="#dbe8cb", label="Accepted legal dumping")],
        loc="outside lower center", ncol=2,
    )
    figure.savefig(root / "gallery.png", dpi=160)
    plt.close(figure)


def build_bank(root: Path, seed: int) -> dict:
    if root.exists():
        raise FileExistsError(f"Choose a new bank directory: {root}")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    records: dict[str, list[dict]] = {}
    registry: list[dict] = []
    distance_rows: list[dict] = []
    scenario_ids: set[str] = set()
    target_ids: set[bytes] = set()
    candidate_id = 0
    family_index = 0

    for split, count in SPLITS.items():
        directory = root / split / "all"
        for folder in (*RESET_ARRAY_FOLDERS, "metadata"):
            (directory / folder).mkdir(parents=True)
        shapes = [SHAPES[(family_index + index) % len(SHAPES)] for index in range(count)]
        family_index += count
        rng.shuffle(shapes)
        keys = np.asarray(exact_reset_keys(count))
        rows = []
        for slot, shape in enumerate(shapes, start=1):
            # Duplicate raster layouts can occur through square symmetries.
            # Resampling only these guarantees disjoint actual reset arrays.
            for _ in range(1000):
                candidate_id += 1
                dig_mask, geometry = foundation_mask(rng, shape)
                if dig_mask.tobytes() not in target_ids:
                    target_ids.add(dig_mask.tobytes())
                    break
            else:
                raise RuntimeError("Could not generate a new foundation layout")
            if label(dig_mask)[1] != 1 or np.any(dig_mask[[0, -1]]) or np.any(dig_mask[:, [0, -1]]):
                raise RuntimeError(f"Invalid connected interior foundation: {geometry}")
            target = np.where(dig_mask, -1, 1).astype(np.int8)
            occupancy = np.zeros_like(target, dtype=np.int8)
            dumpability = np.ones_like(target, dtype=np.bool_)
            action = np.zeros_like(target, dtype=np.int8)
            distance = compute_reward_v2_distance_map(
                target, occupancy, tile_size_m=TILE_SIZE_M,
                distance_ref_m=REWARD_V2_DISTANCE_REF_M,
                distance_bound=REWARD_V2_DISTANCE_BOUND,
            )
            arrays = dict(zip(RESET_ARRAY_FOLDERS, (target, occupancy, dumpability, action, distance)))
            scenario_id = reset_array_scenario_sha256(arrays)
            if scenario_id in scenario_ids:
                raise RuntimeError("Reset-array identity collision across bank splits")
            scenario_ids.add(scenario_id)
            capacity = contained_dump_capacity_sanity_check(
                target, occupancy, dumpability, action,
                minimum_single_layer_ratio=3.0,
            )
            reset_seed = int(keys[slot - 1, 1])
            map_id = f"easy-foundation-{shape}-{candidate_id:05d}"
            identity = {
                "map_id": map_id,
                "source_id": f"procedural:easy-foundation:{seed}:{candidate_id}",
                "scenario_id": scenario_id,
                "split": split,
                "family": "foundation",
                "stratum": "all",
                "primary_cell": f"easy_{shape}",
            }
            row = {
                **identity, "slot_index": slot, "slot_weight": 1.0,
                "identity_slot_multiplicity": 1,
                "episode_id": f"{split}-{map_id}-seed{reset_seed}",
                "reset_seed": reset_seed, "geometry": geometry,
                "dig_cells": int(dig_mask.sum()), "capacity": capacity,
            }
            registry.append(identity)
            rows.append(row)
            for folder, array in arrays.items():
                np.save(directory / folder / f"img_{slot}.npy", array)
            border = build_foundation_border_metadata(dig_mask)
            if not 3 <= len(border["foundation_border_axes_ABC"]) <= 64:
                raise RuntimeError(f"Invalid foundation border metadata: {geometry}")
            write_json(directory / "metadata" / f"trench_{slot}.json", {
                **identity, "geometry": geometry, "axes_ABC": [],
                "trench_axes_count": 0, **border,
            })
            distance_rows.append({
                **identity, "slot_index": slot, "dataset_relative_path": f"{split}/all",
                "distance_path": f"{split}/all/distance/img_{slot}.npy",
                "distance_sha256": sha256_file(directory / "distance" / f"img_{slot}.npy"),
                "required_dig_volume": int(dig_mask.sum()),
                "dig_distance_mean_m": float(distance[dig_mask].mean() * REWARD_V2_DISTANCE_REF_M),
                "dig_distance_max_m": float(distance[dig_mask].max() * REWARD_V2_DISTANCE_REF_M),
                "normalized_distance_max": float(distance.max()),
                "h_reset_over_v0": float(distance[dig_mask].mean()),
            })
        records[split] = rows
        write_jsonl(directory / "manifest.jsonl", rows)
        print(f"Generated {split}: {count} unique maps", flush=True)

    registry_path = root / "source_registry.jsonl"
    write_jsonl(registry_path, registry)
    registry_sha256 = sha256_file(registry_path)
    distance_contract = {
        "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
        "distance_metric": REWARD_V2_DISTANCE_METRIC,
        "distance_normalization": REWARD_V2_DISTANCE_NORMALIZATION,
        "tile_size_m": TILE_SIZE_M,
        "distance_ref_m": REWARD_V2_DISTANCE_REF_M,
        "distance_bound": REWARD_V2_DISTANCE_BOUND,
    }
    loaded_shapes = {}
    for split, count in SPLITS.items():
        directory = root / split / "all"
        write_json(directory / "dataset.json", {
            "schema": "terra_exact_map_dataset_v1",
            "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
            "slot_count": count, "unique_identity_count": count,
            "shape": [MAP_SIZE, MAP_SIZE], **distance_contract,
            "accepted_dump_contract": "exact_visible_dump_v1",
            "minimum_dump_capacity_ratio": 3.0,
            "source_registry": "../../source_registry.jsonl",
            "source_registry_sha256": registry_sha256,
        })
        os.environ["DATASET_SIZE"] = str(count)
        loaded = load_maps_from_disk(
            str(directory), require_trench_metadata=True,
            require_trench_alignment_metadata=True,
            required_distance_protocol_id=REWARD_V2_DISTANCE_PROTOCOL_ID,
        )
        loaded_shapes[split] = [list(array.shape) for array in loaded]
        if np.any(np.asarray(loaded[3]) != -1) or np.any(np.asarray(loaded[5]) < 3):
            raise RuntimeError("Loaded foundation/trench metadata does not match generation")
        print(f"Validated {split}: exact identities, distance arrays, capacity, metadata", flush=True)

    sidecar = root / "distance_sidecar"
    sidecar.mkdir()
    write_json(sidecar / "distance_protocol.json", {
        "schema": "terra_distance_protocol_v1", **distance_contract,
        "sources": "(target > 0) & ~occupancy",
        "traversable": "~occupancy", "neighborhood": 8,
        "cardinal_cost_tiles": 1.0, "diagonal_cost_tiles": float(np.sqrt(2)),
        "obstacle_output": 0.0, "clipping": False,
        "out_of_bounds": "reject",
        "implementation": "terra.env_generation.distance.compute_reward_v2_distance_map",
    })
    write_jsonl(sidecar / "rows.jsonl", distance_rows)
    write_json(sidecar / "dataset.json", {
        "schema": "terra_r2_distance_sidecar_v1", "status": "passed",
        "distance_protocol": "distance_protocol.json",
        "distance_protocol_sha256": sha256_file(sidecar / "distance_protocol.json"),
        "rows": "rows.jsonl", "rows_sha256": sha256_file(sidecar / "rows.jsonl"),
        "datasets": len(SPLITS), "scenarios": len(scenario_ids),
        "scenario_counts": SPLITS, "source_registry_sha256": registry_sha256,
        "physical_identity_contract": "fresh procedural reset arrays, verified by the canonical R2 loader",
        "observed_global_max": max(row["normalized_distance_max"] for row in distance_rows),
    })
    all_rows = [row for rows in records.values() for row in rows]
    areas = [row["dig_cells"] for row in all_rows]
    ratios = [row["capacity"]["single_layer_capacity_ratio"] for row in all_rows]
    summary = {
        "status": "passed", "seed": seed, "generator": str(Path(__file__).resolve()),
        "counts": SPLITS,
        "shape_counts": {split: dict(Counter(row["primary_cell"] for row in rows)) for split, rows in records.items()},
        "unique_map_ids": len({row["map_id"] for row in all_rows}),
        "unique_source_ids": len({row["source_id"] for row in all_rows}),
        "unique_reset_array_ids": len(scenario_ids),
        "unique_target_layouts": len(target_ids), "cross_split_identity_overlap": 0,
        "dig_cells": {"min": min(areas), "max": max(areas), "mean": float(np.mean(areas))},
        "single_layer_dump_capacity_ratio": {"min": min(ratios), "max": max(ratios)},
        "dig_distance_max_m": max(row["dig_distance_max_m"] for row in distance_rows),
        "dig_distance_mean_m": float(np.mean([row["dig_distance_mean_m"] for row in distance_rows])),
        "loaded_array_shapes": loaded_shapes,
        "distance_sidecar_sha256": sha256_file(sidecar / "dataset.json"),
        "geometry": {"grid_cells": [64, 64], "tile_size_m": TILE_SIZE_M, "depth": 1,
                     "obstacle_cells": 0, "accepted_dump": "every cell outside the foundation"},
        "reset": "full resets; normal Terra random robot poses; fixed eval keys from exact_reset_keys",
        "scope": "new layouts of three easy shape families; broad nearby dumping; no claim of remote-dump or unseen-family transfer",
    }
    render_gallery(root, records)
    write_json(root / "validation.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(build_bank(args.output.resolve(), args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
