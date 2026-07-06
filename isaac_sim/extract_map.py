import json
import numpy as np
import jax
import argparse
import pickle
import sys
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the parent directory to the path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.config import BatchConfig, CurriculumGlobalConfig, RewardsType
from terra.actions import TrackedAction, WheeledAction, TrackedActionType, WheeledActionType
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from train_mixed import MixedAgentTrainConfig
from eval_mcts import fix_env_cfg_dtypes, make_mcts_step_fn
sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig


def _to_serializable(obj):
    """Convert JAX/NumPy containers into plain Python types for JSON dumping."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (np.bool_, np.integer, np.floating, jnp.bool_, jnp.integer, jnp.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _timestamped_output_path(output_path: Path) -> Path:
    """Append a finish-time timestamp before the output suffix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}_{timestamp}")


def _mask_to_int_list(value: Any) -> list[list[int]]:
    return np.asarray(value).astype(np.int8).tolist()


def _wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_bucket_to_rad(value: Any, n_bins: int = 12) -> float:
    bucket = float(_to_serializable(value))
    return _wrap_to_pi(bucket * 2.0 * math.pi / float(n_bins))


def _terra_bucket_to_plan_yaw_rad(value: Any, n_bins: int = 12) -> float:
    # Terra bucket 0 points along +plan_y; schema-v2 yaw 0 is +plan_x.
    return _wrap_to_pi(_angle_bucket_to_rad(value, n_bins) + math.pi / 2.0)


def _wheel_bucket_to_rad(value: Any, wheel_step_deg: float = 20.0) -> float:
    bucket = float(_to_serializable(value))
    return _wrap_to_pi(math.radians(bucket * wheel_step_deg))


def _is_load_transition(entry: dict[str, Any]) -> bool:
    loaded_change = entry.get("loaded_state_change", {}) or {}
    before = _as_py_bool(loaded_change.get("before", False))
    after = _as_py_bool(loaded_change.get("after", False))
    return (not before) and after


def _is_unload_transition(entry: dict[str, Any]) -> bool:
    loaded_change = entry.get("loaded_state_change", {}) or {}
    before = _as_py_bool(loaded_change.get("before", False))
    after = _as_py_bool(loaded_change.get("after", False))
    return before and (not after)


def _workspace_type_for_pair(dig_entry: dict[str, Any]) -> str:
    if dig_entry.get("dig_type") == "lift_dug_dirt":
        return "collect_dumped_soil"
    return "excavate"


def _load_plan_alignment(map_path: Path) -> dict[str, Any]:
    """Read TerraMapMaker alignment metadata, with conservative defaults."""
    alignment = {
        "meters_per_tile": 0.1,
        "origin_map_xy_m": [0.0, 0.0],
        "yaw_map_from_plan_rad": 0.0,
    }

    map_json_path = map_path / "metadata" / "map.json"
    try:
        with map_json_path.open("r", encoding="utf-8") as f:
            map_metadata = json.load(f)
        if "meters_per_tile" in map_metadata:
            alignment["meters_per_tile"] = float(map_metadata["meters_per_tile"])
    except Exception:
        pass

    terra_metadata_path = map_path / "metadata" / "terra_metadata.yaml"
    if not terra_metadata_path.exists():
        return alignment

    lines = terra_metadata_path.read_text(encoding="utf-8").splitlines()
    source_resolution = None
    source_size_rows_cols = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("meters_per_tile:"):
            alignment["meters_per_tile"] = float(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("rotation_deg:"):
            rotation_deg = float(stripped.split(":", 1)[1].strip())
            alignment["yaw_map_from_plan_rad"] = math.radians(rotation_deg)
        elif stripped.startswith("terra_origin_map_m:"):
            values = []
            for next_line in lines[idx + 1: idx + 3]:
                item = next_line.strip()
                if item.startswith("-"):
                    values.append(float(item[1:].strip()))
            if len(values) == 2:
                alignment["origin_map_xy_m"] = values
        elif stripped.startswith("resolution_m_per_cell:"):
            source_resolution = float(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("size_rows_cols:"):
            source_size_rows_cols = []
            for next_line in lines[idx + 1: idx + 3]:
                item = next_line.strip()
                if item.startswith("-"):
                    source_size_rows_cols.append(float(item[1:].strip()))

    if source_resolution is not None:
        alignment["meters_per_tile"] = source_resolution
        if len(source_size_rows_cols) == 2:
            rows, cols = source_size_rows_cols
            alignment["origin_map_xy_m"] = [
                -cols * source_resolution / 2.0,
                -rows * source_resolution / 2.0,
            ]

    return alignment


def _to_schema_v2_waypoint(entry: dict[str, Any], workspace_type: str) -> dict[str, Any]:
    agent_state = entry.get("agent_state", {}) or {}
    loaded_change = entry.get("loaded_state_change", {}) or {}
    pos_base = agent_state.get("pos_base", [0.0, 0.0])
    angle_base_rad = agent_state.get("angle_base_rad_override")
    angle_cabin_rad = agent_state.get("angle_cabin_rad_override")
    if angle_base_rad is None:
        angle_base_rad = _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_base", 0.0))
    if angle_cabin_rad is None:
        angle_cabin_rad = _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_cabin", 0.0))

    return {
        "step": int(_to_serializable(entry.get("step", 0))),
        "workspace_type": workspace_type,
        "traversability_mask": _mask_to_int_list(entry.get("traversability_mask")),
        "terrain_modification_mask": _mask_to_int_list(entry.get("terrain_modification_mask")),
        "dug_mask": _mask_to_int_list(entry.get("dug_mask")),
        "dump_mask": _mask_to_int_list(entry.get("dump_mask")),
        "agent_type": int(_to_serializable(entry.get("agent_type", 0))),
        "agent_index": int(_to_serializable(entry.get("agent_index", 0))),
        "agent_state": {
            "pos_base": [float(pos_base[0]), float(pos_base[1])],
            "angle_base_rad": float(angle_base_rad),
            "angle_cabin_rad": float(angle_cabin_rad),
            "wheel_angle_rad": _wheel_bucket_to_rad(agent_state.get("wheel_angle", 0.0)),
            "loaded": _as_py_bool(loaded_change.get("after", False)),
        },
    }


def _schema_v2_waypoints(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    waypoints = []
    pending_dig = None

    for entry in plan:
        if _is_load_transition(entry):
            pending_dig = entry
            continue

        if pending_dig is not None and _is_unload_transition(entry):
            workspace_type = _workspace_type_for_pair(pending_dig)
            waypoints.append(_to_schema_v2_waypoint(pending_dig, workspace_type))
            waypoints.append(_to_schema_v2_waypoint(entry, workspace_type))
            pending_dig = None

    return waypoints


def _plan_to_schema_v2(plan: list[dict[str, Any]], map_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "source_map_frame_id": "map",
        "alignment": _load_plan_alignment(map_path),
        "waypoints": _schema_v2_waypoints(plan),
    }


def _load_single_map_arrays(map_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.load(map_path / "images" / "img_1.npy")
    occupancy = np.load(map_path / "occupancy" / "img_1.npy")
    dumpability = np.load(map_path / "dumpability" / "img_1.npy")
    return target, occupancy.astype(bool), dumpability.astype(bool)


def _as_py_bool(x: Any) -> bool:
    try:
        return bool(np.asarray(x).reshape(-1)[0])
    except Exception:
        return bool(x)


def _mask_any_true(x: Any) -> bool:
    if x is None:
        return False
    try:
        arr = np.asarray(x).astype(bool)
    except Exception:
        return False
    if arr.size == 0:
        return False
    return bool(arr.any())


def _filter_empty_do_actions(plan: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Filter out empty/meaningless DO actions.
    
    Keep:
    - All digging actions (loaded: False → True)
    - All dumping actions (loaded: True → False)
    - Any DO action that modified terrain (tiles changed > 0)
    
    Drop:
    - Only DO actions where loaded state is unchanged AND no tiles were modified
    """
    keep = [False] * len(plan)
    
    stats = {
        "raw_waypoints": len(plan),
        "kept_waypoints": 0,
        "kept_digging": 0,
        "kept_dumping": 0,
        "kept_with_terrain_change": 0,
        "dropped_empty_unchanged": 0,
    }

    for idx, entry in enumerate(plan):
        lsc = entry.get("loaded_state_change", {}) or {}
        before = _as_py_bool(lsc.get("before", False))
        after = _as_py_bool(lsc.get("after", False))
        
        is_dig = (not before) and after
        is_dump = before and (not after)
        has_terrain_change = _mask_any_true(entry.get("terrain_modification_mask"))
        
        if is_dig:
            keep[idx] = True
            stats["kept_digging"] += 1
        elif is_dump:
            keep[idx] = True
            stats["kept_dumping"] += 1
        elif has_terrain_change:
            # Rare case: loaded state unchanged but terrain modified
            keep[idx] = True
            stats["kept_with_terrain_change"] += 1
        else:
            # Empty/unchanged DO action - drop it
            stats["dropped_empty_unchanged"] += 1

    filtered = [entry for i, entry in enumerate(plan) if keep[i]]
    stats["kept_waypoints"] = len(filtered)
    return filtered, stats


def _scalar_float(value: Any, default: float) -> float:
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def _workspace_mask_for_adjustment(entry: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    for key in ("terrain_modification_mask", "actual_terrain_modification_mask", "dug_mask", "dump_mask"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            mask = np.asarray(value).astype(bool)
            if mask.shape != shape:
                mask = mask.reshape(shape)
        except Exception:
            continue
        if mask.any():
            return mask
    return np.zeros(shape, dtype=bool)


def _reach_limits_tiles(
    env_cfgs: Any,
    agent_type: int,
    min_range_tiles_override: float | None,
    max_range_tiles_override: float | None,
) -> tuple[float, float]:
    if min_range_tiles_override is not None and max_range_tiles_override is not None:
        return float(min_range_tiles_override), float(max_range_tiles_override)

    tile_size = _scalar_float(getattr(env_cfgs, "tile_size", None), 0.6875)
    agent_cfg = getattr(env_cfgs, "agent", None)
    agent_width = _scalar_float(getattr(agent_cfg, "width", None), 5.0)
    agent_height = _scalar_float(getattr(agent_cfg, "height", None), 9.0)
    dig_radius_tiles = _scalar_float(getattr(agent_cfg, "dig_radius_tiles", None), 6.0)
    max_agent_dim = max(agent_width / 2.0, agent_height / 2.0)

    if int(agent_type) in (1, 2):
        # Matches Terra's skidsteer/truck cylindrical workspace in state.py.
        min_range_tiles = max_agent_dim - 2.0 + 0.1 * dig_radius_tiles
        max_range_tiles = max_agent_dim - 2.0 + 1.5 * dig_radius_tiles
    else:
        # Matches Terra's excavator workspace in state.py.
        min_range_tiles = max_agent_dim + 0.5 / max(tile_size, 1e-6)
        max_range_tiles = min_range_tiles + dig_radius_tiles

    if min_range_tiles_override is not None:
        min_range_tiles = float(min_range_tiles_override)
    if max_range_tiles_override is not None:
        max_range_tiles = float(max_range_tiles_override)
    return float(min_range_tiles), float(max_range_tiles)


def _base_position_is_traversable(
    candidate: np.ndarray,
    occupancy: np.ndarray,
    traversability_mask: Any,
) -> bool:
    row = int(round(float(candidate[0])))
    col = int(round(float(candidate[1])))
    h, w = occupancy.shape
    if row < 0 or row >= h or col < 0 or col >= w:
        return False
    if bool(occupancy[row, col]):
        return False

    if traversability_mask is None:
        return True
    try:
        traversability = np.asarray(traversability_mask)
        if traversability.shape != occupancy.shape:
            traversability = traversability.reshape(occupancy.shape)
    except Exception:
        return True
    return float(traversability[row, col]) <= 0.0


def _workspace_range_is_valid(
    candidate: np.ndarray,
    workspace_rows_cols: np.ndarray,
    min_allowed_tiles: float,
    max_allowed_tiles: float,
) -> bool:
    closest, furthest = _workspace_range_distances(candidate, workspace_rows_cols)
    return closest >= min_allowed_tiles and furthest <= max_allowed_tiles


def _workspace_range_distances(
    candidate: np.ndarray,
    workspace_rows_cols: np.ndarray,
) -> tuple[float, float]:
    distances = np.linalg.norm(workspace_rows_cols - candidate[None, :], axis=1)
    if distances.size == 0:
        return float("inf"), float("inf")
    return float(np.min(distances)), float(np.max(distances))


def _range_failure_reason(
    closest_tiles: float,
    furthest_tiles: float,
    min_allowed_tiles: float,
    max_allowed_tiles: float,
) -> str:
    min_violation = closest_tiles < min_allowed_tiles
    max_violation = furthest_tiles > max_allowed_tiles
    if min_violation and max_violation:
        return "min_dist_and_max_dist_violated"
    if min_violation:
        return "min_dist_violated"
    if max_violation:
        return "max_dist_violated"
    return "range_ok"


def _entry_plan_yaw_rads(entry: dict[str, Any]) -> tuple[float, float]:
    agent_state = entry.get("agent_state", {}) or {}
    angle_base_rad = agent_state.get("angle_base_rad_override")
    angle_cabin_rad = agent_state.get("angle_cabin_rad_override")
    if angle_base_rad is None:
        angle_base_rad = _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_base", 0.0))
    if angle_cabin_rad is None:
        angle_cabin_rad = _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_cabin", 0.0))
    return float(angle_base_rad), float(angle_cabin_rad)


def _set_entry_pose(
    entry: dict[str, Any],
    pos_base: np.ndarray,
    yaw_rads: tuple[float, float],
) -> tuple[list[float] | None, list[float], float]:
    agent_state = entry.setdefault("agent_state", {})
    old_pos = agent_state.get("pos_base")
    old_base = None
    if old_pos is not None and len(old_pos) >= 2:
        old_base = [float(old_pos[0]), float(old_pos[1])]

    new_pos = [float(pos_base[0]), float(pos_base[1])]
    movement_tiles = 0.0
    if old_base is not None:
        movement_tiles = float(np.linalg.norm(np.asarray(new_pos) - np.asarray(old_base)))

    agent_state["pos_base"] = new_pos
    agent_state["angle_base_rad_override"] = _wrap_to_pi(yaw_rads[0])
    agent_state["angle_cabin_rad_override"] = _wrap_to_pi(yaw_rads[1])
    return old_base, new_pos, movement_tiles


def _paired_dig_dump_indices(plan: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, int]]:
    dig_to_dump = {}
    dump_to_dig = {}
    pending_dig = None
    for idx, entry in enumerate(plan):
        if _is_load_transition(entry):
            pending_dig = idx
        elif pending_dig is not None and _is_unload_transition(entry):
            dig_to_dump[pending_dig] = idx
            dump_to_dig[idx] = pending_dig
            pending_dig = None
    return dig_to_dump, dump_to_dig


def _foundation_border_mask(target: np.ndarray, border_width_tiles: int) -> np.ndarray:
    dig_target = np.asarray(target) < 0
    eroded = dig_target.copy()
    for _ in range(max(0, int(border_width_tiles))):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        neighbor_sum = (
            padded[:-2, :-2].astype(np.int8)
            + padded[:-2, 1:-1].astype(np.int8)
            + padded[:-2, 2:].astype(np.int8)
            + padded[1:-1, :-2].astype(np.int8)
            + padded[1:-1, 1:-1].astype(np.int8)
            + padded[1:-1, 2:].astype(np.int8)
            + padded[2:, :-2].astype(np.int8)
            + padded[2:, 1:-1].astype(np.int8)
            + padded[2:, 2:].astype(np.int8)
        )
        eroded = eroded & (neighbor_sum == 9)
    return dig_target & ~eroded


def _border_dump_facing_yaw_rad(
    entry: dict[str, Any],
    target: np.ndarray,
    border_mask: np.ndarray,
) -> float | None:
    if not _is_load_transition(entry):
        return None
    if entry.get("dig_type") == "lift_dug_dirt":
        return None

    try:
        changed = np.asarray(entry.get("actual_terrain_modification_mask")).astype(bool)
        if changed.shape != target.shape:
            changed = changed.reshape(target.shape)
    except Exception:
        return None

    changed_border = changed & border_mask & (np.asarray(target) < 0)
    if not changed_border.any():
        return None

    dump_rows, dump_cols = np.nonzero(np.asarray(target) > 0)
    if dump_rows.size == 0:
        return None

    rows, cols = np.nonzero(changed_border)
    center = np.asarray([float(rows.mean()), float(cols.mean())])
    dump_center = np.asarray([float(dump_rows.mean()), float(dump_cols.mean())])

    dig_target = (np.asarray(target) < 0).astype(np.float32)
    padded = np.pad(dig_target, 1, mode="constant", constant_values=0.0)
    grad_col = padded[1:-1, 2:] - padded[1:-1, :-2]
    grad_row = padded[2:, 1:-1] - padded[:-2, 1:-1]
    normal = np.asarray([float(grad_row[changed_border].mean()), float(grad_col[changed_border].mean())])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None

    tangent = np.asarray([-normal[1], normal[0]], dtype=float) / norm
    to_dump = dump_center - center
    if float(np.dot(tangent, to_dump)) < 0.0:
        tangent = -tangent

    return _wrap_to_pi(math.atan2(float(tangent[1]), float(tangent[0])))


def _postprocess_base_positions(
    plan: list[dict[str, Any]],
    map_path: Path,
    env_cfgs: Any,
    step_tiles: float,
    margin_tiles: float,
    min_range_tiles_override: float | None,
    max_range_tiles_override: float | None,
    max_distance_m: float | None,
    face_workspace: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target, occupancy, _ = _load_single_map_arrays(map_path)
    shape = occupancy.shape
    tile_size = _scalar_float(getattr(env_cfgs, "tile_size", None), 0.6875)
    border_width_tiles = int(round(_scalar_float(getattr(env_cfgs, "foundation_border_width_tiles", None), 2.0)))
    border_mask = _foundation_border_mask(target, border_width_tiles)
    dig_to_dump, dump_to_dig = _paired_dig_dump_indices(plan)
    stats = {
        "considered": 0,
        "moved": 0,
        "not_adjusted": 0,
        "paired_dump_pose_copied": 0,
        "border_yaw_aligned": 0,
        "skipped_empty_mask": 0,
        "skipped_missing_base": 0,
        "skipped_invalid_range": 0,
        "skipped_blocked": 0,
        "adjustments": [],
        "skipped_adjustments": [],
    }

    for waypoint_idx, entry in enumerate(plan):
        stats["considered"] += 1
        if waypoint_idx in dump_to_dig:
            continue

        agent_state = entry.get("agent_state", {}) or {}
        pos_base = agent_state.get("pos_base")
        if pos_base is None or len(pos_base) < 2:
            stats["skipped_missing_base"] += 1
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "reason": "missing_base",
                }
            )
            continue

        mask = _workspace_mask_for_adjustment(entry, shape)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            stats["skipped_empty_mask"] += 1
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "reason": "empty_workspace_mask",
                    "from": [float(pos_base[0]), float(pos_base[1])],
                }
            )
            continue

        workspace_rows_cols = np.stack([rows.astype(float), cols.astype(float)], axis=1)
        workspace_center = workspace_rows_cols.mean(axis=0)
        base = np.asarray([float(pos_base[0]), float(pos_base[1])], dtype=float)
        from_workspace = base - workspace_center
        center_distance = float(np.linalg.norm(from_workspace))
        if center_distance <= 1e-9:
            stats["skipped_invalid_range"] += 1
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "reason": "base_at_workspace_center",
                    "from": [float(base[0]), float(base[1])],
                }
            )
            continue

        yaw_rads = _entry_plan_yaw_rads(entry)
        border_yaw_rad = _border_dump_facing_yaw_rad(entry, target, border_mask)
        if border_yaw_rad is not None:
            yaw_rads = (border_yaw_rad, border_yaw_rad)

        min_range_tiles, max_range_tiles = _reach_limits_tiles(
            env_cfgs,
            int(_to_serializable(entry.get("agent_type", 0))),
            min_range_tiles_override,
            max_range_tiles_override,
        )
        min_allowed_tiles = min_range_tiles + margin_tiles
        max_allowed_tiles = max_range_tiles - margin_tiles
        if max_distance_m is not None and max_distance_m > 0.0:
            max_allowed_tiles = min(max_allowed_tiles, float(max_distance_m) / max(tile_size, 1e-6))
        if min_allowed_tiles >= max_allowed_tiles:
            stats["skipped_invalid_range"] += 1
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "reason": "invalid_reach_band",
                    "from": [float(base[0]), float(base[1])],
                    "min_allowed_tiles": min_allowed_tiles,
                    "max_allowed_tiles": max_allowed_tiles,
                    "min_allowed_m": min_allowed_tiles * tile_size,
                    "max_allowed_m": max_allowed_tiles * tile_size,
                }
            )
            continue

        target_center_distance = min(
            max(center_distance - step_tiles, min_allowed_tiles),
            max_allowed_tiles,
        )
        desired = workspace_center + from_workspace / center_distance * target_center_distance
        traversability_mask = entry.get("traversability_mask")

        accepted = None
        blocked_candidate_seen = False
        failed_trials = []
        for fraction in (1.0, 0.75, 0.5, 0.25):
            candidate = base + (desired - base) * fraction
            closest_tiles, furthest_tiles = _workspace_range_distances(candidate, workspace_rows_cols)
            range_reason = _range_failure_reason(
                closest_tiles,
                furthest_tiles,
                min_allowed_tiles,
                max_allowed_tiles,
            )
            failed_trials.append(
                {
                    "fraction": fraction,
                    "candidate": [float(candidate[0]), float(candidate[1])],
                    "closest_tiles": closest_tiles,
                    "furthest_tiles": furthest_tiles,
                    "closest_m": closest_tiles * tile_size,
                    "furthest_m": furthest_tiles * tile_size,
                    "reason": range_reason,
                    "range_violation_tiles": (
                        max(0.0, min_allowed_tiles - closest_tiles)
                        + max(0.0, furthest_tiles - max_allowed_tiles)
                    ),
                }
            )
            if range_reason != "range_ok":
                continue
            if not _base_position_is_traversable(candidate, occupancy, traversability_mask):
                blocked_candidate_seen = True
                failed_trials[-1]["reason"] = "blocked_or_non_traversable"
                continue
            accepted = candidate
            break

        if accepted is None:
            accepted_from_failed_trials = True
            if blocked_candidate_seen:
                stats["skipped_blocked"] += 1
            else:
                stats["skipped_invalid_range"] += 1
            accepted = base
            best_trial = min(
                failed_trials,
                key=lambda trial: (
                    trial["range_violation_tiles"],
                    0 if trial["reason"] == "blocked_or_non_traversable" else 1,
                ),
            ) if failed_trials else None
            skip_reason = "blocked_or_non_traversable" if blocked_candidate_seen else "no_valid_candidate"
            if best_trial is not None and best_trial["reason"] != "range_ok":
                skip_reason = best_trial["reason"]
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "agent_index": int(_to_serializable(entry.get("agent_index", 0))),
                    "agent_type": int(_to_serializable(entry.get("agent_type", 0))),
                    "reason": skip_reason,
                    "from": [float(base[0]), float(base[1])],
                    "best_trial": best_trial,
                    "min_allowed_tiles": min_allowed_tiles,
                    "max_allowed_tiles": max_allowed_tiles,
                    "min_allowed_m": min_allowed_tiles * tile_size,
                    "max_allowed_m": max_allowed_tiles * tile_size,
                }
            )
        else:
            accepted_from_failed_trials = False

        movement_tiles = float(np.linalg.norm(accepted - base))
        if movement_tiles <= 1e-9 and border_yaw_rad is None and not accepted_from_failed_trials:
            closest_tiles, furthest_tiles = _workspace_range_distances(base, workspace_rows_cols)
            stats["not_adjusted"] += 1
            stats["skipped_adjustments"].append(
                {
                    "waypoint_index": waypoint_idx,
                    "step": int(_to_serializable(entry.get("step", waypoint_idx))),
                    "agent_index": int(_to_serializable(entry.get("agent_index", 0))),
                    "agent_type": int(_to_serializable(entry.get("agent_type", 0))),
                    "reason": "base_position_unchanged",
                    "from": [float(base[0]), float(base[1])],
                    "closest_tiles": closest_tiles,
                    "furthest_tiles": furthest_tiles,
                    "closest_m": closest_tiles * tile_size,
                    "furthest_m": furthest_tiles * tile_size,
                    "min_allowed_tiles": min_allowed_tiles,
                    "max_allowed_tiles": max_allowed_tiles,
                    "min_allowed_m": min_allowed_tiles * tile_size,
                    "max_allowed_m": max_allowed_tiles * tile_size,
                }
            )
        if movement_tiles > 1e-9 and face_workspace and border_yaw_rad is None:
            to_workspace = workspace_center - accepted
            yaw_rad = math.atan2(float(to_workspace[1]), float(to_workspace[0]))
            yaw_rads = (_wrap_to_pi(yaw_rad), _wrap_to_pi(yaw_rad))

        entries_to_update = [(waypoint_idx, entry)]
        paired_dump_idx = dig_to_dump.get(waypoint_idx)
        if paired_dump_idx is not None:
            entries_to_update.append((paired_dump_idx, plan[paired_dump_idx]))
            stats["paired_dump_pose_copied"] += 1

        for update_idx, update_entry in entries_to_update:
            before, after, entry_movement_tiles = _set_entry_pose(update_entry, accepted, yaw_rads)
            if entry_movement_tiles <= 1e-9 and border_yaw_rad is None and update_idx == waypoint_idx:
                continue
            stats["moved"] += 1
            if border_yaw_rad is not None:
                stats["border_yaw_aligned"] += 1
            stats["adjustments"].append(
                {
                    "waypoint_index": update_idx,
                    "step": int(_to_serializable(update_entry.get("step", update_idx))),
                    "agent_index": int(_to_serializable(update_entry.get("agent_index", 0))),
                    "agent_type": int(_to_serializable(update_entry.get("agent_type", 0))),
                    "paired_with": paired_dump_idx if update_idx == waypoint_idx else waypoint_idx,
                    "from": before,
                    "to": after,
                    "movement_tiles": entry_movement_tiles,
                    "movement_m": entry_movement_tiles * tile_size,
                    "border_yaw_aligned": border_yaw_rad is not None,
                }
            )

    return plan, stats


def _render_plan_gif(
    plan: list[dict[str, Any]],
    map_path: Path,
    out_path: Path,
    scale: int = 8,
    duration_ms: int = 200,
):
    """Render a lightweight plan GIF (DO-waypoints only) without running the environment.

    If the plan has 0 waypoints, render a single frame of the base map so the CLI
    can still produce an artifact without failing.
    """
    from PIL import Image, ImageDraw

    target, occupancy, dumpability = _load_single_map_arrays(map_path)
    h, w = target.shape

    styles = {
        "dig": ((38, 118, 255), (10, 65, 180)),
        "lift": ((150, 72, 210), (88, 34, 145)),
        "dump": ((245, 135, 28), (170, 75, 0)),
        "terrain": ((80, 80, 80), (35, 35, 35)),
    }

    def as_bool_mask(value):
        if value is None:
            return np.zeros((h, w), dtype=bool)
        arr = np.asarray(value).astype(bool)
        if arr.shape != (h, w):
            arr = np.reshape(arr, (h, w))
        return arr

    def waypoint_kind(entry):
        lsc = entry.get("loaded_state_change", {}) or {}
        before = _as_py_bool(lsc.get("before", False))
        after = _as_py_bool(lsc.get("after", False))
        if not before and after:
            return "lift" if entry.get("dig_type") == "lift_dug_dirt" else "dig"
        if before and not after:
            return "dump"
        return "terrain"

    def workspace_mask(entry):
        mask = as_bool_mask(entry.get("terrain_modification_mask"))
        if mask.any():
            return mask
        kind = waypoint_kind(entry)
        if kind in ("dig", "lift"):
            return as_bool_mask(entry.get("dug_mask"))
        if kind == "dump":
            return as_bool_mask(entry.get("dump_mask"))
        return mask

    def base_rgb():
        rgb = np.full((h, w, 3), 245, dtype=np.uint8)
        rgb[occupancy] = (20, 20, 20)
        rgb[target < 0] = (255, 220, 220)
        rgb[target > 0] = (220, 255, 220)
        rgb[(~occupancy) & (~dumpability)] = (235, 235, 235)
        return rgb

    def render_frame(entry):
        pos = entry.get("agent_state", {}).get("pos_base", None)
        agent_row_col = (float(pos[0]), float(pos[1])) if pos is not None else None

        rgb = base_rgb()
        kind = waypoint_kind(entry)
        fill, outline = styles[kind]
        dug_mask = as_bool_mask(entry.get("dug_mask"))
        dump_mask = as_bool_mask(entry.get("dump_mask"))
        mask = workspace_mask(entry)
        rgb[dug_mask] = styles["dig"][0]
        rgb[dump_mask] = styles["dump"][0]

        if mask.any():
            highlight = np.asarray(fill, dtype=np.float32)
            rgb[mask] = np.clip(
                0.35 * rgb[mask].astype(np.float32) + 0.65 * highlight,
                0,
                255,
            ).astype(np.uint8)

        img = Image.fromarray(rgb, mode="RGB").resize((w * scale, h * scale), Image.NEAREST)
        draw = ImageDraw.Draw(img)

        rows, cols = np.nonzero(mask)
        for row, col in zip(rows, cols):
            x0 = int(col * scale)
            y0 = int(row * scale)
            draw.rectangle(
                (x0, y0, x0 + scale - 1, y0 + scale - 1),
                outline=outline,
                width=max(1, scale // 4),
            )

        if agent_row_col is not None:
            # Terra stores pos_base=[row, col]. Convert to image x/y pixels here.
            row, col = agent_row_col
            cx = int(round((col + 0.5) * scale))
            cy = int(round((row + 0.5) * scale))
            r = max(3, scale // 2)
            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=outline,
                outline=(255, 255, 255),
                width=1,
            )

        step = entry.get("step")
        if step is not None:
            draw.text((4, 4), f"step {step}", fill=(0, 0, 0))
        return img

    frames = [render_frame(e) for e in plan] if plan else [render_frame({})]
    durations = duration_ms
    if len(plan) > 1:
        steps = [int(entry.get("step", idx)) for idx, entry in enumerate(plan)]
        gaps = [max(1, b - a) for a, b in zip(steps, steps[1:])]
        gaps.append(1)
        durations = [min(2000, max(duration_ms, duration_ms * gap)) for gap in gaps]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def extract_plan(
    env,
    model,
    model_params,
    env_cfgs,
    rl_config,
    max_frames,
    seed,
    render_rollout_gif: bool = False,
    rollout_gif_every: int = 1,
    use_mcts: bool = False,
):
    """Extract plan by capturing action_map and robot state on DO actions."""
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, 1)  # Just one environment
    timestep = env.reset(env_cfgs, rng_reset)
    
    # Print agent summary at start
    agent_type_names = {0: "excavator", 1: "truck", 2: "skidsteer"}
    num_agents_init = int(np.array(timestep.observation["num_agents"]).reshape(-1)[0])
    print(f"\n=== Agent Configuration ===")
    print(f"Number of agents: {num_agents_init}")
    # Get agent types from observation (agent_states has shape [B, MAX_AGENTS, feat])
    # Feature index 6 is agent_type
    agent_states_init = timestep.observation["agent_states"][0]  # [MAX_AGENTS, feat]
    for i in range(num_agents_init):
        agent_type_val = int(agent_states_init[i, 6])
        agent_type_str = agent_type_names.get(agent_type_val, f"unknown({agent_type_val})")
        print(f"  Agent {i}: {agent_type_str} (type={agent_type_val})")
    print(f"===========================\n")
    
    prev_actions = jnp.zeros(
        (1, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    def _get_current_action_class_and_do(timestep, t_counter):
        """
        Determine the currently acting agent's action class from the *state*
        (not env.batch_cfg), so this works for mixed / overridden action types.
        """
        # Compute agent index - agents act in round-robin order and can't skip
        num_agents = int(np.array(timestep.observation["num_agents"]).reshape(-1)[0])
        agent_index = t_counter % max(num_agents, 1)
        # Access agent state by fixed slot index
        agent_state = timestep.state.agent.agent_states[agent_index]
        action_type_val = int(np.array(agent_state.action_type).flatten()[0])
        if action_type_val == 0:
            return TrackedAction, int(TrackedActionType.DO)
        if action_type_val == 1:
            return WheeledAction, int(WheeledActionType.DO)
        raise ValueError(f"Unknown action_type={action_type_val}")

    def _full_action_map(timestep):
        return jnp.squeeze(timestep.state.world.action_map.map).copy()

    def _single_env_state(timestep):
        def _take_first_env(value):
            if isinstance(value, (jax.Array, np.ndarray)) and value.ndim > 0:
                return value[0]
            return value

        return jax.tree_map(_take_first_env, timestep.state)

    def _state_for_agent(timestep, agent_index):
        state = _single_env_state(timestep)
        return state._replace(
            agent=state.agent._replace(
                current_agent=jnp.asarray(agent_index, dtype=jnp.int32)
            )
        )

    def _full_workspace_mask(timestep, agent_index):
        state = _state_for_agent(timestep, agent_index)
        workspace_mask = state._build_dig_dump_cone().reshape(
            state.world.action_map.map.shape
        ).astype(jnp.bool_)
        obstacle_mask = state.world.padding_mask.map == 1
        return jnp.logical_and(workspace_mask, ~obstacle_mask)

    def _dilate_8_connected(mask):
        padded = jnp.pad(mask.astype(jnp.bool_), 1, mode="constant", constant_values=False)
        return (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )

    def _already_dug_foundation_mask(state):
        return jnp.logical_and(
            state.world.target_map.map < 0,
            state.world.action_map.map < 0,
        )

    def _lift_dumped_dirt_workspace_mask(timestep, agent_index):
        state = _state_for_agent(timestep, agent_index)
        workspace_mask = _full_workspace_mask(timestep, agent_index)
        already_dug_foundation = _already_dug_foundation_mask(state)
        return jnp.logical_and(workspace_mask, ~already_dug_foundation)

    def _dump_workspace_mask(timestep, agent_index):
        state = _state_for_agent(timestep, agent_index)
        workspace_mask = _full_workspace_mask(timestep, agent_index)
        buffered_foundation = _dilate_8_connected(_already_dug_foundation_mask(state))
        return jnp.logical_and(workspace_mask, ~buffered_foundation)

    # Plan storage
    plan = []

    mcts_step = None
    mcts_ppo_diff_count = 0
    if use_mcts:
        mcts_step = make_mcts_step_fn(model, env, rl_config)
        print(
            f"MCTS enabled: num_simulations={rl_config.num_simulations}, "
            f"gamma={rl_config.gamma}"
        )
        print("Warming up MCTS JIT compilation...")
        _, _, _, warm_action, _, _ = mcts_step(model_params, rng, timestep, prev_actions)
        jax.block_until_ready(warm_action)

        # Reset after warmup so the extracted plan starts from the requested seed.
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, 1)
        timestep = env.reset(env_cfgs, rng_reset)
        prev_actions = jnp.zeros(
            (1, rl_config.num_prev_actions),
            dtype=jnp.int32
        )

    def _append_do_plan_entry(
        timestep_before,
        timestep_after,
        t_counter,
        agent_index,
        acting_agent_state_before,
        loaded_before,
        agent_type_before,
        action_map_before,
        traversability_mask,
    ):
        agent_type_str = agent_type_names.get(
            agent_type_before, f"unknown({agent_type_before})"
        )

        # IMPORTANT: After env.step(), observation["agent_states"][0, 0] is the NEXT agent (rotated view).
        # We need to access the acting agent's state by its fixed slot index from timestep.state.
        action_map_after = _full_action_map(timestep_after)
        acting_agent_state_after = timestep_after.state.agent.agent_states[agent_index]
        loaded_after = bool(np.array(acting_agent_state_after.loaded).flatten()[0])

        changed_tiles = action_map_before != action_map_after
        dug_mask = (action_map_after < 0)
        dump_mask = (action_map_after > 0)

        # Determine dig_type for digging actions:
        # "lift_dug_dirt" = picking up previously dumped dirt (action_map_before > 0 in changed tiles)
        # "dig_new_soil" = digging fresh terrain from target
        dig_type = None
        action_type_label = ""
        if not loaded_before and loaded_after:
            dumped_dirt_in_changed_tiles = jnp.sum(action_map_before[changed_tiles] > 0)
            moving_dumped_dirt = dumped_dirt_in_changed_tiles > 0
            dig_type = "lift_dug_dirt" if bool(moving_dumped_dirt) else "dig_new_soil"
            action_type_label = f" (digging: {dig_type})"
        elif loaded_before and not loaded_after:
            action_type_label = " (dumping)"
        else:
            action_type_label = f" (unchanged, loaded={bool(loaded_before)})"

        if loaded_before and not loaded_after:
            terrain_modification_mask = _dump_workspace_mask(
                timestep_before, agent_index
            )
        elif dig_type == "lift_dug_dirt":
            terrain_modification_mask = _lift_dumped_dirt_workspace_mask(
                timestep_before, agent_index
            )
        else:
            # Keep normal foundation digging as the actually changed tiles.
            terrain_modification_mask = changed_tiles.astype(jnp.bool_)

        print(f"DO action at step {t_counter} by agent {agent_index} ({agent_type_str}){action_type_label}, {jnp.sum(changed_tiles)} tiles modified")

        plan_entry = {
            'step': t_counter,
            'traversability_mask': traversability_mask,
            'terrain_modification_mask': terrain_modification_mask,
            'actual_terrain_modification_mask': changed_tiles.astype(jnp.bool_),
            'dug_mask': dug_mask.copy(),
            'dump_mask': dump_mask.copy(),
            'agent_state': {
                'pos_base': [float(np.array(acting_agent_state_before.pos_base).flatten()[0]),
                             float(np.array(acting_agent_state_before.pos_base).flatten()[1])],
                'angle_base': float(np.array(acting_agent_state_before.angle_base).flatten()[0]),
                'angle_cabin': float(np.array(acting_agent_state_before.angle_cabin).flatten()[0]),
                'wheel_angle': float(np.array(acting_agent_state_before.wheel_angle).flatten()[0]),
            },
            'loaded_state_change': {
                'before': loaded_before,
                'after': loaded_after,
            },
            # Additional fields (not in foundation-plan.json but kept for compatibility)
            'agent_type': agent_type_before,
            'agent_index': agent_index,
        }
        if dig_type is not None:
            plan_entry['dig_type'] = dig_type
        plan.append(plan_entry)

    t_counter = 0

    if render_rollout_gif:
        obs0 = dict(timestep.observation)
        obs0["action_map"] = timestep.state.world.action_map.map
        if "interaction_mask" in obs0:
            obs0["interaction_mask"] = jnp.zeros_like(obs0["interaction_mask"])
        env.terra_env.render_obs_pygame(obs0, generate_gif=True)

    while True:
        # Determine current action class from state (tracked vs wheeled)
        action_cls, do_action = _get_current_action_class_and_do(timestep, t_counter)
        num_agents = int(np.array(timestep.observation["num_agents"]).reshape(-1)[0])
        agent_index = t_counter % max(num_agents, 1)

        if use_mcts:
            timestep_before = timestep
            acting_agent_state_before = timestep.state.agent.agent_states[agent_index]
            loaded_before = bool(np.array(acting_agent_state_before.loaded).flatten()[0])
            agent_type_before = int(np.array(acting_agent_state_before.agent_type).flatten()[0])
            action_map_before = _full_action_map(timestep)
            traversability_mask = jnp.squeeze(timestep.observation["traversability_mask"]).copy()

            rng, timestep, prev_actions, action, ppo_action, mcts_action = mcts_step(
                model_params, rng, timestep, prev_actions
            )
            mcts_ppo_diff_count += int(
                (np.asarray(ppo_action) != np.asarray(mcts_action)).sum()
            )

            if int(np.asarray(action).reshape(-1)[0]) == do_action:
                _append_do_plan_entry(
                    timestep_before,
                    timestep,
                    t_counter,
                    agent_index,
                    acting_agent_state_before,
                    loaded_before,
                    agent_type_before,
                    action_map_before,
                    traversability_mask,
                )
        else:
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            # Get action from policy
            # In the multi-agent setup, observations already have the currently acting
            # agent in slot 0 of "agent_states". obs_to_model_input handles this,
            # so we can treat the batch as size 1 and proceed as in single-agent.
            obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, logits_pi = model.apply(model_params, obs_model)
            pi = tfp.distributions.Categorical(logits=logits_pi)
            action = pi.sample(seed=rng_act)

            # Check if DO action and record state BEFORE executing the action
            if action[0] == do_action:
                timestep_before = timestep
                # Access the acting agent's state by its fixed slot index
                acting_agent_state_before = timestep.state.agent.agent_states[agent_index]
                # Use np.array().item() to handle batch dimensions in state fields
                loaded_before = bool(np.array(acting_agent_state_before.loaded).flatten()[0])
                agent_type_before = int(np.array(acting_agent_state_before.agent_type).flatten()[0])

                action_map_before = _full_action_map(timestep)
                traversability_mask = jnp.squeeze(timestep.observation["traversability_mask"]).copy()

                # Update previous actions
                prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
                prev_actions = prev_actions.at[:, 0].set(action)

                # Take step in environment
                rng_step = jax.random.split(rng_step, 1)
                timestep = env.step(
                    timestep, wrap_action(action, action_cls), rng_step
                )

                _append_do_plan_entry(
                    timestep_before,
                    timestep,
                    t_counter,
                    agent_index,
                    acting_agent_state_before,
                    loaded_before,
                    agent_type_before,
                    action_map_before,
                    traversability_mask,
                )
            else:
                # Update previous actions
                prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
                prev_actions = prev_actions.at[:, 0].set(action)

                # Take step in environment
                rng_step = jax.random.split(rng_step, 1)
                timestep = env.step(
                    timestep, wrap_action(action, action_cls), rng_step
                )

        t_counter += 1

        if render_rollout_gif and (t_counter % max(1, int(rollout_gif_every)) == 0):
            obs1 = dict(timestep.observation)
            obs1["action_map"] = timestep.state.world.action_map.map
            if "interaction_mask" in obs1:
                obs1["interaction_mask"] = jnp.zeros_like(obs1["interaction_mask"])
            env.terra_env.render_obs_pygame(obs1, generate_gif=True)


        # Check if done
        if bool(np.asarray(timestep.done)[0]) or t_counter == max_frames:
            break

    if use_mcts:
        total_decisions = max(t_counter, 1)
        pct = 100.0 * mcts_ppo_diff_count / total_decisions
        print(f"MCTS != PPO: {mcts_ppo_diff_count}/{total_decisions} ({pct:.1f}%)")

    return plan


def main():
    def _canon_lists(x):
        """Convert Python lists to tuples/arrays so JAX pytrees stay replace-able."""
        if isinstance(x, list):
            try:
                return jnp.asarray(x)
            except Exception:
                return tuple(_canon_lists(v) for v in x)
        return x

    parser = argparse.ArgumentParser(description="Extract plan from policy")
    parser.add_argument(
        "-policy",
        "--policy_path",
        type=str,
        required=True,
        help="Path to the policy .pkl file"
    )
    # By default, use a map located in a local "test_map" subfolder next to this script.
    default_map_path = str((Path(__file__).parent / "test_map_v3").resolve())
    parser.add_argument(
        "-map",
        "--map_path",
        type=str,
        default=default_map_path,
        help=f"Path to the map file (default: {default_map_path})"
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=350,
        help="Maximum number of steps"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=str(Path(__file__).parent / "plan.pkl"),
        help="Output path for the plan"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Named config preset to match inference_single_map.py behavior.",
    )
    parser.add_argument(
        "--use-mcts",
        dest="use_mcts",
        action="store_true",
        help="Enable MCTS planning at each extraction step. Off by default.",
    )
    parser.add_argument(
        "--num_simulations",
        "-sim",
        type=int,
        default=32,
        help="MCTS simulations per step when --use-mcts is set.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor for MCTS. Defaults to checkpoint config.gamma or 0.99.",
    )
    parser.add_argument(
        "--serialize",
        action="store_true",
        help="Deprecated: JSON is now saved by default next to the PKL."
    )
    parser.add_argument(
        "--render_plan_gif",
        action="store_true",
        help="Render a lightweight plan GIF (DO-waypoints only) next to the output plan.",
    )
    parser.add_argument(
        "--render_rollout_gif",
        action="store_true",
        help="Render a full rollout GIF (every step) using Terra's renderer.",
    )
    parser.add_argument(
        "--rollout_gif_every",
        type=int,
        default=1,
        help="Render every Nth step into the rollout GIF (default: 1).",
    )
    parser.add_argument(
        "--postprocess_base_positions",
        action="store_true",
        help=(
            "After DO extraction/filtering, move each recorded base position slightly "
            "toward its workspace while keeping all workspace tiles within reach."
        ),
    )
    parser.add_argument(
        "--base_adjust_step_tiles",
        type=float,
        default=1.8,
        help="Maximum continuous base-position adjustment toward the workspace, in Terra tile units.",
    )
    parser.add_argument(
        "--base_adjust_margin_tiles",
        type=float,
        default=0.05,
        help="Safety margin applied inside the min/max workspace reach band, in Terra tile units.",
    )
    parser.add_argument(
        "--base_adjust_min_range_tiles",
        type=float,
        default=None,
        help="Optional min allowed distance from adjusted base to closest workspace tile, in tile units.",
    )
    parser.add_argument(
        "--base_adjust_max_range_tiles",
        type=float,
        default=None,
        help="Optional max allowed distance from adjusted base to furthest workspace tile, in tile units.",
    )
    parser.add_argument(
        "--base_adjust_max_distance_m",
        type=float,
        default=7.4,
        help=(
            "Hard cap on distance from adjusted base to the furthest workspace tile, in meters. "
            "Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--base_adjust_face_workspace",
        action="store_true",
        help="When adjusting a base position, emit schema-v2 base/cabin yaw pointing at the workspace.",
    )

    args = parser.parse_args()

    log = load_pkl_object(args.policy_path)
    config = jax.tree_map(_canon_lists, log["train_config"])
    config.num_test_rollouts = 1  # Only one environment
    config.num_devices = 1
    config.num_embeddings_agent_min = 60
    config.num_simulations = args.num_simulations
    if args.gamma is not None:
        config.gamma = args.gamma
    if not hasattr(config, "gamma"):
        config.gamma = 0.99
    print(f"clip_action_maps setting: {config.clip_action_maps}")

    # Create environment
    env_cfgs = jax.tree_map(_canon_lists, log["env_config"])
    def _extract_single_env(x):
        """Extract first env config, handling scalars/bools that can't be subscripted.
        
        Ensures output has at least rank 1 for vmap compatibility.
        """
        try:
            return x[0][None, ...]
        except (TypeError, IndexError):
            # Scalar or bool - wrap in 1D array for vmap compatibility
            return jnp.atleast_1d(x)
    env_cfgs = jax.tree_map(_extract_single_env, env_cfgs)

    batch_cfg = None
    if args.config is not None:
        try:
            from configs.training_configs import get_config

            preset = get_config(args.config)
            print(f"\nLoading config preset: '{args.config}'")

            env_cfgs = env_cfgs._replace(
                agent_types=jnp.asarray(tuple(preset.agent_types), dtype=jnp.int32)[
                    None, ...
                ],
                action_types=jnp.asarray(tuple(preset.action_types), dtype=jnp.int32)[
                    None, ...
                ],
            )

            if preset.maps and len(preset.maps) > 0:
                curriculum_levels = []
                for map_level in preset.maps:
                    rewards_type = (
                        RewardsType.DENSE
                        if map_level.rewards_type == "DENSE"
                        else RewardsType.SPARSE
                    )
                    curriculum_levels.append(
                        {
                            "maps_path": map_level.maps_path,
                            "max_steps_in_episode": map_level.max_steps_in_episode,
                            "rewards_type": rewards_type,
                            "apply_trench_rewards": map_level.apply_trench_rewards,
                        }
                    )

                class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
                    levels = curriculum_levels

                batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())
                first_level = curriculum_levels[0]
                env_cfgs = env_cfgs._replace(
                    max_steps_in_episode=jnp.full(
                        (1,), int(first_level["max_steps_in_episode"]), dtype=jnp.int32
                    ),
                    apply_trench_rewards=jnp.full(
                        (1,), bool(first_level["apply_trench_rewards"]), dtype=jnp.bool_
                    ),
                )
        except Exception as e:
            print(f"Failed to load --config preset '{args.config}': {e}")

    if batch_cfg is None:
        batch_cfg = BatchConfig()

    if args.use_mcts:
        env_cfgs = fix_env_cfg_dtypes(env_cfgs)
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=args.render_rollout_gif,
        n_envs_x_rendering=1,
        n_envs_y_rendering=1,
        display=False,
        shuffle_maps=False,
        single_map_path=args.map_path,
    )

    # Match inference_single_map.py: keep checkpoint env_cfgs fixed during rollout.
    class _NoopCurriculumManager:
        def reset_cfgs(self, env_cfgs):
            return env_cfgs

        def update_cfgs(self, timesteps, rng):
            return timesteps

    env.curriculum_manager = _NoopCurriculumManager()

    # Load neural network
    model = load_neural_network(config, env)
    model_params = log["model"]

    # Extract plan
    plan = extract_plan(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        seed=args.seed,
        render_rollout_gif=args.render_rollout_gif,
        rollout_gif_every=args.rollout_gif_every,
        use_mcts=args.use_mcts,
    )

    raw_len = len(plan)
    plan, filter_stats = _filter_empty_do_actions(plan)
    if raw_len != len(plan):
        print(
            f"Filtered DO waypoints: {filter_stats['kept_waypoints']}/{filter_stats['raw_waypoints']} kept "
            f"({filter_stats['kept_digging']} digging, {filter_stats['kept_dumping']} dumping, "
            f"{filter_stats['kept_with_terrain_change']} terrain-only), "
            f"{filter_stats['dropped_empty_unchanged']} empty/unchanged dropped."
        )

    if args.postprocess_base_positions:
        postprocess_env_cfgs = env.update_env_cfgs(env_cfgs)
        plan, base_adjust_stats = _postprocess_base_positions(
            plan,
            Path(args.map_path),
            postprocess_env_cfgs,
            step_tiles=max(0.0, float(args.base_adjust_step_tiles)),
            margin_tiles=max(0.0, float(args.base_adjust_margin_tiles)),
            min_range_tiles_override=args.base_adjust_min_range_tiles,
            max_range_tiles_override=args.base_adjust_max_range_tiles,
            max_distance_m=args.base_adjust_max_distance_m,
            face_workspace=bool(args.base_adjust_face_workspace),
        )
        print(
            "Post-processed base positions: "
            f"{base_adjust_stats['moved']}/{base_adjust_stats['considered']} moved, "
            f"{base_adjust_stats['not_adjusted']} not adjusted "
            f"({base_adjust_stats['skipped_empty_mask']} empty mask, "
            f"{base_adjust_stats['skipped_missing_base']} missing base, "
            f"{base_adjust_stats['skipped_invalid_range']} range rejected, "
            f"{base_adjust_stats['skipped_blocked']} blocked rejected, "
            f"{base_adjust_stats['paired_dump_pose_copied']} paired dumps copied, "
            f"{base_adjust_stats['border_yaw_aligned']} border-yaw aligned)."
        )
        for adj in base_adjust_stats["adjustments"]:
            before = adj["from"]
            after = adj["to"]
            before_text = "missing"
            if before is not None:
                before_text = f"[{before[0]:.3f}, {before[1]:.3f}]"
            paired_text = ""
            if adj.get("paired_with") is not None:
                paired_text = f" paired_with={adj['paired_with']}"
            yaw_text = " border_yaw" if adj.get("border_yaw_aligned") else ""
            print(
                "  base adjusted "
                f"waypoint={adj['waypoint_index']} step={adj['step']} "
                f"agent={adj['agent_index']} type={adj['agent_type']}"
                f"{paired_text}{yaw_text}: "
                f"{before_text} -> "
                f"[{after[0]:.3f}, {after[1]:.3f}] "
                f"(delta={adj['movement_tiles']:.3f} tiles, {adj['movement_m']:.3f} m)"
            )
        for skipped in base_adjust_stats["skipped_adjustments"]:
            before = skipped.get("from")
            before_text = "missing"
            if before is not None:
                before_text = f"[{before[0]:.3f}, {before[1]:.3f}]"
            trial = skipped.get("best_trial")
            if trial is None:
                closest_tiles = skipped.get("closest_tiles")
                furthest_tiles = skipped.get("furthest_tiles")
                closest_m = skipped.get("closest_m")
                furthest_m = skipped.get("furthest_m")
                trial_text = ""
            else:
                closest_tiles = trial.get("closest_tiles")
                furthest_tiles = trial.get("furthest_tiles")
                closest_m = trial.get("closest_m")
                furthest_m = trial.get("furthest_m")
                trial_text = f" best_fraction={trial['fraction']:.2f}"

            range_text = ""
            if closest_tiles is not None and furthest_tiles is not None:
                range_text = (
                    f" closest={closest_tiles:.3f} tiles/{closest_m:.3f} m "
                    f"furthest={furthest_tiles:.3f} tiles/{furthest_m:.3f} m"
                )
            allowed_text = ""
            if skipped.get("min_allowed_tiles") is not None and skipped.get("max_allowed_tiles") is not None:
                allowed_text = (
                    f" allowed=[{skipped['min_allowed_tiles']:.3f}, "
                    f"{skipped['max_allowed_tiles']:.3f}] tiles/"
                    f"[{skipped['min_allowed_m']:.3f}, {skipped['max_allowed_m']:.3f}] m"
                )

            print(
                "  base not adjusted "
                f"waypoint={skipped['waypoint_index']} step={skipped['step']} "
                f"reason={skipped['reason']}: {before_text}"
                f"{trial_text}{range_text}{allowed_text}"
            )

    # Save plan
    output_path = _timestamped_output_path(Path(args.output_path).expanduser())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[main] Saving PKL plan", flush=True)
    with open(output_path, 'wb') as f:
        pickle.dump(plan, f)

    print(f"Plan extracted and saved to {output_path} (PKL)")
    
    # Always save a JSON representation next to the PKL with the same base name.
    if output_path.suffix:
        json_path = output_path.with_suffix('.json')
    else:
        json_path = output_path.parent / f"{output_path.name}.json"

    plan_json = _plan_to_schema_v2(plan, Path(args.map_path))
    # Use compact array formatting (matching foundation-plan.json)
    json_str = json.dumps(plan_json, indent=2)

    # Compact arrays while preserving dictionary structure
    # Step 1: Compact arrays (content between [ and ])
    # Match multi-line arrays and compact them
    def compact_arrays(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                # Find matching closing bracket
                depth = 1
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == '[':
                        depth += 1
                    elif text[j] == ']':
                        depth -= 1
                    j += 1
                # Extract array content
                array_content = text[i+1:j-1]
                # Only compact if it spans multiple lines
                if '\n' in array_content:
                    # Compact: remove all whitespace
                    compacted = re.sub(r'\s+', '', array_content)
                    result.append('[' + compacted + ']')
                else:
                    result.append(text[i:j])
                i = j
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    json_str = compact_arrays(json_str)

    # Step 2: Ensure dictionary keys are on new lines with proper indentation
    # Fix any dictionary keys that got compacted - each key should be on its own line
    # Pattern: match comma followed by quoted key (any value type, not just objects)
    # We need to handle: ,"key":value (where value can be number, bool, string, array, object)
    def fix_dict_keys(text):
        # First, ensure closing braces are on their own line
        text = re.sub(r'\},\s*"', r'},\n      "', text)
        # Then, ensure each key after a comma is on its own line
        # Match: ,"key": followed by any value (not just {)
        # This handles: ,"key":value where value is number, bool, string, array, or object
        text = re.sub(r',\s*"([^"]+)":\s*', r',\n      "\1": ', text)
        return text

    json_str = fix_dict_keys(json_str)

    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(f"Also saved JSON representation to {json_path}")

    # Optional: render plan GIF from the extracted waypoints
    if args.render_plan_gif:
        gif_path = output_path.with_suffix(".gif") if output_path.suffix else Path(f"{output_path}.gif")
        _render_plan_gif(plan, Path(args.map_path), gif_path)
        print(f"Also saved plan GIF to {gif_path}")

    # Optional: render rollout GIF from Terra renderer
    if args.render_rollout_gif:
        rollout_gif_path = output_path.with_name(output_path.stem + "_rollout.gif")
        env.terra_env.rendering_engine.create_gif(str(rollout_gif_path))
        print(f"Also saved rollout GIF to {rollout_gif_path}")
    
    print(f"Total DO actions (filtered): {len(plan)}")


if __name__ == "__main__":
    main()
