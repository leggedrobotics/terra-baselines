#!/usr/bin/env python3
"""
Standalone script to serialize a plan.pkl file to JSON format.
Uses the same serialization logic as extract_map.py.
"""

import json
import numpy as np
import jax
import argparse
import pickle
import math
from pathlib import Path
import jax.numpy as jnp


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


def _as_py_bool(value):
    try:
        return bool(np.asarray(value).reshape(-1)[0])
    except Exception:
        return bool(value)


def _mask_to_int_list(value):
    return np.asarray(value).astype(np.int8).tolist()


def _wrap_to_pi(angle_rad):
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_bucket_to_rad(value, n_bins=12):
    bucket = float(_to_serializable(value))
    return _wrap_to_pi(bucket * 2.0 * math.pi / float(n_bins))


def _terra_bucket_to_plan_yaw_rad(value, n_bins=12):
    # Terra bucket 0 points along +plan_y; schema-v2 yaw 0 is +plan_x.
    return _wrap_to_pi(_angle_bucket_to_rad(value, n_bins) + math.pi / 2.0)


def _wheel_bucket_to_rad(value, wheel_step_deg=20.0):
    bucket = float(_to_serializable(value))
    return _wrap_to_pi(math.radians(bucket * wheel_step_deg))


def _is_load_transition(entry):
    loaded_change = entry.get("loaded_state_change", {}) or {}
    before = _as_py_bool(loaded_change.get("before", False))
    after = _as_py_bool(loaded_change.get("after", False))
    return (not before) and after


def _is_unload_transition(entry):
    loaded_change = entry.get("loaded_state_change", {}) or {}
    before = _as_py_bool(loaded_change.get("before", False))
    after = _as_py_bool(loaded_change.get("after", False))
    return before and (not after)


def _workspace_type_for_pair(dig_entry):
    if dig_entry.get("dig_type") == "lift_dug_dirt":
        return "collect_dumped_soil"
    return "excavate"


def _load_plan_alignment(map_path: Path | None):
    alignment = {
        "meters_per_tile": 0.1,
        "origin_map_xy_m": [0.0, 0.0],
        "yaw_map_from_plan_rad": 0.0,
    }
    if map_path is None:
        return alignment

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


def _to_schema_v2_waypoint(entry, workspace_type):
    agent_state = entry.get("agent_state", {}) or {}
    loaded_change = entry.get("loaded_state_change", {}) or {}
    pos_base = agent_state.get("pos_base", [0.0, 0.0])

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
            "angle_base_rad": _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_base", 0.0)),
            "angle_cabin_rad": _terra_bucket_to_plan_yaw_rad(agent_state.get("angle_cabin", 0.0)),
            "wheel_angle_rad": _wheel_bucket_to_rad(agent_state.get("wheel_angle", 0.0)),
            "loaded": _as_py_bool(loaded_change.get("after", False)),
        },
    }


def _schema_v2_waypoints(plan):
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


def _plan_to_schema_v2(plan, map_path: Path | None):
    return {
        "schema_version": 2,
        "source_map_frame_id": "map",
        "alignment": _load_plan_alignment(map_path),
        "waypoints": _schema_v2_waypoints(plan),
    }


def _default_map_path(input_path: Path) -> Path | None:
    candidates = [
        input_path.parent,
        input_path.parent / "map_13",
        Path("map_13"),
    ]
    for candidate in candidates:
        if (candidate / "metadata" / "terra_metadata.yaml").exists():
            return candidate
    return None


def _normalize_plan_entry(entry):
    """Normalize plan entry to match foundation-plan.json format:
    - Convert integers to floats for agent_state numeric values
    - Reorder keys to match foundation-plan.json order
    """
    # Convert to serializable first
    entry = _to_serializable(entry)
    
    # Convert agent_state numeric values to floats
    if 'agent_state' in entry:
        agent_state = entry['agent_state']
        if 'pos_base' in agent_state:
            pos_base = agent_state['pos_base']
            if isinstance(pos_base, (list, tuple)):
                agent_state['pos_base'] = [float(x) for x in pos_base]
            elif isinstance(pos_base, (int, np.integer)):
                # Handle case where pos_base might be a single value
                agent_state['pos_base'] = [float(pos_base)]
        
        for key in ['angle_base', 'angle_cabin', 'wheel_angle']:
            if key in agent_state and isinstance(agent_state[key], (int, np.integer)):
                agent_state[key] = float(agent_state[key])
    
    # Reorder keys to match foundation-plan.json: step, traversability_mask, terrain_modification_mask,
    # dug_mask, dump_mask, agent_state, loaded_state_change, then any extra fields
    foundation_order = ['step', 'traversability_mask', 'terrain_modification_mask', 
                       'dug_mask', 'dump_mask', 'agent_state', 'loaded_state_change']
    
    # Get all keys
    all_keys = list(entry.keys())
    # Separate foundation keys and extra keys
    foundation_keys = [k for k in foundation_order if k in entry]
    extra_keys = [k for k in all_keys if k not in foundation_order]
    
    # Create new entry with correct order
    normalized_entry = {}
    for key in foundation_keys:
        normalized_entry[key] = entry[key]
    for key in extra_keys:
        normalized_entry[key] = entry[key]
    
    return normalized_entry


def main():
    parser = argparse.ArgumentParser(description="Serialize plan.pkl to JSON")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the plan.pkl file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: same directory as input with .json extension)"
    )
    parser.add_argument(
        "-m",
        "--map_path",
        type=str,
        default=None,
        help="Map directory for schema-v2 alignment metadata (default: inferred from input location)"
    )

    args = parser.parse_args()

    # Load plan from pickle
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Plan file not found: {input_path}")

    print(f"Loading plan from {input_path}...")
    with open(input_path, 'rb') as f:
        plan = pickle.load(f)

    print(f"Loaded {len(plan)} waypoints")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Save in same directory as input with .json extension
        if input_path.suffix:
            output_path = input_path.with_suffix('.json')
        else:
            output_path = input_path.parent / f"{input_path.name}.json"

    map_path = Path(args.map_path) if args.map_path else _default_map_path(input_path)
    if map_path is None:
        print("No map metadata found; using default schema-v2 alignment.")
    else:
        print(f"Using map alignment from {map_path}")

    # Serialize to schema-v2 JSON with compact arrays.
    print(f"Serializing to schema-v2 JSON...")
    plan_json = _plan_to_schema_v2(plan, map_path)
    
    # Use json.dumps with indent, then compact arrays using regex
    import re
    json_str = json.dumps(plan_json, indent=2)
    
    # Compact arrays while preserving dictionary structure
    # Step 1: Compact arrays (content between [ and ])
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
    def fix_dict_keys(text):
        # First, ensure closing braces are on their own line
        text = re.sub(r'\},\s*"', r'},\n      "', text)
        # Then, ensure each key after a comma is on its own line
        # Match: ,"key": followed by any value (not just {)
        # This handles: ,"key":value where value is number, bool, string, array, or object
        text = re.sub(r',\s*"([^"]+)":\s*', r',\n      "\1": ', text)
        return text
    
    json_str = fix_dict_keys(json_str)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    print(f"Plan serialized and saved to {output_path}")
    print(f"Raw waypoints: {len(plan)}")
    print(f"Schema-v2 waypoints: {len(plan_json['waypoints'])}")


if __name__ == "__main__":
    main()
