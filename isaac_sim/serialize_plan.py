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

    # Serialize to JSON with compact arrays (matching foundation-plan.json format)
    print(f"Serializing to JSON...")
    plan_json = {'waypoints': [_normalize_plan_entry(entry) for entry in plan]}
    
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
    print(f"Total waypoints: {len(plan)}")


if __name__ == "__main__":
    main()

