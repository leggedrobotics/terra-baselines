import json
import numpy as np
import jax
import argparse
import pickle
import sys
import re
from pathlib import Path
from typing import Any

# Add the parent directory to the path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import TrackedAction, WheeledAction, TrackedActionType, WheeledActionType
import jax.numpy as jnp
from utils.utils_ppo import action_type_from_policy_action, obs_to_model_input, policy, wrap_action
from train import TrainConfig  # needed for unpickling checkpoints
from train_mixed import MixedAgentTrainConfig
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

    def render_frame(entry):
        dug_mask = np.asarray(entry.get("dug_mask", np.zeros((h, w), dtype=bool))).astype(bool)
        dump_mask = np.asarray(entry.get("dump_mask", np.zeros((h, w), dtype=bool))).astype(bool)
        pos = entry.get("agent_state", {}).get("pos_base", None)
        agent_xy = (float(pos[0]), float(pos[1])) if pos is not None else None

        rgb = np.full((h, w, 3), 245, dtype=np.uint8)
        rgb[occupancy] = (20, 20, 20)
        rgb[target < 0] = (255, 220, 220)
        rgb[target > 0] = (220, 255, 220)
        rgb[(~occupancy) & (~dumpability)] = (235, 235, 235)
        rgb[dug_mask] = (70, 130, 255)
        rgb[dump_mask] = (255, 170, 60)

        img = Image.fromarray(rgb, mode="RGB").resize((w * scale, h * scale), Image.NEAREST)
        if agent_xy is not None:
            draw = ImageDraw.Draw(img)
            x, y = agent_xy
            cx = int(round(x * scale))
            cy = int(round(y * scale))
            r = max(2, scale // 2)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=2)
        return img

    frames = [render_frame(e) for e in plan] if plan else [render_frame({})]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


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

    # Plan storage
    plan = []

    t_counter = 0

    if render_rollout_gif:
        obs0 = dict(timestep.observation)
        if "interaction_mask" in obs0:
            obs0["interaction_mask"] = jnp.zeros_like(obs0["interaction_mask"])
        env.terra_env.render_obs_pygame(obs0, generate_gif=True)

    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Get action from policy
        # In the multi-agent setup, observations already have the currently acting
        # agent in slot 0 of "agent_states". obs_to_model_input handles this,
        # so we can treat the batch as size 1 and proceed as in single-agent.
        obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
        v, pi = policy(model.apply, model_params, obs_model)
        action = pi.sample(seed=rng_act)
        action_type_sample = action_type_from_policy_action(action)

        # Determine current action class from state (tracked vs wheeled)
        action_cls, do_action = _get_current_action_class_and_do(timestep, t_counter)

        # Check if DO action and record state BEFORE executing the action
        if action_type_sample[0] == do_action:
            # Compute agent index - agents act in round-robin order and can't skip
            num_agents = int(np.array(timestep.observation["num_agents"]).reshape(-1)[0])
            agent_index = t_counter % max(num_agents, 1)
            
            # Access the acting agent's state by its fixed slot index
            acting_agent_state_before = timestep.state.agent.agent_states[agent_index]
            # Use np.array().item() to handle batch dimensions in state fields
            loaded_before = bool(np.array(acting_agent_state_before.loaded).flatten()[0])
            agent_type_before = int(np.array(acting_agent_state_before.agent_type).flatten()[0])
            
            # Agent type names for display
            agent_type_names = {0: "excavator", 1: "truck", 2: "skidsteer"}
            agent_type_str = agent_type_names.get(agent_type_before, f"unknown({agent_type_before})")
            
            action_map_before = jnp.squeeze(timestep.observation["action_map"]).copy()
            traversability_mask = jnp.squeeze(timestep.observation["traversability_mask"]).copy()

            # Update previous actions
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action_type_sample)

            # Take step in environment
            rng_step = jax.random.split(rng_step, 1)
            timestep = env.step(
                timestep, wrap_action(action, action_cls), rng_step
            )

            # Get state AFTER executing the action
            # IMPORTANT: After env.step(), observation["agent_states"][0, 0] is the NEXT agent (rotated view).
            # We need to access the acting agent's state by its fixed slot index from timestep.state.
            action_map_after = jnp.squeeze(timestep.observation["action_map"]).copy()
            # Access the acting agent's state directly from the state object by slot index
            acting_agent_state_after = timestep.state.agent.agent_states[agent_index]
            loaded_after = bool(np.array(acting_agent_state_after.loaded).flatten()[0])

            changed_tiles = action_map_before != action_map_after
            terrain_modification_mask = changed_tiles.astype(jnp.bool_)
            dug_mask = (action_map_after < 0)
            dump_mask = (action_map_after > 0)

            # Determine dig_type for digging actions:
            # "lift_dug_dirt" = picking up previously dumped dirt (action_map_before > 0 in changed tiles)
            # "dig_new_soil" = digging fresh terrain from target
            dig_type = None
            action_type_label = ""
            if not loaded_before and loaded_after:
                # This is a digging action - check if we're lifting dumped dirt or digging new soil
                # Similar logic to state.py: moving_dumped_dirt = selected_tiles_sum > 0
                # where selected_tiles_sum = flattened_action_map @ dig_mask
                dumped_dirt_in_changed_tiles = jnp.sum(action_map_before[changed_tiles] > 0)
                moving_dumped_dirt = dumped_dirt_in_changed_tiles > 0
                dig_type = "lift_dug_dirt" if bool(moving_dumped_dirt) else "dig_new_soil"
                action_type_label = f" (digging: {dig_type})"
            elif loaded_before and not loaded_after:
                action_type_label = " (dumping)"
            else:
                # Case 3: loaded state did not change
                action_type_label = f" (unchanged, loaded={bool(loaded_before)})"
            
            print(f"DO action at step {t_counter} by agent {agent_index} ({agent_type_str}){action_type_label}, {jnp.sum(changed_tiles)} tiles modified")

            # Order keys to match foundation-plan.json format
            # Convert numeric values to floats to match foundation-plan.json
            plan_entry = {
                'step': t_counter,
                'traversability_mask': traversability_mask,
                'terrain_modification_mask': terrain_modification_mask,
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
            # Add dig_type only for digging actions (when we go from unloaded to loaded)
            if dig_type is not None:
                plan_entry['dig_type'] = dig_type
            plan.append(plan_entry)
        else:
            # Update previous actions
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action_type_sample)

            # Take step in environment
            rng_step = jax.random.split(rng_step, 1)
            timestep = env.step(
                timestep, wrap_action(action, action_cls), rng_step
            )

        t_counter += 1

        if render_rollout_gif and (t_counter % max(1, int(rollout_gif_every)) == 0):
            obs1 = dict(timestep.observation)
            if "interaction_mask" in obs1:
                obs1["interaction_mask"] = jnp.zeros_like(obs1["interaction_mask"])
            env.terra_env.render_obs_pygame(obs1, generate_gif=True)


        # Check if done
        if jnp.all(timestep.info["task_done"]).item() or t_counter == max_frames:
            break

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
        default=500,
        help="Maximum number of steps"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./isaac_sim/plan.pkl",
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
        "--serialize",
        action="store_true",
        help="Also save a JSON representation of the plan (default: False, only saves PKL)"
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

    args = parser.parse_args()

    log = load_pkl_object(args.policy_path)
    config = jax.tree_map(_canon_lists, log["train_config"])
    config.num_test_rollouts = 1  # Only one environment
    config.num_devices = 1

    # Disable action map clipping to see full terrain state
    print(f"Original clip_action_maps setting: {config.clip_action_maps}")
    config.clip_action_maps = False
    # Add the missing attribute that the model expects when clipping is disabled
    config.maps_net_normalization_bounds = [-100, 100]  # Reasonable range for terrain heights
    print(f"Modified clip_action_maps setting: {config.clip_action_maps}")
    print(f"Added maps_net_normalization_bounds: {config.maps_net_normalization_bounds}")

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
    env = TerraEnvBatch(
        rendering=args.render_rollout_gif,
        n_envs_x_rendering=1,
        n_envs_y_rendering=1,
        display=False,
        shuffle_maps=False,
        single_map_path=args.map_path,
    )

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

    # Save plan
    output_path = Path(args.output_path)
    print("[main] Saving PKL plan", flush=True)
    with open(output_path, 'wb') as f:
        pickle.dump(plan, f)

    print(f"Plan extracted and saved to {output_path} (PKL)")
    
    # Optionally save a JSON representation for debugging/inspection.
    if args.serialize:
        if output_path.suffix:
            json_path = output_path.with_suffix('.json')
        else:
            json_path = output_path.parent / f"{output_path.name}.json"

        plan_json = {'waypoints': [_to_serializable(entry) for entry in plan]}
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
