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
from utils.utils_ppo import obs_to_model_input, wrap_action
from tensorflow_probability.substrates import jax as tfp
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
    print(f"Using seed={seed}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, 1)  # Just one environment
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (1, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    def _get_current_action_class_and_do(timestep):
        """
        Determine the currently acting agent's action class from the *state*
        (not env.batch_cfg), so this works for mixed / overridden action types.
        """
        # In this codebase, the currently acting agent is always in slot 0.
        a = timestep.state.agent.agent_states[0]
        # Avoid JAX scalar-conversion pitfalls (action_type may be shaped like (1,) or (1,1)).
        action_type_val = int(np.array(a.action_type).reshape(-1)[0])
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
        v, logits_pi = model.apply(model_params, obs_model)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        action = pi.sample(seed=rng_act)

        # Determine current action class from state (tracked vs wheeled)
        action_cls, do_action = _get_current_action_class_and_do(timestep)

        # Check if DO action and record state BEFORE executing the action
        if action[0] == do_action:
            print(f"DO action at step {t_counter}")
            action_map_before = jnp.squeeze(timestep.observation["action_map"]).copy()
            traversability_mask = jnp.squeeze(timestep.observation["traversability_mask"]).copy()
            # Multi-agent: "agent_states" has shape [B, MAX_AGENTS, feat],
            # with the currently acting agent always in slot 0.
            agent_state_before = timestep.observation["agent_states"][0, 0].copy()
            loaded_before = jnp.bool_(agent_state_before[5])
            agent_type_before = jnp.int32(agent_state_before[6])
            # Derive a stable "agent index" by cycling over the number of agents.
            # num_agents comes from the observation; we assume agents act in
            # round-robin order across timesteps.
            num_agents_obs = timestep.observation["num_agents"]
            # num_agents may be a scalar or shape [B]; convert to Python int.
            num_agents = int(np.array(num_agents_obs).reshape(-1)[0])
            agent_index = t_counter % max(num_agents, 1)

            # Update previous actions
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)

            # Take step in environment
            rng_step = jax.random.split(rng_step, 1)
            timestep = env.step(
                timestep, wrap_action(action, action_cls), rng_step
            )

            # Get state AFTER executing the action
            action_map_after = jnp.squeeze(timestep.observation["action_map"]).copy()
            agent_state_after = timestep.observation["agent_states"][0, 0].copy()
            loaded_after = jnp.bool_(agent_state_after[5])

            changed_tiles = action_map_before != action_map_after
            terrain_modification_mask = changed_tiles.astype(jnp.bool_)
            dug_mask = (action_map_after < 0)
            dump_mask = (action_map_after > 0)

            if loaded_before != loaded_after:
                if not loaded_before and loaded_after:
                    print(f"  Digging detected: {jnp.sum(changed_tiles)} tiles modified")
                elif loaded_before and not loaded_after:
                    print(f"  Dumping detected: {jnp.sum(changed_tiles)} tiles modified")
            else:
                # Case 3: loaded state did not change, but still check for modifications
                print(f"  Loaded state unchanged ({loaded_before}), {jnp.sum(changed_tiles)} tiles modified")

            # Order keys to match foundation-plan.json format
            # Convert numeric values to floats to match foundation-plan.json
            plan_entry = {
                'step': t_counter,
                'traversability_mask': traversability_mask,
                'terrain_modification_mask': terrain_modification_mask,
                'dug_mask': dug_mask.copy(),
                'dump_mask': dump_mask.copy(),
                'agent_state': {
                    'pos_base': [float(agent_state_before[0]), float(agent_state_before[1])],
                    'angle_base': float(agent_state_before[2]),
                    'angle_cabin': float(agent_state_before[3]),
                    'wheel_angle': float(agent_state_before[4]),
                },
                'loaded_state_change': {
                    'before': loaded_before,
                    'after': loaded_after,
                },
                # Additional fields (not in foundation-plan.json but kept for compatibility)
                'agent_type': agent_type_before,
                'agent_index': agent_index,
            }
            plan.append(plan_entry)
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
            if "interaction_mask" in obs1:
                obs1["interaction_mask"] = jnp.zeros_like(obs1["interaction_mask"])
            env.terra_env.render_obs_pygame(obs1, generate_gif=True)

        # Single debug line per step: which agent acted and what action was taken
        agent_state_curr = timestep.observation["agent_states"][0, 0]
        agent_type_curr = int(agent_state_curr[6])
        num_agents_obs = timestep.observation["num_agents"]
        num_agents_curr = int(np.array(num_agents_obs).reshape(-1)[0])
        agent_index_curr = (t_counter - 1) % max(num_agents_curr, 1)
        print(
            f"Step {t_counter}: Action={int(action[0])}, "
            f"agent_type={agent_type_curr}, agent_index={agent_index_curr}, "
            f"num_agents={num_agents_curr}"
        )

        # Check if done
        if jnp.all(timestep.info["task_done"]).item() or t_counter == max_frames:
            break

    return plan


def main():
    parser = argparse.ArgumentParser(description="Extract plan from policy")
    parser.add_argument(
        "-policy",
        "--policy_path",
        type=str,
        required=True,
        help="Path to the policy .pkl file"
    )
    # By default, use a map located in a local "test_map" subfolder next to this script.
    default_map_path = str((Path(__file__).parent / "test_map2").resolve())
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

    # Load policy
    log = load_pkl_object(args.policy_path)
    config = log["train_config"]
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
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(lambda x: x[0][None, ...], env_cfgs)
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

    # Save plan
    output_path = Path(args.output_path)
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
    
    print(f"Total DO actions: {len(plan)}")


if __name__ == "__main__":
    main()
