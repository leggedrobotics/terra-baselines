"""
Path visualization script for mixed agent training.
Shows the paths taken by each agent during an episode on a single map.
Uses different colors for different agent types and tracks movement statistics.
"""

import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.state import State
from train import TrainConfig
from terra.config import EnvConfig
import sys
from train_mixed import MixedAgentTrainConfig
from tensorflow_probability.substrates import jax as tfp
import pygame
from PIL import Image
sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig

def rollout_episode_with_paths(
    env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, seed
):
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    t_counter = 0
    reward_seq = []
    obs_seq = []
    state_seq = []
    
    # Path tracking for each agent
    agent_paths = {}  # {agent_idx: [(x, y), ...]}
    agent_distances = {}  # {agent_idx: total_distance}
    agent_do_actions = {}  # {agent_idx: count_of_do_actions}
    agent_types = {}  # {agent_idx: agent_type}
    active_agent_history = []  # Track which agent was active at each step
    
    # Initialize tracking for all agents
    for i in range(4):  # Max 4 agents
        agent_paths[i] = []
        agent_distances[i] = 0.0
        agent_do_actions[i] = 0
        agent_types[i] = None
    
    # Get agent types from checkpoint config (more reliable than observation)
    checkpoint_agent_types = env_cfgs.agent_types
    print(f"Raw agent types from checkpoint: {checkpoint_agent_types}")
    
    # Normalize agent types using the same logic as eval_mixed.py
    def _normalize_types_for_print(x):
        try:
            # Handle tuple of arrays (mixed agent types case)
            if isinstance(x, tuple):
                # If it's a tuple of arrays, preserve the order and count of each agent
                if len(x) > 0 and hasattr(x[0], 'tolist'):
                    import numpy as _np
                    # Take the first element from each array to get the agent type for each agent
                    result = []
                    for arr in x:
                        # Take the first element (all elements in the array should be the same)
                        agent_type = int(_np.array(arr[0]).item())
                        result.append(agent_type)
                    return result
                # Unwrap one level if checkpoint stores as a 1-tuple
                elif len(x) == 1:
                    x = x[0]
                else:
                    return list(x)
            # If batched across envs/devices, take the first slice
            if hasattr(x, "ndim") and x.ndim >= 2:
                x = x[0]
            if hasattr(x, "ndim") and x.ndim == 0:
                import numpy as _np
                return [int(_np.array(x).item())]
            if hasattr(x, "tolist"):
                return list(x.tolist())
            return list(x) if isinstance(x, (list, tuple)) else x
        except Exception:
            return x
    
    normalized_agent_types = _normalize_types_for_print(checkpoint_agent_types)
    print(f"Normalized agent types: {normalized_agent_types}")
    
    # Get tile size from environment config
    tile_size = env_cfgs.tile_size
    if hasattr(tile_size, '__len__') and len(tile_size) > 0:
        tile_size = float(tile_size[0])
    else:
        tile_size = float(tile_size)
    print(f"Tile size: {tile_size}")
    
    # Add initial observation and state (after reset)
    obs_seq.append(timestep.observation)
    state_seq.append(timestep.state)
    
    # Record initial positions
    initial_obs = timestep.observation
    initial_agent_states = initial_obs["agent_states"]  # [B, MAX_AGENTS, feat]
    for i in range(4):
        if i < initial_agent_states.shape[1]:
            agent_state = initial_agent_states[0, i]  # Get agent state for env 0
            if agent_state is not None and len(agent_state) > 6:
                pos_x, pos_y = float(agent_state[0]), float(agent_state[1])
                agent_paths[i].append((pos_x, pos_y))
                agent_types[i] = int(agent_state[6])
    
    prev_positions = {}
    for i in range(4):
        if i < initial_agent_states.shape[1]:
            agent_state = initial_agent_states[0, i]
            if agent_state is not None and len(agent_state) > 6:
                pos_x, pos_y = float(agent_state[0]), float(agent_state[1])
                prev_positions[i] = (pos_x, pos_y)
    
    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            obs = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, logits_pi = model.apply(model_params, obs)
            pi = tfp.distributions.Categorical(logits=logits_pi)
            action = pi.sample(seed=rng_act)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)
        else:
            raise RuntimeError("Model is None!")
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )
        
        t_counter += 1
        
        # COLLECT OBSERVATION AFTER STEP (includes soil mechanics changes)
        obs_seq.append(timestep.observation)
        state_seq.append(timestep.state)
        
        # Track agent movements and actions
        obs = timestep.observation
        current_state = timestep.state
        
        # Get current active agent index from state
        current_agent_idx = int(current_state.agent.current_agent[0])
        
        # Get agent states from observation
        agent_states_batch = obs["agent_states"]  # [B, MAX_AGENTS, feat]
        
        # Debug: print current agent and action every 10 steps
        if t_counter % 10 == 0:
            action_value = int(action[0])
            current_agent_state = agent_states_batch[0, current_agent_idx]
            active_type = int(current_agent_state[6])
            print(f"Step {t_counter}: Active agent {current_agent_idx}, type {active_type}, Action {action_value}")
            
            # Debug: print all agent types
            if t_counter == 10:  # Only print once
                for j in range(4):
                    if j < agent_states_batch.shape[1]:
                        agent_state = agent_states_batch[0, j]
                        if agent_state is not None and len(agent_state) > 6:
                            agent_type = int(agent_state[6])
                            print(f"  Agent {j}: type {agent_type}")
        
        # Record which agent was active at this step
        active_agent_history.append(current_agent_idx)
        
        # Get agent_active mask to determine which agents are in the scene
        agent_active_mask = obs["agent_active"][0]  # [4] - which agents are active
        num_active_agents = int(np.sum(agent_active_mask))
        
        # Debug: print agent_active mask every 10 steps
        if t_counter % 10 == 0:
            print(f"Step {t_counter}: agent_active_mask = {agent_active_mask}, num_active = {num_active_agents}")
        
        # The active agent is always at index 0 in the observation
        # Use modulo to determine which agent is currently active based on step number
        current_agent_idx = t_counter % num_active_agents
        
        # Get the agent state for the currently active agent (always at index 0)
        agent_state = agent_states_batch[0, 0]  # Get agent state for env 0, index 0 (active agent)
        if agent_state is not None and len(agent_state) > 6:  # Check if valid state
            current_pos = (float(agent_state[0]), float(agent_state[1]))  # pos_x, pos_y
            agent_type = int(agent_state[6])  # agent_type
            
            # Use agent type from normalized checkpoint config (more reliable)
            if agent_types[current_agent_idx] is None and current_agent_idx < len(normalized_agent_types):
                agent_types[current_agent_idx] = normalized_agent_types[current_agent_idx]
                print(f"Set agent {current_agent_idx} type to {agent_types[current_agent_idx]} from checkpoint")
            
            # Add current position to path for this agent
            agent_paths[current_agent_idx].append(current_pos)
            
            # Check if agent moved (position changed) for distance calculation
            if current_agent_idx in prev_positions:
                prev_pos = prev_positions[current_agent_idx]
                if prev_pos != current_pos:
                    # Calculate distance moved
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    distance = np.sqrt(dx*dx + dy*dy) * tile_size
                    agent_distances[current_agent_idx] += distance
            
            # Update previous position
            prev_positions[current_agent_idx] = current_pos
            
            # Check for DO actions for the currently active agent
            action_value = int(action[0])
            if action_value == 4 or action_value == 8:  # DO action
                agent_do_actions[current_agent_idx] += 1
            
            # Debug: print tracking info every 10 steps
            if t_counter % 10 == 0:
                print(f"Step {t_counter}: Tracking agent {current_agent_idx}, pos {current_pos}, type {agent_type}")
        
        reward_seq.append(timestep.reward)
        
        if bool(jnp.all(timestep.done)) or t_counter == max_frames:
            break
    
    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return obs_seq, np.cumsum(reward_seq), state_seq, agent_paths, agent_distances, agent_do_actions, agent_types, active_agent_history

def create_path_overlay_gif(obs_seq, state_seq, agent_paths, agent_distances, agent_do_actions, agent_types, 
                           active_agent_history, num_agents, env, out_path):
    """Create a GIF visualization showing progress over time with agent paths."""
    
    # Agent type colors (RGB values for pygame) - more distinct colors
    agent_colors = {
        0: (255, 0, 0),      # Red for Excavator
        1: (0, 255, 0),      # Green for Truck  
        2: (0, 0, 255),      # Blue for Skidsteer
        3: (255, 255, 0),    # Yellow for any other type
        4: (255, 0, 255),    # Magenta fallback
        5: (0, 255, 255),    # Cyan fallback
        6: (255, 128, 0),    # Orange fallback
        7: (128, 0, 255)     # Purple fallback
    }
    
    agent_names = {
        0: 'Excavator',
        1: 'Truck',
        2: 'Skidsteer'
    }
    
    # Scale coordinates to match rendering scale
    tile_size_rendering = 3  # From env.py: MAP_TILES // baseline_map_size = 192//64 = 3
    
    print("Rendering frames with path overlays...")
    
    # Debug: check if rendering engine is available
    rendering_engine = env.terra_env.rendering_engine
    print(f"Rendering engine available: {rendering_engine is not None}")
    if rendering_engine is None:
        print("ERROR: No rendering engine available! Check environment setup.")
        return
    
    # Render each frame with path overlays (like visualize_mixed.py)
    for i, o in enumerate(tqdm(obs_seq, desc="Rendering frames")):
        # Try using state action_map instead of observation action_map
        if i < len(state_seq):
            # Create modified observation with raw state action_map
            modified_obs = dict(o)
            modified_obs['action_map'] = state_seq[i].world.action_map.map
            # Hide interaction cones by zeroing the correct key used by renderer
            if 'interaction_mask' in modified_obs:
                modified_obs['interaction_mask'] = jnp.zeros_like(modified_obs['interaction_mask'])
            
            # Debug: print observation keys to see what we have
            if i == 0:
                print("Available observation keys:", list(modified_obs.keys()))
                for key, value in modified_obs.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            
            # Render the frame FIRST
            env.terra_env.render_obs_pygame(modified_obs, generate_gif=False)  # Don't capture frame yet
            
            # Get the rendered surface from the rendering engine AFTER rendering
            rendering_engine = env.terra_env.rendering_engine
            if rendering_engine is None:
                print("Error: No rendering engine available")
                return
            
            # Get the screen surface (this is what gets captured for the GIF)
            screen_surface = pygame.display.get_surface()
            
            # NOW draw paths on top of the rendered frame
            # Draw path points at the actual agent positions as they're rendered each frame
            
            # Get current agent positions from the observation
            current_obs = obs_seq[i]
            agent_states = current_obs["agent_states"]  # [B, MAX_AGENTS, feat]
            
            # Get which agent is currently active from the active_agent_history
            current_active_agent = active_agent_history[i] if i < len(active_agent_history) else 0
            
            # Current position dots disabled - not needed
            
            # Now draw path lines connecting points from the same agent
            # Draw paths for all agents that have been tracked
            for agent_idx in range(4):  # Check all possible agent slots
                if agent_paths[agent_idx] and len(agent_paths[agent_idx]) > 1:
                    agent_type = agent_types[agent_idx]
                    color = agent_colors.get(agent_type, agent_colors.get(agent_idx % len(agent_colors), (128, 128, 128)))
                    
                    # Debug: print path info for first few frames
                    if i < 3:
                        print(f"Frame {i}, Agent {agent_idx}: path length {len(agent_paths[agent_idx])}, type {agent_type}, color {color}")
                    
                    # Convert all path points to screen coordinates
                    screen_points = []
                    for pos in agent_paths[agent_idx]:
                        px_center, py_center = pos
                        center_x = px_center * tile_size_rendering
                        center_y = py_center * tile_size_rendering
                        
                        # Apply display offsets
                        border_px = 4 * tile_size_rendering
                        # NOTE: The coordinate system is swapped in the agent rendering!
                        # In agent.py line 98: agent_body.append((center_y + rotated_x, center_x + rotated_y))
                        # So we need to swap x and y coordinates
                        screen_x = center_y + border_px
                        screen_y = center_x + border_px
                        
                        screen_points.append((int(screen_x), int(screen_y)))
                    
                    # Only draw path up to current step (progressive path)
                    # Calculate how many points this agent should have shown by frame i
                    # Each agent gets points every num_active_agents steps
                    agent_active_mask = current_obs["agent_active"][0]
                    num_active_agents = int(np.sum(agent_active_mask))
                    
                    # Calculate how many points this agent should have by frame i
                    # Each agent gets points every num_active_agents steps
                    # Agent 0 gets points at steps 0, 2, 4, 6, ... (every 2nd step)
                    # Agent 1 gets points at steps 1, 3, 5, 7, ... (every 2nd step)
                    # So agent_idx gets points at steps: agent_idx, agent_idx + num_active_agents, agent_idx + 2*num_active_agents, ...
                    max_points_for_agent = (i // num_active_agents) + (1 if i % num_active_agents > agent_idx else 0)
                    max_points = min(len(screen_points), max_points_for_agent)
                    
                    # Draw path lines connecting consecutive points from the same agent
                    for j in range(max_points - 1):
                        start_pos = screen_points[j]
                        end_pos = screen_points[j + 1]
                        pygame.draw.line(screen_surface, color, start_pos, end_pos, 2)
                    
                    # Draw start marker (always visible)
                    if len(screen_points) > 0:
                        start_pos = screen_points[0]
                        pygame.draw.circle(screen_surface, color, start_pos, 5)
                    
                    # Don't draw current position marker here since we're already drawing current position dots above
            
            # Debug: print how many paths were drawn
            if i < 3:
                paths_drawn = sum(1 for agent_idx in range(4) if agent_paths[agent_idx] and len(agent_paths[agent_idx]) > 1)
                print(f"Frame {i}: Drew paths for {paths_drawn} agents")
            
            # NOW capture the frame with paths drawn
            frame = pygame.surfarray.array3d(screen_surface)
            rendering_engine.frames.append(frame.swapaxes(0, 1))
        else:
            # Fallback for frames beyond state_seq
            obs_no_interact = dict(o)
            if 'interaction_mask' in obs_no_interact:
                obs_no_interact['interaction_mask'] = jnp.zeros_like(obs_no_interact['interaction_mask'])
            env.terra_env.render_obs_pygame(obs_no_interact, generate_gif=False)
            
            # Capture frame for fallback case too
            screen_surface = pygame.display.get_surface()
            frame = pygame.surfarray.array3d(screen_surface)
            rendering_engine.frames.append(frame.swapaxes(0, 1))
    
    # Create the GIF
    env.terra_env.rendering_engine.create_gif(out_path)
    print(f"Path visualization GIF saved to {out_path}")
    
    # Print summary statistics
    print("\n=== Agent Path Summary ===")
    print(f"Active agents: {num_agents}")
    for agent_idx in range(4):  # Check all possible agent slots
        if agent_paths[agent_idx] and agent_types[agent_idx] is not None:
            agent_type = agent_types[agent_idx]
            agent_name = agent_names.get(agent_type, f'Agent {agent_type}')
            distance = agent_distances[agent_idx]
            do_count = agent_do_actions[agent_idx]
            path_length = len(agent_paths[agent_idx])
            
            print(f"{agent_name} (Agent {agent_idx}):")
            print(f"  - Total distance moved: {distance:.2f} meters")
            print(f"  - Number of DO actions: {do_count}")
            print(f"  - Path points: {path_length}")
            print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="mixed_agents_checkpoint.pkl",
        help="Path to mixed agent trained checkpoint.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=350,
        help="Number of steps.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./agent_paths.gif",
        help="Output path for the path visualization GIF.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    args, _ = parser.parse_known_args()

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = 1  # Single environment
    config.num_devices = 1

    env_cfgs = log["env_config"]
    
    # Debug: print agent types from checkpoint
    print(f"Agent types from checkpoint: {env_cfgs.agent_types}")
    
    # Custom handling for different field types
    def replicate_field(x):
        if x is None:
            return None
        # Handle tuples generically (e.g., agent_types of length 1–4)
        if isinstance(x, tuple):
            return jnp.array(x)[None, ...].repeat(1, 0)
        # Handle scalars (int, float, bool) - just replicate the value
        elif isinstance(x, (int, float, bool)):
            return jnp.array([x])
        # Handle arrays - take first element and replicate
        else:
            return x[0][None, ...].repeat(1, 0)
    
    env_cfgs = jax.tree_map(replicate_field, env_cfgs)
    
    # Create batch config for the environment
    from terra.config import BatchConfig
    batch_cfg = BatchConfig()
    
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=True,  # Enable rendering for GIF creation
        n_envs_x_rendering=1,
        n_envs_y_rendering=1,
        display=False,
        shuffle_maps=False,
    )

    model = load_neural_network(config, env)
    model_params = log["model"]
    
    obs_seq, cum_rewards, state_seq, agent_paths, agent_distances, agent_do_actions, agent_types, active_agent_history = rollout_episode_with_paths(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        seed=args.seed,
    )

    # Get number of active agents from the first observation
    first_obs = obs_seq[0]
    num_agents = int(first_obs["num_agents"][0])  # Number of active agents
    
    # Create path visualization GIF using environment rendering
    create_path_overlay_gif(obs_seq, state_seq, agent_paths, agent_distances, agent_do_actions, agent_types,
                            active_agent_history, num_agents, env, args.out_path)
