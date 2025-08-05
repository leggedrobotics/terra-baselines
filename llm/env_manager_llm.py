import numpy as np
import jax

import jax.numpy as jnp

from terra.env import TerraEnv

from llm.utils_llm import *
from llm.adk_llm import *

import pygame as pg

from llm.env_llm import LargeMapTerraEnv

class EnvironmentsManager:
    """
    Manages completely separate environments for large map and small maps.
    Each environment has its own timestep, configuration, and state.
    Only map data is exchanged between environments.
    """
        
    def __init__(self, seed, global_env_config, small_env_config=None, shuffle_maps=False, rendering=False, display=False, size=64):
        """
        Initialize with separate configurations for large and small environments.
        
        Args:
            seed: Random seed for reproducibility
            global_env_config: Environment configuration for the large global map
            small_env_config: Environment configuration for small maps (or None to derive from global)
            num_partitions: Number of partitions for the large map
            shuffle_maps: Whether to shuffle maps
        """
        self.rng = jax.random.PRNGKey(seed)
        self.global_env_config = global_env_config
        self.shuffle_maps = shuffle_maps
        # Create a custom small environment config if not provided
        if small_env_config is None:
            self.small_env_config = self._derive_small_environment_config()
        else:
            self.small_env_config = small_env_config

        # Overlapping partition data - will be set externally
        self.partitions = []
        self.overlap_map = {}  # Maps partition_id -> set of overlapping partition_ids
        self.overlap_regions = {}  # Cache overlap region calculations
        self.rendering = rendering
        self.display = display
        
        # Initialize the global environment (128x128) with LargeMapTerraEnv
        print("Initializing LargeMapTerraEnv for global environment...")

        self.global_env = LargeMapTerraEnv(
            rendering=rendering,
            n_envs_x_rendering=1,
            n_envs_y_rendering=1,
            display=display,
            shuffle_maps=shuffle_maps,
        )

        self.map_size_px = size  # Set map size based on provided size parameter
        print(f"Global map size: {self.map_size_px}x{self.map_size_px} pixels")
        self.small_agent_config = {
            'height': jnp.array([9], dtype=jnp.int32), 
            'width': jnp.array([5], dtype=jnp.int32)
        }
        self.big_agent_config = {
            'height': jnp.array([19], dtype=jnp.int32), 
            'width': jnp.array([11], dtype=jnp.int32)
        }
        # Initialize the small environment with regular TerraEnv (non-batched)
        print("Initializing TerraEnv for small environment...")
        self.small_env = TerraEnv.new(
            maps_size_px=64,
            rendering=False,
            n_envs_x=1,
            n_envs_y=1,
            display=False,
            agent_config_override=self.small_agent_config
        )
        
        # Store global map data
        self.global_maps = {
            'target_map': None,
            'action_map': None,
            'dumpability_mask': None,
            'dumpability_mask_init': None,
            'padding_mask': None,
            'traversability_mask': None,
            'trench_axes': None,
            'trench_type': None,
        }
        
        # Define partition scheme
        self.partitions = []
        #self._define_partitions()
        
        # Initialize global environment and extract maps
        self._initialize_global_environment()
        
        # Track which environment is currently being displayed
        self.current_display_env = "global"  # or "small"
        
        # Track small environment state
        self.small_env_timestep = None
        self.current_partition_idx = None

        # Agent size configurations
        # self.small_agent_config = {
        #     'height': jnp.array([9], dtype=jnp.int32), 
        #     'width': jnp.array([5], dtype=jnp.int32)
        # }
        # self.big_agent_config = {
        #     'height': jnp.array([19], dtype=jnp.int32), 
        #     'width': jnp.array([11], dtype=jnp.int32)
        # }
        
        #print(f"Agent configs - Small: {self.small_agent_config}, Big: {self.big_agent_config}")

    def _partitions_overlap(self, i: int, j: int) -> bool:
        """Check if two partitions overlap."""
        p1_coords = self.partitions[i]['region_coords']
        p2_coords = self.partitions[j]['region_coords']
        
        y1_start, x1_start, y1_end, x1_end = p1_coords
        y2_start, x2_start, y2_end, x2_end = p2_coords
        
        # print(f"Checking overlap between partition {i}: ({y1_start}, {x1_start}, {y1_end}, {x1_end}) and partition {j}: ({y2_start}, {x2_start}, {y2_end}, {x2_end})")
        
        # Check for overlap - rectangles overlap if they overlap in BOTH dimensions
        y_overlap = (y1_start <= y2_end) and (y2_start <= y1_end)
        x_overlap = (x1_start <= x2_end) and (x2_start <= x1_end)
        
        overlap_exists = y_overlap and x_overlap
        
        # print(f"  Y overlap: {y_overlap} (y1: {y1_start}-{y1_end}, y2: {y2_start}-{y2_end})")
        # print(f"  X overlap: {x_overlap} (x1: {x1_start}-{x1_end}, x2: {x2_start}-{x2_end})")
        # print(f"  Overall overlap: {overlap_exists}")
        
        return overlap_exists

    def _calculate_overlap_region(self, partition_i: int, partition_j: int):
            """
            Calculate the overlapping region between two partitions.
            Returns slices for global coordinates, partition i local coordinates, and partition j local coordinates.
            """
            p1_coords = self.partitions[partition_i]['region_coords']
            p2_coords = self.partitions[partition_j]['region_coords']
            
            y1_start, x1_start, y1_end, x1_end = p1_coords
            y2_start, x2_start, y2_end, x2_end = p2_coords
            
            # print(f"Calculating overlap region between partition {partition_i} and {partition_j}")
            # print(f"  Partition {partition_i}: ({y1_start}, {x1_start}) to ({y1_end}, {x1_end})")
            # print(f"  Partition {partition_j}: ({y2_start}, {x2_start}) to ({y2_end}, {x2_end})")
            
            # Find intersection in global coordinates
            overlap_y_start = max(y1_start, y2_start)
            overlap_x_start = max(x1_start, x2_start)
            overlap_y_end = min(y1_end, y2_end)
            overlap_x_end = min(x1_end, x2_end)
            
            # print(f"  Global overlap region: ({overlap_y_start}, {overlap_x_start}) to ({overlap_y_end}, {overlap_x_end})")
            
            # Check if there's actual overlap
            if overlap_y_start > overlap_y_end or overlap_x_start > overlap_x_end:
                print(f"  No actual overlap!")
                return None
            
            # Convert to local coordinates for each partition
            local_i_y_start = overlap_y_start - y1_start
            local_i_x_start = overlap_x_start - x1_start
            local_i_y_end = overlap_y_end - y1_start
            local_i_x_end = overlap_x_end - x1_start
            
            local_j_y_start = overlap_y_start - y2_start
            local_j_x_start = overlap_x_start - x2_start
            local_j_y_end = overlap_y_end - y2_start
            local_j_x_end = overlap_x_end - x2_start
            
            # print(f"  Partition {partition_i} local overlap: ({local_i_y_start}, {local_i_x_start}) to ({local_i_y_end}, {local_i_x_end})")
            # print(f"  Partition {partition_j} local overlap: ({local_j_y_start}, {local_j_x_start}) to ({local_j_y_end}, {local_j_x_end})")
            
            return {
                'global_slice': (slice(overlap_y_start, overlap_y_end + 1), 
                            slice(overlap_x_start, overlap_x_end + 1)),
                'partition_i_slice': (slice(local_i_y_start, local_i_y_end + 1), 
                                    slice(local_i_x_start, local_i_x_end + 1)),
                'partition_j_slice': (slice(local_j_y_start, local_j_y_end + 1),
                                    slice(local_j_x_start, local_j_x_end + 1)),
                'overlap_bounds': (overlap_y_start, overlap_x_start, overlap_y_end, overlap_x_end)
            }

    def set_partitions(self, partitions):
        """
        Set the partitions and compute overlap relationships.
        """
        print(f"\n=== SETTING PARTITIONS ===")
        self.partitions = partitions
        
        print(f"Partitions set:")
        for i, partition in enumerate(self.partitions):
            print(f"  Partition {i}: {partition}")
        
        # Use the fixed overlap computation
        self._compute_overlap_relationships_fixed()
        
        print(f"Set {len(self.partitions)} partitions with overlaps computed.")

    def _compute_overlap_relationships(self):
        """
        Compute which partitions overlap with each other and cache overlap regions.
        """
        print(f"\n=== COMPUTING OVERLAP RELATIONSHIPS ===")
        
        self.overlap_map = {i: set() for i in range(len(self.partitions))}
        self.overlap_regions = {}
        
        for i in range(len(self.partitions)):
            for j in range(i + 1, len(self.partitions)):
                print(f"\nChecking partitions {i} and {j}:")
                
                if self._partitions_overlap(i, j):
                    self.overlap_map[i].add(j)
                    self.overlap_map[j].add(i)
                    
                    # Cache the overlap region calculation
                    overlap_info = self._calculate_overlap_region(i, j)
                    if overlap_info is not None:
                        self.overlap_regions[(i, j)] = overlap_info
                        self.overlap_regions[(j, i)] = overlap_info  # Symmetric
                        print(f"  Stored overlap info for partitions {i} <-> {j}")
                    else:
                        print(f"  Could not calculate overlap region!")
                else:
                    print(f"  No overlap detected")
        
        # Print final overlap information
        print(f"\n=== FINAL OVERLAP RELATIONSHIPS ===")
        for i, partition in enumerate(self.partitions):
            overlaps = list(self.overlap_map[i])
            print(f"Partition {i}: region={partition['region_coords']}, overlaps with {overlaps}")
        
        print(f"Total overlap regions cached: {len(self.overlap_regions)}")

    def initialize_with_fixed_overlaps(self, partitions):
        """
        Initialize partitions with fixed overlap detection.
        """
        
        # Set partitions using the fixed method
        self.set_partitions(partitions)
        
    def step_simple(self, partition_idx: int, action, partition_states: dict):
        """
        Simple step function - just steps the environment without any synchronization.
        Synchronization happens separately.
        """
        partition_state = partition_states[partition_idx]
        current_state = partition_state['timestep'].state
        current_env_cfg = partition_state['timestep'].env_cfg
        
        # Extract required data for step
        current_target_map = current_state.world.target_map.map
        current_padding_mask = current_state.world.padding_mask.map
        current_dumpability_mask_init = current_state.world.dumpability_mask_init.map
        current_trench_axes = current_state.world.trench_axes
        current_trench_type = current_state.world.trench_type
        current_action_map = current_state.world.action_map.map

        # Step the environment
        new_timestep = self.small_env.step(
            state=current_state,
            action=action,
            target_map=current_target_map,
            padding_mask=current_padding_mask,
            trench_axes=current_trench_axes,
            trench_type=current_trench_type,
            dumpability_mask_init=current_dumpability_mask_init,
            action_map=current_action_map,
            env_cfg=current_env_cfg
        )
        
        return new_timestep

    def _create_clean_env_config(self):
        """Create a clean environment config for 64x64 maps without batch dimensions"""

    
        # If you have a reference to the original config structure, use it
        # Otherwise, create a minimal one
        try:
            # Try to create from the global config but clean it up
            base_cfg = self.small_env_config if hasattr(self, 'small_env_config') else self.global_env_config
        
            # Remove any batch dimensions by taking the first element
            def unbatch(x):
                if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] == 1:
                    return x[0]
                return x
            
            clean_cfg = jax.tree_map(unbatch, base_cfg)
            return clean_cfg
        
        except Exception as e:
            print(f"Warning: Could not clean config: {e}")
            # Return the original config and hope for the best
            return self.global_env_config
        
    def initialize_small_environment(self, partition_idx):
        """
        Initialize the small environment with map data from a specific global map partition.
        Uses TerraEnv (non-batched) for better performance and simpler interface.
        """
        if partition_idx < 0 or partition_idx >= len(self.partitions):
            raise ValueError(f"Invalid partition index: {partition_idx}")

        partition = self.partitions[partition_idx]
        region_coords = partition['region_coords']
        custom_pos = partition['start_pos']
        custom_angle = partition['start_angle']

        # Extract sub-maps from global maps (64x64)
        # sub_maps = {
        #     'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),
        #     'action_map': create_sub_task_action_map_64x64(self.global_maps['action_map'], region_coords),
        #     'dumpability_mask': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask'], region_coords),
        #     'dumpability_mask_init': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask_init'], region_coords),
        #     'padding_mask': create_sub_task_padding_mask_64x64(self.global_maps['padding_mask'], region_coords),
        #     'traversability_mask': create_sub_task_traversability_mask_64x64(self.global_maps['traversability_mask'], region_coords),
        # }
        #print("Self.map_size_px:", self.map_size_px)

        if self.map_size_px == 64:
            sub_maps = {
                'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),                              #ok
                'action_map': self.global_maps['action_map'],
                'dumpability_mask': self.global_maps['dumpability_mask'],
                'dumpability_mask_init': self.global_maps['dumpability_mask_init'],
                'padding_mask': self.global_maps['padding_mask'],
                'traversability_mask': self.global_maps['traversability_mask'],                                                             #OK, keep the full traversability mask
            }
        else:
            sub_maps = {
                'target_map': create_sub_task_target_map_64x64_fixed(self.global_maps['target_map'], region_coords),
                'action_map': create_sub_task_action_map_64x64_fixed(self.global_maps['action_map'], region_coords),
                'dumpability_mask': create_sub_task_dumpability_mask_64x64_fixed(self.global_maps['dumpability_mask'], region_coords),
                'dumpability_mask_init': create_sub_task_dumpability_mask_64x64_fixed(self.global_maps['dumpability_mask_init'], region_coords),
                'padding_mask': create_sub_task_padding_mask_64x64_fixed(self.global_maps['padding_mask'], region_coords),
                'traversability_mask': create_sub_task_traversability_mask_64x64_fixed(self.global_maps['traversability_mask'], region_coords),
            }
            # Create all sub-maps consistently as 64x64
            # sub_maps = create_sub_task_maps_64x64_consistent(self.global_maps, region_coords)
    
            # # Debug: verify shapes
            # debug_map_shapes(sub_maps, f"Partition {partition_idx}")

        # save_mask(np.array(sub_maps['target_map']),'target', 'after_init', partition_idx, 0)
        # save_mask(np.array(sub_maps['action_map']),'action', 'after_init', partition_idx, 0)
        # save_mask(np.array(sub_maps['dumpability_mask']),'dumpability', 'after_init', partition_idx, 0)
        # save_mask(np.array(sub_maps['dumpability_mask_init']),'dumpability_init', 'after_init', partition_idx, 0)
        # save_mask(np.array(sub_maps['padding_mask']),'padding', 'after_init', partition_idx, 0)
        # save_mask(np.array(sub_maps['traversability_mask']),'traversability', 'after_init', partition_idx, 0)

        #DIAGNOSTIC: Check sub-map validity
        # print(f"=== SUB-MAP DIAGNOSTICS ===")
        # for name, map_data in sub_maps.items():
        #     print(f"{name}:")
        #     print(f"  Shape: {map_data.shape}")
        

        # Fix trench data shapes - remove batch dimension for single environment
        trench_axes = self.global_maps['trench_axes']
        trench_type = self.global_maps['trench_type']
    
        # Remove batch dimension if present
        if trench_axes.shape[0] == 1:
            trench_axes = trench_axes[0]  # Shape: (3, 3) instead of (1, 3, 3)
        if trench_type.shape[0] == 1:
            trench_type = trench_type[0]  # Shape: () instead of (1,)
        trench_axes = trench_axes.astype(jnp.float32)
        trench_type = trench_type.astype(jnp.int32)
        
        # Reset the small environment using TerraEnv's interface (no batching)
        clean_env_cfg = self._create_clean_env_config()
        print(f"Environment config created")

        self.rng, reset_key = jax.random.split(self.rng)

        try:
            print("Resetting small environment with custom map data...")
            
            # Use TerraEnv's reset method directly - much cleaner interface
            small_timestep = self.small_env.reset(
                key=reset_key,
                target_map=sub_maps['target_map'],
                padding_mask=sub_maps['padding_mask'],
                trench_axes=trench_axes,
                trench_type=trench_type,
                dumpability_mask_init=sub_maps['dumpability_mask_init'],
                action_map=sub_maps['action_map'],
                env_cfg=clean_env_cfg,
                custom_pos=custom_pos,
                custom_angle=custom_angle
            )
            print("Small environment reset successfully.")

            # Store current small environment state
            self.small_env_timestep = small_timestep
            self.current_partition_idx = partition_idx
            
            # Set partition status to active
            self.partitions[partition_idx]['status'] = 'active'
            
            # Switch display to small environment
            self.current_display_env = "small"
            return small_timestep
            
        except Exception as e:
            import traceback
            print(f"Error initializing small environment: {e}")
            print(traceback.format_exc())
            raise

    def _update_world_map(self, world_state, map_name: str, new_map):
        """
        Helper method to update a specific map in the world state.
        This creates a new world state with the updated map.
        """
        # Get the current map object
        current_map_obj = getattr(world_state, map_name)
        
        # Create new map object with updated data
        updated_map_obj = current_map_obj._replace(map=new_map)
        
        # Create new world state with updated map
        updated_world = world_state._replace(**{map_name: updated_map_obj})
        
        return updated_world

    def _derive_small_environment_config(self):
        """
        Derive a configuration for small environments based on the global config.
        Returns a modified config with appropriate size settings.
        """
        # Create a copy of the global environment config
        small_config = jax.tree_map(lambda x: x, self.global_env_config)
        
        # Modify map size and other relevant parameters
        # This requires knowledge of the config structure
        if hasattr(small_config, 'maps') and hasattr(small_config.maps, 'edge_length_px'):
            small_config = small_config._replace(
                maps=small_config.maps._replace(
                    edge_length_px=jnp.array([64], dtype=jnp.int32)
                )
            )
        
        # If map_size is a separate attribute
        if hasattr(small_config, 'map_size'):
            small_config = small_config._replace(map_size=64)
            
        return small_config
    
    def _initialize_global_environment(self):
        """Initialize the global environment with proper batching"""
        self.rng, reset_key = jax.random.split(self.rng)
    
        # Create array of keys for batching consistency
        reset_keys = jax.random.split(reset_key, 1)  # Shape: (1, 2)
    
        print("Initializing global environment...")
        global_timestep = self.global_env.reset(self.global_env_config, reset_keys)
    
        # Extract and store global map data
        self.global_maps['target_map'] = global_timestep.state.world.target_map.map[0].copy()
        self.global_maps['action_map'] = global_timestep.state.world.action_map.map[0].copy()
        self.global_maps['dumpability_mask'] = global_timestep.state.world.dumpability_mask.map[0].copy()
        self.global_maps['dumpability_mask_init'] = global_timestep.state.world.dumpability_mask_init.map[0].copy()
        self.global_maps['padding_mask'] = global_timestep.state.world.padding_mask.map[0].copy()
        self.global_maps['traversability_mask'] = global_timestep.state.world.traversability_mask.map[0].copy()
        self.global_maps['trench_axes'] = global_timestep.state.world.trench_axes.copy()
        self.global_maps['trench_type'] = global_timestep.state.world.trench_type.copy()
    
        # Store global timestep
        self.global_timestep = global_timestep
    
        print("Global environment initialized successfully.")
        #print(f"Initial target map has {jnp.sum(self.global_maps['target_map'] < 0)} dig targets")

        return self.global_timestep
        
    def map_position_small_to_global(self, small_pos, region_coords):
        """
        Map agent position from small map coordinates to global map coordinates.
        Assumes the small map places the region at (0,0), so we need to add offsets.
        Returns position in (x, y) format for rendering.
        """
        y_start, x_start, y_end, x_end = region_coords
        
        # Extract position values - assuming agent position is [x, y]
        if hasattr(small_pos, 'shape'):
            if len(small_pos.shape) == 1 and small_pos.shape[0] == 2:
                local_x = float(small_pos[0])
                local_y = float(small_pos[1])
            else:
                local_x = float(small_pos.flatten()[0])
                local_y = float(small_pos.flatten()[1])
        else:
            local_x = float(small_pos[0])
            local_y = float(small_pos[1])
        
        # Add region offset to get global position
        # global_x = local_x + x_start
        # global_y = local_y + y_start

        # Adjust global coordinates based on map size
        if self.map_size_px == 128:
            global_x = local_x + y_start
            global_y = local_y + x_start
        else:
            global_x = local_x 
            global_y = local_y

        # global_x = local_x 
        # global_y = local_y 
        #print(f"Mapping small position {small_pos} to global coordinates: ({global_x}, {global_y}) with region {region_coords}")
        
        # Ensure position is within valid bounds
        #global_x = max(0, min(63, global_x))
        #global_y = max(0, min(63, global_y))
        
        # Return as (x, y) for rendering
        #return (int(global_x), int(global_y))
                # Return as (y, x) for rendering
        return (int(global_y), int(global_x))

    def is_small_task_completed(self):
        """Check if the current small environment task is completed."""
        if self.small_env_timestep is None:
            return False
        
        # Handle both scalar and array cases for done flag
        done_value = self.small_env_timestep.done
        if isinstance(done_value, jnp.ndarray):
            if done_value.shape == ():  # Scalar array
                return bool(done_value)
            elif len(done_value.shape) > 0:  # Array with dimensions
                return bool(done_value[0])
            else:
                return bool(done_value)
        else:
            return bool(done_value)
        
    def _update_global_environment_display_with_all_agents(self, partition_states):
        """
        Update the global environment display with ALL active agents.
        Fixed to handle initialization errors properly.
        """
        try:
            self.rng, reset_key = jax.random.split(self.rng)
            reset_keys = jax.random.split(reset_key, 1)

            # Collect all active agent positions and angles
            all_agent_positions = []
            all_agent_angles_base = []
            all_agent_angles_cabin = []
            all_agent_loaded = []
        
            for partition_idx, partition_state in partition_states.items():
                if partition_state['status'] == 'active' and partition_state['timestep'] is not None:
                    # Get agent state from this partition
                    small_agent_state = partition_state['timestep'].state.agent.agent_state
                    partition = self.partitions[partition_idx]
                    region_coords = partition['region_coords']

                    small_pos = small_agent_state.pos_base
                    small_angle_base = small_agent_state.angle_base
                    small_angle_cabin = small_agent_state.angle_cabin
                    small_loaded = small_agent_state.loaded
                    # print("original small pos:", small_pos)
                    # print("original small angle base:", small_angle_base)
                    # print("original small angle cabin:", small_angle_cabin)
                
                    # Map position to global coordinates
                    global_pos = self.map_position_small_to_global(small_pos, region_coords)
                
                    # Handle angle extraction
                    if hasattr(small_angle_base, 'shape'):
                        if small_angle_base.shape == ():
                            angle_base_val = int(small_angle_base)
                        elif len(small_angle_base.shape) >= 1:
                            angle_base_val = int(small_angle_base.flatten()[0])
                        else:
                            angle_base_val = 0.0
                    else:
                        angle_base_val = int(small_angle_base)

                    if hasattr(small_angle_cabin, 'shape'):
                        if small_angle_cabin.shape == ():
                            angle_cabin_val = int(small_angle_cabin)
                        elif len(small_angle_cabin.shape) >= 1:
                            angle_cabin_val = int(small_angle_cabin.flatten()[0])
                        else:
                            angle_cabin_val = 0.0
                    else:
                        angle_cabin_val = int(small_angle_cabin)
                    
                    if hasattr(small_loaded, 'shape'):
                        if small_loaded.shape == ():
                            small_loaded = int(small_loaded)
                        elif len(small_loaded.shape) >= 1:
                            small_loaded = int(small_loaded.flatten()[0])
                        else:
                            small_loaded = False
                    else:
                        small_loaded = int(small_loaded)

                    #print(global_pos, angle_base_val, angle_cabin_val, small_loaded)
                
                    all_agent_positions.append(global_pos)
                    all_agent_angles_base.append(angle_base_val)
                    all_agent_angles_cabin.append(angle_cabin_val)
                    all_agent_loaded.append(small_loaded)
                
                    print(f"Agent {partition_idx} at global position: {global_pos}, angle base: {angle_base_val}, angle cabin: {angle_cabin_val}, loaded: {small_loaded}")

            # Update global maps from small environments incrementally
            if self.map_size_px == 64:
                self.update_global_maps_from_all_small_environments_small(partition_states)
            else:
                self.update_global_maps_from_all_small_environments_big(partition_states)

            # Use first agent for reset position (others will be added during rendering)
            custom_pos = all_agent_positions[0] if all_agent_positions else None
            custom_angle = all_agent_angles_base[0] if all_agent_angles_base else None

            # Reset global environment with updated maps
            self.global_timestep = self.global_env.reset_with_map_override(
                self.global_env_config,
                reset_keys,
                custom_pos=custom_pos,
                custom_angle=custom_angle,
                target_map_override=self.global_maps['target_map'],
                traversability_mask_override=self.global_maps['traversability_mask'],
                padding_mask_override=self.global_maps['padding_mask'],
                dumpability_mask_override=self.global_maps['dumpability_mask'],
                dumpability_mask_init_override=self.global_maps['dumpability_mask_init'],
                action_map_override=self.global_maps['action_map'],
                agent_config_override=self.small_agent_config
            )
        
            # Store all agent positions for rendering - Initialize these attributes
            if not hasattr(self.global_env, 'all_agent_positions'):
                self.global_env.all_agent_positions = []
            if not hasattr(self.global_env, 'all_agent_angles_base'):
                self.global_env.all_agent_angles_base = []
            if not hasattr(self.global_env, 'all_agent_angles_cabin'):
                self.global_env.all_agent_angles_cabin = []
            if not hasattr(self.global_env, 'all_agent_loaded'):
                self.global_env.all_agent_loaded = []
                
            self.global_env.all_agent_positions = all_agent_positions
            self.global_env.all_agent_angles_base = all_agent_angles_base
            self.global_env.all_agent_angles_cabin = all_agent_angles_cabin
            self.global_env.all_agent_loaded = all_agent_loaded

            print(f"Global environment updated with {len(all_agent_positions)} active agents.")
        
        except Exception as e:
            print(f"Warning: Could not update global environment display: {e}")
            import traceback
            traceback.print_exc()
    
    def update_global_maps_from_all_small_environments_small(self, partition_states):
        """
        Update global maps with changes from ALL active small environments.
        Fixed to handle shape mismatches by properly extracting the correct region size.
        """
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active' and partition_state['timestep'] is not None:
                partition = self.partitions[partition_idx]
                y_start, x_start, y_end, x_end = partition['region_coords']
            
                # Calculate actual region dimensions
                # region_height = y_end - y_start + 1
                # region_width = x_end - x_start + 1
            
                # print(f"Partition {partition_idx} region: ({y_start}, {x_start}) to ({y_end}, {x_end})")
                # print(f"Expected region size: {region_height} x {region_width}")
            
                region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        
                # Get current state from small environment
                small_state = partition_state['timestep'].state
            
                # Extract the maps from small environment (these are 64x64)
                small_maps = {
                    'dumpability_mask': small_state.world.dumpability_mask.map,
                    'target_map': small_state.world.target_map.map,
                    'action_map': small_state.world.action_map.map,
                    'traversability_mask': small_state.world.traversability_mask.map,
                    'padding_mask': small_state.world.padding_mask.map,
                }

                
            
                # print(f"Small environment map shapes:")
                # for name, map_data in small_maps.items():
                #     print(f"  {name}: {map_data.shape}")
            
                # Extract only the relevant portion from the 64x64 small maps
                # that corresponds to the actual region size
                # extract_height = min(region_height, 64)
                # extract_width = min(region_width, 64)
            
                for map_name, small_map in small_maps.items():
                    # Extract the portion that matches the region size
                    #extracted_region = small_map[:extract_height, :extract_width]
                    extracted_region = small_map[region_slice]
                    #print(map_name, extracted_region.shape, region_slice)

                
                    #print(f"  Extracted {map_name}: {extracted_region.shape} -> Global region: {region_height}x{region_width}")
                    self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)

                    # Update the global map with the extracted region
                    # try:
                    #     self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)
                    # except ValueError as e:
                    #     #print(f"  WARNING: Shape mismatch for {map_name}: {e}")
                    #     # Try to handle the mismatch by padding or cropping
                    #     if extracted_region.shape[0] != region_height or extracted_region.shape[1] != region_width:
                    #         # Pad or crop to match the region size
                    #         if extracted_region.shape[0] < region_height or extracted_region.shape[1] < region_width:
                    #             # Pad with zeros
                    #             padded_region = jnp.zeros((region_height, region_width), dtype=extracted_region.dtype)
                    #             padded_region = padded_region.at[:extracted_region.shape[0], :extracted_region.shape[1]].set(extracted_region)
                    #             extracted_region = padded_region
                    #         else:
                    #             # Crop to fit
                    #             extracted_region = extracted_region[:region_height, :region_width]
                        
                    #         #print(f"  Adjusted {map_name}: {extracted_region.shape}")
                    #         self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)

    def update_global_maps_from_all_small_environments_big(self, partition_states):
        """
        FIXED: Update global maps with changes from ALL active small environments.
        Properly handles coordinate transformation between 64x64 partition maps and 128x128 global maps.
        """
        #print(f"🔄 Updating global maps from {len(partition_states)} partitions")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active' and partition_state['timestep'] is not None:
                try:
                    partition = self.partitions[partition_idx]
                    y_start, x_start, y_end, x_end = partition['region_coords']
                    
                    #print(f"  📦 Processing partition {partition_idx}: region ({y_start}, {x_start}) to ({y_end}, {x_end})")
                    
                    # Calculate actual region dimensions
                    region_height = y_end - y_start + 1
                    region_width = x_end - x_start + 1
                    
                    #print(f"     Region size: {region_height} x {region_width}")
                    
                    # Get current state from small environment
                    small_state = partition_state['timestep'].state
                    
                    # Extract the maps from small environment (these are 64x64)
                    small_maps = {
                        'dumpability_mask': small_state.world.dumpability_mask.map,
                        'target_map': small_state.world.target_map.map,
                        'action_map': small_state.world.action_map.map,
                        'traversability_mask': small_state.world.traversability_mask.map,
                        'padding_mask': small_state.world.padding_mask.map,
                    }
                    
                    # print(f"     Small environment map shapes:")
                    # for name, map_data in small_maps.items():
                    #     print(f"       {name}: {map_data.shape}")
                    
                    # CRITICAL FIX: Extract the correct portion from the 64x64 partition maps
                    # We need to extract a region-sized portion from the 64x64 map, NOT use global coordinates
                    
                    for map_name, small_map in small_maps.items():
                        if small_map.shape != (64, 64):
                            #print(f"     ⚠️  WARNING: {map_name} has unexpected shape {small_map.shape}, skipping")
                            continue
                        
                        # FIXED: Extract region from the 64x64 local map using LOCAL coordinates
                        if region_height <= 64 and region_width <= 64:
                            # Extract the relevant portion from the TOP-LEFT of the 64x64 map
                            # This corresponds to the actual region data
                            extracted_region = small_map[:region_height, :region_width]
                            #print(f"     ✅ Extracted {map_name}: {extracted_region.shape} from local coordinates")
                        else:
                            #print(f"     ❌ Region size {region_height}x{region_width} exceeds 64x64, skipping {map_name}")
                            continue
                        
                        # Now update the global map using GLOBAL coordinates
                        global_region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
                        
                        try:
                            # Verify global map exists and has correct shape
                            if map_name not in self.global_maps:
                                #print(f"     ❌ Global map {map_name} not found")
                                continue
                                
                            global_map = self.global_maps[map_name]
                            if global_map.shape != (128, 128):
                                #print(f"     ⚠️  Global map {map_name} has unexpected shape {global_map.shape}")
                                continue
                            
                            # Update the global map
                            self.global_maps[map_name] = global_map.at[global_region_slice].set(extracted_region)
                            #print(f"     ✅ Updated global {map_name} at {global_region_slice} with {extracted_region.shape} data")
                            
                        except Exception as update_error:
                            print(f"     ❌ Failed to update global {map_name}: {update_error}")
                            print(f"        Global map shape: {self.global_maps[map_name].shape}")
                            print(f"        Global slice: {global_region_slice}")
                            print(f"        Extracted region shape: {extracted_region.shape}")
                            continue
                    
                    #print(f"  ✅ Completed partition {partition_idx}")
                    
                except Exception as partition_error:
                    print(f"  ❌ Failed to process partition {partition_idx}: {partition_error}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        #print(f"🏁 Global maps update completed")

    def render_global_environment_with_multiple_agents(self, partition_states, VISUALIZE_PARTITIONS=False):
        """
        Update and render global environment showing ALL active excavators.
        Fixed to handle missing attributes gracefully.
        """
        # First update with all agents
        self._update_global_environment_display_with_all_agents(partition_states)

        # Then render with multiple agents
        try:
            obs = self.global_timestep.observation
            info = self.global_timestep.info
        
            # Pass additional agent positions to the rendering system
            if (hasattr(self.global_env, 'all_agent_positions') and 
                hasattr(self.global_env, 'all_agent_angles_base') and
                hasattr(self.global_env, 'all_agent_angles_cabin') and 
                hasattr(self.global_env, 'all_agent_loaded')):
                
                # Add all agent positions to the info for rendering
                info['additional_agents'] = {
                    'positions': self.global_env.all_agent_positions,
                    'angles base': self.global_env.all_agent_angles_base,
                    'angles cabin': self.global_env.all_agent_angles_cabin,
                    'loaded': self.global_env.all_agent_loaded
                }
                #print(f"Passing {len(self.global_env.all_agent_positions)} agents to renderer")
            else:
                print("Warning: Agent attributes not properly initialized for rendering")
                # Initialize empty lists to prevent further errors
                info['additional_agents'] = {
                    'positions': [],
                    'angles base': [],
                    'angles cabin': [],
                    'loaded': []
                }

            if VISUALIZE_PARTITIONS:
                info['show_partitions'] = True
                info['partitions'] = self.partitions  # Just pass the whole partition list

            # Pass agent config to rendering
            info['agent_config'] = self.small_agent_config
            
            if self.rendering:
                self.global_env.terra_env.render_obs_pygame(obs, info)
    
        except Exception as e:
            print(f"Global rendering error: {e}")
            import traceback
            traceback.print_exc()

    def render_all_partition_views_grid(self, partition_states):
        """
        Render all active partition views in a grid layout.
        This shows what each agent sees simultaneously.
        """

        
        active_partitions = [idx for idx, state in partition_states.items() 
                            if state['status'] == 'active']
        
        if not active_partitions:
            return
        
        # Get screen dimensions
        screen = pg.display.get_surface()
        if screen is None:
            return
        
        screen_width, screen_height = screen.get_size()
        
        # Calculate grid layout
        num_partitions = len(active_partitions)
        cols = min(2, num_partitions)  # Max 2 columns
        rows = (num_partitions + cols - 1) // cols
        
        # Calculate size for each partition view
        partition_width = screen_width // cols
        partition_height = screen_height // rows
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Render each partition
        for i, partition_idx in enumerate(active_partitions):
            partition_state = partition_states[partition_idx]
            
            # Calculate position in grid
            col = i % cols
            row = i // cols
            x_offset = col * partition_width
            y_offset = row * partition_height
            
            # Render this partition's view
            self._render_single_partition_view(
                screen, partition_state, partition_idx,
                x_offset, y_offset, partition_width, partition_height
            )
        
        pg.display.flip()

    def _render_single_partition_view(self, screen, partition_state, partition_idx,
                                    x_offset, y_offset, width, height):
        """
        Render a single partition's view within the given screen area.
        """
        # Get the maps from the partition
        current_timestep = partition_state['timestep']
        world = current_timestep.state.world
        agent_state = current_timestep.state.agent.agent_state
        
        # Extract maps
        target_map = world.target_map.map
        action_map = world.action_map.map
        traversability_mask = world.traversability_mask.map
        agent_pos = agent_state.pos_base
        
        #print(current_timestep.observation)
        # target_map = current_timestep.observation['target_map']
        # action_map = current_timestep.observation['action_map']
        # traversability_mask = current_timestep.observation['traversability_mask']

        # Map dimensions
        map_height, map_width = target_map.shape
        
        # Calculate tile size to fit in available space
        tile_width = (width - 40) // map_width  # Leave 40 pixels for margins
        tile_height = (height - 60) // map_height  # Leave 60 pixels for title and info
        tile_size = max(2, min(tile_width, tile_height))
        
        # Center the map in the available space
        map_pixel_width = map_width * tile_size
        map_pixel_height = map_height * tile_size
        map_x = x_offset + (width - map_pixel_width) // 2
        map_y = y_offset + 40  # Leave space for title
        
        # Draw title
        font = pg.font.Font(None, 32)
        title = f"Partition {partition_idx}"
        text = font.render(title, True, (255, 255, 255))
        screen.blit(text, (x_offset + 10, y_offset + 5))
        
        # Draw the map
        for y in range(map_height):
            for x in range(map_width):
                # Get cell values
                target_val = target_map[y, x]
                action_val = action_map[y, x]
                traversable = traversability_mask[y, x]
                
                # Determine color based on cell state
                if traversable == -1:  # Agent position
                    color = (255, 100, 255)  # Magenta
                elif traversable == 1:   # Obstacle (including other agents)
                    color = (255, 50, 50)   # Red
                elif action_val > 0:     # Dumped soil
                    color = (139, 69, 19)   # Brown
                elif action_val == -1:   # Dug area
                    color = (101, 67, 33)   # Dark brown
                elif target_val == -1:   # Target to dig
                    color = (255, 255, 0)   # Yellow
                elif target_val == 1:    # Target to dump
                    color = (0, 255, 0)     # Green
                else:                    # Free space
                    color = (220, 220, 220) # Light gray
                
                # Draw the tile
                rect = pg.Rect(
                    map_x + x * tile_size,
                    map_y + y * tile_size,
                    tile_size,
                    tile_size
                )
                pg.draw.rect(screen, color, rect)
        
        # Draw border around map
        border_rect = pg.Rect(map_x - 1, map_y - 1, 
                            map_pixel_width + 2, map_pixel_height + 2)
        pg.draw.rect(screen, (255, 255, 255), border_rect, 1)
        
        # Draw agent position and stats
        small_font = pg.font.Font(None, 20)
        
        # Agent position
        pos_text = f"Agent: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})"
        pos_surface = small_font.render(pos_text, True, (255, 255, 255))
        screen.blit(pos_surface, (x_offset + 10, y_offset + height - 40))
        
        # Obstacle count (red cells = other agents + terrain obstacles)
        obstacle_count = np.sum(traversability_mask == 1)
        obstacle_text = f"Red obstacles: {obstacle_count}"
        obstacle_surface = small_font.render(obstacle_text, True, (255, 100, 100))
        screen.blit(obstacle_surface, (x_offset + 10, y_offset + height - 20))
    
    def _should_show_agent_in_partition(self, partition_idx, agent_y, agent_x):
        """
        Determine if an agent at the given position should be visible to the partition.
        
        For global maps, you might want to:
        1. Show all agents everywhere (return True)
        2. Show agents only within a certain distance of the partition's region
        3. Show agents only within the partition's assigned region
        """
        # Option 1: Show all agents everywhere (recommended for global maps)
        return True
        
        # Option 2: Show agents within partition region + buffer
        # if partition_idx < len(self.partitions):
        #     partition = self.partitions[partition_idx]
        #     y_start, x_start, y_end, x_end = partition['region_coords']
        #     
        #     # Add buffer around partition region
        #     buffer = 10
        #     return (y_start - buffer <= agent_y <= y_end + buffer and 
        #             x_start - buffer <= agent_x <= x_end + buffer)
        # 
        # return False

    def initialize_base_traversability_masks(self, partition_states):
        """
        Store the initial clean traversability masks for each partition.
        This captures the original terrain obstacles before any agent synchronization.
        Call this ONCE after partition initialization but BEFORE any agent sync.
        """
        if not hasattr(self, 'base_traversability_masks'):
            self.base_traversability_masks = {}
        
        #print("Initializing base traversability masks...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Get the current traversability mask
                current_mask = partition_state['timestep'].state.world.traversability_mask.map.copy()
                
                # Clean ANY agent markers to get pure terrain
                # -1 = agent position (clear to 0)
                # 1 = could be terrain or agent obstacles (assume terrain at initialization)
                # 0 = free space (keep as is)
                
                # Create a completely clean mask - only terrain obstacles, no agents
                clean_mask = jnp.where(
                    current_mask == -1,  # Remove any agent positions
                    0,  # Set to free space
                    jnp.where(
                        current_mask == 1,  # Keep terrain obstacles
                        1,
                        0  # Everything else becomes free space
                    )
                )
                
                self.base_traversability_masks[partition_idx] = clean_mask
                #print(f"  Stored clean base mask for partition {partition_idx}")
                
                # Count terrain obstacles for verification
                terrain_obstacles = jnp.sum(clean_mask == 1)
                #print(f"    Terrain obstacles: {terrain_obstacles}")

    def _update_partition_with_other_agents(self, target_partition_idx, target_partition_state, 
                                        all_occupied_cells, partition_states):
        """
        Update a partition's traversability mask to show other agents as obstacles.
        Now properly preserves original terrain obstacles.
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from the clean base mask (has original terrain obstacles but no agent obstacles)
        if hasattr(self, 'base_traversability_masks') and target_partition_idx in self.base_traversability_masks:
            # Start from clean base (original terrain obstacles preserved)
            current_traversability = self.base_traversability_masks[target_partition_idx].copy()
            
            # Add back the current agent's position (-1)
            original_traversability = current_timestep.state.world.traversability_mask.map
            agent_mask = (original_traversability == -1)
            current_traversability = jnp.where(
                agent_mask,
                -1,  # Restore current agent position
                current_traversability  # Keep clean base with terrain obstacles
            )
        else:
            # Fallback: use current mask but this might not work perfectly
            print(f"Warning: No base mask for partition {target_partition_idx}, using current mask")
            current_traversability = current_timestep.state.world.traversability_mask.map.copy()
            
            # Try to clear only agent obstacles (this is less reliable)
            # Keep -1 (current agent) and assume original 1s are terrain
            # This fallback is not ideal - base masks are recommended
        
        # STEP 2: Add current positions of OTHER agents as obstacles
        agents_added = 0
        cells_added = 0
        
        for other_partition_idx, occupied_cells in all_occupied_cells.items():
            if other_partition_idx == target_partition_idx:
                continue  # Don't add self as obstacle
                
            for cell_y, cell_x in occupied_cells:
                # Check if this cell should be visible in this partition
                if self._should_show_agent_in_partition(target_partition_idx, cell_y, cell_x):
                    # Check bounds
                    if (0 <= cell_y < current_traversability.shape[0] and 
                        0 <= cell_x < current_traversability.shape[1]):
                        # Mark as obstacle (value = 1) - this represents another agent
                        # Only set if it's currently free space (0) to avoid overwriting terrain
                        if current_traversability[cell_y, cell_x] == 0:
                            current_traversability = current_traversability.at[cell_y, cell_x].set(1)
                            cells_added += 1
            
            if cells_added > 0:
                agents_added += 1
        
        # STEP 3: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            current_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        if agents_added > 0:
            print(f"  ✓ Added {agents_added} other agents ({cells_added} cells) to partition {target_partition_idx}")

    def sync_agents_in_global_environment(self, partition_states):
        """
        Updated synchronization that preserves original terrain obstacles.
        """
        print(f"\n=== SYNCING AGENTS IN GLOBAL ENVIRONMENT ===")
        
        # Initialize base masks if not done yet (IMPORTANT: do this early, before any sync operations)
        if not hasattr(self, 'base_traversability_masks'):
            print("Initializing base traversability masks...")
            self.initialize_base_traversability_masks(partition_states)
        
        # Collect all active agent positions and their occupied cells
        all_agent_positions = {}
        all_occupied_cells = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            traversability = current_timestep.state.world.traversability_mask.map
            
            # Find where this agent is (value = -1)
            agent_mask = (traversability == -1)
            agent_positions = jnp.where(agent_mask)
            
            if len(agent_positions[0]) > 0:
                # Store agent position info
                all_agent_positions[partition_idx] = {
                    'positions': agent_positions,
                    'count': len(agent_positions[0])
                }
                
                # Store occupied cells for this agent
                occupied_cells = []
                for i in range(len(agent_positions[0])):
                    cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
                    occupied_cells.append(cell)
                all_occupied_cells[partition_idx] = occupied_cells
        
        # Update each partition's traversability mask with other agents
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_with_other_agents(
                target_partition_idx, target_partition_state, 
                all_occupied_cells, partition_states
            )
        
        print(f"Agent synchronization completed for {len(all_agent_positions)} active agents")

    def sync_targets_across_partitions(self, partition_states):
        """
        Synchronize targets across partitions by marking other partitions' targets as obstacles.
        This prevents agents from working on targets assigned to other partitions.
        """
        print(f"\n=== SYNCING TARGETS ACROSS PARTITIONS ===")
        
        # Collect all partition targets
        all_partition_targets = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            target_map = current_timestep.state.world.target_map.map
            
            # Store the target map for this partition
            all_partition_targets[partition_idx] = target_map
        
        # Update each partition to mark other partitions' targets as obstacles
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_with_other_targets(
                target_partition_idx, target_partition_state, 
                all_partition_targets, partition_states
            )
        
        print(f"Target synchronization completed for {len(all_partition_targets)} active partitions")

    def _update_partition_with_other_targets(self, target_partition_idx, target_partition_state, 
                                        all_partition_targets, partition_states):
        """
        Update a partition's traversability mask to mark other partitions' targets as obstacles.
        """
        current_timestep = target_partition_state['timestep']
        
        # Start from the current traversability mask
        current_traversability = current_timestep.state.world.traversability_mask.map.copy()
        
        targets_blocked = 0
        cells_blocked = 0
        
        # Add other partitions' targets as obstacles
        for other_partition_idx, other_target_map in all_partition_targets.items():
            if other_partition_idx == target_partition_idx:
                continue  # Don't block own targets
            
            # Get the current partition's own target map to avoid conflicts
            own_target_map = current_timestep.state.world.target_map.map
            
            # Find other partition's targets (both dig targets -1 and dump targets 1)
            other_dig_targets = (other_target_map == -1)
            other_dump_targets = (other_target_map == 1)
            other_all_targets = other_dig_targets | other_dump_targets
            
            # Only block targets that are not also targets in current partition
            own_targets = (own_target_map == -1) | (own_target_map == 1)
            
            # Find positions to block: other partition's targets that aren't current partition's targets
            positions_to_block = other_all_targets & ~own_targets
            
            # Find valid positions to block in the traversability mask
            for y in range(current_traversability.shape[0]):
                for x in range(current_traversability.shape[1]):
                    if positions_to_block[y, x]:
                        # Only mark as obstacle if it's currently free space (0) or traversable
                        # Don't overwrite agent positions (-1) or existing obstacles (1)
                        if current_traversability[y, x] == 0:
                            current_traversability = current_traversability.at[y, x].set(1)
                            cells_blocked += 1
            
            if cells_blocked > 0:
                targets_blocked += 1
        
        # Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            current_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        if targets_blocked > 0:
            print(f"  ✓ Blocked {cells_blocked} target cells from {targets_blocked} other partitions in partition {target_partition_idx}")

    # def step_with_full_global_sync(self, partition_idx: int, action, partition_states: dict):
    #     """
    #     UPDATED: Enhanced step function that properly handles dumped soil as obstacles.
    #     """
    #     # Step 1: Take the action in the partition
    #     new_timestep = self.step_simple(partition_idx, action, partition_states)
        
    #     # Step 2: Update the partition state
    #     partition_states[partition_idx]['timestep'] = new_timestep
        
    #     # Step 3: Extract changes from this partition and update global maps
    #     self._update_global_maps_from_single_partition(partition_idx, partition_states)
        
    #     # Step 4: Propagate global map changes to ALL partitions (EXCLUDING traversability)
    #     self._sync_all_partitions_from_global_maps_excluding_traversability(partition_states)
        
    #     # Step 5: Properly sync agent positions AND dumped soil obstacles
    #     self._sync_agent_positions_across_partitions(partition_states)
        
    #     # Step 6: Update observations to match synced states
    #     self._update_all_observations(partition_states)
        
    #     return new_timestep
    
    def step_with_full_global_sync(self, partition_idx: int, action, partition_states: dict):
        """
        FIXED: Synchronize BEFORE stepping to prevent collisions.
        """
        # Step 1: Sync current positions BEFORE any movement
        self._sync_agent_positions_across_partitions(partition_states)
        # print(f"  ✓ Synchronized agent positions before action in partition {partition_idx}")
        
        # # Step 2: Update observations so agents see synchronized state
        self._update_all_observations(partition_states)
        #print(f"  ✓ Updated observations for partition {partition_idx}")
        
        # Step 3: NOW take the action with proper obstacle awareness
        new_timestep = self.step_simple(partition_idx, action, partition_states)
        #print(f"  ✓ Took action in partition {partition_idx}")
        
        # # Step 4: Update the partition state
        partition_states[partition_idx]['timestep'] = new_timestep
        # print(f"  ✓ Updated partition state for partition {partition_idx}")
        
        # # Step 5: Extract changes and update global maps
        if self.map_size_px == 64:
            self._update_global_maps_from_single_partition_small(partition_idx, partition_states)
            #print(f"  ✓ Updated SMALL global maps from partition {partition_idx}")
        else:
            self._update_global_maps_from_single_partition_big(partition_idx, partition_states)
            #print(f"  ✓ Updated BIG global maps from partition {partition_idx}")
        
        # # Step 6: Propagate changes to other partitions
        if self.overlap_regions != {} and self.map_size_px == 64:
            self._sync_all_partitions_from_global_maps_excluding_traversability(partition_states)
            #print(f"  ✓ Synced all partitions from global maps (excluding traversability)")
        
        return new_timestep
    
    
    def _update_global_maps_from_single_partition_big_fixed(self, source_partition_idx, partition_states):
        """
        FIXED: Update global maps with proper handling for overlapping regions.
        This version correctly handles coordinate mapping and overlap synchronization.
        """
        if source_partition_idx not in partition_states:
            return
            
        source_state = partition_states[source_partition_idx]['timestep'].state
        partition = self.partitions[source_partition_idx]
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        
        #print(f"  Updating global maps from partition {source_partition_idx}")
        #print(f"    Partition region_coords: {region_coords}")
        
        # Calculate the actual region dimensions
        region_height = y_end - y_start + 1
        region_width = x_end - x_start + 1
        
        #print(f"    Region dimensions in global map: {region_height}x{region_width}")
        
        # Define which maps to update globally
        maps_to_update = [
            'action_map', 
            'dumpability_mask',
            'dumpability_mask_init',
            'target_map'  # Include target_map for overlap sync
        ]
        
        # Update each map in the global storage
        for map_name in maps_to_update:
            # Get the current map from the partition (this is always 64x64)
            partition_map = getattr(source_state.world, map_name).map
            #print(f"    {map_name} partition shape: {partition_map.shape}")
            
            # CRITICAL FIX: The partition map is 64x64, which represents the ENTIRE partition view
            # We can only update the portion of the global map that fits within 64x64
            # If the region is larger than 64x64, we can only update a 64x64 portion
            
            # Calculate how much we can actually update
            update_height = min(64, region_height)
            update_width = min(64, region_width)
            
            #print(f"    Will update {update_height}x{update_width} region in global map")
            
            # Extract the data to update (the full partition map up to the update size)
            data_to_update = partition_map[:update_height, :update_width]
            
            # Define the target slice in the global map (only update what we can)
            global_region_slice = (
                slice(y_start, y_start + update_height), 
                slice(x_start, x_start + update_width)
            )
            
            # Update the global map with the extracted region
            try:
                self.global_maps[map_name] = self.global_maps[map_name].at[global_region_slice].set(data_to_update)
                #print(f"    ✓ Updated global {map_name} from partition {source_partition_idx}")
            except Exception as e:
                print(f"    ✗ Error updating global {map_name}: {e}")
                # print(f"      Global map shape: {self.global_maps[map_name].shape}")
                # print(f"      Target slice: {global_region_slice}")
                # print(f"      Data to update shape: {data_to_update.shape}")


    def step_with_full_global_sync_fixed(self, partition_idx: int, action, partition_states: dict):
        """
        FIXED: Enhanced step function with proper overlap synchronization for big maps.
        """
        # Step 1: Sync current positions BEFORE any movement
        self._sync_agent_positions_across_partitions(partition_states)
        
        # Step 2: Update observations so agents see synchronized state
        self._update_all_observations(partition_states)
        
        # Step 3: Take the action with proper obstacle awareness
        new_timestep = self.step_simple(partition_idx, action, partition_states)
        
        # Step 4: Update the partition state
        partition_states[partition_idx]['timestep'] = new_timestep
        
        # Step 5: Extract changes and update global maps
        if self.map_size_px == 64:
            self._update_global_maps_from_single_partition_small(partition_idx, partition_states)
        else:
            self._update_global_maps_from_single_partition_big_fixed(partition_idx, partition_states)
        
        # Step 6: Sync overlapping regions for big maps
        if self.map_size_px > 64 and self.overlap_regions:
            self._sync_overlapping_regions_big_maps_fixed(partition_idx, partition_states)
        
        # Step 7: Propagate changes to other partitions
        if self.overlap_regions != {}:
            self._sync_all_partitions_from_global_maps_excluding_traversability(partition_states)
        
        return new_timestep


    def _sync_overlapping_regions_big_maps_fixed(self, source_partition_idx, partition_states):
        """
        FIXED: Synchronize overlapping regions between partitions for big maps.
        Handles the case where regions are larger than 64x64.
        """
        #print(f"  Syncing overlapping regions from partition {source_partition_idx}")
        
        # Get all partitions that overlap with the source
        overlapping_partitions = self.overlap_map.get(source_partition_idx, set())
        
        for target_partition_idx in overlapping_partitions:
            if (target_partition_idx not in partition_states or 
                partition_states[target_partition_idx]['status'] != 'active'):
                continue
            
            # Get overlap information
            overlap_key = (min(source_partition_idx, target_partition_idx), 
                        max(source_partition_idx, target_partition_idx))
            
            if overlap_key not in self.overlap_regions:
                continue
            
            overlap_info = self.overlap_regions[overlap_key]
            
            # Determine which partition is source and which is target
            if source_partition_idx < target_partition_idx:
                source_slice = overlap_info['partition_i_slice']
                target_slice = overlap_info['partition_j_slice']
            else:
                source_slice = overlap_info['partition_j_slice']
                target_slice = overlap_info['partition_i_slice']
            
            # Sync the overlapping region
            self._sync_single_overlap_region_fixed(
                source_partition_idx, target_partition_idx,
                source_slice, target_slice,
                partition_states
            )


    def _sync_single_overlap_region_fixed(self, source_idx, target_idx, source_slice, target_slice, partition_states):
        """
        FIXED: Sync a single overlapping region from source to target partition.
        Handles shape mismatches by only syncing the actual overlapping data.
        """
        source_state = partition_states[source_idx]['timestep'].state
        target_state = partition_states[target_idx]['timestep'].state
        
        # Maps to sync (excluding traversability which is handled separately)
        maps_to_sync = ['action_map', 'dumpability_mask', 'dumpability_mask_init']
        
        #print(f"    Syncing overlap from partition {source_idx} to {target_idx}")
        
        for map_name in maps_to_sync:
            try:
                # Get source and target maps
                source_map = getattr(source_state.world, map_name).map
                target_map = getattr(target_state.world, map_name).map.copy()
                
                # Extract the actual shapes of the slices
                source_y_slice, source_x_slice = source_slice
                target_y_slice, target_x_slice = target_slice
                
                # Calculate actual dimensions of overlap region
                source_height = min(source_y_slice.stop - source_y_slice.start, source_map.shape[0] - source_y_slice.start)
                source_width = min(source_x_slice.stop - source_x_slice.start, source_map.shape[1] - source_x_slice.start)
                target_height = min(target_y_slice.stop - target_y_slice.start, target_map.shape[0] - target_y_slice.start)
                target_width = min(target_x_slice.stop - target_x_slice.start, target_map.shape[1] - target_x_slice.start)
                
                # Use the minimum dimensions to ensure compatibility
                sync_height = min(source_height, target_height)
                sync_width = min(source_width, target_width)
                
                # Create adjusted slices
                adj_source_slice = (
                    slice(source_y_slice.start, source_y_slice.start + sync_height),
                    slice(source_x_slice.start, source_x_slice.start + sync_width)
                )
                adj_target_slice = (
                    slice(target_y_slice.start, target_y_slice.start + sync_height),
                    slice(target_x_slice.start, target_x_slice.start + sync_width)
                )
                
                # Extract overlapping region from source
                source_overlap_data = source_map[adj_source_slice]
                
                # Update target map with source data
                target_map = target_map.at[adj_target_slice].set(source_overlap_data)
                
                # Update the world state
                updated_world = self._update_world_map(target_state.world, map_name, target_map)
                target_state = target_state._replace(world=updated_world)
                
                #print(f"      ✓ Synced {map_name} ({sync_height}x{sync_width} region)")
                
            except Exception as e:
                print(f"      ✗ Error syncing {map_name}: {e}")
        
        # Special handling for target_map - merge instead of overwrite
        try:
            source_target_map = source_state.world.target_map.map
            target_target_map = target_state.world.target_map.map.copy()
            
            # Use adjusted slices for target map as well
            source_overlap_targets = source_target_map[adj_source_slice]
            target_overlap_targets = target_target_map[adj_target_slice]
            
            # Merge logic: keep dig targets (-1) from both, prioritize source for conflicts
            merged_targets = jnp.where(
                (target_overlap_targets == -1) | (source_overlap_targets == -1),
                -1,  # Keep dig targets from either partition
                source_overlap_targets  # Otherwise use source
            )
            
            target_target_map = target_target_map.at[adj_target_slice].set(merged_targets)
            updated_world = self._update_world_map(target_state.world, 'target_map', target_target_map)
            target_state = target_state._replace(world=updated_world)
            
            #print(f"      ✓ Merged target_map ({sync_height}x{sync_width} region)")
            
        except Exception as e:
            print(f"      ✗ Error merging target_map: {e}")
        
        # Update the partition state
        updated_timestep = partition_states[target_idx]['timestep']._replace(state=target_state)
        partition_states[target_idx]['timestep'] = updated_timestep
        
        #print(f"    ✓ Synced overlap region")


    def _calculate_overlap_region_fixed(self, partition_i: int, partition_j: int):
        """
        FIXED: Calculate the overlapping region between two partitions with correct coordinate mapping.
        Handles cases where partitions are larger than 64x64.
        """
        p1_coords = self.partitions[partition_i]['region_coords']
        p2_coords = self.partitions[partition_j]['region_coords']
        
        y1_start, x1_start, y1_end, x1_end = p1_coords
        y2_start, x2_start, y2_end, x2_end = p2_coords
        
        # Find intersection in global coordinates
        overlap_y_start = max(y1_start, y2_start)
        overlap_x_start = max(x1_start, x2_start)
        overlap_y_end = min(y1_end, y2_end)
        overlap_x_end = min(x1_end, x2_end)
        
        # Check if there's actual overlap
        if overlap_y_start > overlap_y_end or overlap_x_start > overlap_x_end:
            return None
        
        # Calculate local coordinates relative to each partition's origin
        # But limit to 64x64 since that's the actual partition map size
        local_i_y_start = overlap_y_start - y1_start
        local_i_x_start = overlap_x_start - x1_start
        local_i_y_end = overlap_y_end - y1_start
        local_i_x_end = overlap_x_end - x1_start
        
        local_j_y_start = overlap_y_start - y2_start
        local_j_x_start = overlap_x_start - x2_start
        local_j_y_end = overlap_y_end - y2_start
        local_j_x_end = overlap_x_end - x2_start
        
        # CRITICAL: Ensure local coordinates don't exceed 64x64 bounds
        # The partition maps are always 64x64, even if the region is larger
        local_i_y_start = max(0, min(local_i_y_start, 63))
        local_i_x_start = max(0, min(local_i_x_start, 63))
        local_i_y_end = max(0, min(local_i_y_end, 63))
        local_i_x_end = max(0, min(local_i_x_end, 63))
        
        local_j_y_start = max(0, min(local_j_y_start, 63))
        local_j_x_start = max(0, min(local_j_x_start, 63))
        local_j_y_end = max(0, min(local_j_y_end, 63))
        local_j_x_end = max(0, min(local_j_x_end, 63))
        
        return {
            'global_slice': (slice(overlap_y_start, overlap_y_end + 1), 
                            slice(overlap_x_start, overlap_x_end + 1)),
            'partition_i_slice': (slice(local_i_y_start, local_i_y_end + 1), 
                                slice(local_i_x_start, local_i_x_end + 1)),
            'partition_j_slice': (slice(local_j_y_start, local_j_y_end + 1),
                                slice(local_j_x_start, local_j_x_end + 1)),
            'overlap_bounds': (overlap_y_start, overlap_x_start, overlap_y_end, overlap_x_end)
        }


    def _compute_overlap_relationships_fixed(self):
        """
        FIXED: Compute overlap relationships with corrected coordinate mapping.
        """
        print(f"\n=== COMPUTING OVERLAP RELATIONSHIPS ===")
        
        self.overlap_map = {i: set() for i in range(len(self.partitions))}
        self.overlap_regions = {}
        
        for i in range(len(self.partitions)):
            for j in range(i + 1, len(self.partitions)):
                print(f"\nChecking partitions {i} and {j}:")
                
                if self._partitions_overlap(i, j):
                    self.overlap_map[i].add(j)
                    self.overlap_map[j].add(i)
                    
                    # Use the fixed overlap calculation
                    overlap_info = self._calculate_overlap_region_fixed(i, j)
                    if overlap_info is not None:
                        self.overlap_regions[(i, j)] = overlap_info
                        self.overlap_regions[(j, i)] = overlap_info  # Symmetric
                        
                        # Debug: print overlap details
                        print(f"  Overlap found:")
                        print(f"    Global region: {overlap_info['overlap_bounds']}")
                        print(f"    Partition {i} local slice: {overlap_info['partition_i_slice']}")
                        print(f"    Partition {j} local slice: {overlap_info['partition_j_slice']}")
                    else:
                        print(f"  Could not calculate overlap region!")
                else:
                    print(f"  No overlap detected")
        
        print(f"\n=== FINAL OVERLAP RELATIONSHIPS ===")
        for i, partition in enumerate(self.partitions):
            overlaps = list(self.overlap_map[i])
            print(f"Partition {i}: region={partition['region_coords']}, overlaps with {overlaps}")
        
        print(f"Total overlap regions cached: {len(self.overlap_regions)}")

    def _sync_all_partitions_from_global_maps_excluding_traversability_fixed(self, partition_states):
        """
        FIXED: Synchronize ALL partitions with updated global maps, handling 64x64 partition maps correctly.
        """
        #print(f"  Syncing global maps to all partitions (excluding traversability)")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            # Get current state
            current_timestep = target_partition_state['timestep']
            current_state = current_timestep.state
            
            # Create updated world state with global maps but preserve partition-specific targets
            updated_world = self._create_world_with_global_maps_preserve_targets_fixed(
                current_state.world, target_partition_idx
            )
            
            # Create updated state and timestep
            updated_state = current_state._replace(world=updated_world)
            updated_timestep = current_timestep._replace(state=updated_state)
            
            # Update the partition state
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            #print(f"    Synced global maps to partition {target_partition_idx}")


    def _create_world_with_global_maps_preserve_targets_fixed(self, current_world, partition_idx):
        """
        FIXED: Create a new world state that uses global maps but correctly handles 64x64 partition size.
        """
        # Get partition info
        partition = self.partitions[partition_idx]
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        
        # Calculate region dimensions
        region_height = min(64, y_end - y_start + 1)
        region_width = min(64, x_end - x_start + 1)
        
        #print(f"    Creating world for partition {partition_idx}, extracting {region_height}x{region_width} from global")
        
        # Get the original partition-specific target map
        if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
            partition_target_map = self.partition_target_maps[partition_idx]
        else:
            partition_target_map = current_world.target_map.map
        
        # Extract 64x64 regions from global maps for this partition
        extracted_maps = {}
        
        for map_name in ['action_map', 'dumpability_mask', 'dumpability_mask_init', 'padding_mask']:
            global_map = self.global_maps[map_name]
            
            # Extract the region from global map
            extracted_region = global_map[y_start:y_start + region_height, x_start:x_start + region_width]
            
            # If extracted region is smaller than 64x64, pad it
            if extracted_region.shape != (64, 64):
                # Create a 64x64 map with appropriate default values
                if map_name == 'action_map':
                    default_val = 0  # Free space
                elif 'dumpability' in map_name:
                    default_val = 0 if 'dumpability_mask' in map_name else 1  # Can't dump / can dump for init
                elif map_name == 'padding_mask':
                    default_val = 1  # Non-traversable padding
                else:
                    default_val = 0
                    
                padded_map = jnp.full((64, 64), default_val, dtype=extracted_region.dtype)
                padded_map = padded_map.at[:extracted_region.shape[0], :extracted_region.shape[1]].set(extracted_region)
                extracted_maps[map_name] = padded_map
            else:
                extracted_maps[map_name] = extracted_region
        
        # Create updated world with extracted maps
        updated_world = current_world._replace(
            target_map=current_world.target_map._replace(map=partition_target_map),  # Keep partition-specific
            action_map=current_world.action_map._replace(map=extracted_maps['action_map']),
            dumpability_mask=current_world.dumpability_mask._replace(map=extracted_maps['dumpability_mask']),
            dumpability_mask_init=current_world.dumpability_mask_init._replace(map=extracted_maps['dumpability_mask_init']),
            padding_mask=current_world.padding_mask._replace(map=extracted_maps['padding_mask'])
            # NOTE: traversability_mask is handled separately by agent sync
        )
        
        return updated_world


    def step_with_full_global_sync_fixed_v2(self, partition_idx: int, action, partition_states: dict):
        """
        FIXED V2: Enhanced step function with better error handling for shape mismatches.
        """
        try:
            # Step 1: Sync current positions BEFORE any movement
            self._sync_agent_positions_across_partitions(partition_states)
            
            # Step 2: Update observations so agents see synchronized state
            self._update_all_observations(partition_states)
            
            # Step 3: Take the action with proper obstacle awareness
            new_timestep = self.step_simple(partition_idx, action, partition_states)
            
            # Step 4: Update the partition state
            partition_states[partition_idx]['timestep'] = new_timestep
            
            # Step 5: Extract changes and update global maps
            if self.map_size_px == 64:
                self._update_global_maps_from_single_partition_small(partition_idx, partition_states)
            else:
                self._update_global_maps_from_single_partition_big_fixed(partition_idx, partition_states)
            
            # Step 6: Sync overlapping regions for big maps
            if self.map_size_px > 64 and self.overlap_regions:
                self._sync_overlapping_regions_big_maps_fixed(partition_idx, partition_states)
            
            # Step 7: Propagate changes to other partitions (using fixed version)
            if self.overlap_regions != {}:
                self._sync_all_partitions_from_global_maps_excluding_traversability_fixed(partition_states)
            
            return new_timestep
            
        except Exception as e:
            print(f"    ERROR in step_with_full_global_sync for partition {partition_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Return the timestep even if sync failed
            return new_timestep if 'new_timestep' in locals() else partition_states[partition_idx]['timestep']


        
    def _update_partition_traversability_with_dumped_soil_and_dig_targets(self, target_partition_idx, target_partition_state, 
                                                                     all_agent_positions, partition_states):
        """
        FIXED: Clean approach to updating traversability that includes:
        1. Original terrain obstacles
        2. Dumped soil as obstacles  
        3. Other agents as obstacles
        4. OTHER PARTITIONS' DIG TARGETS (-1) as obstacles (FIXED - only dig targets, not dump targets)
        
        Traversability logic:
        - 0: Free space (can drive through)
        - 1: Obstacles (terrain + other agents + dumped soil + other partitions' dig targets)
        - -1: Current agent position
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from completely clean base mask (original terrain only)
        if target_partition_idx in self.base_traversability_masks:
            clean_traversability = self.base_traversability_masks[target_partition_idx].copy()
            #print(f"    Starting from clean base for partition {target_partition_idx}")
        else:
            #print(f"    WARNING: No base mask for partition {target_partition_idx}, creating clean mask")
            current_mask = current_timestep.state.world.traversability_mask.map
            clean_traversability = jnp.where(
                (current_mask == -1) | (current_mask == 1),  # Remove all agent markers
                0,  # Set to free space
                current_mask  # Keep original terrain
            )
        
        # STEP 2: Add dumped soil areas as obstacles
        action_map = current_timestep.state.world.action_map.map
        dumped_areas = (action_map > 0)  # Positive values = dumped soil
        
        # Mark dumped soil areas as obstacles (1)
        clean_traversability = jnp.where(
            dumped_areas,
            1,  # Dumped soil = obstacle
            clean_traversability  # Keep existing values
        )
        
        dumped_obstacle_count = jnp.sum(dumped_areas)
        # if dumped_obstacle_count > 0:
        #     print(f"    Added {dumped_obstacle_count} dumped soil obstacles to partition {target_partition_idx}")
        
        # STEP 3: FIXED - Add other partitions' DIG TARGETS (-1) as obstacles, but NOT dump targets (1)
        other_dig_targets_blocked = 0
        
        for other_partition_idx, other_partition_state in partition_states.items():
            if (other_partition_idx == target_partition_idx or 
                other_partition_state['status'] != 'active'):
                continue
                
            # Get the original target map for the other partition
            if hasattr(self, 'partition_target_maps') and other_partition_idx in self.partition_target_maps:
                other_target_map = self.partition_target_maps[other_partition_idx]
                
                # FIXED: Only block dig targets (-1), NOT dump targets (1)
                other_dig_targets = (other_target_map == -1)  # Only dig targets
                # Note: We don't block dump targets (other_target_map == 1) because agents can potentially traverse dump areas
                
                # Mark dig targets as obstacles in current partition's traversability
                clean_traversability = jnp.where(
                    other_dig_targets,
                    1,  # Other partitions' dig targets = obstacles
                    clean_traversability  # Keep existing values
                )
                
                dig_targets_blocked_from_this_partition = jnp.sum(other_dig_targets)
                other_dig_targets_blocked += dig_targets_blocked_from_this_partition
                
                #if dig_targets_blocked_from_this_partition > 0:
                    #print(f"    Blocked {dig_targets_blocked_from_this_partition} DIG TARGETS from partition {other_partition_idx}")
        
        # if other_dig_targets_blocked > 0:
        #     #print(f"    Total other dig targets blocked: {other_dig_targets_blocked}")
        
        # STEP 4: Add THIS partition's agent positions as agents (-1)
        if target_partition_idx in all_agent_positions:
            own_positions = all_agent_positions[target_partition_idx]
            for cell_y, cell_x in own_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    clean_traversability = clean_traversability.at[cell_y, cell_x].set(-1)
            
            #print(f"    Added {len(own_positions)} own agent cells to partition {target_partition_idx}")
        
        # STEP 5: Add OTHER agents as OBSTACLES (1), not agents
        other_agents_added = 0
        other_cells_added = 0
        
        for other_partition_idx, other_positions in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue  # Skip own agent
                
            for cell_y, cell_x in other_positions:
                # Check bounds
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    
                    # Add as OBSTACLE (1), not agent (-1)
                    # Only if it's currently free space (0) - don't overwrite own agent or existing obstacles
                    current_value = clean_traversability[cell_y, cell_x]
                    if current_value == 0:  # Free space
                        clean_traversability = clean_traversability.at[cell_y, cell_x].set(1)
                        other_cells_added += 1
                    #elif current_value == -1:  # Don't overwrite own agent
                        #print(f"      Conflict: Other agent at own agent position ({cell_y}, {cell_x})")
            
            if other_cells_added > 0:
                other_agents_added += 1
        
        # STEP 6: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            clean_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        # Debug output with all obstacle types
        terrain_count = jnp.sum(self.base_traversability_masks.get(target_partition_idx, jnp.zeros_like(clean_traversability)) == 1)
        dumped_soil_count = jnp.sum(dumped_areas)
        own_agent_count = jnp.sum(clean_traversability == -1)
        other_obstacle_count = other_cells_added
        total_obstacles = jnp.sum(clean_traversability == 1)
        free_space = jnp.sum(clean_traversability == 0)
        total_cells = clean_traversability.size
        
        # print(f"    Partition {target_partition_idx} traversability summary:")
        # print(f"      Original terrain obstacles: {terrain_count}")
        # print(f"      Dumped soil obstacles: {dumped_soil_count}")
        # print(f"      Other partitions' DIG TARGETS blocked: {other_dig_targets_blocked}")
        # print(f"      Other agents (as obstacles): {other_obstacle_count}")
        # print(f"      Total obstacles: {total_obstacles}")
        # print(f"      Own agent cells: {own_agent_count}")
        # print(f"      Free space: {free_space} ({free_space/total_cells:.1%})")
        
        # if free_space < total_cells * 0.3:  # Less than 30% free space
        #     print(f"      ⚠️  Warning: Low free space percentage")
        # else:
        #     print(f"      ✅ Good free space percentage")

    def _sync_all_partitions_from_global_maps_excluding_traversability(self, partition_states):
        """
        UPDATED: Synchronize ALL partitions with updated global maps, but preserve partition-specific targets.
        """
        #print(f"  Syncing global maps to all partitions (excluding traversability and preserving partition targets)")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            # Get current state
            current_timestep = target_partition_state['timestep']
            current_state = current_timestep.state
            
            # Create updated world state with global maps but preserve partition-specific targets
            updated_world = self._create_world_with_global_maps_preserve_targets(current_state.world, target_partition_idx)
            
            # Create updated state and timestep
            updated_state = current_state._replace(world=updated_world)
            updated_timestep = current_timestep._replace(state=updated_state)
            
            # Update the partition state
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            #print(f"    Synced global maps to partition {target_partition_idx} (targets preserved)")

    def _update_all_observations(self, partition_states):
        """
        Update observations for all partitions to match their synced states.
        """
        #print(f"  Updating observations for all partitions")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            
            # Create updated observation that matches the synced state
            updated_observation = self._create_observation_from_synced_state(
                current_timestep.observation, 
                current_timestep.state.world
            )
            
            # Update the timestep with the new observation
            updated_timestep = current_timestep._replace(observation=updated_observation)
            partition_states[partition_idx]['timestep'] = updated_timestep

    def _update_global_maps_from_single_partition_small(self, source_partition_idx, partition_states):
        """
        FIXED: Update global maps but handle target_map specially.
        Target maps should remain partition-specific and not be fully synchronized.
        """
        if source_partition_idx not in partition_states:
            return
            
        source_state = partition_states[source_partition_idx]['timestep'].state
        partition = self.partitions[source_partition_idx]
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        
        #print(f"  Updating global maps from partition {source_partition_idx}")
        
        # Define which maps to update globally (EXCLUDE target_map)
        maps_to_update = [
            'action_map', 
            'dumpability_mask',
            'dumpability_mask_init'
        ]
        
        # Update each map in the global storage (EXCLUDING target_map)
        for map_name in maps_to_update:
            # Get the current map from the partition
            partition_map = getattr(source_state.world, map_name).map
            #print(map_name, partition_map.shape)
            
            # Extract the region that corresponds to this partition
            region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
            partition_region = partition_map[region_slice]
            #print(f"    Extracted region {region_slice} from partition {source_partition_idx} for {map_name}")
            
            # Update the global map with this region
            self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(partition_region)
            
            #print(f"    Updated global {map_name} from partition {source_partition_idx}")
        
        # Handle target_map specially - update global but don't sync back to other partitions
        target_map = source_state.world.target_map.map
        region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        target_region = target_map[region_slice]
        
        # Update global target map for tracking purposes, but partitions keep their own
        self.global_maps['target_map'] = self.global_maps['target_map'].at[region_slice].set(target_region)
        #print(f"    Updated global target_map from partition {source_partition_idx} (for tracking only)")

    def _update_global_maps_from_single_partition_big(self, source_partition_idx, partition_states):
        """
        FIXED: Update global maps but handle coordinate mapping correctly.
        The partition maps are always 64x64, but we need to map them back to the correct
        global coordinates based on the partition's region_coords.
        """
        if source_partition_idx not in partition_states:
            return
            
        source_state = partition_states[source_partition_idx]['timestep'].state
        partition = self.partitions[source_partition_idx]
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        
        # print(f"  Updating global maps from partition {source_partition_idx}")
        # print(f"    Partition region_coords: {region_coords}")
        
        # Calculate the actual region dimensions
        region_height = y_end - y_start + 1
        region_width = x_end - x_start + 1
        
        # print(f"    Region dimensions: {region_height}x{region_width}")
        
        # Define which maps to update globally (EXCLUDE target_map)
        maps_to_update = [
            'action_map', 
            'dumpability_mask',
            'dumpability_mask_init'
        ]
        
        # Update each map in the global storage (EXCLUDING target_map)
        for map_name in maps_to_update:
            # Get the current map from the partition (this is always 64x64)
            partition_map = getattr(source_state.world, map_name).map
            #print(f"    {map_name} partition shape: {partition_map.shape}")
            
            # FIXED: Extract only the relevant portion from the 64x64 partition map
            # The partition map contains the region data, but it might be padded to 64x64
            # We need to extract only the actual region size from the partition map
            
            # Calculate how much of the partition map corresponds to the actual region
            extract_height = min(region_height, 64)
            extract_width = min(region_width, 64)
            
            # Extract the relevant portion from the partition map
            # This assumes the region data starts at (0,0) in the partition map
            extracted_region = partition_map[:extract_height, :extract_width]
            
            # print(f"    Extracted region shape: {extracted_region.shape}")
            # print(f"    Target global region: {region_height}x{region_width}")
            
            # Define the target slice in the global map
            global_region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
            
            # Verify the shapes match
            target_shape = (region_height, region_width)
            if extracted_region.shape != target_shape:
                print(f"    WARNING: Shape mismatch - extracted: {extracted_region.shape}, target: {target_shape}")
                # Handle the mismatch by padding or cropping
                if extracted_region.shape[0] < target_shape[0] or extracted_region.shape[1] < target_shape[1]:
                    # Pad with the appropriate default value
                    if map_name == 'action_map':
                        default_val = 0  # Free space
                    elif 'dumpability' in map_name:
                        default_val = 0  # Can't dump
                    else:
                        default_val = 0
                    
                    padded_region = jnp.full(target_shape, default_val, dtype=extracted_region.dtype)
                    padded_region = padded_region.at[:extracted_region.shape[0], :extracted_region.shape[1]].set(extracted_region)
                    extracted_region = padded_region
                else:
                    # Crop to fit
                    extracted_region = extracted_region[:target_shape[0], :target_shape[1]]
            
            #print(f"    Final extracted region shape: {extracted_region.shape}")
            
            # Update the global map with the extracted region
            try:
                self.global_maps[map_name] = self.global_maps[map_name].at[global_region_slice].set(extracted_region)
                #print(f"    ✓ Updated global {map_name} from partition {source_partition_idx}")
            except Exception as e:
                print(f"    ✗ Error updating global {map_name}: {e}")
                # print(f"      Global map shape: {self.global_maps[map_name].shape}")
                # print(f"      Target slice: {global_region_slice}")
                # print(f"      Extracted region shape: {extracted_region.shape}")
        
        # Handle target_map specially - update global but don't sync back to other partitions
        try:
            target_map = source_state.world.target_map.map
            extract_height = min(region_height, 64)
            extract_width = min(region_width, 64)
            target_region = target_map[:extract_height, :extract_width]
            
            # Pad or crop as needed
            if target_region.shape != (region_height, region_width):
                if target_region.shape[0] < region_height or target_region.shape[1] < region_width:
                    padded_region = jnp.ones((region_height, region_width), dtype=target_region.dtype)  # Default to dump areas
                    padded_region = padded_region.at[:target_region.shape[0], :target_region.shape[1]].set(target_region)
                    target_region = padded_region
                else:
                    target_region = target_region[:region_height, :region_width]
            
            # Update global target map for tracking purposes
            global_region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
            self.global_maps['target_map'] = self.global_maps['target_map'].at[global_region_slice].set(target_region)
            #print(f"    ✓ Updated global target_map from partition {source_partition_idx} (for tracking only)")
            
        except Exception as e:
            print(f"    ✗ Error updating global target_map: {e}")
    
    def _create_observation_from_synced_state(self, original_observation, synced_world):
        """
        Create an updated observation dictionary that reflects the synced world state.
        """
        # Start with the original observation (copy all fields)
        updated_observation = {}
        for key, value in original_observation.items():
            updated_observation[key] = value
        
        # Update the critical fields with synced data
        updated_observation['traversability_mask'] = synced_world.traversability_mask.map
        updated_observation['action_map'] = synced_world.action_map.map
        updated_observation['target_map'] = synced_world.target_map.map
        updated_observation['dumpability_mask'] = synced_world.dumpability_mask.map
        updated_observation['padding_mask'] = synced_world.padding_mask.map
        
        return updated_observation

    def _sync_agent_positions_across_partitions(self, partition_states):
        """
        UPDATED: Properly sync agent positions with dumped soil and dig target blocking only.
        """
        #print(f"  Syncing agent positions, dumped soil, and blocking other DIG TARGETS across all partitions")
        
        # Ensure base masks are initialized
        if not hasattr(self, 'base_traversability_masks'):
            #print("  WARNING: Base masks not initialized, initializing now...")
            self.initialize_base_traversability_masks(partition_states)
        
        # Collect all current agent positions
        all_agent_positions = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            traversability = current_timestep.state.world.traversability_mask.map
            
            # Find this partition's agent positions (value = -1)
            agent_mask = (traversability == -1)
            agent_positions = jnp.where(agent_mask)
            
            if len(agent_positions[0]) > 0:
                occupied_cells = []
                for i in range(len(agent_positions[0])):
                    cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
                    occupied_cells.append(cell)
                all_agent_positions[partition_idx] = occupied_cells
                #print(f"    Agent {partition_idx}: {len(occupied_cells)} occupied cells")
        
        # Update each partition with clean traversability including only dig target obstacles
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_traversability_with_dumped_soil_and_dig_targets(
                target_partition_idx, target_partition_state, 
                all_agent_positions, partition_states
            )

    def initialize_partition_specific_target_maps(self, partition_states):
        """
        Store the original partition-specific target maps.
        Each partition should only see their own targets, never targets from other partitions.
        Call this ONCE after partition initialization.
        """
        if not hasattr(self, 'partition_target_maps'):
            self.partition_target_maps = {}
        
        #print("Storing partition-specific target maps...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Store the original target map for this partition
                original_target_map = partition_state['timestep'].state.world.target_map.map.copy()
                self.partition_target_maps[partition_idx] = original_target_map
                
                # Count targets for verification
                dig_targets = jnp.sum(original_target_map == -1)
                dump_targets = jnp.sum(original_target_map == 1)
                
                #print(f"  Partition {partition_idx}: {dig_targets} dig targets, {dump_targets} dump targets")

    def _create_world_with_global_maps_preserve_targets(self, current_world, partition_idx):
        """
        FIXED: Create a new world state that uses global maps but preserves partition-specific targets.
        """
        # Get the original partition-specific target map
        if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
            partition_target_map = self.partition_target_maps[partition_idx]
            #print(f"    Preserving original target map for partition {partition_idx}")
        else:
            # Fallback to current target map
            partition_target_map = current_world.target_map.map
            #print(f"    WARNING: Using current target map for partition {partition_idx} (no stored original)")
        
        updated_world = current_world._replace(
            target_map=current_world.target_map._replace(map=partition_target_map),  # Keep partition-specific
            action_map=current_world.action_map._replace(map=self.global_maps['action_map']),
            dumpability_mask=current_world.dumpability_mask._replace(map=self.global_maps['dumpability_mask']),
            dumpability_mask_init=current_world.dumpability_mask_init._replace(map=self.global_maps['dumpability_mask_init']),
            padding_mask=current_world.padding_mask._replace(map=self.global_maps['padding_mask'])
            # NOTE: traversability_mask will be handled separately by agent sync
        )
        
        return updated_world
    
    def assign_exclusive_targets_in_overlaps(self, partition_states):
        """
        Assign targets in overlapping regions exclusively to one partition.
        This prevents conflicts and double work.
        """
        print("\n=== ASSIGNING EXCLUSIVE TARGETS IN OVERLAPPING REGIONS ===")
        
        # Track which targets have been assigned
        assigned_targets = {}  # (y, x) -> partition_idx
        
        # First pass: identify all targets in overlapping regions
        overlap_targets = {}  # (y, x) -> list of partition_idx that can see it
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            partition = self.partitions[partition_idx]
            region_coords = partition['region_coords']
            y_start, x_start, y_end, x_end = region_coords
            
            # Get this partition's target map
            if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
                target_map = self.partition_target_maps[partition_idx]
                
                # Find all dig targets in this partition
                dig_targets = jnp.where(target_map == -1)
                
                for i in range(len(dig_targets[0])):
                    local_y = int(dig_targets[0][i])
                    local_x = int(dig_targets[1][i])
                    
                    # Convert to global coordinates
                    global_y = y_start + local_y
                    global_x = x_start + local_x
                    
                    # Check if this target is in an overlap region
                    for other_idx in self.overlap_map.get(partition_idx, set()):
                        other_partition = self.partitions[other_idx]
                        other_y_start, other_x_start, other_y_end, other_x_end = other_partition['region_coords']
                        
                        # Check if this global coordinate is within the other partition's region
                        if (other_y_start <= global_y <= other_y_end and 
                            other_x_start <= global_x <= other_x_end):
                            # This target is in an overlap region
                            coord = (global_y, global_x)
                            if coord not in overlap_targets:
                                overlap_targets[coord] = []
                            overlap_targets[coord].append(partition_idx)
        
        # Second pass: assign overlapping targets based on strategy
        targets_reassigned = 0
        for (global_y, global_x), partition_list in overlap_targets.items():
            if len(partition_list) > 1:
                # Multiple partitions can see this target - assign to one
                assigned_partition = self._choose_partition_for_target(
                    global_y, global_x, partition_list, partition_states
                )
                assigned_targets[(global_y, global_x)] = assigned_partition
                targets_reassigned += 1
                
                print(f"  Target at ({global_y}, {global_x}) assigned to partition {assigned_partition} "
                    f"(was visible to: {partition_list})")
        
        # Third pass: update partition target maps to remove non-assigned targets
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            partition = self.partitions[partition_idx]
            region_coords = partition['region_coords']
            y_start, x_start, y_end, x_end = region_coords
            
            # Get current target map
            if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
                target_map = self.partition_target_maps[partition_idx].copy()
                modified = False
                
                # Check each target in this partition
                dig_targets = jnp.where(target_map == -1)
                for i in range(len(dig_targets[0])):
                    local_y = int(dig_targets[0][i])
                    local_x = int(dig_targets[1][i])
                    global_y = y_start + local_y
                    global_x = x_start + local_x
                    
                    # If this target was assigned to another partition, remove it
                    if ((global_y, global_x) in assigned_targets and 
                        assigned_targets[(global_y, global_x)] != partition_idx):
                        # Change from dig target (-1) to free space (0) or dump area (1)
                        target_map = target_map.at[local_y, local_x].set(0)
                        modified = True
                
                if modified:
                    # Update the stored partition target map
                    self.partition_target_maps[partition_idx] = target_map
                    
                    # Update the actual partition's target map
                    current_timestep = partition_state['timestep']
                    updated_world = self._update_world_map(
                        current_timestep.state.world, 
                        'target_map', 
                        target_map
                    )
                    updated_state = current_timestep.state._replace(world=updated_world)
                    updated_timestep = current_timestep._replace(state=updated_state)
                    partition_states[partition_idx]['timestep'] = updated_timestep
        
        print(f"  Total targets reassigned: {targets_reassigned}")
        print(f"=== TARGET ASSIGNMENT COMPLETE ===\n")
        
        return assigned_targets


    def _choose_partition_for_target(self, global_y, global_x, partition_list, partition_states):
        """
        Choose which partition should handle a target in an overlapping region.
        
        Strategies:
        1. Closest agent
        2. Least loaded partition (fewer remaining targets)
        3. First-come-first-served (lowest partition index)
        4. Based on partition efficiency/performance
        """
        # Strategy 1: Assign to partition with closest agent
        min_distance = float('inf')
        chosen_partition = partition_list[0]
        
        for partition_idx in partition_list:
            if partition_states[partition_idx]['status'] != 'active':
                continue
                
            # Get agent position
            agent_state = partition_states[partition_idx]['timestep'].state.agent.agent_state
            agent_pos = agent_state.pos_base
            
            # Calculate distance to target
            # Note: agent position might need coordinate transformation
            partition = self.partitions[partition_idx]
            y_start, x_start, _, _ = partition['region_coords']
            
            # Convert agent local position to global
            agent_global_y = y_start + agent_pos[0]
            agent_global_x = x_start + agent_pos[1]
            
            distance = jnp.sqrt((agent_global_y - global_y)**2 + (agent_global_x - global_x)**2)
            
            if distance < min_distance:
                min_distance = distance
                chosen_partition = partition_idx
        
        return chosen_partition


    def _choose_partition_by_load(self, partition_list, partition_states):
        """
        Alternative strategy: Choose partition with fewer remaining targets.
        """
        min_targets = float('inf')
        chosen_partition = partition_list[0]
        
        for partition_idx in partition_list:
            if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
                target_map = self.partition_target_maps[partition_idx]
                remaining_targets = jnp.sum(target_map == -1)
                
                if remaining_targets < min_targets:
                    min_targets = remaining_targets
                    chosen_partition = partition_idx
        
        return chosen_partition


    def initialize_partition_specific_target_maps_with_exclusive_assignment(self, partition_states):
        """
        Enhanced version that assigns exclusive targets after initialization.
        """
        # First, do the regular initialization
        if not hasattr(self, 'partition_target_maps'):
            self.partition_target_maps = {}
        
        print("Storing partition-specific target maps...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Store the original target map for this partition
                original_target_map = partition_state['timestep'].state.world.target_map.map.copy()
                self.partition_target_maps[partition_idx] = original_target_map
                
                # Count targets for verification
                dig_targets = jnp.sum(original_target_map == -1)
                dump_targets = jnp.sum(original_target_map == 1)
                
                print(f"  Partition {partition_idx}: {dig_targets} dig targets, {dump_targets} dump targets")
        
        # Then assign exclusive targets in overlapping regions
        if self.overlap_regions:
            self.assign_exclusive_targets_in_overlaps(partition_states)

