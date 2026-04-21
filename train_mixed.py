#!/usr/bin/env python3
"""
Training script for mixed-agent environments using a unified network with agent-type conditioning.

================================================================================
CONFIGURATION SYSTEM
================================================================================

Configurations are defined in: configs/training_configs.yaml

This YAML file contains named presets that specify:
- agent_types: Which agents to use (0=excavator, 1=truck, 2=skidsteer)
- action_types: Movement type per agent (0=tracked, 1=wheeled)
- reward_multipliers: Tuning parameters for reward shaping
- maps: Which map datasets to train on
- capacity overrides: truck_capacity, skidsteer_capacity, truck_road_restricted

Available Presets (run `python configs/training_configs.py` for details):
---------------------------------------------------------------------------
  Solo:        solo_excavator, solo_skidsteer
  Two-agent:   excavator_skidsteer, excavator_truck, excavator_truck_roads, dual_excavator
  Three-agent: excavators_truck
  Trench:      trench_excavator
  Wheeled:     wheeled_excavator

Check configs/training_configs.yaml for more details.

================================================================================
QUICK START
================================================================================

# Use a preset configuration
python train_mixed.py --config excavator_truck

# Use a preset with custom name for wandb
python train_mixed.py --config excavator_skidsteer --name "my-experiment"

# Override a specific parameter from the preset
python train_mixed.py --config excavator_truck --transport_relocate_mult 2.5

# List all available presets
python configs/training_configs.py

================================================================================
MANUAL OVERRIDES (without using presets)
================================================================================

# Two agents: excavator + skidsteer (tracked)
python train_mixed.py --agent_types "(0,2)" --action_types "(0,0)"

# Four agents: 2 excavators + 2 skidsteers with mixed movement
python train_mixed.py --agent_types "(0,2,0,2)" --action_types "(0,1,0,1)"

================================================================================
ADDING NEW CONFIGURATIONS
================================================================================

Edit configs/training_configs.yaml to add new presets:

    my_new_config:
      description: My custom training setup
      agent_types: [0, 2, 2]
      action_types: [0, 0, 0]
      reward_multipliers:
        dump_bonus_mult: 0.5
        excavator_relocate_dumped_mult: 0.3
        excavator_relocate_dug_dirt_mult: 1.5
        transport_relocate_mult: 2.0
      maps:
        - path: foundations_dumpzones_v3
          max_steps: 900

================================================================================
REFERENCE
================================================================================

Agent Types:
  0 = Excavator (digs and dumps)
  1 = Truck (transport, road-restricted optional)
  2 = Skidsteer (transport)

Action Types:
  0 = Tracked movement
  1 = Wheeled movement

Reward Multipliers:
  dump_bonus_mult              - Bonus for correct dumping
  excavator_relocate_dumped_mult   - Excavator reward for moving already-dumped dirt
  excavator_relocate_dug_dirt_mult - Excavator reward for moving freshly dug dirt
  transport_relocate_mult      - Transport agent (truck/skidsteer) relocating dirt reward
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig, BatchConfig, Rewards, CurriculumGlobalConfig, RewardsType
from flax.training.train_state import TrainState
import optax
import wandb
import eval_ppo
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm
from functools import partial
from flax.jax_utils import replicate, unreplicate
from flax import struct
import utils.helpers as helpers
from utils.utils_ppo import select_action_ppo, wrap_action, obs_to_model_input, policy
import json
import os
import shutil
import tempfile
from pathlib import Path

# Import the base training infrastructure
from train import get_curriculum_levels, calculate_gae, ppo_update_networks, Transition

jax.config.update("jax_threefry_partitionable", True)

def safe_jax_to_python(value):
    """Safely convert JAX arrays to Python scalars"""
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            # If it's an array with multiple elements, take the first one
            if hasattr(value, 'shape') and value.shape:
                return value.ravel()[0].item()
            else:
                return float(value)
    elif hasattr(value, '__array__'):
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)
    else:
        return value


def _sorted_map_indices(images_dir: Path) -> list[int]:
    indices = []
    for image_path in images_dir.glob("img_*.npy"):
        try:
            indices.append(int(image_path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def _load_optional_array(path: Path):
    if path.exists():
        return np.load(path)
    return None


def _copy_or_fill_array(src: Path | None, dst: Path, fallback_array) -> None:
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    else:
        np.save(dst, fallback_array)


def _maybe_load_single_map_metadata(map_path: Path):
    metadata_file = map_path / "metadata" / "map.json"
    if metadata_file.exists():
        return metadata_file
    flat_metadata_file = map_path / "metadata.json"
    if flat_metadata_file.exists():
        return flat_metadata_file
    return None


def _metadata_has_trench_axes(metadata_path: Path | None) -> bool:
    if metadata_path is None or not metadata_path.exists():
        return False
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        return False
    trench_axes = metadata.get("axes_ABC")
    return isinstance(trench_axes, list) and len(trench_axes) > 0


def _build_mixed_dataset_pool(
    curriculum_levels: list[dict],
    target_map_path: str,
    replay_map_count: int,
    target_map_repeat: int,
) -> tuple[list[dict], str, int]:
    dataset_root = os.getenv("DATASET_PATH", "")
    if not dataset_root:
        raise RuntimeError("DATASET_PATH must be set to build a mixed target-map training pool.")

    if replay_map_count <= 0:
        raise ValueError("replay_map_count must be > 0 when building a mixed dataset pool.")
    if target_map_repeat <= 0:
        raise ValueError("target_map_repeat must be > 0 when building a mixed dataset pool.")
    if not curriculum_levels:
        raise ValueError("curriculum_levels_override must be set when building a mixed dataset pool.")

    target_map_dir = Path(target_map_path).resolve()
    if not target_map_dir.exists():
        raise FileNotFoundError(f"Target map path does not exist: {target_map_dir}")

    target_image = _load_optional_array(target_map_dir / "images" / "img_1.npy")
    if target_image is None:
        target_image = _load_optional_array(target_map_dir / "image.npy")
    if target_image is None:
        raise FileNotFoundError(f"Could not find target-map image data under {target_map_dir}")

    target_occupancy = _load_optional_array(target_map_dir / "occupancy" / "img_1.npy")
    if target_occupancy is None:
        target_occupancy = _load_optional_array(target_map_dir / "occupancy.npy")
    target_dumpability = _load_optional_array(target_map_dir / "dumpability" / "img_1.npy")
    if target_dumpability is None:
        target_dumpability = _load_optional_array(target_map_dir / "dumpability.npy")
    target_distance = _load_optional_array(target_map_dir / "distance" / "img_1.npy")
    if target_distance is None:
        target_distance = _load_optional_array(target_map_dir / "distance.npy")
    if target_occupancy is None or target_dumpability is None or target_distance is None:
        raise FileNotFoundError(
            f"Target map at {target_map_dir} is missing occupancy, dumpability, or distance data."
        )
    target_actions = _load_optional_array(target_map_dir / "actions" / "img_1.npy")
    if target_actions is None:
        target_actions = _load_optional_array(target_map_dir / "actions.npy")
    if target_actions is None:
        target_actions = np.zeros_like(target_image)
    target_metadata = _maybe_load_single_map_metadata(target_map_dir)
    target_metadata_has_trench_axes = _metadata_has_trench_axes(target_metadata)

    temp_root = Path(tempfile.mkdtemp(prefix="terra_mixed_pool_", dir="/tmp"))
    mixed_levels = []
    mixed_pool_size = replay_map_count + target_map_repeat

    for level_idx, level in enumerate(curriculum_levels):
        source_dir = Path(dataset_root) / level["maps_path"]
        if not source_dir.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {source_dir}")

        indices = _sorted_map_indices(source_dir / "images")
        if not indices:
            raise RuntimeError(f"No dataset maps found under {source_dir / 'images'}")
        selected_indices = indices[-min(replay_map_count, len(indices)) :]
        if len(selected_indices) < replay_map_count:
            selected_indices.extend([selected_indices[-1]] * (replay_map_count - len(selected_indices)))

        level_dir = temp_root / f"level_{level_idx}"
        for subdir in ["images", "occupancy", "dumpability", "distance", "actions", "metadata"]:
            (level_dir / subdir).mkdir(parents=True, exist_ok=True)

        dataset_has_actions = (source_dir / "actions").exists()
        dataset_metadata_files = [source_dir / "metadata" / f"trench_{i}.json" for i in selected_indices]
        keep_metadata = target_metadata_has_trench_axes and all(
            path.exists() for path in dataset_metadata_files
        )

        out_idx = 1
        for src_idx in selected_indices:
            image_path = source_dir / "images" / f"img_{src_idx}.npy"
            occupancy_path = source_dir / "occupancy" / f"img_{src_idx}.npy"
            dumpability_path = source_dir / "dumpability" / f"img_{src_idx}.npy"
            distance_path = source_dir / "distance" / f"img_{src_idx}.npy"
            if not all(path.exists() for path in [image_path, occupancy_path, dumpability_path, distance_path]):
                raise FileNotFoundError(f"Dataset map {src_idx} in {source_dir} is incomplete.")

            shutil.copy2(image_path, level_dir / "images" / f"img_{out_idx}.npy")
            shutil.copy2(occupancy_path, level_dir / "occupancy" / f"img_{out_idx}.npy")
            shutil.copy2(dumpability_path, level_dir / "dumpability" / f"img_{out_idx}.npy")
            shutil.copy2(distance_path, level_dir / "distance" / f"img_{out_idx}.npy")

            if dataset_has_actions:
                _copy_or_fill_array(
                    source_dir / "actions" / f"img_{src_idx}.npy",
                    level_dir / "actions" / f"img_{out_idx}.npy",
                    np.zeros_like(target_image),
                )
            else:
                np.save(level_dir / "actions" / f"img_{out_idx}.npy", np.zeros_like(target_image))

            if keep_metadata:
                shutil.copy2(
                    source_dir / "metadata" / f"trench_{src_idx}.json",
                    level_dir / "metadata" / f"trench_{out_idx}.json",
                )
            out_idx += 1

        for _ in range(target_map_repeat):
            np.save(level_dir / "images" / f"img_{out_idx}.npy", target_image)
            np.save(level_dir / "occupancy" / f"img_{out_idx}.npy", target_occupancy)
            np.save(level_dir / "dumpability" / f"img_{out_idx}.npy", target_dumpability)
            np.save(level_dir / "distance" / f"img_{out_idx}.npy", target_distance)
            np.save(level_dir / "actions" / f"img_{out_idx}.npy", target_actions)
            if keep_metadata:
                shutil.copy2(
                    target_metadata,
                    level_dir / "metadata" / f"trench_{out_idx}.json",
                )
            out_idx += 1

        if not keep_metadata:
            shutil.rmtree(level_dir / "metadata")

        mixed_level = dict(level)
        mixed_level["maps_path"] = f"level_{level_idx}"
        mixed_levels.append(mixed_level)

    return mixed_levels, str(temp_root), mixed_pool_size


@dataclass 
class MixedAgentTrainConfig:
    """Configuration for training mixed agent environments
    
    Supports loading from named presets via --config <name>.
    See configs/training_configs.py for available presets.
    """
    name: str
    num_devices: int = 0
    project: str = "mixed-agents"
    group: str = "tracked-skidsteer"
    num_envs_per_device: int = 2048
    num_steps: int = 32  
    update_epochs: int = 2 
    num_minibatches: int = 16 
    total_timesteps: int = 50_000_000_000  
    lr: float = 3e-4   
    clip_eps: float = 0.2 
    gamma: float = 0.9984 
    gae_lambda: float = 0.95
    ent_coef: float = 0.06  
    vf_coef: float = 2.0 
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = 100
    checkpoint_interval: int = 100
    
    # Model settings optimized for mixed agents
    num_prev_actions = 10  # will be overridden to 5 * num_agents at runtime
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    maps_net_normalization_bounds = [-10, 10]  # Required field for network initialization
    loaded_max = 100
    num_rollouts_eval = 200  # max length of an episode in Terra for eval
    cache_clear_interval = 1000  # Less frequent cache clearing for speed
    # Entropy scheduler (cosine decay)
    ent_schedule_start: float = 0.15
    ent_schedule_end: float = 0.005
    ent_schedule_steps: int = 9500
    
    # Removed agent-type curriculum; use override only
    # Optional override to specify an arbitrary list of agent types, e.g. (2,0,0,2)
    agent_types_override: tuple | None = None
    # Optional override to specify action types for each agent, e.g. (0,1,0,1) for tracked/wheeled
    action_types_override: tuple | None = None
    # Debug assertions and one-time validations
    debug: bool = False
    
    # Checkpoint loading
    resume_from: str | None = None  # Path to a checkpoint .pkl to resume from
    load_env_from_checkpoint: bool = True  # If true, use env_config from checkpoint
    
    # Named configuration preset (loads from configs/training_configs.py)
    config_name: str | None = None  # e.g., "excavator_truck", "solo_excavator"
    
    # Reward multipliers (can be set via config preset or CLI)
    dump_bonus_mult: float | None = None
    excavator_relocate_dumped_mult: float | None = None
    excavator_relocate_dug_dirt_mult: float | None = None
    transport_relocate_mult: float | None = None
    
    # Capacity overrides
    truck_capacity: int | None = None
    skidsteer_capacity: int | None = None
    truck_road_restricted: bool | None = None
    
    # Curriculum/maps override (from YAML config)
    # Format: list of dicts with keys: maps_path, max_steps_in_episode, rewards_type, apply_trench_rewards
    curriculum_levels_override: list | None = None
    # Optional single-map training path. When set, map loading uses this path directly
    # and does not rely on DATASET_PATH / DATASET_SIZE.
    single_map_path: str | None = None
    # Mixed fine-tuning mode: sample recent config maps and oversample the target map.
    replay_map_count: int = 0
    target_map_repeat: int = 0


    def __post_init__(self):
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.num_updates = (
            self.total_timesteps // (self.num_steps * self.num_envs)
        ) // self.num_devices

        print(f"Devices: {jax.devices()}")
        print(f"Mixed Agent Training - Devices: {self.num_devices}, Updates: {self.num_updates}")
        print(f"Using overridden agent types: {self.agent_types_override}")

    # make object subscriptable - required for compatibility with existing code
    def __getitem__(self, key):
        return getattr(self, key)


def create_mixed_agent_env_config(
    agent_types=(0, 2), 
    action_types=(0, 0),
    # Optional reward multipliers
    dump_bonus_mult=None,
    excavator_relocate_dumped_mult=None,
    excavator_relocate_dug_dirt_mult=None,
    transport_relocate_mult=None,
    # Optional capacity overrides
    truck_capacity=None,
    skidsteer_capacity=None,
    truck_road_restricted=None,
):
    """Create environment configuration optimized for mixed agent training
    
    Args:
        agent_types: Tuple of agent type IDs (0=excavator, 1=truck, 2=skidsteer)
        action_types: Tuple of action type IDs (0=tracked, 1=wheeled)
        dump_bonus_mult: Multiplier for dump rewards
        excavator_relocate_dumped_mult: Multiplier for excavator relocating dumped material
        excavator_relocate_dug_dirt_mult: Multiplier for excavator relocating dug dirt
        transport_relocate_mult: Multiplier for transport relocation rewards
        truck_capacity: Override for truck capacity
        skidsteer_capacity: Override for skidsteer capacity
        truck_road_restricted: Whether trucks are restricted to roads
    """
    
    # Use the existing dense rewards from config
    env_config = EnvConfig()  # This automatically uses Rewards.dense() which includes all our rewards
    
    # Set the agent types from the training configuration
    env_config = env_config._replace(agent_types=agent_types)
    
    # Set the action types from the training configuration
    env_config = env_config._replace(action_types=action_types)
    
    # Apply reward multipliers if provided
    if dump_bonus_mult is not None:
        env_config = env_config._replace(dump_bonus_mult=dump_bonus_mult)
    if excavator_relocate_dumped_mult is not None:
        env_config = env_config._replace(excavator_relocate_dumped_mult=excavator_relocate_dumped_mult)
    if excavator_relocate_dug_dirt_mult is not None:
        env_config = env_config._replace(excavator_relocate_dug_dirt_mult=excavator_relocate_dug_dirt_mult)
    if transport_relocate_mult is not None:
        env_config = env_config._replace(transport_relocate_mult=transport_relocate_mult)
    
    # Apply capacity overrides if provided
    if truck_capacity is not None:
        env_config = env_config._replace(truck_capacity=truck_capacity)
    if skidsteer_capacity is not None:
        env_config = env_config._replace(skidsteer_capacity=skidsteer_capacity)
    if truck_road_restricted is not None:
        env_config = env_config._replace(truck_road_restricted=truck_road_restricted)
    
    return env_config


class ConfigurableAgentManager:
    """Simplified: agent types come only from override or defaults."""
    
    def __init__(self, config: MixedAgentTrainConfig):
        self.config = config
    
    def get_current_agent_types(self, *_, **__) -> tuple[int, int]:
        if self.config.agent_types_override is not None:
            ats = tuple(self.config.agent_types_override)
        else:
            ats = EnvConfig().agent_types
        # Ensure we always return a 2-tuple for prints; extra types still supported elsewhere
        if len(ats) >= 2:
            return (int(ats[0]), int(ats[1]))
        if len(ats) == 1:
            return (int(ats[0]), int(ats[0]))
        return (0, 2)
    
    def get_current_action_types(self, *_, **__) -> tuple[int, int]:
        if self.config.action_types_override is not None:
            ats = tuple(self.config.action_types_override)
        else:
            # Default to tracked actions (0) for all agents
            ats = (0, 0)
        # Ensure we always return a 2-tuple for prints; extra types still supported elsewhere
        if len(ats) >= 2:
            return (int(ats[0]), int(ats[1]))
        if len(ats) == 1:
            return (int(ats[0]), int(ats[0]))
        return (0, 0)




def make_mixed_agent_states(config: MixedAgentTrainConfig, env_params: EnvConfig = None, env_params_override: EnvConfig = None):
    """Initialize states for mixed agent training - compatible with make_states interface"""
    curriculum_levels = config.curriculum_levels_override
    single_map_path = config.single_map_path

    if (
        single_map_path is not None
        and config.replay_map_count > 0
        and config.target_map_repeat > 0
    ):
        curriculum_levels, mixed_dataset_root, mixed_pool_size = _build_mixed_dataset_pool(
            curriculum_levels=curriculum_levels,
            target_map_path=single_map_path,
            replay_map_count=config.replay_map_count,
            target_map_repeat=config.target_map_repeat,
        )
        os.environ["DATASET_PATH"] = mixed_dataset_root
        os.environ["DATASET_SIZE"] = str(mixed_pool_size)
        single_map_path = None
        print(
            "📍 Using mixed target-map pool: "
            f"{config.replay_map_count} recent maps + "
            f"{config.target_map_repeat} target-map repeats per curriculum level"
        )
        print(f"📍 Mixed dataset root: {mixed_dataset_root}")

    # Create batch config - override curriculum levels if provided
    if curriculum_levels is not None and len(curriculum_levels) > 0:
        # Create custom CurriculumGlobalConfig with the override levels
        custom_curriculum = CurriculumGlobalConfig()
        # We need to create a new class with the overridden levels since it's a NamedTuple
        # NamedTuples can't have mutable class attributes overridden, so we work around this
        class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
            levels = curriculum_levels
        
        batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())
        print(f"📍 Using maps from config: {[lvl['maps_path'] for lvl in curriculum_levels]}")
    else:
        batch_cfg = BatchConfig()
        print("📍 Using default maps from config.py")
    
    # Initialize environment with configurable agents
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        shuffle_maps=False,
        single_map_path=single_map_path,
    )
    if single_map_path is not None:
        print(f"📍 Using single map path: {single_map_path}")
    
    # Get environment parameters with agent types from config
    if env_params is None:
        if env_params_override is not None:
            # Use environment config from checkpoint
            env_params = env_params_override
            print("Using environment config from checkpoint")
        else:
            # Use override if provided; else default EnvConfig agent_types
            if config.agent_types_override is not None:
                agent_types = tuple(config.agent_types_override)
            else:
                agent_types = EnvConfig().agent_types
            
            # Use action types override if provided, otherwise use default (0,0)
            action_types = config.action_types_override if config.action_types_override is not None else (0, 0)
            env_params = create_mixed_agent_env_config(
                agent_types=agent_types, 
                action_types=action_types,
                # Pass reward multipliers from config
                dump_bonus_mult=config.dump_bonus_mult,
                excavator_relocate_dumped_mult=config.excavator_relocate_dumped_mult,
                excavator_relocate_dug_dirt_mult=config.excavator_relocate_dug_dirt_mult,
                transport_relocate_mult=config.transport_relocate_mult,
                # Pass capacity overrides
                truck_capacity=config.truck_capacity,
                skidsteer_capacity=config.skidsteer_capacity,
                truck_road_restricted=config.truck_road_restricted,
            )
            # Verbose training configuration summary
            type_names = {0: "Excavator", 1: "Truck", 2: "SkidSteer"}
            print("🧩 Agent Types (effective):", agent_types)
            print("🧩 Agent Types (names):", 
                  " + ".join(type_names.get(t, f"Unknown({t})") for t in agent_types))
            if config.agent_types_override is not None:
                print("✅ Using --agent_types override")
            
            # Print action types information
            action_type_names = {0: "Tracked", 1: "Wheeled"}
            print("🚗 Action Types (effective):", action_types)
            print("🚗 Action Types (names):", 
                  " + ".join(action_type_names.get(t, f"Unknown({t})") for t in action_types))
            if config.action_types_override is not None:
                print("✅ Using --action_types override")
            else:
                print("🚗 Using default action types (all tracked)")
            
            # Print reward multipliers if any were set
            if any([config.dump_bonus_mult, config.excavator_relocate_dumped_mult,
                    config.excavator_relocate_dug_dirt_mult, config.transport_relocate_mult]):
                print("📊 Reward Multipliers:")
                if config.dump_bonus_mult is not None:
                    print(f"   dump_bonus_mult: {config.dump_bonus_mult}")
                if config.excavator_relocate_dumped_mult is not None:
                    print(f"   excavator_relocate_dumped_mult: {config.excavator_relocate_dumped_mult}")
                if config.excavator_relocate_dug_dirt_mult is not None:
                    print(f"   excavator_relocate_dug_dirt_mult: {config.excavator_relocate_dug_dirt_mult}")
                if config.transport_relocate_mult is not None:
                    print(f"   transport_relocate_mult: {config.transport_relocate_mult}")
    
    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    print("⏱️  Batching env_params...", flush=True)
    t_env_params = time.time()
    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None, None]
        .repeat(num_devices, 0)
        .repeat(num_envs_per_device, 1),
        env_params,
    )
    print(
        f"⏱️  Batching env_params done in {time.time() - t_env_params:.2f}s",
        flush=True,
    )
    
    print(f"Mixed Agent Environment - Tile size shape: {env_params.tile_size.shape}", flush=True)

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    # Infer num_prev_actions as 5 per agent without triggering a reset/pmap
    try:
        MAX_AGENTS = 4
        # Priority 1: explicit override
        if config.agent_types_override is not None:
            na = len(tuple(config.agent_types_override))
        # Priority 2: env params we just batched include agent_types with trailing num_agents dim
        elif hasattr(env_params, 'agent_types') and hasattr(env_params.agent_types, 'shape'):
            na = int(env_params.agent_types.shape[-1])
        # Priority 3: batch config on env, if present as a tuple/list
        elif hasattr(env.batch_cfg, 'agent_types') and isinstance(env.batch_cfg.agent_types, (tuple, list)):
            na = len(env.batch_cfg.agent_types)
        else:
            na = MAX_AGENTS
        na = max(1, min(MAX_AGENTS, int(na)))
        config.num_prev_actions = int(5 * na)
        print(f"Setting num_prev_actions to {config.num_prev_actions} (5 per agent × {na} agents)", flush=True)
    except Exception as e:
        print(f"Warning: failed to infer num_agents for num_prev_actions ({e}); keeping {config.num_prev_actions}", flush=True)

    # Create the unified network with agent type features (now that num_prev_actions is set)
    print("⏱️  Initializing model...", flush=True)
    t_model_init = time.time()
    network, network_params = get_model_ready(_rng, config, env)
    print(
        f"⏱️  Model init done in {time.time() - t_model_init:.2f}s",
        flush=True,
    )
    # Debug: print number of actions for current action type (kept as requested)
    try:
        num_actions_debug = env.batch_cfg.action_type.get_num_actions()
        print(f"🛠️ Debug: Number of actions = {num_actions_debug}", flush=True)
    except Exception as e:
        print(f"🛠️ Debug: Failed to read number of actions: {e}", flush=True)
    
    # Optimizer with mixed agent considerations
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.lr, eps=1e-5),
    )
    
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )
    
    
    print(f"Network: Unified with agent type conditioning", flush=True)
    
    return rng, env, env_params, train_state


def train_mixed_agents(config: MixedAgentTrainConfig):
    """Main training function for mixed agents - with full feature parity to original train.py"""
    
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
        tags=["mixed-agents", "tracked-excavator", "skid-steer", "unified-network"]
    )
    
    # Log source files - same as original train.py
    train_py_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terra", "terra", "config.py")
    models_path = os.path.join(os.path.dirname(__file__), "utils", "models.py")
    
    code_artifact = wandb.Artifact(name="mixed_agent_source_code", type="code")
    
    for file_path, name in [(train_py_path, "train_mixed_agents.py"), 
                           (config_path, "config.py"),
                           (models_path, "models.py")]:
        if os.path.exists(file_path):
            code_artifact.add_file(file_path, name=name)
    
    if code_artifact.files:
        run.log_artifact(code_artifact)

    # Optionally load checkpoint before creating states
    checkpoint = None
    env_params_override = None
    if config.resume_from is not None and os.path.exists(config.resume_from):
        try:
            checkpoint = helpers.load_pkl_object(config.resume_from)
            if config.load_env_from_checkpoint and "env_config" in checkpoint:
                env_params_override = checkpoint["env_config"]
            print(f"Loaded checkpoint from {config.resume_from}")
        except Exception as e:
            print(f"Failed to load checkpoint from {config.resume_from}: {e}")

    # Initialize training components (optionally with env override)
    rng, env, env_params, train_state = make_mixed_agent_states(
        config, env_params_override=env_params_override
    )

    # If checkpoint has model params, overwrite initialized params
    if checkpoint is not None and "model" in checkpoint:
        try:
            train_state = train_state.replace(params=checkpoint["model"])
            print("Replaced model parameters from checkpoint.")
        except Exception as e:
            print(f"Failed to set model parameters from checkpoint: {e}")

    # Removed agent-type curriculum monitoring
    
    def log_environment_metrics(timestep, update_num):
        """Log environment metrics for all mixed agent training"""
        try:
            # Basic episode metrics
            episode_done = timestep.done
            completion_rate = safe_jax_to_python(jnp.mean(episode_done))
            
            # Log the metrics - ensure step is always positive and increasing
            if update_num > 0:  # Only log if we have a valid step number
                wandb.log({
                    "progress/episode_completion_rate": completion_rate,
                }, step=update_num)
                
        except Exception as e:
            # Log the error but don't crash the training
            print(f"⚠️  Warning: Failed to log environment metrics at step {update_num}: {e}")
            # Optionally log a minimal set of metrics without step to avoid the warning
            try:
                wandb.log({
                    "progress/episode_completion_rate": 0.0,
                })
            except:
                pass  # If even this fails, just continue training
    
    def make_mixed_agent_train(env, env_params, config):
        def train(rng: jax.Array, train_state: TrainState):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(
                _rng, config.num_envs_per_device * config.num_devices
            )
            reset_rng = reset_rng.reshape(
                (config.num_devices, config.num_envs_per_device, -1)
            )

            # TERRA: Reset envs
            (
                env_params_reset,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
            ) = env.prepare_reset(env_params, reset_rng)
            reset_fn_p = jax.pmap(env.reset_prepared, axis_name="devices")
            timestep = reset_fn_p(
                env_params_reset,
                reset_rng,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
            )
            # Removed one-time debug sanity prints
            
            # Initialize reward_components in timestep.info to maintain consistent pytree structure
            # This prevents JAX scan errors when reward_components is added later
            if hasattr(timestep, 'info') and isinstance(timestep.info, dict):
                # Add empty reward_components to match the structure produced in env.step/state._get_reward
                # Shapes follow timestep.reward's batch shape; agent vectors add a MAX_AGENTS axis (4)
                batch_shape = timestep.reward.shape
                MAX_AGENTS = 4
                dummy_components = {
                    "agent_rewards": jnp.zeros(batch_shape + (MAX_AGENTS,), dtype=jnp.float32),
                    "agent_active": jnp.zeros(batch_shape + (MAX_AGENTS,), dtype=jnp.int32),
                    "num_agents": jnp.zeros(batch_shape, dtype=jnp.int32),
                    "terminal": jnp.zeros_like(timestep.reward),
                    "trench": jnp.zeros_like(timestep.reward),
                    "existence": jnp.zeros_like(timestep.reward),
                }
                # Create new timestep with reward_components added to info
                timestep = timestep._replace(
                    info={**timestep.info, "reward_components": dummy_components}
                )
            prev_actions = jnp.zeros(
                (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
            )
            prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))

            # TRAIN LOOP
            @partial(jax.pmap, axis_name="devices")
            def _update_step(runner_state, ent_coef_current, update_idx):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, step_idx):
                    rng, train_state, prev_timestep, prev_actions, prev_reward = runner_state

                    # SELECT ACTION
                    rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                    action, log_prob, value, _ = select_action_ppo(
                        train_state, prev_timestep.observation, prev_actions, _rng_model, config
                    )

                    # STEP ENV
                    _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                    action_env = wrap_action(action, env.batch_cfg.action_type)
                    timestep = env.step(prev_timestep, action_env, _rng_env)

                    # Removed SWAP debug prints
                    transition = Transition(
                        done=timestep.done,
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_actions=prev_actions,
                        prev_reward=prev_reward,
                    )

                    # UPDATE PREVIOUS ACTIONS
                    prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
                    prev_actions = prev_actions.at[..., 0].set(action)

                    runner_state = (rng, train_state, timestep, prev_actions, timestep.reward)
                    return runner_state, transition

                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(
                    _env_step, runner_state, jnp.arange(config.num_steps), config.num_steps
                )

                # Distribute terminal reward to the last num_agents alternating steps
                # This attributes the shared terminal to all agents that acted in the final round
                done_seq = transitions.done            # [seq, batch]
                reward_seq = transitions.reward        # [seq, batch]
                # Terminal bonus only on terminal steps
                terminal_bonus = jnp.where(done_seq, reward_seq, 0.0)  # [seq, batch]

                # Get num_agents per env (assumed constant across sequence); shape [batch]
                # transitions.obs stores prev_timestep.observation
                num_agents_per_env = transitions.obs["num_agents"][0]  # [batch]
                # Clip to supported window 1..MAX_AGENTS
                MAX_AGENTS = 4
                num_agents_per_env = jnp.clip(num_agents_per_env.astype(jnp.int32), 1, MAX_AGENTS)

                # Build windowed backfill: sum of shifts 0..K-1, where K=num_agents_per_env
                # shifted[k] = terminal_bonus shifted backward by k (zeros for first k rows)
                def shift_back(arr, k):
                    zeros = jnp.zeros_like(arr[:k])
                    return jnp.concatenate([zeros, arr[:-k]], axis=0) if k > 0 else arr

                shifted_stack = []
                # Start at k=1 to avoid including the unshifted terminal bonus (no double-count)
                for k in range(1, MAX_AGENTS):
                    shifted_k = shift_back(terminal_bonus, k)  # [seq, batch]
                    # Mask this shift per env if k < num_agents
                    use_k = (num_agents_per_env > k).astype(jnp.float32)  # [batch]
                    shifted_k = shifted_k * use_k  # broadcast over seq
                    shifted_stack.append(shifted_k)
                window_sum = jnp.stack(shifted_stack, axis=0).sum(axis=0)  # [seq, batch]

                # With k starting at 1, we don't include the terminal step itself in the sum
                augmented_reward = reward_seq + window_sum
                transitions = transitions.replace(reward=augmented_reward)

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_actions, prev_reward = runner_state
                rng, _rng = jax.random.split(rng)
                _, _, last_val, _ = select_action_ppo(
                    train_state, timestep.observation, prev_actions, _rng, config
                )
                advantages, targets = calculate_gae(
                    transitions, last_val, config.gamma, config.gae_lambda
                )

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            advantages=advantages,
                            targets=targets,
                            config=config,
                            ent_coef_override=ent_coef_current,
                        )
                        return new_train_state, update_info

                    rng, train_state, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    # [num_minibatches, minibatch_size, seq_len, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(
                            x, (config.num_minibatches, -1) + x.shape[1:]
                        ),
                        shuffled_batch,
                    )
                    train_state, update_info = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )

                    update_state = (rng, train_state, transitions, advantages, targets)
                    return update_state, update_info

                # [seq_len, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config.update_epochs
                )

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

                # Explained variance between value predictions and returns
                # Use transitions and targets from current update_state (first device in pmap)
                _, _, transitions_ev, _, targets_ev = update_state
                vpred = transitions_ev.value
                vtrue = targets_ev
                vpred_flat = vpred.reshape(-1)
                vtrue_flat = vtrue.reshape(-1)
                var_y = jnp.var(vtrue_flat)
                explained_var = 1 - jnp.var(vtrue_flat - vpred_flat) / (var_y + 1e-8)
                # Attach to loss_info for logging
                loss_info = dict(loss_info)
                loss_info["explained_variance"] = explained_var

                rng, train_state = update_state[:2]
                # EVALUATE AGENT
                rng, _rng = jax.random.split(rng)

                runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
                return runner_state, loss_info

            # Setup runner state for multiple devices
            rng, rng_rollout = jax.random.split(rng)
            rng = jax.random.split(rng, num=config.num_devices)
            train_state = replicate(train_state, jax.local_devices()[: config.num_devices])
            runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
            
            # Entropy scheduler: cosine decay using config variables
            ent_start = float(config.ent_schedule_start)
            ent_end = float(config.ent_schedule_end)
            ent_T = float(config.ent_schedule_steps)

            for i in tqdm(range(config.num_updates), desc="Training"):
                f = min(1.0, i / ent_T) if ent_T > 0 else 1.0
                # Cosine decay: starts at ent_start when f=0, ends at ent_end when f=1
                ent_coef_current = ent_end + 0.5 * (ent_start - ent_end) * (1.0 + jnp.cos(jnp.pi * f))
                # Linear decay from ent_start to ent_end over ent_T updates
                # ent_coef_current = ent_start + (ent_end - ent_start) * f
                # Broadcast scalar to devices for pmap input
                ent_broadcast = jnp.array([ent_coef_current] * config.num_devices)
                start_time = time.time()
                runner_state, loss_info = jax.block_until_ready(
                    _update_step(runner_state, ent_broadcast, jnp.array([i] * config.num_devices))
                )
                end_time = time.time()

                iteration_duration = end_time - start_time
                iterations_per_second = 1 / iteration_duration
                steps_per_second = (
                    iterations_per_second
                    * config.num_steps
                    * config.num_envs
                    * config.num_devices
                )

                tqdm.write(f"Steps/s: {steps_per_second:.2f}")

                # Use data from the first device for stats and eval
                loss_info_single = unreplicate(loss_info)
                runner_state_single = unreplicate(runner_state)
                _, train_state, timestep, prev_actions = runner_state_single[:4]
                env_params_single = timestep.env_cfg

                if i % config.log_train_interval == 0:
                    # Consolidated logging to prevent step ordering issues
                    curriculum_levels = get_curriculum_levels(
                        env_params_single, env.batch_cfg.curriculum_global.levels
                    )
                    
                    # Start with base metrics
                    log_dict = {
                        "performance/steps_per_second": steps_per_second,
                        "performance/iterations_per_second": iterations_per_second,
                        "curriculum_levels": curriculum_levels,
                        "lr": config.lr,
                        "sched/entropy_coef": float(ent_coef_current),
                        **loss_info_single,
                    }
                    
                    # Removed fixed agent1/agent2 type metrics to support dynamic agent counts
                    
                    # Add environment metrics (without separate wandb.log call)
                    try:
                        completion_rate = safe_jax_to_python(jnp.mean(timestep.done))
                        log_dict["progress/episode_completion_rate"] = completion_rate
                    except Exception:
                        pass

                    # Add reward breakdown logging (without separate wandb.log call)
                    try:
                        reward_components = None
                        if hasattr(timestep, "reward_components"):
                            reward_components = timestep.reward_components
                        elif hasattr(timestep, "info") and isinstance(timestep.info, dict):
                            rc = timestep.info.get("reward_components", None)
                            reward_components = rc

                        if reward_components is not None:
                            # Support per-agent rewards vector and masks
                            breakdown_means = {}
                            agent_rewards = reward_components.get("agent_rewards", None)
                            agent_active = reward_components.get("agent_active", None)
                            num_agents = reward_components.get("num_agents", None)
                            # Scalar components
                            for k in ["terminal", "trench", "existence"]:
                                if k in reward_components:
                                    breakdown_means[k] = safe_jax_to_python(reward_components[k])
                            # Vector per-agent rewards
                            if agent_rewards is not None:
                                try:
                                    ar = agent_rewards
                                    # Log each available agent index separately
                                    for idx in range(ar.shape[-1]):
                                        key = f"agent_{idx}"
                                        breakdown_means[f"{key}"] = safe_jax_to_python(jnp.mean(ar[..., idx]))
                                except Exception:
                                    pass
                            # Add to main log dict
                            for k, v in breakdown_means.items():
                                log_dict[f"rewards/{k}"] = v
                            # Also log masks if present
                            if agent_active is not None:
                                try:
                                    log_dict["agents/active_count"] = safe_jax_to_python(jnp.sum(agent_active))
                                except Exception:
                                    pass
                            if num_agents is not None:
                                try:
                                    log_dict["agents/num_agents"] = safe_jax_to_python(num_agents)
                                except Exception:
                                    pass
                            
                            
                            
                                
                            # Skip bar chart for now to avoid step conflicts
                    except Exception:
                        pass
                    
                    # Single consolidated wandb.log call
                    wandb.log(log_dict, step=i)

                if i % config.checkpoint_interval == 0:
                    checkpoint = {
                        "train_config": config,
                        "env_config": env_params_single,
                        "model": runner_state_single[1].params,
                        "loss_info": loss_info_single,
                    }
                    helpers.save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")

                if i % config.log_eval_interval == 0:
                    # Eval runs as pmap across all devices. Feed replicated train_state
                    # and env_params (pulled from the unreduced runner_state) and a
                    # per-device rng. Results come back with a leading device axis; we
                    # aggregate across devices host-side so reward/action % are global.
                    train_state_replicated = runner_state[1]
                    env_params_replicated = runner_state[2].env_cfg
                    rng_rollout_per_device = jax.random.split(
                        rng_rollout, num=config.num_devices
                    )
                    eval_stats_per_device = eval_ppo.rollout(
                        rng_rollout_per_device,
                        env,
                        env_params_replicated,
                        train_state_replicated,
                        config,
                    )
                    # Combine per-device stats: sums for counts/rewards, max/min for
                    # extrema, and keep length from device 0 (identical across devices).
                    eval_stats = eval_stats_per_device._replace(
                        max_reward=eval_stats_per_device.max_reward.max(),
                        min_reward=eval_stats_per_device.min_reward.min(),
                        reward=eval_stats_per_device.reward.sum(),
                        length=eval_stats_per_device.length[0],
                        episodes=eval_stats_per_device.episodes.sum(),
                        positive_terminations=eval_stats_per_device.positive_terminations.sum(),
                        terminations=eval_stats_per_device.terminations.sum(),
                        positive_terminations_steps=eval_stats_per_device.positive_terminations_steps.sum(),
                        action_0=eval_stats_per_device.action_0.sum(),
                        action_1=eval_stats_per_device.action_1.sum(),
                        action_2=eval_stats_per_device.action_2.sum(),
                        action_3=eval_stats_per_device.action_3.sum(),
                        action_4=eval_stats_per_device.action_4.sum(),
                        action_5=eval_stats_per_device.action_5.sum(),
                        action_6=eval_stats_per_device.action_6.sum(),
                        action_7=eval_stats_per_device.action_7.sum(),
                    )

                    # Total envs that contributed to the sums across all devices.
                    n = config.num_envs_per_device * config.num_devices * eval_stats.length
                    avg_positive_episode_length = jnp.where(
                        eval_stats.positive_terminations > 0,
                        eval_stats.positive_terminations_steps / eval_stats.positive_terminations,
                        jnp.zeros_like(eval_stats.positive_terminations_steps)
                    )
                    loss_info_single.update(
                        {
                            "eval/rewards": eval_stats.reward / n,
                            "eval/max_reward": eval_stats.max_reward,
                            "eval/min_reward": eval_stats.min_reward,
                            "eval/lengths": eval_stats.length,
                            "eval/FORWARD %": eval_stats.action_0 / n,
                            "eval/BACKWARD %": eval_stats.action_1 / n,
                            "eval/CLOCK %": eval_stats.action_2 / n,
                            "eval/ANTICLOCK %": eval_stats.action_3 / n,
                            "eval/CABIN_CLOCK %": eval_stats.action_4 / n,
                            "eval/CABIN_ANTICLOCK %": eval_stats.action_5 / n,
                            "eval/DO": eval_stats.action_6 / n,
                            "eval/DO_NOTHING %": eval_stats.action_7 / n,
                            "eval/positive_terminations": eval_stats.positive_terminations
                            / (config.num_envs_per_device * config.num_devices),
                            "eval/total_terminations": eval_stats.terminations
                            / (config.num_envs_per_device * config.num_devices),
                            "eval/avg_positive_episode_length": avg_positive_episode_length
                        }
                    )

                    wandb.log(loss_info_single)

                # Clear JAX caches and run garbage collection to stabilize memory use
                if i % config.cache_clear_interval == 0:
                    jax.clear_caches()
                    import gc
                    gc.collect()

            return {"runner_state": runner_state_single, "loss_info": loss_info_single}

        return train
    
    train_fn = make_mixed_agent_train(env, env_params, config)
    
    def train_with_monitoring(rng, train_state):
        return train_fn(rng, train_state)

    
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Environments per device: {config.num_envs_per_device}")
    print(f"   - Total environments: {config.num_envs}")
    print(f"   - Training steps: {config.num_steps}")
    print(f"   - Total timesteps: {config.total_timesteps:,}")
    print(f"   - Learning rate: {config.lr}")
    
    print("=" * 60)
    print("🚀 Starting Mixed Agent Training...")
    print("⚙️  JAX is now compiling the control-flow graph. This is normal and taking a few minutes...", flush=True)

    try:
        t = time.time()
        train_info = jax.block_until_ready(train_with_monitoring(rng, train_state))
        elapsed_time = time.time() - t
        print(f"✅ Mixed agent training completed in {elapsed_time:.2f}s")
        
        # Save final checkpoint with special naming - enhanced metadata
        try:
            at_final = train_info["runner_state"][2].env_cfg.agent_types
            if hasattr(at_final, 'shape') and len(at_final.shape) > 1:
                a1 = int(jnp.mean(at_final[0, :, 0]))
                a2 = int(jnp.mean(at_final[0, :, 1]))
            else:
                a1 = int(at_final[0])
                a2 = int(at_final[1])
            type_names = {0: "Excavator", 1: "Truck", 2: "SkidSteer"}
            agent_types_str = f"{type_names.get(a1, 'unknown')}_{type_names.get(a2, 'unknown')}"
        except Exception:
            agent_types_str = "unknown_unknown"
        
        final_checkpoint = {
            "train_config": config,
            "env_config": train_info["runner_state"][2].env_cfg,  # timestep.env_cfg
            "model": train_info["runner_state"][1].params,
            "loss_info": train_info["loss_info"],
            "agent_types": agent_types_str,
            "network_type": "unified_with_agent_type_conditioning",
            "training_duration": elapsed_time,
            "final_reward": train_info.get("final_reward", None)
        }
        helpers.save_pkl_object(final_checkpoint, f"checkpoints/{config.name}_FINAL.pkl")
        print(f"💾 Final mixed agent model saved to checkpoints/{config.name}_FINAL.pkl")
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted. Finalizing...")
    finally:
        run.finish()
        print("📈 Wandb session finished.")


if __name__ == "__main__":
    DT = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    import argparse

    parser = argparse.ArgumentParser(description="Train mixed agent policies (Tracked + Skid Steer)")
    parser.add_argument(
        "-n", "--name", type=str, default="mixed-agents-skidsteer-skidsteer",
        help="Experiment name"
    )
    parser.add_argument(
        "-m", "--machine", type=str, default="local",
        help="Machine identifier"
    )
    parser.add_argument(
        "-d", "--num_devices", type=int, default=0,
        help="Number of devices to use. If 0, uses all available devices."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_envs_per_device", type=int, default=1024,
        help="Number of parallel envs per device"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=50_000_000_000,
        help="Total environment timesteps across all devices"
    )
    parser.add_argument(
        "--agent_types", type=str, default=None,   # 0=excavator, 1=truck, 2=skidsteer
        help="Override agent types with a Python tuple, e.g. '(2,0,2,0)'. Overrides --config."
    )
    parser.add_argument(
        "--action_types", type=str, default=None,
        help="Override action types with a Python tuple, e.g. '(1,)' for wheeled. Overrides --config."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable one-time sanity assertions/prints for agent ordering and masks"
    )
    
    # Named configuration preset
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Load a named training config preset (e.g., 'excavator_truck', 'solo_excavator'). "
             "Run 'python configs/training_configs.py' to see available presets."
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help=(
            "Optional single-map folder/file path for training. "
            "By default this enables pure single-map training. "
            "If --replay_map_count and --target_map_repeat are also set, "
            "the map is mixed with recent config-dataset maps instead."
        ),
    )
    parser.add_argument(
        "--replay_map_count",
        type=int,
        default=0,
        help=(
            "When used with --map_path, keep the last N maps from each config dataset "
            "and mix them with repeated copies of the target map."
        ),
    )
    parser.add_argument(
        "--target_map_repeat",
        type=int,
        default=0,
        help=(
            "When used with --map_path and --replay_map_count, add the target map this many "
            "times to the mixed training pool."
        ),
    )
    
    # Reward multiplier arguments
    parser.add_argument(
        "--dump_bonus_mult", type=float, default=None,
        help="Multiplier for dump rewards (overrides config preset)"
    )
    parser.add_argument(
        "--excavator_relocate_dumped_mult", type=float, default=None,
        help="Multiplier for excavator relocating dumped material (overrides config preset)"
    )
    parser.add_argument(
        "--excavator_relocate_dug_dirt_mult", type=float, default=None,
        help="Multiplier for excavator relocating dug dirt (overrides config preset)"
    )
    parser.add_argument(
        "--transport_relocate_mult", type=float, default=None,
        help="Multiplier for transport relocation rewards (overrides config preset)"
    )
    
    # Checkpoint loading arguments
    parser.add_argument(
        "-r", "--resume_from", type=str, default=None,
        help="Path to a checkpoint .pkl to resume training from."
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--load_env_from_checkpoint",
        dest="load_env_from_checkpoint",
        action="store_true",
        help="Load env_config from the checkpoint (default)."
    )
    env_group.add_argument(
        "--no-load-env-from-checkpoint",
        dest="load_env_from_checkpoint",
        action="store_false",
        help="Do not load env_config from checkpoint; use default/current EnvConfig()."
    )
    
    args, _ = parser.parse_known_args()
    
    # default to True unless explicitly disabled
    if args.load_env_from_checkpoint is None:
        args.load_env_from_checkpoint = True
    
    # Initialize config values from preset if --config is provided
    agent_types_override = None
    action_types_override = None
    dump_bonus_mult = None
    excavator_relocate_dumped_mult = None
    excavator_relocate_dug_dirt_mult = None
    transport_relocate_mult = None
    truck_capacity = None
    skidsteer_capacity = None
    truck_road_restricted = None
    curriculum_levels_override = None
    
    if args.config is not None:
        try:
            from configs.training_configs import get_config, list_configs
            preset = get_config(args.config)
            print(f"\n📦 Loading config preset: '{args.config}'")
            print(f"   Description: {preset.description}")
            
            # Apply preset values
            agent_types_override = preset.agent_types
            action_types_override = preset.action_types
            
            # Apply reward multipliers from preset
            dump_bonus_mult = preset.reward_multipliers.dump_bonus_mult
            excavator_relocate_dumped_mult = preset.reward_multipliers.excavator_relocate_dumped_mult
            excavator_relocate_dug_dirt_mult = preset.reward_multipliers.excavator_relocate_dug_dirt_mult
            transport_relocate_mult = preset.reward_multipliers.transport_relocate_mult
            
            # Apply capacity overrides from preset
            truck_capacity = preset.truck_capacity
            skidsteer_capacity = preset.skidsteer_capacity
            truck_road_restricted = preset.truck_road_restricted
            
            # Apply maps/curriculum from preset (convert MapLevel objects to dict format)
            if preset.maps and len(preset.maps) > 0:
                from terra.config import RewardsType
                curriculum_levels_override = []
                for map_level in preset.maps:
                    # Convert rewards_type string to enum
                    rewards_type = RewardsType.DENSE if map_level.rewards_type == "DENSE" else RewardsType.SPARSE
                    curriculum_levels_override.append({
                        "maps_path": map_level.maps_path,
                        "max_steps_in_episode": map_level.max_steps_in_episode,
                        "rewards_type": rewards_type,
                        "apply_trench_rewards": map_level.apply_trench_rewards,
                    })
            
        except ImportError as e:
            print(f"⚠️  Failed to import training configs: {e}")
            print("   Make sure configs/training_configs.py exists")
        except ValueError as e:
            print(f"⚠️  {e}")
            print("   Run 'python configs/training_configs.py' to see available presets")
    
    # Override with explicit CLI arguments (these take precedence over preset)
    if args.agent_types is not None:
        try:
            import ast
            parsed = ast.literal_eval(args.agent_types)
            # Normalize to a tuple of ints; accept tuple, list, or single int
            if isinstance(parsed, tuple):
                agent_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, list):
                agent_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, (int,)):
                agent_types_override = (int(parsed),)
            else:
                raise ValueError("--agent_types must be a tuple/list like (2,0,0,2) or a single int like (0)")
            print(f"➡️  CLI override agent types: {agent_types_override}")
        except Exception as e:
            print(f"⚠️  Failed to parse --agent_types '{args.agent_types}': {e}")
    
    if args.action_types is not None:
        try:
            import ast
            parsed = ast.literal_eval(args.action_types)
            # Normalize to a tuple of ints; accept tuple, list, or single int
            if isinstance(parsed, tuple):
                action_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, list):
                action_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, (int,)):
                action_types_override = (int(parsed),)
            else:
                raise ValueError("--action_types must be a tuple/list like (0,1,0,1) or a single int like (0)")
            print(f"➡️  CLI override action types: {action_types_override}")
        except Exception as e:
            print(f"⚠️  Failed to parse --action_types '{args.action_types}': {e}")

    if (args.replay_map_count > 0 or args.target_map_repeat > 0) and args.map_path is None:
        raise ValueError("Mixed target-map replay requires --map_path.")
    if (args.replay_map_count > 0 or args.target_map_repeat > 0) and args.config is None:
        raise ValueError("Mixed target-map replay requires --config so the dataset source is defined.")
    if (args.replay_map_count > 0) != (args.target_map_repeat > 0):
        raise ValueError("Set both --replay_map_count and --target_map_repeat to use mixed target-map replay.")
    
    # CLI reward multiplier overrides take precedence
    if args.dump_bonus_mult is not None:
        dump_bonus_mult = args.dump_bonus_mult
    if args.excavator_relocate_dumped_mult is not None:
        excavator_relocate_dumped_mult = args.excavator_relocate_dumped_mult
    if args.excavator_relocate_dug_dirt_mult is not None:
        excavator_relocate_dug_dirt_mult = args.excavator_relocate_dug_dirt_mult
    if args.transport_relocate_mult is not None:
        transport_relocate_mult = args.transport_relocate_mult
    
    # Use default agent types if nothing was set
    if agent_types_override is None:
        agent_types_override = (0,)  # Default to single excavator
    if action_types_override is None:
        action_types_override = (0,)  # Default to tracked
    
    # Validate: skidsteers (agent_type=2) cannot use wheeled movement (action_type=1)
    for i in range(min(len(agent_types_override), len(action_types_override))):
        if agent_types_override[i] == 2 and action_types_override[i] == 1:
            raise ValueError(
                f"Agent {i}: Skidsteer (agent_type=2) does not support wheeled movement "
                f"(action_type=1). Skidsteers require tracked movement (action_type=0) "
                f"for auto-load, push-mode, and reverse-dump mechanics."
            )
    
    name = f"{args.name}-{args.machine}-{DT}"
    
    config = MixedAgentTrainConfig(
        name=name, 
        num_devices=args.num_devices,
        lr=args.lr,
        num_envs_per_device=args.num_envs_per_device,
        total_timesteps=args.total_timesteps,
        resume_from=args.resume_from,
        load_env_from_checkpoint=args.load_env_from_checkpoint,
        agent_types_override=agent_types_override,
        action_types_override=action_types_override,
        debug=args.debug,
        config_name=args.config,
        dump_bonus_mult=dump_bonus_mult,
        excavator_relocate_dumped_mult=excavator_relocate_dumped_mult,
        excavator_relocate_dug_dirt_mult=excavator_relocate_dug_dirt_mult,
        transport_relocate_mult=transport_relocate_mult,
        truck_capacity=truck_capacity,
        skidsteer_capacity=skidsteer_capacity,
        truck_road_restricted=truck_road_restricted,
        curriculum_levels_override=curriculum_levels_override,
        single_map_path=args.map_path,
        replay_map_count=args.replay_map_count,
        target_map_repeat=args.target_map_repeat,
    )
    
    train_mixed_agents(config) 
