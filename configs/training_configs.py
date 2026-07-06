"""
Training Configuration Loader for Terra Multi-Agent Training

Loads configurations from training_configs.yaml.

Usage:
    python train_mixed.py --config excavator_truck
    python train_mixed.py --config excavator_skidsteer
    python train_mixed.py --config solo_excavator
    
    # List all available configs:
    python configs/training_configs.py
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import yaml
except ImportError:
    print("Warning: pyyaml not installed. Run: pip install pyyaml")
    yaml = None


@dataclass
class RewardMultipliers:
    """Reward multipliers that tune agent behavior incentives."""
    dump_bonus_mult: float = 0.5
    excavator_relocate_dumped_mult: float = 0.2
    excavator_relocate_dug_dirt_mult: float = 1.5
    transport_relocate_mult: float = 1.5


@dataclass
class MapLevel:
    """Configuration for a single curriculum level."""
    maps_path: str
    max_steps_in_episode: int = 800
    rewards_type: str = "DENSE"
    apply_trench_rewards: bool = False


@dataclass
class CurriculumSettings:
    """Terra CurriculumManager thresholds (per parallel env)."""
    increase_level_threshold: int = 20
    decrease_level_threshold: int = 80
    last_level_type: str = "random"  # "random" | "none"


@dataclass
class TrainingConfig:
    """Complete training configuration for a named setup."""
    name: str
    description: str
    
    # Agent configuration
    agent_types: Tuple[int, ...]
    action_types: Tuple[int, ...]
    
    # Map configuration (curriculum levels)
    maps: List[MapLevel] = field(default_factory=list)
    curriculum: CurriculumSettings = field(default_factory=CurriculumSettings)
    
    # Reward multipliers
    reward_multipliers: RewardMultipliers = field(default_factory=RewardMultipliers)
    
    # Optional overrides for EnvConfig fields
    truck_capacity: Optional[int] = None
    skidsteer_capacity: Optional[int] = None
    truck_road_restricted: Optional[bool] = None
    foundation_dump_overlap_threshold: Optional[float] = None


# Cache for loaded configs
_TRAINING_CONFIGS: Dict[str, TrainingConfig] = {}
_CONFIGS_LOADED = False


def _get_yaml_path() -> Path:
    """Get the path to the YAML config file."""
    return Path(__file__).parent / "training_configs.yaml"


def _load_configs_from_yaml() -> Dict[str, TrainingConfig]:
    """Load all configurations from the YAML file."""
    global _TRAINING_CONFIGS, _CONFIGS_LOADED
    
    if _CONFIGS_LOADED:
        return _TRAINING_CONFIGS
    
    if yaml is None:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")
    
    yaml_path = _get_yaml_path()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        raw_configs = yaml.safe_load(f)
    
    for name, cfg in raw_configs.items():
        # Parse reward multipliers
        rm_data = cfg.get('reward_multipliers', {})
        reward_multipliers = RewardMultipliers(
            dump_bonus_mult=rm_data.get('dump_bonus_mult', 0.5),
            excavator_relocate_dumped_mult=rm_data.get('excavator_relocate_dumped_mult', 0.2),
            excavator_relocate_dug_dirt_mult=rm_data.get('excavator_relocate_dug_dirt_mult', 1.5),
            transport_relocate_mult=rm_data.get('transport_relocate_mult', 1.5),
        )
        
        # Parse maps
        maps = []
        for m in cfg.get('maps', []):
            maps.append(MapLevel(
                maps_path=m.get('path', ''),
                max_steps_in_episode=m.get('max_steps', 800),
                rewards_type=m.get('rewards_type', 'DENSE'),
                apply_trench_rewards=m.get('apply_trench_rewards', False),
            ))

        cur_data = cfg.get('curriculum', {})
        curriculum = CurriculumSettings(
            increase_level_threshold=cur_data.get('increase_level_threshold', 20),
            decrease_level_threshold=cur_data.get('decrease_level_threshold', 80),
            last_level_type=cur_data.get('last_level_type', 'random'),
        )
        
        # Create TrainingConfig
        _TRAINING_CONFIGS[name] = TrainingConfig(
            name=name,
            description=cfg.get('description', ''),
            agent_types=tuple(cfg.get('agent_types', [0])),
            action_types=tuple(cfg.get('action_types', [0])),
            maps=maps,
            curriculum=curriculum,
            reward_multipliers=reward_multipliers,
            truck_capacity=cfg.get('truck_capacity'),
            skidsteer_capacity=cfg.get('skidsteer_capacity'),
            truck_road_restricted=cfg.get('truck_road_restricted'),
            foundation_dump_overlap_threshold=cfg.get('foundation_dump_overlap_threshold'),
        )
    
    _CONFIGS_LOADED = True
    return _TRAINING_CONFIGS


def get_config(name: str) -> TrainingConfig:
    """Get a training configuration by name."""
    configs = _load_configs_from_yaml()
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return configs[name]


def list_configs() -> List[str]:
    """Get a list of all available configuration names."""
    configs = _load_configs_from_yaml()
    return sorted(configs.keys())


def print_config_summary():
    """Print a summary of all available configurations."""
    configs = _load_configs_from_yaml()
    
    print("\n" + "=" * 70)
    print("Available Training Configurations")
    print("=" * 70)
    
    agent_names = {0: "Excav", 1: "Truck", 2: "Skid"}
    
    for name in sorted(configs.keys()):
        cfg = configs[name]
        agents_str = "+".join(agent_names.get(a, f"?{a}") for a in cfg.agent_types)
        maps_str = ", ".join(m.maps_path for m in cfg.maps)
        
        print(f"\n  {name}")
        print(f"    Description: {cfg.description}")
        print(f"    Agents: {agents_str} ({len(cfg.agent_types)} agents)")
        print(f"    Maps: {maps_str}")
        print(f"    Reward Mults: dump={cfg.reward_multipliers.dump_bonus_mult}, "
              f"transport={cfg.reward_multipliers.transport_relocate_mult}")
    
    print("\n" + "=" * 70)
    print(f"\nConfig file: {_get_yaml_path()}")
    print("=" * 70 + "\n")


# For backwards compatibility
TRAINING_CONFIGS = property(lambda self: _load_configs_from_yaml())


def register_config(config: TrainingConfig) -> TrainingConfig:
    """Register a training configuration programmatically (for backwards compatibility)."""
    configs = _load_configs_from_yaml()
    configs[config.name] = config
    return config


if __name__ == "__main__":
    print_config_summary()
