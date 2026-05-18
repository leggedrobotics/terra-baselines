"""
Training configuration presets for Terra multi-agent training.

Configurations are defined in training_configs.yaml.

Available functions:
    get_config(name): Get a training configuration by name
    list_configs(): Get a list of all available configuration names
    print_config_summary(): Print a summary of all available configurations

Usage:
    # In train_mixed.py:
    python train_mixed.py --config excavator_truck
    
    # Or programmatically:
    from configs.training_configs import get_config
    config = get_config("excavator_truck")
    
    # List available configs:
    python configs/training_configs.py
"""

from .training_configs import (
    TrainingConfig,
    RewardMultipliers,
    MapLevel,
    get_config,
    list_configs,
    print_config_summary,
)

__all__ = [
    "TrainingConfig",
    "RewardMultipliers", 
    "MapLevel",
    "get_config",
    "list_configs",
    "print_config_summary",
]
