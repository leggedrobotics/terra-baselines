#!/usr/bin/env python3
"""
Example showing how to use action types for manual play without the training script.
This demonstrates how the tuple default (0,0) works for action types.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from terra.terra.config import EnvConfig
from terra.terra.state import State
import jax
import jax.numpy as jnp

def example_manual_play():
    """Example of creating environments with different action type configurations."""
    
    print("Example: Manual play with different action type configurations")
    print("=" * 60)
    
    # Create a simple test environment
    key = jax.random.PRNGKey(42)
    target_map = jnp.zeros((20, 20), dtype=jnp.int32)
    padding_mask = jnp.ones((20, 20), dtype=jnp.int32)
    trench_axes = jnp.zeros((20, 20), dtype=jnp.int32)
    trench_type = jnp.zeros((20, 20), dtype=jnp.int32)
    dumpability_mask_init = jnp.zeros((20, 20), dtype=jnp.int32)
    action_map = jnp.zeros((20, 20), dtype=jnp.int32)
    
    # Example 1: Default configuration (all tracked)
    print("\n1. Default configuration (all tracked):")
    env_cfg_default = EnvConfig()
    print(f"   Agent types: {env_cfg_default.agent_types}")
    print(f"   Action types: {env_cfg_default.action_types}")
    
    state = State.new(
        key, env_cfg_default, target_map, padding_mask, trench_axes, 
        trench_type, dumpability_mask_init, action_map
    )
    
    for i, agent_state in enumerate(state.agent.agent_states[:2]):
        agent_type = int(agent_state.agent_type[0])
        action_type = int(agent_state.action_type[0])
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type}")
    
    # Example 2: Custom configuration with mixed action types
    print("\n2. Custom configuration (mixed action types):")
    env_cfg_mixed = EnvConfig(
        agent_types=(0, 1),  # excavator + truck
        action_types=(0, 1)  # tracked + wheeled
    )
    print(f"   Agent types: {env_cfg_mixed.agent_types}")
    print(f"   Action types: {env_cfg_mixed.action_types}")
    
    state = State.new(
        key, env_cfg_mixed, target_map, padding_mask, trench_axes, 
        trench_type, dumpability_mask_init, action_map
    )
    
    for i, agent_state in enumerate(state.agent.agent_states[:2]):
        agent_type = int(agent_state.agent_type[0])
        action_type = int(agent_state.action_type[0])
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type}")
    
    # Example 3: All wheeled configuration
    print("\n3. All wheeled configuration:")
    env_cfg_wheeled = EnvConfig(
        agent_types=(0, 1),  # excavator + truck
        action_types=(1, 1)  # both wheeled
    )
    print(f"   Agent types: {env_cfg_wheeled.agent_types}")
    print(f"   Action types: {env_cfg_wheeled.action_types}")
    
    state = State.new(
        key, env_cfg_wheeled, target_map, padding_mask, trench_axes, 
        trench_type, dumpability_mask_init, action_map
    )
    
    for i, agent_state in enumerate(state.agent.agent_states[:2]):
        agent_type = int(agent_state.agent_type[0])
        action_type = int(agent_state.action_type[0])
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type}")
    
    print("\n✅ All examples completed successfully!")
    print("\nKey points:")
    print("- Default action_types is (0,0) - all agents use tracked movement")
    print("- You can override action_types in EnvConfig for manual play")
    print("- Action types are independent of agent types")
    print("- 0=tracked, 1=wheeled movement")

if __name__ == "__main__":
    example_manual_play()
