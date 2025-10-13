#!/usr/bin/env python3
"""
Test script to verify action types functionality works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from terra.terra.agent import Agent
from terra.terra.config import EnvConfig
import jax
import jax.numpy as jnp

def test_action_types():
    """Test that action types are correctly set in agent initialization."""
    
    # Create a simple test environment config
    env_cfg = EnvConfig()
    
    # Test data
    key = jax.random.PRNGKey(42)
    padding_mask = jnp.ones((20, 20), dtype=jnp.int32)
    action_map = jnp.zeros((20, 20), dtype=jnp.int32)
    
    print("Testing action types functionality...")
    
    # Test 1: Default behavior (all tracked)
    print("\n1. Testing default behavior (all tracked):")
    agent, _ = Agent.new(
        key, env_cfg, 10, 10, padding_mask, action_map,
        agent_types=(0, 2),  # excavator + skidsteer
        action_types=(0, 0)  # should default to tracked
    )
    
    for i, state in enumerate(agent.agent_states[:2]):  # Only check first 2 agents
        agent_type = int(state.agent_type[0])
        action_type = int(state.action_type[0])
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type}")
        assert action_type == 0, f"Expected action_type=0 (tracked), got {action_type}"
    
    # Test 2: Mixed action types
    print("\n2. Testing mixed action types:")
    agent, _ = Agent.new(
        key, env_cfg, 10, 10, padding_mask, action_map,
        agent_types=(0, 2),  # excavator + skidsteer
        action_types=(0, 1)  # tracked + wheeled
    )
    
    for i, state in enumerate(agent.agent_states[:2]):  # Only check first 2 agents
        agent_type = int(state.agent_type[0])
        action_type = int(state.action_type[0])
        expected_action_type = (0, 1)[i]  # Expected based on action_types=(0,1)
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type} (expected {expected_action_type})")
        assert action_type == expected_action_type, f"Expected action_type={expected_action_type}, got {action_type}"
    
    # Test 3: All wheeled
    print("\n3. Testing all wheeled:")
    agent, _ = Agent.new(
        key, env_cfg, 10, 10, padding_mask, action_map,
        agent_types=(0, 2),  # excavator + skidsteer
        action_types=(1, 1)  # both wheeled
    )
    
    for i, state in enumerate(agent.agent_states[:2]):  # Only check first 2 agents
        agent_type = int(state.agent_type[0])
        action_type = int(state.action_type[0])
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type}")
        assert action_type == 1, f"Expected action_type=1 (wheeled), got {action_type}"
    
    # Test 4: More agents than action_types (should default remaining to tracked)
    print("\n4. Testing more agents than action_types (should default remaining to tracked):")
    agent, _ = Agent.new(
        key, env_cfg, 10, 10, padding_mask, action_map,
        agent_types=(0, 1, 2, 0),  # 4 agents: excavator, truck, skidsteer, excavator
        action_types=(1, 0)  # only 2 action types specified
    )
    
    for i, state in enumerate(agent.agent_states[:4]):  # Check all 4 agents
        agent_type = int(state.agent_type[0])
        action_type = int(state.action_type[0])
        expected_action_type = (1, 0, 0, 0)[i]  # First 2 from action_types, rest default to 0
        expected_agent_type = (0, 1, 2, 0)[i]  # excavator, truck, skidsteer, excavator
        print(f"   Agent {i}: agent_type={agent_type}, action_type={action_type} (expected {expected_action_type})")
        assert action_type == expected_action_type, f"Expected action_type={expected_action_type}, got {action_type}"
        assert agent_type == expected_agent_type, f"Expected agent_type={expected_agent_type}, got {agent_type}"
    
    print("\n✅ All tests passed! Action types functionality is working correctly.")

if __name__ == "__main__":
    test_action_types()
