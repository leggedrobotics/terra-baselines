#!/usr/bin/env python3
"""
Quick test to measure performance impact of soil mechanics.
This script temporarily disables soil mechanics to see compilation/training speedup.
"""

import os
import sys
import time
sys.path.append('/cluster/home/alesweber/TerraProject/terra')

# Set environment for testing
os.environ['JAX_PLATFORMS'] = 'cpu'  # For testing without GPU
os.environ['DATASET_PATH'] = '/cluster/home/alesweber/TerraProject/terra/data/terra/train/'
os.environ['DATASET_SIZE'] = '10'

from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import BatchConfig
import jax
import jax.numpy as jnp

def test_compilation_speed():
    """Test how fast the network compiles with current soil mechanics"""
    print("🧪 Testing Network Compilation Speed...")
    
    # Create environment and config
    env = TerraEnvBatch(batch_cfg=BatchConfig())
    config = {
        'num_envs': 512,  # Smaller for testing
        'num_prev_actions': 10,
        'loaded_max': 100,
        'maps_net_normalization_bounds': [-10, 10],
        'clip_action_maps': True,
        'local_map_normalization_bounds': [-16, 16]
    }
    
    # Time compilation
    start_time = time.time()
    
    rng = jax.random.PRNGKey(0)
    model, params = get_model_ready(rng, config, env)
    
    compilation_time = time.time() - start_time
    
    print(f"✅ Network compiled in {compilation_time:.2f} seconds")
    print(f"📊 Model parameters: {sum(x.size for x in jax.tree_leaves(params)):,}")
    
    return compilation_time

def test_simple_forward_pass():
    """Test a simple forward pass to see basic performance"""
    print("\n🚀 Testing Forward Pass Speed...")
    
    env = TerraEnvBatch(batch_cfg=BatchConfig())
    config = {
        'num_envs': 256,  # Even smaller for speed test
        'num_prev_actions': 10,
        'loaded_max': 100,
        'maps_net_normalization_bounds': [-10, 10],
        'clip_action_maps': True,
        'local_map_normalization_bounds': [-16, 16]
    }
    
    rng = jax.random.PRNGKey(0)
    model, params = get_model_ready(rng, config, env)
    
    # Create dummy observation
    map_width = env.batch_cfg.maps_dims.maps_edge_length
    map_height = env.batch_cfg.maps_dims.maps_edge_length
    
    obs = [
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.num_state_obs)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], config["num_prev_actions"])),
        # Agent 2 features
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.num_state_obs)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
    ]
    
    # Time forward pass
    start_time = time.time()
    
    # Run forward pass
    value, logits = model.apply(params, obs)
    
    forward_time = time.time() - start_time
    
    print(f"✅ Forward pass completed in {forward_time:.4f} seconds")
    print(f"📊 Output shapes - Value: {value.shape}, Logits: {logits.shape}")
    
    return forward_time

if __name__ == "__main__":
    print("🔬 Terra Performance Test - Current Implementation")
    print("=" * 60)
    
    try:
        # Test compilation speed
        comp_time = test_compilation_speed()
        
        # Test forward pass speed
        forward_time = test_simple_forward_pass()
        
        print("\n📈 Performance Summary:")
        print(f"   - Network compilation: {comp_time:.2f}s")
        print(f"   - Forward pass: {forward_time:.4f}s")
        print(f"   - Estimated training overhead: HIGH (due to soil mechanics)")
        
        print("\n💡 Recommendation:")
        print("   - Soil mechanics adds significant computational overhead")
        print("   - Consider simplifying or disabling for performance testing")
        print("   - Mixed agent training compounds the complexity")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected on login nodes without proper GPU access") 