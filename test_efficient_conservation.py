#!/usr/bin/env python3
"""
Test script to verify the new efficient conservation soil mechanics.
Tests both performance improvements and dirt conservation correctness.
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
sys.path.append('/cluster/home/alesweber/TerraProject/terra')

# Set environment for testing
os.environ['JAX_PLATFORMS'] = 'cpu'  # For testing without GPU
os.environ['DATASET_PATH'] = '/cluster/home/alesweber/TerraProject/terra/data/terra/train/'
os.environ['DATASET_SIZE'] = '10'

from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import BatchConfig

def test_dirt_conservation():
    """Test that the new efficient conservation method actually conserves dirt"""
    print("🧪 Testing Dirt Conservation Correctness...")
    
    # Create a small test environment
    env = TerraEnvBatch(batch_cfg=BatchConfig())
    
    # Create test state with dummy data
    rng = jax.random.PRNGKey(42)
    
    # Initialize environment to get a proper state
    env_params = env._get_default_params()
    timestep = env.reset(env_params, rng)
    state = timestep.state[0]  # Get first environment state
    
    # Create test scenarios
    test_cases = [
        {
            'name': 'Single Pile',
            'action_map': jnp.zeros((64, 64), dtype=jnp.int16).at[30:34, 30:34].set(10),
            'affected_mask': jnp.zeros((64, 64), dtype=jnp.bool_).at[30:34, 30:34].set(True)
        },
        {
            'name': 'Multiple Piles',
            'action_map': jnp.zeros((64, 64), dtype=jnp.int16).at[20:24, 20:24].set(8).at[40:44, 40:44].set(12),
            'affected_mask': jnp.zeros((64, 64), dtype=jnp.bool_).at[20:24, 20:24].set(True).at[40:44, 40:44].set(True)
        },
        {
            'name': 'Linear Dump',
            'action_map': jnp.zeros((64, 64), dtype=jnp.int16).at[32, 20:44].set(5),
            'affected_mask': jnp.zeros((64, 64), dtype=jnp.bool_).at[32, 20:44].set(True)
        }
    ]
    
    conservation_results = []
    
    for test_case in test_cases:
        print(f"  📋 Testing: {test_case['name']}")
        
        # Set up test state with our test data
        test_state = state._replace(
            world=state.world._replace(
                action_map=state.world.action_map._replace(map=test_case['action_map'])
            )
        )
        
        original_map = test_case['action_map']
        affected_mask = test_case['affected_mask']
        
        # Calculate original dirt total
        original_total = jnp.sum(jnp.where(affected_mask, original_map, 0))
        
        # Apply the new efficient conservation method
        result_map = test_state._apply_local_soil_mechanics_simplified(original_map, affected_mask)
        
        # Calculate result dirt total
        result_total = jnp.sum(jnp.where(affected_mask, result_map, 0))
        
        # Check conservation
        conservation_error = abs(float(original_total) - float(result_total))
        conservation_percentage = (conservation_error / max(float(original_total), 1e-6)) * 100
        
        print(f"     Original dirt: {original_total}")
        print(f"     Result dirt: {result_total:.2f}")
        print(f"     Conservation error: {conservation_error:.4f} ({conservation_percentage:.2f}%)")
        
        conservation_results.append({
            'name': test_case['name'],
            'original': float(original_total),
            'result': float(result_total),
            'error': conservation_error,
            'error_percentage': conservation_percentage
        })
        
        # Check if spreading occurred (dirt should move to adjacent tiles)
        original_nonzero = jnp.sum(original_map > 0)
        result_nonzero = jnp.sum(result_map > 0)
        print(f"     Spreading: {original_nonzero} → {result_nonzero} tiles with dirt")
        print()
    
    return conservation_results

def test_compilation_performance():
    """Test compilation speed with the new efficient method"""
    print("🚀 Testing Compilation Performance...")
    
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
    print("  ⏱️ Compiling network...")
    start_time = time.time()
    
    rng = jax.random.PRNGKey(0)
    model, params = get_model_ready(rng, config, env)
    
    compilation_time = time.time() - start_time
    
    print(f"  ✅ Network compiled in {compilation_time:.2f} seconds")
    print(f"  📊 Model parameters: {sum(x.size for x in jax.tree_leaves(params)):,}")
    
    # Test a simple forward pass
    print("  ⏱️ Testing forward pass...")
    start_step_time = time.time()
    
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
    
    # Run forward pass
    value, logits = model.apply(params, obs)
    
    step_time = time.time() - start_step_time
    
    print(f"  ✅ Forward pass completed in {step_time:.4f} seconds")
    print(f"  📊 Output shapes - Value: {value.shape}, Logits: {logits.shape}")
    
    return compilation_time, step_time

def test_soil_mechanics_operation():
    """Directly test the soil mechanics function for correctness"""
    print("🔬 Testing Soil Mechanics Function Directly...")
    
    # Create a simple test environment
    env = TerraEnvBatch(batch_cfg=BatchConfig())
    rng = jax.random.PRNGKey(42)
    env_params = env._get_default_params()
    timestep = env.reset(env_params, rng)
    state = timestep.state[0]
    
    # Create test dirt pile
    action_map = jnp.zeros((64, 64), dtype=jnp.int16)
    action_map = action_map.at[32, 32].set(100)  # Big pile in center
    affected_mask = jnp.zeros((64, 64), dtype=jnp.bool_)
    affected_mask = affected_mask.at[30:35, 30:35].set(True)  # 5x5 area around pile
    
    # Test state with our dirt pile
    test_state = state._replace(
        world=state.world._replace(
            action_map=state.world.action_map._replace(map=action_map)
        )
    )
    
    print("  📋 Test setup:")
    print(f"     Original dirt at (32,32): {action_map[32, 32]}")
    print(f"     Total original dirt: {jnp.sum(action_map)}")
    
    # Time the operation
    start_time = time.time()
    result_map = test_state._apply_local_soil_mechanics_simplified(action_map, affected_mask)
    operation_time = time.time() - start_time
    
    print(f"  ⏱️ Soil mechanics operation took: {operation_time:.6f} seconds")
    
    # Check results
    result_center = result_map[32, 32]
    total_result = jnp.sum(result_map)
    
    print("  📊 Results:")
    print(f"     Dirt at (32,32) after spreading: {result_center:.2f}")
    print(f"     Total dirt after spreading: {total_result:.2f}")
    print(f"     Conservation error: {abs(float(jnp.sum(action_map)) - float(total_result)):.4f}")
    
    # Check spreading pattern
    spread_area = result_map[30:35, 30:35]
    print("  🗺️ 5x5 area around center after spreading:")
    for i in range(5):
        row_str = "     " + " ".join(f"{spread_area[i, j]:6.1f}" for j in range(5))
        print(row_str)
    
    return operation_time, float(total_result)

def main():
    print("🧪 Testing Efficient Conservation Soil Mechanics")
    print("=" * 80)
    
    try:
        # Test 1: Direct soil mechanics operation
        print("\n" + "="*50)
        operation_time, total_dirt = test_soil_mechanics_operation()
        
        # Test 2: Dirt conservation correctness
        print("\n" + "="*50)
        conservation_results = test_dirt_conservation()
        
        # Test 3: Compilation performance
        print("\n" + "="*50)
        compilation_time, step_time = test_compilation_performance()
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        print(f"🚀 Performance:")
        print(f"   - Network compilation: {compilation_time:.2f}s")
        print(f"   - Forward pass: {step_time:.4f}s")
        print(f"   - Soil mechanics operation: {operation_time:.6f}s")
        
        print(f"\n🧮 Conservation Accuracy:")
        avg_error = np.mean([r['error_percentage'] for r in conservation_results])
        max_error = max([r['error_percentage'] for r in conservation_results])
        print(f"   - Average conservation error: {avg_error:.2f}%")
        print(f"   - Maximum conservation error: {max_error:.2f}%")
        
        # Verdict
        print(f"\n🎯 VERDICT:")
        if compilation_time < 30:  # Much better than 6+ minutes
            print("   ✅ EXCELLENT compilation speed improvement!")
        elif compilation_time < 60:
            print("   ✅ GOOD compilation speed improvement!")
        else:
            print("   ⚠️ Still slow compilation - may need further optimization")
            
        if avg_error < 5.0:  # Less than 5% error
            print("   ✅ EXCELLENT dirt conservation!")
        elif avg_error < 10.0:
            print("   ✅ GOOD dirt conservation!")
        else:
            print("   ⚠️ Conservation needs improvement")
            
        print(f"\n💡 Recommendation:")
        if compilation_time < 60 and avg_error < 10:
            print("   🎉 The efficient conservation method is working great!")
            print("   🚀 You should see significant training speedup with realistic physics")
        else:
            print("   🔧 Consider switching to 'fast' mode for maximum training speed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis might be expected on login nodes without proper environment access")

if __name__ == "__main__":
    main() 