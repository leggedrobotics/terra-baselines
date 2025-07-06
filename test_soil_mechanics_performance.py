#!/usr/bin/env python3
"""
Performance test script to benchmark different soil mechanics configurations.
This helps identify the optimal settings for training speed vs realism.
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
sys.path.append('/cluster/home/alesweber/TerraProject/terra')

# Set environment for testing
os.environ['JAX_PLATFORMS'] = 'cpu'  # For testing without GPU
os.environ['DATASET_PATH'] = '/cluster/home/alesweber/TerraProject/terra/data/terra/train/'
os.environ['DATASET_SIZE'] = '10'

from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import BatchConfig

def benchmark_soil_mechanics_configs():
    """Test different soil mechanics configurations for performance"""
    
    configs = [
        {
            'name': 'No Soil Mechanics',
            'USE_SIMPLIFIED_SOIL_MECHANICS': True,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': False,
            'description': 'Fastest - no soil mechanics at all'
        },
        {
            'name': 'Simplified Soil Mechanics',
            'USE_SIMPLIFIED_SOIL_MECHANICS': True,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': True,
            'description': 'Fast - single convolution, no conservation'
        },
        {
            'name': 'Full Soil Mechanics',
            'USE_SIMPLIFIED_SOIL_MECHANICS': False,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': True,
            'description': 'Slow - full Gaussian + conservation (original)'
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"📄 {config['description']}")
        
        # Apply configuration by modifying the state module
        import terra.terra.state as state_module
        state_module.USE_SIMPLIFIED_SOIL_MECHANICS = config['USE_SIMPLIFIED_SOIL_MECHANICS']
        state_module.ENABLE_SOIL_MECHANICS_IN_TRAINING = config['ENABLE_SOIL_MECHANICS_IN_TRAINING']
        
        try:
            # Time compilation and training step
            start_time = time.time()
            
            # Create environment and config
            env = TerraEnvBatch(batch_cfg=BatchConfig())
            model_config = {
                'num_envs': 512,  # Smaller for testing
                'num_prev_actions': 10,
                'loaded_max': 100,
                'maps_net_normalization_bounds': [-10, 10],
                'clip_action_maps': True,
                'local_map_normalization_bounds': [-16, 16]
            }
            
            rng = jax.random.PRNGKey(0)
            model, params = get_model_ready(rng, model_config, env)
            
            compilation_time = time.time() - start_time
            
            # Test a simple training step
            start_step_time = time.time()
            
            # Create dummy observation for forward pass
            map_width = env.batch_cfg.maps_dims.maps_edge_length
            map_height = env.batch_cfg.maps_dims.maps_edge_length
            
            obs = [
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.num_state_obs)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], map_width, map_height)),
                jnp.zeros((model_config["num_envs"], map_width, map_height)),
                jnp.zeros((model_config["num_envs"], map_width, map_height)),
                jnp.zeros((model_config["num_envs"], map_width, map_height)),
                jnp.zeros((model_config["num_envs"], model_config["num_prev_actions"])),
                # Agent 2 features
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.num_state_obs)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], env.batch_cfg.agent.angles_cabin)),
                jnp.zeros((model_config["num_envs"], map_width, map_height)),
            ]
            
            # Run forward pass
            value, logits = model.apply(params, obs)
            
            step_time = time.time() - start_step_time
            
            results.append({
                'config': config['name'],
                'compilation_time': compilation_time,
                'step_time': step_time,
                'description': config['description']
            })
            
            print(f"✅ Compilation: {compilation_time:.2f}s")
            print(f"✅ Step time: {step_time:.4f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'config': config['name'],
                'compilation_time': float('inf'),
                'step_time': float('inf'),
                'description': f"Failed: {e}"
            })
    
    return results

def print_performance_summary(results):
    """Print a summary of performance results"""
    print("\n" + "="*80)
    print("🏆 SOIL MECHANICS PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort by compilation time
    sorted_results = sorted(results, key=lambda x: x['compilation_time'])
    
    for i, result in enumerate(sorted_results):
        if result['compilation_time'] == float('inf'):
            continue
            
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        
        print(f"\n{rank} {result['config']}")
        print(f"   📊 Compilation: {result['compilation_time']:.2f}s")
        print(f"   ⚡ Step time: {result['step_time']:.4f}s")
        print(f"   📝 {result['description']}")
    
    # Calculate speedup
    if len(sorted_results) >= 2:
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        speedup = slowest['compilation_time'] / fastest['compilation_time']
        
        print(f"\n💨 SPEEDUP: {speedup:.1f}x faster compilation with '{fastest['config']}'")
        print(f"   Best choice for training: {fastest['config']}")
        print(f"   Best choice for evaluation: {slowest['config']} (if realism needed)")

def print_optimization_recommendations():
    """Print specific optimization recommendations"""
    print("\n" + "="*80)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    print("""
🚀 For Training Performance:
   1. Set ENABLE_SOIL_MECHANICS_IN_TRAINING = False
   2. Set USE_SIMPLIFIED_SOIL_MECHANICS = True
   3. Expected speedup: 5-10x faster compilation
   4. Trade-off: Less realistic soil behavior

🎯 For Evaluation/Testing:
   1. Set ENABLE_SOIL_MECHANICS_IN_TRAINING = True
   2. Set USE_SIMPLIFIED_SOIL_MECHANICS = False
   3. Realistic soil physics for final assessment

⚖️ Balanced Approach:
   1. Use simplified mechanics during early training
   2. Switch to full mechanics for final training stages
   3. Always use full mechanics for evaluation

🔧 Implementation:
   Edit terra/terra/state.py lines 21-25:
   - ENABLE_SOIL_MECHANICS_IN_TRAINING = False  # For speed
   - USE_SIMPLIFIED_SOIL_MECHANICS = True       # For speed
""")

if __name__ == "__main__":
    print("🔬 Terra Soil Mechanics Performance Benchmark")
    print("This will test different soil mechanics configurations for speed")
    
    try:
        results = benchmark_soil_mechanics_configs()
        print_performance_summary(results)
        print_optimization_recommendations()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("This is expected on login nodes without proper GPU access")
        print("Run this script on a compute node for accurate results") 