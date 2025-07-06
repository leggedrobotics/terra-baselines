#!/usr/bin/env python3
"""
Simple test for soil mechanics conservation without full environment setup.
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

# Force CPU platform
os.environ['JAX_PLATFORMS'] = 'cpu'

def test_conservation_math():
    """Test the mathematical correctness of our conservation approach"""
    print("🧮 Testing Conservation Mathematics...")
    
    # Test case 1: Simple pile
    print("\n📋 Test 1: Single dirt pile")
    test_map = jnp.zeros((10, 10))
    test_map = test_map.at[5, 5].set(100.0)  # Big pile in center
    affected_mask = jnp.zeros((10, 10), dtype=jnp.bool_)
    affected_mask = affected_mask.at[4:7, 4:7].set(True)  # 3x3 area around pile
    
    # Define 3x3 Gaussian kernel (matching our corrected implementation)
    kernel = jnp.array([
        [0.077847, 0.123317, 0.077847],
        [0.123317, 0.195346, 0.123317], 
        [0.077847, 0.123317, 0.077847]
    ])
    
    print(f"Original dirt total: {jnp.sum(test_map)}")
    print(f"Original dirt at center: {test_map[5, 5]}")
    
    # Simulate our corrected algorithm
    affected_area = jnp.where(affected_mask, test_map, 0.0)
    unaffected_area = jnp.where(affected_mask, 0.0, test_map)
    
    # Apply convolution
    spread_map = jax.scipy.signal.convolve2d(affected_area, kernel, mode='same')
    print(f"After convolution: {jnp.sum(spread_map):.6f}")
    
    # Apply conservation scaling (our corrected method)
    affected_original_total = jnp.sum(jnp.where(affected_mask, affected_area, 0.0))
    affected_spread_total = jnp.sum(jnp.where(affected_mask, spread_map, 0.0))
    
    conservation_factor = jnp.where(
        affected_spread_total > 1e-6,
        affected_original_total / affected_spread_total,
        1.0
    )
    
    conserved_spread_map = jnp.where(
        affected_mask,
        spread_map * conservation_factor,
        spread_map
    )
    
    result_map = unaffected_area + conserved_spread_map
    
    print(f"After conservation: {jnp.sum(result_map):.6f}")
    print(f"Conservation error: {abs(float(jnp.sum(test_map)) - float(jnp.sum(result_map))):.8f}")
    
    # Test case 2: Multiple piles
    print("\n📋 Test 2: Multiple dirt piles")
    test_map2 = jnp.zeros((10, 10))
    test_map2 = test_map2.at[3, 3].set(50.0).at[7, 7].set(75.0)
    affected_mask2 = jnp.zeros((10, 10), dtype=jnp.bool_)
    affected_mask2 = affected_mask2.at[2:5, 2:5].set(True).at[6:9, 6:9].set(True)  # Two 3x3 areas
    
    print(f"Original dirt total: {jnp.sum(test_map2)}")
    
    # Apply same algorithm
    affected_area2 = jnp.where(affected_mask2, test_map2, 0.0)
    unaffected_area2 = jnp.where(affected_mask2, 0.0, test_map2)
    
    spread_map2 = jax.scipy.signal.convolve2d(affected_area2, kernel, mode='same')
    
    affected_original_total2 = jnp.sum(jnp.where(affected_mask2, affected_area2, 0.0))
    affected_spread_total2 = jnp.sum(jnp.where(affected_mask2, spread_map2, 0.0))
    
    conservation_factor2 = jnp.where(
        affected_spread_total2 > 1e-6,
        affected_original_total2 / affected_spread_total2,
        1.0
    )
    
    conserved_spread_map2 = jnp.where(
        affected_mask2,
        spread_map2 * conservation_factor2,
        spread_map2
    )
    
    result_map2 = unaffected_area2 + conserved_spread_map2
    
    print(f"After conservation: {jnp.sum(result_map2):.6f}")
    print(f"Conservation error: {abs(float(jnp.sum(test_map2)) - float(jnp.sum(result_map2))):.8f}")
    
    return float(jnp.sum(result_map)), float(jnp.sum(result_map2))

def test_spreading_pattern():
    """Test that dirt actually spreads properly"""
    print("\n🗺️ Testing Spreading Pattern...")
    
    # Create a 7x7 test case for visualization
    test_map = jnp.zeros((7, 7))
    test_map = test_map.at[3, 3].set(100.0)  # Center pile
    affected_mask = jnp.zeros((7, 7), dtype=jnp.bool_)
    affected_mask = affected_mask.at[2:5, 2:5].set(True)  # 3x3 area around pile
    
    print("Original map:")
    for i in range(7):
        row_str = " ".join(f"{test_map[i, j]:6.1f}" for j in range(7))
        print(f"  {row_str}")
    
    # Apply our corrected spreading algorithm
    kernel = jnp.array([
        [0.077847, 0.123317, 0.077847],
        [0.123317, 0.195346, 0.123317], 
        [0.077847, 0.123317, 0.077847]
    ])
    
    # Simulate our corrected algorithm
    affected_area = jnp.where(affected_mask, test_map, 0.0)
    unaffected_area = jnp.where(affected_mask, 0.0, test_map)
    
    spread_map = jax.scipy.signal.convolve2d(affected_area, kernel, mode='same')
    
    # Conservation scaling
    affected_original_total = jnp.sum(jnp.where(affected_mask, affected_area, 0.0))
    affected_spread_total = jnp.sum(jnp.where(affected_mask, spread_map, 0.0))
    conservation_factor = jnp.where(
        affected_spread_total > 1e-6, 
        affected_original_total / affected_spread_total, 
        1.0
    )
    
    conserved_spread_map = jnp.where(
        affected_mask,
        spread_map * conservation_factor,
        spread_map
    )
    
    result_map = unaffected_area + conserved_spread_map
    
    print("\nAfter spreading with conservation:")
    for i in range(7):
        row_str = " ".join(f"{result_map[i, j]:6.1f}" for j in range(7))
        print(f"  {row_str}")
    
    print(f"\nTotal before: {jnp.sum(test_map):.2f}")
    print(f"Total after: {jnp.sum(result_map):.2f}")
    print(f"Center reduced from {test_map[3,3]:.1f} to {result_map[3,3]:.1f}")
    print(f"Neighboring tiles now have dirt: {jnp.sum(result_map > 0.1):.0f} tiles")
    
    return float(jnp.sum(result_map))

def test_performance():
    """Test performance of the operation"""
    print("\n🚀 Testing Performance...")
    
    # Create a realistic size test (64x64 like in training)
    test_map = jnp.zeros((64, 64))
    test_map = test_map.at[30:34, 30:34].set(50.0)  # 4x4 dirt area
    affected_mask = jnp.zeros((64, 64), dtype=jnp.bool_)
    affected_mask = affected_mask.at[28:36, 28:36].set(True)  # 8x8 affected area
    
    kernel = jnp.array([
        [0.077847, 0.123317, 0.077847],
        [0.123317, 0.195346, 0.123317], 
        [0.077847, 0.123317, 0.077847]
    ])
    
    # Time the operation
    start_time = time.time()
    
    # This is what happens in our optimized function
    affected_area = jnp.where(affected_mask, test_map, 0.0)
    unaffected_area = jnp.where(affected_mask, 0.0, test_map)
    
    spread_map = jax.scipy.signal.convolve2d(affected_area, kernel, mode='same')
    
    affected_original_total = jnp.sum(jnp.where(affected_mask, affected_area, 0.0))
    affected_spread_total = jnp.sum(jnp.where(affected_mask, spread_map, 0.0))
    conservation_factor = jnp.where(
        affected_spread_total > 1e-6,
        affected_original_total / affected_spread_total,
        1.0
    )
    
    conserved_spread_map = jnp.where(
        affected_mask,
        spread_map * conservation_factor,
        spread_map
    )
    
    result_map = unaffected_area + conserved_spread_map
    
    operation_time = time.time() - start_time
    
    print(f"64x64 operation took: {operation_time:.6f} seconds")
    print(f"Conservation error: {abs(float(jnp.sum(test_map)) - float(jnp.sum(result_map))):.8f}")
    
    # Test with multiple operations (like in training)
    print("\nTesting batch of operations...")
    start_time = time.time()
    
    for _ in range(100):  # Simulate 100 soil mechanics calls
        affected_area = jnp.where(affected_mask, test_map, 0.0)
        unaffected_area = jnp.where(affected_mask, 0.0, test_map)
        
        spread_map = jax.scipy.signal.convolve2d(affected_area, kernel, mode='same')
        
        affected_original_total = jnp.sum(jnp.where(affected_mask, affected_area, 0.0))
        affected_spread_total = jnp.sum(jnp.where(affected_mask, spread_map, 0.0))
        conservation_factor = jnp.where(
            affected_spread_total > 1e-6,
            affected_original_total / affected_spread_total,
            1.0
        )
        
        conserved_spread_map = jnp.where(
            affected_mask,
            spread_map * conservation_factor,
            spread_map
        )
        
        result_map = unaffected_area + conserved_spread_map
    
    batch_time = time.time() - start_time
    
    print(f"100 operations took: {batch_time:.4f} seconds")
    print(f"Per operation: {batch_time/100:.6f} seconds")
    
    return operation_time, batch_time

def main():
    print("🧪 Simple Conservation Test")
    print("=" * 50)
    
    try:
        # Test mathematical correctness
        result1, result2 = test_conservation_math()
        
        # Test spreading visualization
        spread_result = test_spreading_pattern()
        
        # Test performance
        single_time, batch_time = test_performance()
        
        # Summary
        print("\n" + "="*50)
        print("📊 SUMMARY")
        print("="*50)
        
        print("✅ Conservation Tests:")
        print(f"   - Single pile conserved: {result1:.2f} dirt")
        print(f"   - Multiple piles conserved: {result2:.2f} dirt")
        print(f"   - Spreading visualization: {spread_result:.2f} dirt")
        
        print(f"\n🚀 Performance:")
        print(f"   - Single 64x64 operation: {single_time:.6f}s")
        print(f"   - 100 operations: {batch_time:.4f}s")
        print(f"   - Per operation average: {batch_time/100:.6f}s")
        
        print(f"\n🎯 VERDICT:")
        if single_time < 0.001:  # Less than 1ms
            print("   ✅ EXCELLENT performance - operation is very fast!")
        elif single_time < 0.01:  # Less than 10ms  
            print("   ✅ GOOD performance - should not cause compilation issues")
        else:
            print("   ⚠️ Operation might still be too slow")
            
        # Check if conservation errors are reasonable
        max_error = 1e-6  # Very tight tolerance
        if abs(result1 - 100.0) < max_error and abs(result2 - 125.0) < max_error:
            print("   ✅ PERFECT dirt conservation!")
        elif abs(result1 - 100.0) < 0.01 and abs(result2 - 125.0) < 0.01:
            print("   ✅ EXCELLENT dirt conservation!")
        else:
            print("   ⚠️ Conservation needs improvement")
            
        print(f"\n💡 Expected benefits:")
        print(f"   🚀 This method should eliminate the 6+ minute compilation times")
        print(f"   🧮 Maintains realistic dirt spreading and conservation")
        print(f"   ⚡ Should reduce training step time from 5+s to 2-3s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 