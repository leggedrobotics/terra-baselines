#!/usr/bin/env python3
"""
Aggregate results from parallel SLURM jobs
Run this after all parallel jobs complete
"""

import json
import numpy as np
import argparse
import os
import datetime
from glob import glob

def load_map_results(results_dir):
    """Load all individual map results"""
    pattern = os.path.join(results_dir, "map_*_results.json")
    result_files = sorted(glob(pattern))
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    print(f"Found {len(result_files)} result files")
    
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_results.append(data)
                print(f"Loaded {file_path}: Map {data['map_idx']}, best coverage {data['best_result']['coverage']:.4f}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def compute_final_statistics(all_results):
    """Compute final statistics from all maps using the exact same function as the original"""
    
    # Import the original function
    from llm.eval_llm import compute_stats_llm
    
    # Extract best results from each map
    best_results = [result['best_result'] for result in all_results]
    
    # Create the exact same lists as in the original code
    best_episode_done_once_list = [r['episode_done_once'] for r in best_results]
    best_episode_length_list = [r['episode_length'] for r in best_results]
    best_move_cumsum_list = [r['move_cumsum'] for r in best_results]
    best_do_cumsum_list = [r['do_cumsum'] for r in best_results]
    best_areas_list = [r['areas'] for r in best_results]
    best_dig_tiles_per_target_map_init_list = [r['dig_tiles_per_target_map_init'] for r in best_results]
    best_dug_tiles_per_action_map_list = [r['dug_tiles_per_action_map'] for r in best_results]
    
    # Calculate coverages for additional reporting
    coverages = []
    for dug, total in zip(best_dug_tiles_per_action_map_list, best_dig_tiles_per_target_map_init_list):
        if total > 0:
            coverages.append(dug / total)
        else:
            coverages.append(0.0)
    
    # Call the original compute_stats_llm function with the exact same parameters
    print(f"\n{'='*80}")
    print(f"CALLING ORIGINAL compute_stats_llm FUNCTION")
    print(f"{'='*80}")
    
    compute_stats_llm(
        best_episode_done_once_list, 
        best_episode_length_list, 
        best_move_cumsum_list,
        best_do_cumsum_list, 
        best_areas_list, 
        best_dig_tiles_per_target_map_init_list,
        best_dug_tiles_per_action_map_list
    )
    
    # Also compute some additional statistics for saving
    stats = {
        'n_maps': len(all_results),
        'coverage_stats': {
            'mean': float(np.mean(coverages)),
            'std': float(np.std(coverages)),
            'min': float(np.min(coverages)),
            'max': float(np.max(coverages)),
            'median': float(np.median(coverages)),
            'q25': float(np.percentile(coverages, 25)),
            'q75': float(np.percentile(coverages, 75))
        },
        'episode_stats': {
            'completion_rate': float(np.mean(best_episode_done_once_list)),
            'mean_length': float(np.mean(best_episode_length_list)),
            'std_length': float(np.std(best_episode_length_list))
        },
        'action_stats': {
            'mean_move_cumsum': float(np.mean(best_move_cumsum_list)),
            'mean_do_cumsum': float(np.mean(best_do_cumsum_list))
        },
        'intervention_stats': {
            'total_interventions': sum(r.get('total_interventions', 0) for r in best_results),
            'maps_with_interventions': sum(1 for r in best_results if r.get('total_interventions', 0) > 0)
        },
        # Store the arrays that were passed to compute_stats_llm
        'compute_stats_arrays': {
            'best_episode_done_once_list': best_episode_done_once_list,
            'best_episode_length_list': best_episode_length_list,
            'best_move_cumsum_list': best_move_cumsum_list,
            'best_do_cumsum_list': best_do_cumsum_list,
            'best_areas_list': best_areas_list,
            'best_dig_tiles_per_target_map_init_list': best_dig_tiles_per_target_map_init_list,
            'best_dug_tiles_per_action_map_list': best_dug_tiles_per_action_map_list
        }
    }
    
    return stats, best_results, coverages

def print_statistics(stats, model_name):
    """Print formatted statistics - the original compute_stats_llm already printed detailed stats"""
    print(f"\n{'='*80}")
    print(f"ADDITIONAL COVERAGE STATISTICS - {model_name}")
    print(f"{'='*80}")
    print(f"Number of maps processed: {stats['n_maps']}")
    
    print(f"\nBest Partition Coverage Statistics:")
    print(f"  Mean coverage: {stats['coverage_stats']['mean']:.4f} ± {stats['coverage_stats']['std']:.4f}")
    print(f"  Median coverage: {stats['coverage_stats']['median']:.4f}")
    print(f"  Min coverage: {stats['coverage_stats']['min']:.4f}")
    print(f"  Max coverage: {stats['coverage_stats']['max']:.4f}")
    print(f"  Q25-Q75: {stats['coverage_stats']['q25']:.4f} - {stats['coverage_stats']['q75']:.4f}")
    
    if stats['intervention_stats']['total_interventions'] > 0:
        print(f"\nIntervention Statistics:")
        print(f"  Total interventions: {stats['intervention_stats']['total_interventions']}")
        print(f"  Maps with interventions: {stats['intervention_stats']['maps_with_interventions']}/{stats['n_maps']}")
        print(f"  Intervention rate: {stats['intervention_stats']['total_interventions']/sum(stats['compute_stats_arrays']['best_episode_length_list']):.1%} per step")

def save_aggregated_results(all_results, stats, coverages, model_name, output_dir):
    """Save aggregated results to file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    aggregated_data = {
        'timestamp': timestamp,
        'model_name': model_name,
        'statistics': stats,
        'best_coverages': coverages,
        'individual_map_results': all_results,
        'metadata': {
            'aggregation_script': 'aggregate_parallel_results.py',
            'aggregation_time': datetime.datetime.now().isoformat()
        }
    }
    
    # Save main aggregated results
    aggregated_filename = os.path.join(output_dir, f"aggregated_results_{model_name.replace('/', '_')}_{timestamp}.json")
    with open(aggregated_filename, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    
    # Save summary statistics in a more readable format
    summary_filename = os.path.join(output_dir, f"summary_{model_name.replace('/', '_')}_{timestamp}.txt")
    with open(summary_filename, 'w') as f:
        f.write(f"Aggregated Results Summary - {model_name}\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Number of maps processed: {stats['n_maps']}\n\n")
        
        f.write("Coverage Statistics:\n")
        f.write(f"  Mean: {stats['coverage_stats']['mean']:.4f} ± {stats['coverage_stats']['std']:.4f}\n")
        f.write(f"  Median: {stats['coverage_stats']['median']:.4f}\n")
        f.write(f"  Min: {stats['coverage_stats']['min']:.4f}\n")
        f.write(f"  Max: {stats['coverage_stats']['max']:.4f}\n")
        f.write(f"  Q25-Q75: {stats['coverage_stats']['q25']:.4f} - {stats['coverage_stats']['q75']:.4f}\n\n")
        
        f.write("Episode Statistics:\n")
        f.write(f"  Completion rate: {stats['episode_stats']['completion_rate']:.1%}\n")
        f.write(f"  Mean episode length: {stats['episode_stats']['mean_length']:.1f} ± {stats['episode_stats']['std_length']:.1f}\n\n")
        
        f.write("Action Statistics:\n")
        f.write(f"  Mean move cumsum: {stats['action_stats']['mean_move_cumsum']:.1f}\n")
        f.write(f"  Mean do cumsum: {stats['action_stats']['mean_do_cumsum']:.1f}\n\n")
        
        if stats['intervention_stats']['total_interventions'] > 0:
            f.write("Intervention Statistics:\n")
            f.write(f"  Total interventions: {stats['intervention_stats']['total_interventions']}\n")
            f.write(f"  Maps with interventions: {stats['intervention_stats']['maps_with_interventions']}/{stats['n_maps']}\n\n")
        
        f.write("Individual Map Coverages:\n")
        for i, coverage in enumerate(coverages):
            f.write(f"  Map {i:2d}: {coverage:.4f}\n")
    
    return aggregated_filename, summary_filename

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from parallel SLURM jobs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_parallel",
        help="Directory containing individual map result files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (used for file naming and display)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="aggregated_results",
        help="Directory to save aggregated results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build full results directory path
    results_dir = os.path.join(args.results_dir, args.model_name.replace('/', '_'))
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print(f"Available directories in {args.results_dir}:")
        if os.path.exists(args.results_dir):
            for item in os.listdir(args.results_dir):
                item_path = os.path.join(args.results_dir, item)
                if os.path.isdir(item_path):
                    print(f"  {item}")
        return
    
    print(f"Loading results from: {results_dir}")
    
    try:
        # Load all map results
        all_results = load_map_results(results_dir)
        
        if not all_results:
            print("No valid results found!")
            return
        
        # Compute statistics
        stats, best_results, coverages = compute_final_statistics(all_results)
        
        # Print statistics
        print_statistics(stats, args.model_name)
        
        # Save aggregated results
        aggregated_file, summary_file = save_aggregated_results(
            all_results, stats, coverages, args.model_name, args.output_dir
        )
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {aggregated_file}")
        print(f"  Summary: {summary_file}")
        
        print(f"\n{'='*80}")
        print("AGGREGATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())