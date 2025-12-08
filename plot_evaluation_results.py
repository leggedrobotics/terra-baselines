#!/usr/bin/env python3
"""
Script to generate comparison graphs from evaluation results.
Compares single excavator, excavator+truck, and excavator+skidsteer configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set academic/publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('seaborn-v0_8')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
})

def parse_evaluation_file(file_path):
    """Parse evaluation file and extract key metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract metrics using string parsing
    metrics = {}
    
    # Completion rate
    completion_line = [line for line in content.split('\n') if 'Completion:' in line]
    if completion_line:
        metrics['completion'] = float(completion_line[0].split(':')[1].strip().replace('%', ''))
    
    # Path efficiency
    path_eff_line = [line for line in content.split('\n') if 'Path efficiency:' in line]
    if path_eff_line:
        metrics['path_efficiency'] = float(path_eff_line[0].split(':')[1].strip().split()[0])
    
    # Workspace efficiency
    workspace_eff_line = [line for line in content.split('\n') if 'Workspaces efficiency:' in line]
    if workspace_eff_line:
        metrics['workspace_efficiency'] = float(workspace_eff_line[0].split(':')[1].strip().split()[0])
    
    # Goal efficiency
    goal_eff_line = [line for line in content.split('\n') if 'Goal efficiency (1/steps): mean=' in line]
    if goal_eff_line:
        metrics['goal_efficiency'] = float(goal_eff_line[0].split('mean=')[1])
    
    # Steps to completion
    steps_line = [line for line in content.split('\n') if 'Avg steps till completion:' in line]
    if steps_line:
        metrics['avg_steps'] = float(steps_line[0].split(':')[1].strip())
    
    # Per-agent metrics
    excavator_line = [line for line in content.split('\n') if 'Excavator:' in line]
    if excavator_line:
        parts = excavator_line[0].split(',')
        metrics['excavator_move_m'] = float(parts[0].split('move_m=')[1])
        if len(parts) > 3:
            metrics['excavator_do_events'] = int(parts[3].split('do_events=')[1])
        else:
            metrics['excavator_do_events'] = 0
    
    truck_line = [line for line in content.split('\n') if 'Truck:' in line]
    if truck_line:
        parts = truck_line[0].split(',')
        metrics['truck_move_m'] = float(parts[0].split('move_m=')[1])
        if len(parts) > 3:
            metrics['truck_do_events'] = int(parts[3].split('do_events=')[1])
        else:
            metrics['truck_do_events'] = 0
    
    skidsteer_line = [line for line in content.split('\n') if 'Skidsteer:' in line]
    if skidsteer_line:
        parts = skidsteer_line[0].split(',')
        metrics['skidsteer_move_m'] = float(parts[0].split('move_m=')[1])
        if len(parts) > 3:
            metrics['skidsteer_do_events'] = int(parts[3].split('do_events=')[1])
        else:
            metrics['skidsteer_do_events'] = 0
    
    return metrics

def create_comparison_plots():
    """Create comprehensive comparison plots."""
    
    # File paths
    files = {
        'Single Excavator': '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/46747639_eval - excav.out',
        'Excavator + Truck': '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/46747118_eval - excav truck.out',
        'Excavator + Skidsteer': '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/46883081_eval - excav skid.out'
    }
    
    # Parse all files
    data = {}
    for name, file_path in files.items():
        data[name] = parse_evaluation_file(file_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.suptitle('Multi-Agent Excavation Performance Comparison', fontsize=13, fontweight='bold', y=0.995)
    
    # Academic color palette
    colors = ['#3d5a80', '#5a9f4b', '#c47c24']  # Muted blue, green, orange
    
    # 1. Workspace Efficiency Comparison (Lower is Better)
    ax1 = axes[0, 0]
    configs = list(data.keys())
    workspace_effs = [data[config]['workspace_efficiency'] for config in configs]
    bars1 = ax1.bar(range(len(configs)), workspace_effs, width=0.6, color=colors, 
                    edgecolor='black', linewidth=1.0, alpha=0.85)
    ax1.set_title('Workspace Efficiency (Lower is Better)', fontweight='bold', pad=10)
    ax1.set_ylabel('Efficiency Score', fontweight='medium')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=0)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars1, workspace_effs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='semibold', fontsize=9)
    
    # 2. Meters Moved Comparison
    ax2 = axes[0, 1]
    excavator_moves = [data[config]['excavator_move_m'] for config in configs]
    truck_moves = [data[config].get('truck_move_m', 0) for config in configs]
    skidsteer_moves = [data[config].get('skidsteer_move_m', 0) for config in configs]
    
    x = np.arange(len(configs))
    width = 0.24
    offset = 0.28  # Distance from center for side bars
    
    # Use consistent colors: excavator matches main blue, truck green, skidsteer orange
    agent_colors = ['#3d5a80', '#5a9f4b', '#c47c24']  # Matching colors from main palette
    bars2a = ax2.bar(x - offset, excavator_moves, width, label='Excavator', 
                     color=agent_colors[0], edgecolor='black', linewidth=1.0, alpha=0.85)
    bars2b = ax2.bar(x, truck_moves, width, label='Truck', 
                     color=agent_colors[1], edgecolor='black', linewidth=1.0, alpha=0.85)
    bars2c = ax2.bar(x + offset, skidsteer_moves, width, label='Skidsteer', 
                     color=agent_colors[2], edgecolor='black', linewidth=1.0, alpha=0.85)
    
    ax2.set_title('Distance Moved by Agent Type', fontweight='bold', pad=10)
    ax2.set_ylabel('Meters Moved', fontweight='medium')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=0)
    ax2.legend(loc='lower left', frameon=True, edgecolor='gray', fancybox=False)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bars in [bars2a, bars2b, bars2c]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, height + 10, 
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='semibold')
    
    # Add line connecting excavator bars to show trend
    excavator_positions = x - offset  # positions of excavator bars
    ax2.plot(excavator_positions, excavator_moves, color=agent_colors[0], 
             marker='o', markersize=8, linewidth=2, alpha=0.7, zorder=5)
    
    # 3. Excavator Lift/Dump Events (Actions/2)
    ax3 = axes[1, 0]
    excavator_events = [data[config]['excavator_do_events'] for config in configs]
    bars3 = ax3.bar(range(len(configs)), excavator_events, width=0.6, color=colors, 
                    edgecolor='black', linewidth=1.0, alpha=0.85)
    ax3.set_title('Excavator Lift/Dump Events', fontweight='bold', pad=10)
    ax3.set_ylabel('Number of Events', fontweight='medium')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=0)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bar, value in zip(bars3, excavator_events):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value}', ha='center', va='bottom', fontweight='semibold', fontsize=9)
    
    # Add lines and labels showing percentage decrease from baseline
    baseline = excavator_events[0]  # Single excavator baseline
    x_positions = range(len(configs))
    
    # Draw lines from baseline to each bar
    for i in range(1, len(excavator_events)):
        pct_decrease = ((excavator_events[i] - baseline) / baseline) * 100
        ax3.plot([0, i], [baseline, excavator_events[i]], 
                color='red', linestyle='--', linewidth=1.5, alpha=0.6, zorder=3)
        # Add annotation with percentage (positioned higher to avoid bar labels)
        ax3.text(i, excavator_events[i] + 15, f'{pct_decrease:.1f}%', 
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # 4. Completion Rate
    ax4 = axes[1, 1]
    completion_rates = [data[config]['completion'] for config in configs]
    bars4 = ax4.bar(range(len(configs)), completion_rates, width=0.6, color=colors, 
                    edgecolor='black', linewidth=1.0, alpha=0.85)
    ax4.set_title('Task Completion Rate', fontweight='bold', pad=10)
    ax4.set_ylabel('Completion %', fontweight='medium')
    ax4.set_ylim(0, 105)
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=0)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bar, value in zip(bars4, completion_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='semibold', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot with high quality
    output_path = '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/evaluation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for thesis (vector format)
    pdf_path = '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/evaluation_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for config, metrics in data.items():
        print(f"\n{config}:")
        print(f"  Completion Rate: {metrics['completion']:.1f}%")
        print(f"  Workspace Efficiency (lower=better): {metrics['workspace_efficiency']:.1f}")
        print(f"  Path Efficiency: {metrics['path_efficiency']:.1f}")
        print(f"  Avg Steps to Completion: {metrics['avg_steps']:.0f}")
        print(f"  Excavator Move Distance: {metrics['excavator_move_m']:.1f}m")
        print(f"  Excavator Lift/Dump Events: {metrics['excavator_do_events']}")
        
        if 'truck_move_m' in metrics and metrics['truck_move_m'] > 0:
            print(f"  Truck Move Distance: {metrics['truck_move_m']:.1f}m")
            print(f"  Truck Lift/Dump Events: {metrics['truck_do_events']}")
        
        if 'skidsteer_move_m' in metrics and metrics['skidsteer_move_m'] > 0:
            print(f"  Skidsteer Move Distance: {metrics['skidsteer_move_m']:.1f}m")
            print(f"  Skidsteer Lift/Dump Events: {metrics['skidsteer_do_events']}")

if __name__ == "__main__":
    create_comparison_plots()
