#!/usr/bin/env python3
"""
Script to visualize terrain modification masks from a plan extracted by extract_plan.py
"""

import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_plan(plan_path):
    """Load the plan from a pickle file."""
    with open(plan_path, 'rb') as f:
        plan = pickle.load(f)
    return plan


def plot_terrain_modifications(plan, output_dir=None, show_plots=True):
    """
    Plot terrain modification masks for each step in the plan.

    Args:
        plan: List of plan entries from extract_plan.py (terrain modification actions)
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots interactively
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    print(f"Found {len(plan)} terrain modification actions in the plan")

    # Filter out entries with no terrain modifications
    modified_entries = []
    for entry in plan:
        terrain_mask = np.array(entry['terrain_modification_mask'])
        if np.sum(terrain_mask) > 0:
            modified_entries.append(entry)

    print(f"{len(modified_entries)} of these actually modified terrain")

    if len(modified_entries) == 0:
        print("No terrain modification actions found in the plan.")
        return

    for i, entry in enumerate(modified_entries):
        # Check if we have terrain change values for enhanced visualization
        if 'terrain_change_values' in entry:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            has_change_values = True
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            has_change_values = False

        # Get data
        terrain_mask = np.array(entry['terrain_modification_mask'])
        traversability_mask = np.array(entry['traversability_mask'])
        agent_state = entry['agent_state']
        step = entry['step']

        # Plot 1: Binary modification mask
        im1 = ax1.imshow(terrain_mask, cmap='Reds', interpolation='nearest')
        ax1.set_title(f'Terrain Modification Mask (Binary)\nStep {step} (Entry {i+1}/{len(modified_entries)})')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')

        # Add agent position
        agent_y, agent_x = agent_state['pos_base']  # Note: Terra coordinates are (y, x)
        loaded_before = entry['loaded_state_change']['before']
        loaded_after = entry['loaded_state_change']['after']
        ax1.plot(agent_x, agent_y, 'bo', markersize=8, label=f'Agent (before: {loaded_before}, after: {loaded_after})')
        ax1.legend()

        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Modified (1) / Unmodified (0)')

        # Plot 2: Traversability mask for context
        im2 = ax2.imshow(traversability_mask, cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Traversability Mask\nStep {step}')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')

        # Add agent position on traversability map too
        ax2.plot(agent_x, agent_y, 'ro', markersize=8, label=f'Agent (before: {loaded_before}, after: {loaded_after})')
        ax2.legend()

        # Add colorbar
        plt.colorbar(im2, ax=ax2, label='Traversable (1) / Non-traversable (0)')

        # Plot 3: Actual change values (if available)
        if has_change_values:
            terrain_changes = np.array(entry['terrain_change_values'])
            # Use a diverging colormap to show positive and negative changes
            max_abs_change = np.max(np.abs(terrain_changes)) if np.any(terrain_changes != 0) else 1
            im3 = ax3.imshow(terrain_changes, cmap='RdBu_r', interpolation='nearest',
                            vmin=-max_abs_change, vmax=max_abs_change)
            ax3.set_title(f'Terrain Change Values\nStep {step}')
            ax3.set_xlabel('X coordinate')
            ax3.set_ylabel('Y coordinate')

            # Add agent position
            ax3.plot(agent_x, agent_y, 'ko', markersize=8, label=f'Agent (before: {loaded_before}, after: {loaded_after})')
            ax3.legend()

            # Add colorbar
            plt.colorbar(im3, ax=ax3, label='Change in terrain value')

        # Add info text - determine action type based on loaded state change
        loaded_change = entry['loaded_state_change']
        if not loaded_change['before'] and loaded_change['after']:
            action_type = "Digging"
        elif loaded_change['before'] and not loaded_change['after']:
            action_type = "Dumping"
        else:
            action_type = "Unknown terrain modification"

        num_modified = int(np.sum(terrain_mask))

        # Add statistics about changes if available
        if has_change_values:
            terrain_changes = np.array(entry['terrain_change_values'])
            changed_values = terrain_changes[terrain_changes != 0]
            if len(changed_values) > 0:
                stats_text = f' | Min: {np.min(changed_values):.3f}, Max: {np.max(changed_values):.3f}, Mean: {np.mean(np.abs(changed_values)):.3f}'
            else:
                stats_text = ''
        else:
            stats_text = ''

        fig.suptitle(f'{action_type} - {num_modified} tiles modified{stats_text}', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save plot if output directory specified
        if output_dir:
            filename = f'digging_step_{step:04d}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {filepath}")

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()


def plot_all_modifications_overlay(plan, output_dir=None, show_plots=True):
    """
    Create an overlay plot showing all digging locations across all steps.

    Args:
        plan: List of plan entries from extract_plan.py (only digging actions)
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots interactively
    """
    if len(plan) == 0:
        print("No plan entries found.")
        return

    # Get the shape from the first entry
    first_entry = plan[0]
    terrain_shape = np.array(first_entry['terrain_modification_mask']).shape

    # Accumulate all modifications
    all_modifications = np.zeros(terrain_shape)
    modification_count = np.zeros(terrain_shape)

    for entry in plan:
        terrain_mask = np.array(entry['terrain_modification_mask'])
        all_modifications = np.logical_or(all_modifications, terrain_mask > 0)
        modification_count += terrain_mask

    # Create the overlay plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot binary overlay (any modification)
    im1 = ax1.imshow(all_modifications, cmap='Reds', interpolation='nearest')
    ax1.set_title('All Digging Locations\n(Binary Overlay)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=ax1, label='Dug (1) / Undug (0)')

    # Plot modification count
    im2 = ax2.imshow(modification_count, cmap='hot', interpolation='nearest')
    ax2.set_title('Digging Frequency\n(Number of times dug)')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(im2, ax=ax2, label='Number of digging actions')

    total_modified_tiles = int(np.sum(all_modifications))
    max_modifications = int(np.max(modification_count))
    fig.suptitle(f'Total: {total_modified_tiles} tiles dug, Max digging actions per tile: {max_modifications}', 
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save plot if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        filename = 'all_digging_locations_overlay.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved overlay plot to {filepath}")

    # Show plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize digging actions from extracted plan")
    parser.add_argument(
        "plan_path",
        type=str,
        help="Path to the plan .pkl file generated by extract_plan.py"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        help="Directory to save plots (optional)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively (useful when saving only)"
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Only generate the overlay plot, not individual step plots"
    )

    args = parser.parse_args()

    # Load the plan
    print(f"Loading plan from {args.plan_path}")
    plan = load_plan(args.plan_path)

    show_plots = not args.no_show

    if not args.overlay_only:
        # Plot individual digging actions
        print("Generating individual digging action plots...")
        plot_terrain_modifications(plan, args.output_dir, show_plots)

    # Plot overlay of all digging locations
    print("Generating digging overlay plot...")
    plot_all_modifications_overlay(plan, args.output_dir, show_plots)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
