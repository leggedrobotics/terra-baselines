#!/usr/bin/env python3
"""
Visualize soil collapse mechanics - how height differences spread after digging/dumping.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import signal

# Set academic style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
})

def simulate_soil_collapse(map_2d, affected_mask, n_iters=3):
    """
    Simulate soil collapse mechanics from Terra.
    Moves dirt between neighbors when height difference >= 2.
    """
    def collapse_step(i, map_2d):
        """One iteration of soil collapse."""
        result = map_2d.copy()
        mask = affected_mask.astype(bool)
        
        # Check each direction and move dirt
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Get neighbor heights by rolling
            shifted = np.roll(result, shift=(dy, dx), axis=(0, 1))
            diff = shifted - result
            neighbor_mask = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            
            # Move dirt if height difference >= 2
            move = (diff >= 2) & mask & neighbor_mask
            result = result + move.astype(result.dtype)
            result = np.roll(result, shift=(dy, dx), axis=(0, 1)) - move.astype(result.dtype)
            result = np.roll(result, shift=(-dy, -dx), axis=(0, 1))
        
        return result
    
    # Apply collapse iterations
    for i in range(n_iters):
        map_2d = collapse_step(i, map_2d)
    
    return map_2d

def expand_mask_for_soil_mechanics(mask, validity_mask, kernel_size=3):
    """Expand mask to include neighbors."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # Apply 2D convolution
    expanded_float = signal.correlate2d(
        mask.astype(np.float32), kernel, mode='same'
    )
    
    expanded = expanded_float > 0
    
    # Only include valid tiles
    return np.logical_and(expanded, validity_mask)

def visualize_soil_collapse():
    """Create visualization of soil collapse mechanics."""
    
    # Create example scenario: 15x15 grid
    size = 15
    map_2d = np.zeros((size, size), dtype=np.int32)  # Start with ground level (height 0)
    
    # Create a tall dirt pile in center (5 units tall)
    center_y, center_x = size // 2, size // 2
    radius = 2  # Smaller pile for clearer visualization
    
    affected_mask = np.zeros((size, size), dtype=bool)
    affected_mask[center_y-radius:center_y+radius+1, 
                  center_x-radius:center_x+radius+1] = True
    
    # Create initial tall pile
    map_2d[affected_mask] = 5  # 5-unit tall pile
    
    map_before = map_2d.copy()
    
    # Validity mask (all tiles are valid for this example)
    validity_mask = np.ones((size, size), dtype=bool)
    expanded_mask = expand_mask_for_soil_mechanics(affected_mask, validity_mask)
    
    # Apply soil collapse - the pile spreads outward
    map_after_collapse = simulate_soil_collapse(map_2d.copy(), expanded_mask, n_iters=5)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle('Soil Collapse Mechanics', fontsize=14, fontweight='bold', y=1.02)
    
    # Use gradient blue colormap
    cmap = plt.cm.Blues
    
    # Plot 1: Before (initial tall pile)
    im1 = axes[0].imshow(map_before, cmap=cmap, vmin=0, vmax=5, origin='lower')
    axes[0].set_title('Initial Tall Pile\n(Height = 5)', fontweight='bold', pad=10)
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Distance')
    
    # Mark affected area with less obvious but visible color
    x_coords, y_coords = np.where(affected_mask)
    for x, y in zip(x_coords, y_coords):
        axes[0].add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, 
                                         fill=False, edgecolor='darkslategray', linewidth=1.5))
    
    # Plot 2: Expanded mask (intermediate step)
    im2 = axes[1].imshow(map_before, cmap=cmap, vmin=0, vmax=5, origin='lower')
    axes[1].set_title('Expanded Zone\n(Collapse area)', fontweight='bold', pad=10)
    axes[1].set_xlabel('Distance')
    
    # Mark expanded area to show where collapse will happen
    x_expanded, y_expanded = np.where(expanded_mask)
    for x, y in zip(x_expanded, y_expanded):
        if affected_mask[y, x]:
            color = 'darkslategray'
            linewidth = 1.5
            linestyle = '-'
        else:
            color = 'mediumslateblue'
            linewidth = 1.2
            linestyle = '--'
        axes[1].add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, 
                                       fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))
    
    # Plot 3: After collapse
    im3 = axes[2].imshow(map_after_collapse, cmap=cmap, vmin=0, vmax=5, origin='lower')
    axes[2].set_title('After Soil Collapse\n(Spreads outward)', fontweight='bold', pad=10)
    axes[2].set_xlabel('Distance')
    
    # Calculate and show height differences
    # Show the gradient clearly
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='lightgray')
    
    # Add colorbar on the right side, outside all plots
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Dirt Height', rotation=270, labelpad=15, fontweight='medium')
    
    # Add annotations showing key points
    # Plot 1 (Before): Show height at pile
    axes[0].text(center_x, center_y, '5', ha='center', va='center', 
                fontweight='bold', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='darkblue'))
    axes[0].text(size-1, 0, 'h=0', ha='right', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot 2 (Expanded zone): Show same pile, marked zones
    axes[1].text(center_x, center_y, '5', ha='center', va='center', 
                fontweight='bold', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='darkblue'))
    
    # Plot 3 (After collapse): Show height values
    center_height = int(map_after_collapse[center_y, center_x])
    axes[2].text(center_x, center_y, f'{center_height}', ha='center', va='center', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Show heights at edges in final plot
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if not (dx == 0 and dy == 0):
                y, x = center_y + dy * 2, center_x + dx * 2
                if 0 <= y < size and 0 <= x < size:
                    height = int(map_after_collapse[y, x])
                    if height > 0:
                        axes[2].text(x, y, f'{height}', ha='center', va='center', 
                                   fontsize=8, color='darkblue', fontweight='bold')
    
    axes[2].text(size-1, 0, 'h=0', ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add legends - with visible but not too thick colors
    darkslategray_patch = mpatches.Patch(color='darkslategray', label='Original Pile', fill=False, linewidth=2)
    mediumslateblue_patch = mpatches.Patch(color='mediumslateblue', label='Expanded Zone', fill=False, 
                                           linestyle='--', linewidth=2)
    
    # Add legend to first plot with better visibility
    leg1 = axes[0].legend(handles=[darkslategray_patch], loc='lower left', frameon=True, 
                          fontsize=10, edgecolor='black', facecolor='white', 
                          framealpha=0.95, borderpad=0.5)
    leg1.get_frame().set_linewidth(1.5)
    for line in leg1.get_lines():
        line.set_linewidth(2)
    
    # Add combined legend to second plot
    leg2 = axes[1].legend(handles=[darkslategray_patch, mediumslateblue_patch], loc='lower left', frameon=True, 
                          fontsize=10, edgecolor='black', facecolor='white', 
                          framealpha=0.95, borderpad=0.5)
    leg2.get_frame().set_linewidth(1.5)
    for line in leg2.get_lines():
        line.set_linewidth(2)
    
    # Adjust layout to account for the colorbar
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    
    # Save the plot
    output_path = '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/soil_collapse_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Soil collapse visualization saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/soil_collapse_visualization.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SOIL COLLAPSE SUMMARY")
    print("="*60)
    print(f"Initial total dirt: {np.sum(map_before)}")
    print(f"Dirt after collapse: {np.sum(map_after_collapse)}")
    print(f"Dirt preserved: {np.sum(map_before) == np.sum(map_after_collapse)}")
    print(f"Height gradient (before): max={np.max(map_before)}, min={np.min(map_before)}")
    print(f"Height gradient (after): max={np.max(map_after_collapse)}, min={np.min(map_after_collapse)}")
    print(f"Area coverage (before): {np.sum(map_before > 0)} tiles")
    print(f"Area coverage (after): {np.sum(map_after_collapse > 0)} tiles")

if __name__ == "__main__":
    visualize_soil_collapse()

