#!/usr/bin/env python3
"""
Visualization: Relocation potential explained via distance map.

Figure layout (2x2):
 - Top-left: Dirt pile far from dump zone
 - Bottom-left: Distance map (normalized) with dirt region highlighted
 - Top-right: Dirt pile moved closer to dump zone
 - Bottom-right: Distance map (same dump zone) with new dirt region highlighted

Relocation potential = sum over non-dump tiles of positive dirt * distance_to_dump.
We use the normalized taxicab distance from `terra/tools/generate_distance_maps.py`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Reuse the distance computation util
PKG_ROOT = Path(__file__).resolve().parents[1]  # TerraProject/terra-baselines -> TerraProject
TOOLS_PATH = PKG_ROOT / "terra" / "tools"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

from generate_distance_maps import compute_distance_map_taxicab  # noqa: E402


def _make_nonuniform_pile(height: int, width: int, center: tuple[int, int], radius: int = 3, seed: int = 42) -> np.ndarray:
    """
    Create an irregular, non-uniform dirt pile using a radial falloff modulated by noise.
    Returns an int32 action map (positive values mean available dirt).
    """
    rng = np.random.default_rng(seed)
    cy, cx = center
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Smooth radial falloff in [0,1]
    base = np.clip((radius + 1 - dist), 0, None)
    base = base / base.max() if base.max() > 0 else base

    # Irregularity via noise; slightly blur by averaging neighboring noise samples
    raw_noise = rng.random((height, width)).astype(np.float32)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()
    # simple convolution using padding='edge'
    pad = 1
    padded = np.pad(raw_noise, pad_width=pad, mode='edge')
    smooth_noise = (
        kernel[0, 0] * padded[0:height, 0:width] +
        kernel[0, 1] * padded[0:height, 1:width+1] +
        kernel[0, 2] * padded[0:height, 2:width+2] +
        kernel[1, 0] * padded[1:height+1, 0:width] +
        kernel[1, 1] * padded[1:height+1, 1:width+1] +
        kernel[1, 2] * padded[1:height+1, 2:width+2] +
        kernel[2, 0] * padded[2:height+2, 0:width] +
        kernel[2, 1] * padded[2:height+2, 1:width+1] +
        kernel[2, 2] * padded[2:height+2, 2:width+2]
    )

    # Combine falloff and noise for an irregular mask
    shape_score = base * (0.7 + 0.6 * smooth_noise) - 0.25
    mask = (shape_score > 0.25)

    # Intensity: modulate base by noise to make non-uniform heights
    intensity = (base ** 1.2) * (0.6 + 0.8 * smooth_noise) * 6.0
    intensity = np.clip(intensity, 0, None)

    pile = np.zeros((height, width), dtype=np.int32)
    pile[mask] = intensity[mask].astype(np.int32)
    return pile


def _make_dump_zone(height: int, width: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Return a target_map (1 in dump zone, 0 elsewhere). Rect is (y0, x0, y1, x1)."""
    y0, x0, y1, x1 = rect
    target = np.zeros((height, width), dtype=np.int32)
    target[y0:y1, x0:x1] = 1
    return target


def _relocation_potential(action_map: np.ndarray, target_map: np.ndarray, dist_map: np.ndarray) -> float:
    """
    Sum over non-dump tiles of positive dirt times distance to nearest dump.
    action_map: int32 (can be >= 0)
    target_map: int32 with 1 in dump zone
    dist_map: float32 normalized distance in [0,1]
    """
    non_dump = target_map <= 0
    positive_dirt = np.clip(action_map, a_min=0, a_max=None).astype(np.float32)
    return float(np.sum(positive_dirt[non_dump] * dist_map[non_dump]))


def visualize_distance_potential(output_png: str) -> None:
    # Visual style similar to soil collapse visualization
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
    })

    height, width = 64, 64

    # Fixed dump zone (14x14) slightly offset from edges
    dump_rect = (46, 46, 60, 60)  # (y0, x0, y1, x1)
    target_map = _make_dump_zone(height, width, dump_rect)

    # Two dirt pile centers: far and closer (increase spacing; keep away from borders/legends)
    far_center = (18, 16)
    near_center = (38, 40)

    action_far = _make_nonuniform_pile(height, width, far_center, radius=7, seed=7)
    action_near = _make_nonuniform_pile(height, width, near_center, radius=7, seed=11)

    # Distance map normalized by 24 tiles (explicit)
    # Compute raw cityblock distance, then divide by 24.0
    dump = target_map > 0
    if _HAS_SCIPY:
        dist_raw = ndi.distance_transform_cdt(~dump, metric='taxicab').astype(np.float32)
    else:
        h, w = target_map.shape
        big = np.int32(10**6)
        dist_i = np.where(dump, 0, big).astype(np.int32)
        for i in range(h):
            for j in range(w):
                if dist_i[i, j] == 0:
                    continue
                left = dist_i[i - 1, j] + 1 if i > 0 else big
                up = dist_i[i, j - 1] + 1 if j > 0 else big
                dist_i[i, j] = min(dist_i[i, j], left, up)
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                current = dist_i[i, j]
                right = dist_i[i + 1, j] + 1 if i < h - 1 else big
                down = dist_i[i, j + 1] + 1 if j < h - 1 else big
                dist_i[i, j] = min(current, right, down)
        dist_raw = dist_i.astype(np.float32)
    # Use a larger normalization to avoid saturating many tiles at 1.0
    dist_map = dist_raw / 80.0

    pot_far = _relocation_potential(action_far, target_map, dist_map)
    pot_near = _relocation_potential(action_near, target_map, dist_map)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle('Relocation Potential via Distance Map', fontsize=14, fontweight='bold', y=0.98)

    # Common helpers
    def draw_dump_outline(ax):
        y0, x0, y1, x1 = dump_rect
        rect = mpatches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                  fill=False, edgecolor='black', linewidth=2.0)
        ax.add_patch(rect)

    def show_dirt(ax, action_map: np.ndarray, title: str, pot_value: float):
        im = ax.imshow(action_map, cmap=plt.cm.Blues, origin='lower')
        ax.set_title(f"{title}\nPotential = {pot_value:.2f}", fontweight='bold', pad=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        draw_dump_outline(ax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='lightgray')
        return im

    def show_distance(ax, dist_map: np.ndarray, action_map: np.ndarray, title: str, pot_value: float):
        im = ax.imshow(dist_map, cmap=plt.cm.viridis, origin='lower', vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # Overlay semi-transparent highlight where dirt lies
        mask = action_map > 0
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = np.array([1.0, 0.1, 0.1, 0.5], dtype=np.float32)  # red highlight with higher alpha
        ax.imshow(overlay, origin='lower')
        # Place potential text slightly to the side of the highlighted region
        ys, xs = np.where(mask)
        if ys.size > 0:
            y_c = int(np.median(ys))
            x_c = int(np.median(xs))
            # Offset to the right by more tiles, clipped to bounds
            height, width = mask.shape
            x_t = int(np.clip(x_c + 10, 0, width - 1))
            y_t = int(np.clip(y_c, 0, height - 1))
            ax.annotate(f"{pot_value:.2f}", xy=(x_c, y_c), xytext=(x_t, y_t),
                        textcoords='data', ha='left', va='center', color='white',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.6))
        draw_dump_outline(ax)
        ax.set_aspect('equal')
        ax.grid(False)
        return im

    im00 = show_dirt(axes[0, 0], action_far, 'Dirt pile farther from dump', pot_far)
    im10 = show_distance(axes[1, 0], dist_map, action_far, 'Distance map (far dirt)', pot_far)

    im01 = show_dirt(axes[0, 1], action_near, 'Dirt pile moved closer', pot_near)
    im11 = show_distance(axes[1, 1], dist_map, action_near, 'Distance map (close dirt)', pot_near)

    # Colorbars
    cb0 = fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb0.set_label('Dirt units', rotation=270, labelpad=12)

    cb1 = fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb1.set_label('Normalized distance', rotation=270, labelpad=12)

    cb2 = fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb2.set_label('Dirt units', rotation=270, labelpad=12)

    cb3 = fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb3.set_label('Normalized distance', rotation=270, labelpad=12)

    # Legends
    dump_patch = mpatches.Patch(facecolor='none', edgecolor='black', label='Dump zone')
    dirt_patch = mpatches.Patch(color='royalblue', label='Dirt (intensity = units)')
    highlight_patch = mpatches.Patch(color=(1.0, 0.1, 0.1, 0.5), label='Highlighted dirt region')
    axes[0, 0].legend(handles=[dump_patch, dirt_patch], loc='lower left')
    axes[1, 0].legend(handles=[dump_patch, highlight_patch], loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    out_png = os.path.abspath(output_png)
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Console summary
    print("Relocation potential (normalized distance weighting):")
    print(f"  Far dirt  : {pot_far:.4f}")
    print(f"  Near dirt : {pot_near:.4f}")
    print(f"  Reduction : {pot_far - pot_near:.4f}")


if __name__ == "__main__":
    png_path = "/cluster/project/rsl/alesweber/TerraProject/terra-baselines/distance_potential_visualization.png"
    visualize_distance_potential(png_path)


