#!/usr/bin/env python3
"""
Dummy visualization of an obstacle-aware geodesic distance map on a snake-like grid.
Generates a 64x64 map with a sinusoidal/snake corridor, sets a border-aligned dump zone,
computes geodesic distances (4-neighborhood BFS) to the dump zone avoiding obstacles,
and saves a heatmap visualization similar in spirit to other distance visualizers.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def build_snake_obstacles(
    h: int,
    w: int,
    min_gap: int = 8,
    max_gap: int = 16,
    gap_width: int = 10,
    clearance_top: int = 2,
    clearance_bottom: int = 6,
    num_walls: int | None = None,
) -> np.ndarray:
    """
    Create thin (1-pixel) horizontal wall rows every `wall_gap` tiles with a shifting
    gap to form a snake corridor. This blocks any direct straight-line path to the
    dump zone and forces a detour.

    Returns a boolean array (h, w) where True indicates an obstacle (wall pixel).
    """
    obstacles = np.zeros((h, w), dtype=bool)
    base_margin = max(2, gap_width + 1)
    rng = np.random.default_rng()
    wall_idx = 0
    y_start = max(min_gap, clearance_top)
    y_end = max(y_start, h - clearance_bottom)

    def carve_wall_at(y: int, wall_idx: int):
        if y < 0 or y >= h:
            return False
        if obstacles[y].any():
            return False
        obstacles[y, :] = True
        local_gap = int(np.clip(gap_width + rng.integers(-2, 3), 3, max(3, gap_width + 3)))
        half = max(1, local_gap // 2)
        margin = int(np.clip(base_margin + rng.integers(-1, 2), 2, max(2, base_margin + 2)))
        base_center = margin if (wall_idx % 2 == 0) else (w - margin)
        jitter = int(rng.integers(-4, 5))
        center = int(np.clip(base_center + jitter, margin // 2, w - margin // 2))
        x0 = int(np.clip(center - half, 0, w - 1))
        x1 = int(np.clip(center + half, 0, w - 1))
        obstacles[y, x0:x1 + 1] = False
        return True

    if num_walls is not None:
        # Place exactly num_walls rows, spaced a bit further for more clearance
        start = y_start + 8   # extra space from dump zone
        gap = 12              # wider spacing to not be too close
        ys = [start + i * gap for i in range(num_walls)]
        for yi in ys:
            if yi >= y_end:
                break
            if carve_wall_at(int(yi), wall_idx):
                wall_idx += 1
        return obstacles

    y = y_start
    while y < y_end:
        # Randomly skip some walls to reduce obstacle density
        if rng.random() < 0.25:
            y += int(rng.integers(min_gap, max_gap + 1))
            continue
        if carve_wall_at(y, wall_idx):
            wall_idx += 1
        # advance y by a random gap to achieve random vertical spacing
        y += int(rng.integers(min_gap, max_gap + 1))

    return obstacles


def make_dump_zone(h: int, w: int, edge: str = "top", size=(8, 15)) -> np.ndarray:
    """
    Create a rectangular dump zone flush with one border. size=(short, long) where the long
    side aligns with the chosen border.
    Returns a boolean array (h, w) where True indicates dump zone tiles.
    """
    short, long = size
    dz = np.zeros((h, w), dtype=bool)

    if edge in ("top", "bottom"):
        hh, ww = short, long
        x0 = w // 2 - ww // 2
        x1 = x0 + ww
        if edge == "top":
            y0, y1 = 0, hh
        else:
            y1, y0 = h, h - hh
        dz[y0:y1, x0:x1] = True
    else:  # left/right
        hh, ww = long, short
        y0 = h // 2 - hh // 2
        y1 = y0 + hh
        if edge == "left":
            x0, x1 = 0, ww
        else:
            x1, x0 = w, w - ww
        dz[y0:y1, x0:x1] = True

    return dz


def geodesic_distance_to_targets(obstacles: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute 4-neighborhood geodesic distance on a grid from any target cell (distance 0)
    to all other cells, avoiding obstacles (True = obstacle).
    Unreachable cells are set to np.inf.
    """
    h, w = obstacles.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)

    q = deque()
    ys, xs = np.where(targets)
    for y, x in zip(ys, xs):
        if not obstacles[y, x]:
            dist[y, x] = 0.0
            q.append((y, x))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not obstacles[ny, nx]:
                if dist[ny, nx] > d + 1.0:
                    dist[ny, nx] = d + 1.0
                    q.append((ny, nx))

    return dist


def main():
    h, w = 64, 64
    # Build snake corridor obstacles
    obstacles = build_snake_obstacles(
        h,
        w,
        min_gap=10,
        max_gap=18,
        gap_width=12,
        clearance_top=6,   # extra space above first obstacle
        clearance_bottom=4,
        num_walls=3,
    )

    # Dump zone on top edge (8x15, long side along edge) — matches current dataset convention
    dump_zone = make_dump_zone(h, w, edge="top", size=(8, 15))

    # Ensure dump zone isn't marked as obstacle
    obstacles = obstacles.copy()
    obstacles[dump_zone] = False

    # Exactly 3 obstacle lines are generated in build_snake_obstacles with num_walls=3

    # Geodesic distance to dump zone
    dist = geodesic_distance_to_targets(obstacles, dump_zone)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Mask unreachable as max value for colormap clipping
    finite = np.isfinite(dist)
    max_val = np.max(dist[finite]) if np.any(finite) else 1.0
    vis = dist.copy()
    vis[~finite] = max_val

    im = ax.imshow(vis, cmap="viridis", origin="upper")
    ax.set_title("Geodesic Distance to Dump Zone (Snake Obstacles)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay thin, black obstacles (alpha=1 on obstacle pixels)
    obs_overlay = np.zeros((*obstacles.shape, 4), dtype=float)
    obs_overlay[..., 3] = obstacles.astype(float) * 1.0  # alpha channel
    # black color is already zeros in RGB
    ax.imshow(obs_overlay, origin="upper")
    # Draw dump zone as black outline only (no fill)
    ys, xs = np.where(dump_zone)
    if ys.size > 0 and xs.size > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='black', linewidth=2)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Tiles to Dump Zone")

    out_dir = Path(__file__).parent
    out_path = out_dir / "distance_geodesic_snake.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved geodesic distance visualization to: {out_path}")


if __name__ == "__main__":
    main()


