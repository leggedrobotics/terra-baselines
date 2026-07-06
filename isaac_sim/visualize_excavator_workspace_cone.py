#!/usr/bin/env python3
"""Render Terra excavator dig/dump workspace cones for selected radii.

This is intentionally standalone: it copies the workspace math from
terra/terra/state.py instead of importing Terra/JAX or running an env.
"""

import math
import struct
import zlib
from pathlib import Path


MAP_SIZE = 64
FOUNDATION_SIZE = 14
EDGE_LENGTH_M = 44.0
TILE_SIZE = EDGE_LENGTH_M / MAP_SIZE
AGENT_WIDTH_TILES = 5
AGENT_HEIGHT_TILES = 9
ANGLES_CABIN = 12

AGENT_POS = (32, 18)  # [row, col], left of the centered foundation.
BASE_ANGLE_RAD = 0.0
CABIN_ANGLE_RAD = 0.0
ARM_ANGLE_RAD = 0.0

RADII = (4, 5, 6)
SCALE = 10
PANEL_GAP = 18
TOP_PAD = 26
BOTTOM_PAD = 32


def apply_rot_transl(theta: float, origin_xy: tuple[float, float], point_xy: tuple[float, float]) -> tuple[float, float]:
    """Copy terra.utils.apply_rot_transl for one point."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = point_xy[0] - origin_xy[0]
    dy = point_xy[1] - origin_xy[1]
    return (
        cos_t * dx + sin_t * dy,
        -sin_t * dx + cos_t * dy,
    )


def local_cartesian_to_cyl(local_xy: tuple[float, float]) -> tuple[float, float]:
    """Copy terra.utils.apply_local_cartesian_to_cyl for one point."""
    x, y = local_xy
    return math.sqrt(x * x + y * y), math.atan2(-x, y)


def foundation_tiles() -> set[tuple[int, int]]:
    start = MAP_SIZE // 2 - FOUNDATION_SIZE // 2
    end = start + FOUNDATION_SIZE
    return {(row, col) for row in range(start, end) for col in range(start, end)}


def workspace_mask(dig_radius_tiles: int) -> set[tuple[int, int]]:
    current_pos = ((AGENT_POS[0] + 0.5) * TILE_SIZE, (AGENT_POS[1] + 0.5) * TILE_SIZE)
    max_agent_dim = max(AGENT_WIDTH_TILES / 2.0, AGENT_HEIGHT_TILES / 2.0)
    min_distance_from_agent = TILE_SIZE * max_agent_dim
    fixed_extension = 0.5  # meters, as in terra/terra/state.py

    r_min = fixed_extension + min_distance_from_agent
    r_max = fixed_extension + min_distance_from_agent + dig_radius_tiles * TILE_SIZE
    theta_max = 2.0 * math.pi / ANGLES_CABIN
    theta_min = -theta_max

    agent_width = AGENT_WIDTH_TILES * TILE_SIZE
    agent_height = AGENT_HEIGHT_TILES * TILE_SIZE
    eps = TILE_SIZE / 2.0
    exclude_x_half = math.floor((agent_width + eps) / 2.0)
    exclude_y_half = math.floor((agent_height + eps) / 2.0)

    mask = set()
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            point = ((row + 0.5) * TILE_SIZE, (col + 0.5) * TILE_SIZE)
            local_arm = apply_rot_transl(ARM_ANGLE_RAD, current_pos, point)
            radius, theta = local_cartesian_to_cyl(local_arm)
            in_cyl = r_min <= radius <= r_max and theta_min <= theta <= theta_max

            local_base = apply_rot_transl(BASE_ANGLE_RAD, current_pos, point)
            outside_agent_x = local_base[0] >= exclude_x_half or local_base[0] <= -exclude_x_half
            outside_agent_y = local_base[1] >= exclude_y_half or local_base[1] <= -exclude_y_half
            outside_agent_body = outside_agent_x or outside_agent_y

            if in_cyl and outside_agent_body:
                mask.add((row, col))
    return mask


def set_pixel(rgb: bytearray, width: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width:
        idx = (y * width + x) * 3
        rgb[idx:idx + 3] = bytes(color)


def fill_rect(
    rgb: bytearray,
    width: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, y0), max(0, y1)):
        row_start = y * width * 3
        for x in range(max(0, x0), min(width, x1)):
            idx = row_start + x * 3
            rgb[idx:idx + 3] = bytes(color)


def draw_digit(rgb: bytearray, width: int, x: int, y: int, digit: str, color: tuple[int, int, int], scale: int = 2) -> None:
    patterns = {
        "0": ("111", "101", "101", "101", "111"),
        "1": ("010", "110", "010", "010", "111"),
        "2": ("111", "001", "111", "100", "111"),
        "3": ("111", "001", "111", "001", "111"),
        "4": ("101", "101", "111", "001", "001"),
        "5": ("111", "100", "111", "001", "111"),
        "6": ("111", "100", "111", "101", "111"),
        "7": ("111", "001", "001", "010", "010"),
        "8": ("111", "101", "111", "101", "111"),
        "9": ("111", "101", "111", "001", "111"),
        ".": ("000", "000", "000", "000", "010"),
        "m": ("000", "101", "111", "101", "101"),
        "=": ("000", "111", "000", "111", "000"),
        "r": ("000", "110", "101", "100", "100"),
    }
    pattern = patterns.get(digit)
    if pattern is None:
        return
    for py, line in enumerate(pattern):
        for px, bit in enumerate(line):
            if bit == "1":
                fill_rect(rgb, width, x + px * scale, y + py * scale, x + (px + 1) * scale, y + (py + 1) * scale, color)


def draw_text(rgb: bytearray, width: int, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
    cursor = x
    for char in text:
        if char == " ":
            cursor += 4 * scale
            continue
        draw_digit(rgb, width, cursor, y, char, color, scale)
        cursor += 4 * scale


def write_png(path: Path, width: int, height: int, rgb: bytearray) -> None:
    rows = []
    for y in range(height):
        rows.append(b"\x00" + bytes(rgb[y * width * 3:(y + 1) * width * 3]))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def render() -> Path:
    panel_px = MAP_SIZE * SCALE
    width = len(RADII) * panel_px + (len(RADII) - 1) * PANEL_GAP
    height = TOP_PAD + panel_px + BOTTOM_PAD
    rgb = bytearray([248, 248, 244] * width * height)

    foundation = foundation_tiles()
    colors = {
        "grid": (214, 214, 210),
        "foundation": (245, 190, 190),
        "workspace": (70, 130, 230),
        "overlap": (145, 72, 190),
        "agent": (30, 30, 30),
        "text": (30, 30, 30),
    }

    for panel_idx, radius in enumerate(RADII):
        x_panel = panel_idx * (panel_px + PANEL_GAP)
        y_panel = TOP_PAD
        mask = workspace_mask(radius)

        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                x0 = x_panel + col * SCALE
                y0 = y_panel + row * SCALE
                color = (252, 252, 249)
                if (row, col) in foundation:
                    color = colors["foundation"]
                if (row, col) in mask:
                    color = colors["overlap"] if (row, col) in foundation else colors["workspace"]
                fill_rect(rgb, width, x0, y0, x0 + SCALE, y0 + SCALE, color)

        for i in range(MAP_SIZE + 1):
            gx = x_panel + i * SCALE
            gy = y_panel + i * SCALE
            fill_rect(rgb, width, gx, y_panel, gx + 1, y_panel + panel_px, colors["grid"])
            fill_rect(rgb, width, x_panel, gy, x_panel + panel_px, gy + 1, colors["grid"])

        ar, ac = AGENT_POS
        fill_rect(
            rgb,
            width,
            x_panel + (ac - AGENT_WIDTH_TILES // 2) * SCALE,
            y_panel + (ar - AGENT_HEIGHT_TILES // 2) * SCALE,
            x_panel + (ac + AGENT_WIDTH_TILES // 2 + 1) * SCALE,
            y_panel + (ar + AGENT_HEIGHT_TILES // 2 + 1) * SCALE,
            colors["agent"],
        )

        r_min_m = 0.5 + max(AGENT_WIDTH_TILES / 2.0, AGENT_HEIGHT_TILES / 2.0) * TILE_SIZE
        r_max_m = r_min_m + radius * TILE_SIZE
        draw_text(rgb, width, x_panel + 6, 6, f"r={radius}", colors["text"], 2)
        draw_text(rgb, width, x_panel + 110, 6, f"{r_max_m:.1f}m", colors["text"], 2)
        draw_text(rgb, width, x_panel + 6, TOP_PAD + panel_px + 8, f"{len(mask)}", colors["text"], 2)

    output = Path(__file__).with_name("excavator_workspace_cone_r4_r5_r6.png")
    write_png(output, width, height, rgb)
    return output


if __name__ == "__main__":
    print(render())
