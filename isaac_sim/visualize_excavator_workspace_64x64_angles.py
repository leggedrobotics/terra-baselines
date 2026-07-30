#!/usr/bin/env python3
"""Render the 64x64 Terra excavator workspace at the new meter scale.

The script is dependency-free on purpose. It mirrors the excavator workspace
math from terra/terra/state.py and writes a PNG contact sheet showing all 12
cabin angles around one correctly scaled excavator footprint.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


MAP_SIZE = 64
METERS_PER_TILE = 0.572
EDGE_LENGTH_M = MAP_SIZE * METERS_PER_TILE

EXCAVATOR_LONG_SIDE_M = 6.08
EXCAVATOR_SHORT_SIDE_M = 3.50

AGENT_HEIGHT_TILES = 11
AGENT_WIDTH_TILES = 7
ANGLES_CABIN = 12
DIG_RADIUS_TILES = 5
FIXED_EXTENSION_M = 0.5

AGENT_POS = (32, 32)  # [row, col]
FOUNDATION_SIZE_TILES = 14

SCALE = 7
LABEL_H = 34
PANEL_PAD = 14
PANEL_GAP = 18
COLS = 4
ROWS = 3

PANEL_MAP_PX = MAP_SIZE * SCALE
PANEL_W = PANEL_MAP_PX
PANEL_H = LABEL_H + PANEL_MAP_PX + 30
CANVAS_W = COLS * PANEL_W + (COLS - 1) * PANEL_GAP + 2 * PANEL_PAD
CANVAS_H = ROWS * PANEL_H + (ROWS - 1) * PANEL_GAP + 2 * PANEL_PAD + 48


Color = tuple[int, int, int]


FONT = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "00000", "01100", "01000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def apply_rot_transl(theta: float, origin_xy: tuple[float, float], point_xy: tuple[float, float]) -> tuple[float, float]:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = point_xy[0] - origin_xy[0]
    dy = point_xy[1] - origin_xy[1]
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


def local_cartesian_to_cyl(local_xy: tuple[float, float]) -> tuple[float, float]:
    x, y = local_xy
    return math.sqrt(x * x + y * y), math.atan2(-x, y)


def angle_idx_to_rad(angle_idx: int) -> float:
    angle = 2.0 * math.pi * angle_idx / ANGLES_CABIN
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def agent_tile_dim(physical_m: float) -> int:
    rounded = round(physical_m / METERS_PER_TILE)
    return rounded if rounded % 2 != 0 else rounded + 1


def blend(base: Color, over: Color, alpha: float) -> Color:
    return tuple(round((1 - alpha) * b + alpha * o) for b, o in zip(base, over))


def fill_rect(rgb: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, y0)
    y1 = max(0, y1)
    for y in range(y0, y1):
        row_start = y * width * 3
        for x in range(x0, x1):
            idx = row_start + x * 3
            rgb[idx:idx + 3] = bytes(color)


def draw_text(rgb: bytearray, width: int, x: int, y: int, text: str, color: Color, scale: int = 2) -> None:
    cursor = x
    for char in text.upper():
        pattern = FONT.get(char, FONT[" "])
        for py, line in enumerate(pattern):
            for px, bit in enumerate(line):
                if bit == "1":
                    fill_rect(
                        rgb,
                        width,
                        cursor + px * scale,
                        y + py * scale,
                        cursor + (px + 1) * scale,
                        y + (py + 1) * scale,
                        color,
                    )
        cursor += 6 * scale


def draw_line(rgb: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        fill_rect(rgb, width, x0, y0, x0 + 1, y0 + 1, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def body_tiles() -> set[tuple[int, int]]:
    row_c, col_c = AGENT_POS
    rows = range(row_c - AGENT_HEIGHT_TILES // 2, row_c + AGENT_HEIGHT_TILES // 2 + 1)
    cols = range(col_c - AGENT_WIDTH_TILES // 2, col_c + AGENT_WIDTH_TILES // 2 + 1)
    return {(row, col) for row in rows for col in cols}


def foundation_tiles() -> set[tuple[int, int]]:
    start = MAP_SIZE // 2 - FOUNDATION_SIZE_TILES // 2
    end = start + FOUNDATION_SIZE_TILES
    return {(row, col) for row in range(start, end) for col in range(start, end)}


def workspace_mask(angle_idx: int) -> set[tuple[int, int]]:
    current_pos = ((AGENT_POS[0] + 0.5) * METERS_PER_TILE, (AGENT_POS[1] + 0.5) * METERS_PER_TILE)
    max_agent_dim = max(AGENT_WIDTH_TILES / 2.0, AGENT_HEIGHT_TILES / 2.0)
    min_distance_from_agent = METERS_PER_TILE * max_agent_dim

    r_min = FIXED_EXTENSION_M + min_distance_from_agent
    r_max = r_min + DIG_RADIUS_TILES * METERS_PER_TILE
    theta_max = 2.0 * math.pi / ANGLES_CABIN
    theta_min = -theta_max

    agent_width_m = AGENT_WIDTH_TILES * METERS_PER_TILE
    agent_height_m = AGENT_HEIGHT_TILES * METERS_PER_TILE
    eps = METERS_PER_TILE / 2.0
    exclude_x_half = math.floor((agent_width_m + eps) / 2.0)
    exclude_y_half = math.floor((agent_height_m + eps) / 2.0)

    theta = angle_idx_to_rad(angle_idx)
    mask = set()
    radius_by_tile = {}
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            point = ((row + 0.5) * METERS_PER_TILE, (col + 0.5) * METERS_PER_TILE)
            local_arm = apply_rot_transl(theta, current_pos, point)
            radius, local_theta = local_cartesian_to_cyl(local_arm)
            radius_by_tile[(row, col)] = radius
            in_cyl = r_min <= radius <= r_max and theta_min <= local_theta <= theta_max

            local_base = apply_rot_transl(0.0, current_pos, point)
            outside_agent_x = local_base[0] >= exclude_x_half or local_base[0] <= -exclude_x_half
            outside_agent_y = local_base[1] >= exclude_y_half or local_base[1] <= -exclude_y_half
            if in_cyl and (outside_agent_x or outside_agent_y):
                mask.add((row, col))

    cleaned = set()
    inner_band_limit = r_min + METERS_PER_TILE
    for row, col in mask:
        neighbor_count_4 = sum(
            (nr, nc) in mask
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        )
        remove = radius_by_tile[(row, col)] < inner_band_limit and neighbor_count_4 <= 1
        if not remove:
            cleaned.add((row, col))
    return cleaned


def panel_origin(panel_idx: int) -> tuple[int, int]:
    row = panel_idx // COLS
    col = panel_idx % COLS
    x = PANEL_PAD + col * (PANEL_W + PANEL_GAP)
    y = PANEL_PAD + 48 + row * (PANEL_H + PANEL_GAP)
    return x, y


def draw_panel(rgb: bytearray, angle_idx: int, panel_idx: int) -> None:
    colors = {
        "panel": (252, 253, 252),
        "grid": (218, 223, 220),
        "grid_8": (178, 187, 183),
        "border": (74, 86, 82),
        "target": (239, 170, 150),
        "workspace": (54, 129, 214),
        "overlap": (121, 67, 163),
        "body": (34, 38, 42),
        "cab": (238, 175, 54),
        "text": (27, 33, 35),
    }
    x0, y0 = panel_origin(panel_idx)
    map_x = x0
    map_y = y0 + LABEL_H
    target = foundation_tiles()
    body = body_tiles()
    workspace = workspace_mask(angle_idx)

    fill_rect(rgb, CANVAS_W, x0, y0, x0 + PANEL_W, y0 + PANEL_H, colors["panel"])

    degree = round(math.degrees(angle_idx_to_rad(angle_idx)))
    if degree == -180:
        degree = 180
    draw_text(rgb, CANVAS_W, x0, y0 + 4, f"CABIN {degree:+d} DEG", colors["text"], 2)
    draw_text(rgb, CANVAS_W, x0 + 210, y0 + 4, f"{len(workspace)} TILES", colors["text"], 2)

    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            color = (247, 250, 248)
            if (row, col) in target:
                color = colors["target"]
            if (row, col) in workspace:
                color = colors["overlap"] if (row, col) in target else colors["workspace"]
                if (row, col) not in target:
                    color = blend((247, 250, 248), color, 0.82)
            if (row, col) in body:
                color = colors["body"]
            px = map_x + col * SCALE
            py = map_y + row * SCALE
            fill_rect(rgb, CANVAS_W, px, py, px + SCALE, py + SCALE, color)

    for i in range(MAP_SIZE + 1):
        grid_color = colors["grid_8"] if i % 8 == 0 else colors["grid"]
        gx = map_x + i * SCALE
        gy = map_y + i * SCALE
        fill_rect(rgb, CANVAS_W, gx, map_y, gx + 1, map_y + PANEL_MAP_PX, grid_color)
        fill_rect(rgb, CANVAS_W, map_x, gy, map_x + PANEL_MAP_PX, gy + 1, grid_color)

    fill_rect(rgb, CANVAS_W, map_x, map_y, map_x + PANEL_MAP_PX, map_y + 2, colors["border"])
    fill_rect(rgb, CANVAS_W, map_x, map_y + PANEL_MAP_PX - 2, map_x + PANEL_MAP_PX, map_y + PANEL_MAP_PX, colors["border"])
    fill_rect(rgb, CANVAS_W, map_x, map_y, map_x + 2, map_y + PANEL_MAP_PX, colors["border"])
    fill_rect(rgb, CANVAS_W, map_x + PANEL_MAP_PX - 2, map_y, map_x + PANEL_MAP_PX, map_y + PANEL_MAP_PX, colors["border"])

    center_x = map_x + round((AGENT_POS[1] + 0.5) * SCALE)
    center_y = map_y + round((AGENT_POS[0] + 0.5) * SCALE)
    angle = angle_idx_to_rad(angle_idx)
    arrow_len = 8 * SCALE
    arrow_x = center_x + round(math.sin(angle) * arrow_len)
    arrow_y = center_y - round(math.cos(angle) * arrow_len)
    draw_line(rgb, CANVAS_W, center_x, center_y, arrow_x, arrow_y, colors["cab"])
    fill_rect(rgb, CANVAS_W, center_x - 3, center_y - 3, center_x + 4, center_y + 4, colors["cab"])

    scale_tiles = 10.0 / METERS_PER_TILE
    sx0 = map_x + 10
    sy0 = map_y + PANEL_MAP_PX + 14
    sx1 = sx0 + round(scale_tiles * SCALE)
    draw_line(rgb, CANVAS_W, sx0, sy0, sx1, sy0, colors["border"])
    draw_line(rgb, CANVAS_W, sx0, sy0 - 5, sx0, sy0 + 5, colors["border"])
    draw_line(rgb, CANVAS_W, sx1, sy0 - 5, sx1, sy0 + 5, colors["border"])
    draw_text(rgb, CANVAS_W, sx1 + 8, sy0 - 8, "10 M", colors["text"], 1)


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
    rgb = bytearray([(244, 247, 245)[i % 3] for i in range(CANVAS_W * CANVAS_H * 3)])
    title = f"64X64 MAP  {METERS_PER_TILE:.3f} M/TILE  EDGE {EDGE_LENGTH_M:.2f} M"
    subtitle = (
        f"EXCAVATOR {AGENT_HEIGHT_TILES}X{AGENT_WIDTH_TILES} TILES "
        f"({EXCAVATOR_LONG_SIDE_M:.2f}X{EXCAVATOR_SHORT_SIDE_M:.2f} M), "
        f"WORKSPACE R={DIG_RADIUS_TILES} TILES"
    )
    draw_text(rgb, CANVAS_W, PANEL_PAD, 14, title, (24, 31, 33), 2)
    draw_text(rgb, CANVAS_W, PANEL_PAD, 34, subtitle, (72, 82, 82), 1)

    for idx in range(ANGLES_CABIN):
        draw_panel(rgb, idx, idx)

    output = Path(__file__).with_name("excavator_workspace_64x64_0p572_angles.png")
    write_png(output, CANVAS_W, CANVAS_H, rgb)
    return output


if __name__ == "__main__":
    print(render())
