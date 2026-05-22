#!/usr/bin/env python3
"""
Render a static Terra plan manual from an extracted waypoint plan.

The input plan is the PKL or JSON produced by isaac_sim/extract_map.py. The
output is a single PNG that overlays each DO waypoint's changed workspace on the
map, labels it in execution order, and marks the base position used for that
workspace. This is meant as an inspection/manual artifact, not a rollout render.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


KIND_STYLES = {
    "dig": {
        "label": "dig",
        "fill": (38, 118, 255),
        "outline": (10, 65, 180),
    },
    "lift": {
        "label": "lift",
        "fill": (150, 72, 210),
        "outline": (88, 34, 145),
    },
    "dump": {
        "label": "dump",
        "fill": (245, 135, 28),
        "outline": (170, 75, 0),
    },
    "terrain": {
        "label": "terrain",
        "fill": (80, 80, 80),
        "outline": (35, 35, 35),
    },
}


def _load_plan(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "waypoints" in data:
            data = data["waypoints"]
        if not isinstance(data, list):
            raise ValueError(f"Expected a list or {{'waypoints': list}} in {path}")
        return data

    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _load_map_arrays(map_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.load(map_path / "images" / "img_1.npy")
    occupancy_path = map_path / "occupancy" / "img_1.npy"
    dumpability_path = map_path / "dumpability" / "img_1.npy"

    occupancy = (
        np.load(occupancy_path).astype(bool)
        if occupancy_path.exists()
        else np.zeros_like(target, dtype=bool)
    )
    dumpability = (
        np.load(dumpability_path).astype(bool)
        if dumpability_path.exists()
        else np.ones_like(target, dtype=bool)
    )
    return target, occupancy, dumpability


def _as_bool_array(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value).astype(bool)
    if arr.shape != shape:
        arr = np.reshape(arr, shape)
    return arr


def _as_py_bool(value: Any) -> bool:
    try:
        return bool(np.asarray(value).reshape(-1)[0])
    except Exception:
        return bool(value)


def _waypoint_kind(entry: dict[str, Any]) -> str:
    loaded_change = entry.get("loaded_state_change", {}) or {}
    before = _as_py_bool(loaded_change.get("before", False))
    after = _as_py_bool(loaded_change.get("after", False))
    if not before and after:
        return "lift" if entry.get("dig_type") == "lift_dug_dirt" else "dig"
    if before and not after:
        return "dump"
    return "terrain"


def _workspace_mask(entry: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = _as_bool_array(entry.get("terrain_modification_mask"), shape)
    if mask.any():
        return mask

    # Fallback for older plans that may not contain terrain_modification_mask.
    kind = _waypoint_kind(entry)
    if kind in ("dig", "lift"):
        return _as_bool_array(entry.get("dug_mask"), shape)
    if kind == "dump":
        return _as_bool_array(entry.get("dump_mask"), shape)
    return mask


def _base_rgb(target: np.ndarray, occupancy: np.ndarray, dumpability: np.ndarray) -> np.ndarray:
    rgb = np.full((*target.shape, 3), 246, dtype=np.uint8)
    rgb[target < 0] = (255, 224, 224)
    rgb[target > 0] = (226, 250, 224)
    rgb[(~occupancy) & (~dumpability)] = (224, 224, 224)
    rgb[occupancy] = (25, 25, 25)
    return rgb


def _alpha_fill(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    color_arr = np.asarray(color, dtype=np.float32)
    base = rgb[mask].astype(np.float32)
    rgb[mask] = np.clip((1.0 - alpha) * base + alpha * color_arr, 0, 255).astype(np.uint8)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    padded = np.pad(mask, 1, constant_values=False)
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~neighbors


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def _agent_pos(entry: dict[str, Any]) -> tuple[float, float] | None:
    pos = (entry.get("agent_state") or {}).get("pos_base")
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    # Terra stores pos_base=[x, y] where x is row/axis-0 and y is col/axis-1.
    return float(arr[0]), float(arr[1])


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2 - 1), text, font=font, fill=fill)


def _draw_number_badge(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    font: ImageFont.ImageFont,
    radius: int,
) -> None:
    x, y = center_xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=max(1, radius // 4),
    )
    _draw_centered_text(draw, center_xy, label, font, (255, 255, 255))


def render_plan_manual(
    plan: list[dict[str, Any]],
    map_path: Path,
    output_path: Path,
    scale: int = 10,
    alpha: float = 0.38,
    label_mode: str = "index",
    draw_legend: bool = True,
) -> None:
    target, occupancy, dumpability = _load_map_arrays(map_path)
    h, w = target.shape
    shape = (h, w)
    rgb = _base_rgb(target, occupancy, dumpability)

    parsed = []
    for order_idx, entry in enumerate(plan, start=1):
        kind = _waypoint_kind(entry)
        mask = _workspace_mask(entry, shape)
        parsed.append((order_idx, entry, kind, mask))
        _alpha_fill(rgb, mask, KIND_STYLES[kind]["fill"], alpha)

    image = Image.fromarray(rgb, mode="RGB").resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    label_font = _load_font(max(9, int(scale * 1.45)), bold=True)
    small_font = _load_font(max(9, int(scale * 1.15)))
    badge_radius = max(5, int(scale * 1.1))

    for order_idx, entry, kind, mask in parsed:
        style = KIND_STYLES[kind]
        boundary = _mask_boundary(mask)
        rows, cols = np.nonzero(boundary)
        for row, col in zip(rows, cols):
            x0 = int(col * scale)
            y0 = int(row * scale)
            draw.rectangle(
                (x0, y0, x0 + scale - 1, y0 + scale - 1),
                outline=style["outline"],
                width=max(1, scale // 4),
            )

        workspace_center = _centroid(mask)
        agent_center = _agent_pos(entry)
        label_center = workspace_center or agent_center
        if label_center is None:
            continue

        label_row, label_col = label_center
        label_xy = ((label_col + 0.5) * scale, (label_row + 0.5) * scale)

        if agent_center is not None:
            agent_row, agent_col = agent_center
            agent_xy = ((agent_col + 0.5) * scale, (agent_row + 0.5) * scale)
            marker_r = max(2, scale // 3)
            draw.ellipse(
                (
                    agent_xy[0] - marker_r,
                    agent_xy[1] - marker_r,
                    agent_xy[0] + marker_r,
                    agent_xy[1] + marker_r,
                ),
                fill=style["outline"],
                outline=(255, 255, 255),
                width=1,
            )
            if workspace_center is not None:
                draw.line((agent_xy[0], agent_xy[1], label_xy[0], label_xy[1]), fill=style["outline"], width=1)

        text = str(order_idx) if label_mode == "index" else str(entry.get("step", order_idx))
        _draw_number_badge(draw, label_xy, text, style["fill"], style["outline"], label_font, badge_radius)

    if draw_legend:
        legend_width = max(180, scale * 20)
        legend = Image.new("RGB", (image.width + legend_width, image.height), (255, 255, 255))
        legend.paste(image, (0, 0))
        draw = ImageDraw.Draw(legend)
        x = image.width + 14
        y = 14
        title_font = _load_font(max(11, int(scale * 1.25)), bold=True)
        draw.text((x, y), "Plan manual", font=title_font, fill=(20, 20, 20))
        y += int(scale * 2.4)
        counts = {kind: 0 for kind in KIND_STYLES}
        for _, _, kind, _ in parsed:
            counts[kind] += 1
        for kind, style in KIND_STYLES.items():
            if counts[kind] == 0:
                continue
            draw.rectangle((x, y, x + 14, y + 14), fill=style["fill"], outline=style["outline"])
            draw.text((x + 22, y - 1), f"{style['label']}: {counts[kind]}", font=small_font, fill=(35, 35, 35))
            y += int(scale * 2.2)
        y += int(scale * 0.8)
        draw.text((x, y), f"waypoints: {len(plan)}", font=small_font, fill=(35, 35, 35))
        y += int(scale * 1.8)
        draw.text((x, y), "dot = base", font=small_font, fill=(35, 35, 35))
        y += int(scale * 1.8)
        draw.text((x, y), "number = workspace", font=small_font, fill=(35, 35, 35))
        image = legend

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static PNG manual from an extracted Terra plan.")
    parser.add_argument("-p", "--plan", required=True, type=Path, help="Path to terra_plan.pkl or terra_plan.json.")
    parser.add_argument("-m", "--map_path", required=True, type=Path, help="Map directory used to extract the plan.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--scale", type=int, default=10, help="Pixels per map tile in the output PNG.")
    parser.add_argument("--alpha", type=float, default=0.38, help="Workspace overlay opacity in [0, 1].")
    parser.add_argument(
        "--label-mode",
        choices=("index", "step"),
        default="index",
        help="Use compact waypoint order labels or original env step numbers.",
    )
    parser.add_argument("--no-legend", action="store_true", help="Do not append the right-side legend.")
    args = parser.parse_args()

    plan = _load_plan(args.plan)
    output = args.output
    if output is None:
        output = args.plan.with_name(f"{args.plan.stem}_manual.png")

    render_plan_manual(
        plan=plan,
        map_path=args.map_path,
        output_path=output,
        scale=max(1, args.scale),
        alpha=float(np.clip(args.alpha, 0.0, 1.0)),
        label_mode=args.label_mode,
        draw_legend=not args.no_legend,
    )
    print(f"Saved plan manual PNG to {output}")


if __name__ == "__main__":
    main()
