#!/usr/bin/env python3
"""Replay a fixed-bank trench benchmark and build a trajectory GIF gallery.

The fixed-bank JSON remains the quantitative source of truth.  This tool reads
that receipt, selects representative successes and stalls without hand-picking
map identities, replays the *full* panel so selected trajectories retain the
same reset and step RNG contract, and renders a portable browser gallery.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_fixed_bank import (  # noqa: E402
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
    sha256_file,
)
from eval_mcts import _apply_in_batch_chunks  # noqa: E402
from train import TrainConfig  # noqa: E402
from train_mixed import (  # noqa: E402
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.accepted_bank import V8_RELEASE_ID, load_accepted_bank  # noqa: E402
from utils.helpers import load_pkl_object  # noqa: E402
from utils.models import validate_model_params_match  # noqa: E402
from utils.utils_ppo import (
    _config_option,
    obs_to_model_input,
    wrap_action,
)  # noqa: E402

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

SCHEMA = "terra_trench_benchmark_gallery_v1"
ACTION_NAMES = {
    -1: "initial",
    0: "forward",
    1: "backward",
    2: "base clockwise",
    3: "base anticlockwise",
    4: "cabin clockwise",
    5: "cabin anticlockwise",
    6: "DO",
    7: "do nothing",
}
DO_ACTION = 6
MAP_SIZE = 384
CELL_SIZE = MAP_SIZE // 64
CANVAS_SIZE = (574, 468)


def safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in slug.split("-") if part)


def trench_geometry(condition_id: str) -> str:
    if not condition_id.startswith("trn-"):
        raise ValueError(f"not a trench condition: {condition_id}")
    body = condition_id[4:]
    for geometry in ("straight", "tee", "seg2", "seg3", "net3", "net4"):
        if body == geometry or body.startswith(geometry + "-"):
            return geometry
    raise ValueError(f"unknown trench geometry: {condition_id}")


def _median_row(rows: list[dict], key) -> dict | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (key(row), int(row["slot_index"])))
    return ordered[(len(ordered) - 1) // 2]


def select_representatives(per_map: list[dict], per_condition: int) -> list[dict]:
    """Select outcome-balanced, deterministic representatives per condition."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_map:
        if row.get("family") == "trench":
            grouped[row["primary_cell"]].append(row)

    selected: list[dict] = []
    for condition in sorted(grouped):
        rows = grouped[condition]
        successes = [row for row in rows if bool(row["success"])]
        failures = [row for row in rows if not bool(row["success"])]
        candidates: list[tuple[str, dict | None]] = [
            ("median success", _median_row(successes, lambda row: row["steps"])),
            ("typical stall", _median_row(failures, lambda row: row["dig_fraction"])),
            (
                "highest-progress stall",
                max(
                    failures,
                    key=lambda row: (row["dig_fraction"], -int(row["slot_index"])),
                    default=None,
                ),
            ),
            (
                "fast success",
                min(
                    successes,
                    key=lambda row: (row["steps"], int(row["slot_index"])),
                    default=None,
                ),
            ),
            (
                "lowest-progress stall",
                min(
                    failures,
                    key=lambda row: (row["dig_fraction"], int(row["slot_index"])),
                    default=None,
                ),
            ),
            ("slow success", _median_row(successes, lambda row: -row["steps"])),
        ]
        seen: set[int] = set()
        condition_selected: list[dict] = []
        for role, candidate in candidates:
            if candidate is None:
                continue
            slot = int(candidate["slot_index"])
            if slot in seen:
                continue
            enriched = dict(candidate)
            enriched["gallery_role"] = role
            enriched["geometry"] = trench_geometry(condition)
            enriched["structural_exclusion"] = condition.startswith("trn-net4-")
            condition_selected.append(enriched)
            seen.add(slot)
            if len(condition_selected) == per_condition:
                break
        if len(condition_selected) < min(per_condition, len(rows)):
            for candidate in sorted(
                rows,
                key=lambda row: (
                    bool(row["success"]),
                    float(row["dig_fraction"]),
                    int(row["slot_index"]),
                ),
            ):
                slot = int(candidate["slot_index"])
                if slot in seen:
                    continue
                enriched = dict(candidate)
                enriched["gallery_role"] = "coverage fill"
                enriched["geometry"] = trench_geometry(condition)
                enriched["structural_exclusion"] = condition.startswith("trn-net4-")
                condition_selected.append(enriched)
                seen.add(slot)
                if len(condition_selected) == per_condition:
                    break
        selected.extend(condition_selected)
    return selected


def _load_benchmark(
    path: Path,
    checkpoint: Path,
    *,
    panel_family: str,
    accepted_panel: str,
    terra_revision: str,
) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("benchmark JSON must contain exactly one checkpoint record")
    record = payload[0]
    checkpoint_sha = sha256_file(checkpoint)
    if record.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("benchmark/checkpoint SHA-256 mismatch")
    if not bool(record.get("deterministic")):
        raise ValueError("gallery requires a deterministic fixed-bank benchmark")
    if not bool(record.get("exact_manifest_enumeration")):
        raise ValueError("benchmark did not enumerate the exact manifest")
    if not isinstance(record.get("per_map"), list) or not record["per_map"]:
        raise ValueError("benchmark has no per-map records")
    if record.get("split") != accepted_panel:
        raise ValueError("benchmark accepted panel does not match gallery request")
    accepted_bank = record.get("accepted_bank")
    if not isinstance(accepted_bank, dict):
        raise ValueError("benchmark has no accepted-bank provenance")
    if accepted_bank.get("evaluation_panel_family") != panel_family:
        raise ValueError("benchmark panel family does not match gallery request")
    if accepted_bank.get("terra_revision") != terra_revision:
        raise ValueError(
            "benchmark Terra protocol revision does not match gallery request"
        )
    return record


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_SMALL = _font(12)
FONT_BODY = _font(14)
FONT_TITLE = _font(16, bold=True)


def _completion(target: np.ndarray, action_map: np.ndarray) -> float:
    target_cells = target < 0
    total = int(target_cells.sum())
    if total == 0:
        return 1.0
    return float((target_cells & (action_map < 0)).sum() / total)


def _draw_legend(draw: ImageDraw.ImageDraw, y: int) -> None:
    items = [
        ((216, 92, 80), "remaining"),
        ((54, 126, 184), "dug"),
        ((174, 111, 45), "soil"),
        ((80, 155, 93), "dump zone"),
        ((55, 59, 68), "obstacle"),
    ]
    x = 14
    for color, label in items:
        draw.rectangle((x, y + 2, x + 11, y + 13), fill=color)
        draw.text((x + 16, y), label, font=FONT_SMALL, fill=(54, 57, 64))
        x += 16 + int(draw.textlength(label, font=FONT_SMALL)) + 18


def render_frame(
    snapshot: dict[str, np.ndarray],
    local_index: int,
    meta: dict,
    path_points: list[tuple[float, float]],
    *,
    step: int,
    horizon: int,
    action: int,
    action_had_effect: bool,
    done: bool,
    succeeded: bool,
    pre_alignment_valid: bool,
    pre_yaw_error: float,
    pre_standoff_error: float,
) -> Image.Image:
    target = np.asarray(snapshot["target_map"][local_index])
    action_map = np.asarray(snapshot["action_map"][local_index])
    padding = np.asarray(snapshot["padding_mask"][local_index])
    dumpability = np.asarray(snapshot["dumpability_mask"][local_index])
    interaction = np.asarray(snapshot["interaction_mask"][local_index]) > 0
    axes = np.asarray(snapshot["trench_axes"][local_index])
    trench_type = int(np.asarray(snapshot["trench_type"][local_index]).reshape(-1)[0])
    agent = np.asarray(snapshot["agent_states"][local_index, 0])

    grid = np.empty((64, 64, 3), dtype=np.uint8)
    grid[:] = (238, 231, 211)
    grid[dumpability == 0] = (211, 207, 197)
    grid[target > 0] = (80, 155, 93)
    grid[target < 0] = (216, 92, 80)
    grid[(target < 0) & (action_map < 0)] = (54, 126, 184)
    grid[(target >= 0) & (action_map < 0)] = (72, 111, 151)
    positive = action_map > 0
    if np.any(positive):
        height = np.clip(
            action_map.astype(np.float32) / max(float(action_map.max()), 1.0), 0, 1
        )
        soil = np.stack(
            (
                154 + 42 * height,
                92 + 42 * height,
                31 + 32 * height,
            ),
            axis=-1,
        ).astype(np.uint8)
        grid[positive] = soil[positive]
    if np.any(interaction):
        overlay = np.array((67, 190, 205), dtype=np.float32)
        grid[interaction] = (
            0.68 * grid[interaction].astype(np.float32) + 0.32 * overlay
        ).astype(np.uint8)
    grid[padding > 0] = (55, 59, 68)

    map_image = Image.fromarray(grid, mode="RGB").resize(
        (MAP_SIZE, MAP_SIZE), Image.Resampling.NEAREST
    )
    image = Image.new("RGB", CANVAS_SIZE, (247, 245, 239))
    image.paste(map_image, (12, 60))
    draw = ImageDraw.Draw(image)

    condition = str(meta["primary_cell"])
    outcome = "SUCCESS" if bool(meta["success"]) else "STALL"
    outcome_color = (30, 126, 73) if bool(meta["success"]) else (174, 57, 53)
    draw.text((14, 9), condition, font=FONT_TITLE, fill=(31, 34, 40))
    draw.text(
        (14, 32),
        f"slot {int(meta['slot_index'])}  ·  {meta['gallery_role']}",
        font=FONT_BODY,
        fill=(77, 80, 87),
    )
    badge_width = int(draw.textlength(outcome, font=FONT_TITLE)) + 18
    draw.rounded_rectangle(
        (CANVAS_SIZE[0] - badge_width - 14, 9, CANVAS_SIZE[0] - 14, 34),
        radius=7,
        fill=outcome_color,
    )
    draw.text(
        (CANVAS_SIZE[0] - badge_width - 5, 12),
        outcome,
        font=FONT_TITLE,
        fill=(255, 255, 255),
    )

    # Section centerlines provide a direct visual reference for alignment.
    for section in axes[: max(0, trench_type)]:
        if section.shape[0] < 8 or not np.all(np.isfinite(section[3:7])):
            continue
        y1, x1, y2, x2 = [float(value) for value in section[3:7]]
        draw.line(
            (
                12 + (x1 + 0.5) * CELL_SIZE,
                60 + (y1 + 0.5) * CELL_SIZE,
                12 + (x2 + 0.5) * CELL_SIZE,
                60 + (y2 + 0.5) * CELL_SIZE,
            ),
            fill=(247, 206, 70),
            width=2,
        )

    if len(path_points) > 1:
        path_xy = [
            (12 + (col + 0.5) * CELL_SIZE, 60 + (row + 0.5) * CELL_SIZE)
            for row, col in path_points
        ]
        draw.line(path_xy, fill=(31, 42, 61), width=2)

    row, col = float(agent[0]), float(agent[1])
    heading_index = float(agent[2])
    angle = 2.0 * math.pi * heading_index / 12.0
    cx = 12 + (col + 0.5) * CELL_SIZE
    cy = 60 + (row + 0.5) * CELL_SIZE
    radius = 7
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(26, 31, 43),
        outline=(255, 255, 255),
        width=2,
    )
    dx = math.cos(angle) * 18
    dy = -math.sin(angle) * 18
    draw.line((cx, cy, cx + dx, cy + dy), fill=(255, 255, 255), width=3)

    panel_x = 410
    current_completion = _completion(target, action_map)
    lines = [
        ("step", f"{step}/{horizon}"),
        ("action", ACTION_NAMES.get(action, str(action))),
        ("effect", "yes" if action_had_effect else "no"),
        ("aligned", "yes" if pre_alignment_valid else "no"),
        ("yaw error", f"{pre_yaw_error:.3f}"),
        ("standoff err", f"{pre_standoff_error:.3f}"),
        ("dig progress", f"{100 * current_completion:.1f}%"),
        ("final dig", f"{100 * float(meta['dig_fraction']):.1f}%"),
        ("final steps", str(int(meta["steps"]))),
        ("no-effect", str(int(meta["no_effect_action_count"]))),
    ]
    draw.text((panel_x, 65), "trajectory", font=FONT_TITLE, fill=(31, 34, 40))
    y = 94
    for label, value in lines:
        draw.text((panel_x, y), label, font=FONT_SMALL, fill=(101, 102, 108))
        draw.text((panel_x, y + 15), value, font=FONT_BODY, fill=(31, 34, 40))
        y += 34
    if bool(meta.get("structural_exclusion")):
        draw.rounded_rectangle((panel_x, 414, 562, 442), radius=6, fill=(231, 188, 55))
        draw.text(
            (panel_x + 8, 421),
            "net4 diagnostic only",
            font=FONT_SMALL,
            fill=(46, 40, 20),
        )
    elif done:
        terminal = "completed" if succeeded else "horizon stall"
        draw.text((panel_x, 422), terminal, font=FONT_TITLE, fill=outcome_color)

    _draw_legend(draw, 449)
    return image


def _selected_snapshot(timestep, selected: np.ndarray) -> dict[str, np.ndarray]:
    obs = timestep.observation
    payload = {
        "target_map": obs["target_map"][selected],
        "action_map": timestep.state.world.action_map.map[selected],
        "padding_mask": obs["padding_mask"][selected],
        "dumpability_mask": obs["dumpability_mask"][selected],
        "interaction_mask": obs["interaction_mask"][selected],
        "agent_states": obs["agent_states"][selected],
        "trench_axes": timestep.state.world.trench_axes[selected],
        "trench_type": timestep.state.world.trench_type[selected],
    }
    return jax.device_get(payload)


def _append_frame(writer: dict, frame: Image.Image, duration_ms: int) -> None:
    writer["frames"].append(frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    writer["durations"].append(duration_ms)


def replay_and_render(
    *,
    checkpoint_path: Path,
    bank_root: Path,
    panel_family: str,
    accepted_panel: str,
    terra_revision: str,
    benchmark: dict,
    selection: list[dict],
    horizon: int,
    seed: int,
    frame_every: int,
    output_dir: Path,
) -> list[dict]:
    dataset = json.loads((bank_root / "dataset.json").read_text())
    release_id = dataset.get("release_id")
    accepted_bank = load_accepted_bank(
        bank_root,
        "G-UNIFORM",
        terra_revision,
        curriculum_stage="full" if release_id == V8_RELEASE_ID else None,
        evaluation_panel_family=panel_family,
    )
    panel = next(
        item for item in accepted_bank.evaluation_panels if item.name == accepted_panel
    )
    directory = bank_root / panel.maps_path
    manifest_rows = load_manifest(directory)
    count = len(manifest_rows)
    if count != len(benchmark["per_map"]):
        raise ValueError("benchmark/panel slot-count mismatch")
    if sha256_file(directory / "manifest.jsonl") != benchmark["manifest_sha256"]:
        raise ValueError("benchmark/panel manifest SHA-256 mismatch")

    checkpoint = load_pkl_object(str(checkpoint_path))
    train_config = checkpoint["train_config"]
    _validate_checkpoint_architecture(checkpoint, train_config)
    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(count)
    config = configure_for_bank(train_config, panel.maps_path, count)
    _validate_checkpoint_architecture(checkpoint, config)
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    validate_model_params_match(
        initialized_state.params, checkpoint["model"], str(checkpoint_path)
    )
    model = SimpleNamespace(apply=initialized_state.apply_fn)

    raw_gate = getattr(env_params, "enforce_trench_dig_alignment", None)
    if raw_gate is None or not bool(np.ravel(np.asarray(raw_gate))[0]):
        raise RuntimeError(
            "gallery checkpoint did not resolve to the strict trench gate"
        )

    map_keys = exact_reset_keys(count)
    state_keys = manifest_environment_keys(
        manifest_rows, count, accepted_bank.environment_protocol_sha256
    )
    timestep, env_params, _ = prepare_manifest_episode_reset(
        env, env_params, map_keys, state_keys
    )

    selected_zero = np.asarray(
        [int(meta["slot_index"]) - 1 for meta in selection], dtype=np.int32
    )
    if np.any(selected_zero < 0) or np.any(selected_zero >= count):
        raise ValueError("selection contains an out-of-range slot")
    if len(set(selected_zero.tolist())) != len(selected_zero):
        raise ValueError("selection contains duplicate slots")
    meta_by_zero = {int(meta["slot_index"]) - 1: meta for meta in selection}
    local_by_zero = {slot: index for index, slot in enumerate(selected_zero.tolist())}
    writers = {
        slot: {"meta": meta_by_zero[slot], "frames": [], "durations": [], "path": []}
        for slot in selected_zero.tolist()
    }

    rng = jrandom.PRNGKey(seed)
    rng, _ = jrandom.split(rng)
    prev_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    terminated = jnp.zeros(count, dtype=jnp.bool_)
    succeeded = jnp.zeros(count, dtype=jnp.bool_)
    lengths = jnp.zeros(count, dtype=jnp.int32)

    initial_snapshot = _selected_snapshot(timestep, selected_zero)
    initial_agents = np.asarray(initial_snapshot["agent_states"])
    for slot, writer in writers.items():
        local = local_by_zero[slot]
        writer["path"].append(
            (float(initial_agents[local, 0, 0]), float(initial_agents[local, 0, 1]))
        )
        obs = timestep.observation
        frame = render_frame(
            initial_snapshot,
            local,
            writer["meta"],
            writer["path"],
            step=0,
            horizon=horizon,
            action=-1,
            action_had_effect=False,
            done=False,
            succeeded=False,
            pre_alignment_valid=bool(
                np.asarray(obs["fresh_trench_dig_alignment_valid"])[slot] > 0.5
            ),
            pre_yaw_error=float(np.asarray(obs["fresh_trench_dig_yaw_error"])[slot]),
            pre_standoff_error=float(
                np.asarray(obs["fresh_trench_dig_standoff_error"])[slot]
            ),
        )
        _append_frame(writer, frame, 240)

    for step in range(1, horizon + 1):
        active = ~terminated
        pre_valid = np.asarray(
            timestep.observation["fresh_trench_dig_alignment_valid"]
        ).reshape(-1)
        pre_yaw = np.asarray(
            timestep.observation["fresh_trench_dig_yaw_error"]
        ).reshape(-1)
        pre_standoff = np.asarray(
            timestep.observation["fresh_trench_dig_standoff_error"]
        ).reshape(-1)

        rng, _rng_act, rng_step = jrandom.split(rng, 3)
        obs_model = obs_to_model_input(timestep.observation, prev_actions, config)
        _, logits = _apply_in_batch_chunks(model, checkpoint["model"], obs_model)
        if _config_option(config, "action_logit_masking", False):
            logits = jnp.where(obs_model[22], logits, jnp.float32(-1e9))
        action = jnp.argmax(logits, axis=-1)
        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)
        candidate = env.step_no_reset(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            jrandom.split(rng_step, count),
        )

        def preserve_inactive(previous, candidate_leaf):
            if not hasattr(candidate_leaf, "shape"):
                return candidate_leaf
            if candidate_leaf.ndim == 0 or candidate_leaf.shape[0] != count:
                return candidate_leaf
            mask = active.reshape((count,) + (1,) * (candidate_leaf.ndim - 1))
            return jnp.where(mask, candidate_leaf, previous)

        timestep = jax.tree_util.tree_map(preserve_inactive, timestep, candidate)
        prev_actions = jnp.where(
            timestep.done[:, None], jnp.zeros_like(prev_actions), prev_actions
        )
        step_done = timestep.done
        step_success = timestep.info["task_done"]
        lengths += active.astype(jnp.int32)
        succeeded |= active & step_success.astype(jnp.bool_)
        newly_done = np.asarray(active & step_done, dtype=bool)
        terminated |= active & step_done.astype(jnp.bool_)

        action_host = np.asarray(action, dtype=np.int32)
        effect_host = np.asarray(timestep.info["action_had_effect"], dtype=bool)
        positions = np.asarray(timestep.observation["agent_states"])[
            selected_zero, 0, :2
        ]
        for slot, writer in writers.items():
            if writer.get("closed"):
                continue
            local = local_by_zero[slot]
            writer["path"].append(
                (float(positions[local, 0]), float(positions[local, 1]))
            )

        capture_slots = [
            slot
            for slot, writer in writers.items()
            if not writer.get("closed")
            and (
                step % frame_every == 0
                or int(action_host[slot]) == DO_ACTION
                or bool(newly_done[slot])
                or step == horizon
            )
        ]
        if capture_slots:
            snapshot = _selected_snapshot(timestep, selected_zero)
            succeeded_host = np.asarray(succeeded, dtype=bool)
            for slot in capture_slots:
                writer = writers[slot]
                local = local_by_zero[slot]
                frame = render_frame(
                    snapshot,
                    local,
                    writer["meta"],
                    writer["path"],
                    step=step,
                    horizon=horizon,
                    action=int(action_host[slot]),
                    action_had_effect=bool(effect_host[slot]),
                    done=bool(newly_done[slot]) or step == horizon,
                    succeeded=bool(succeeded_host[slot]),
                    pre_alignment_valid=bool(pre_valid[slot] > 0.5),
                    pre_yaw_error=float(pre_yaw[slot]),
                    pre_standoff_error=float(pre_standoff[slot]),
                )
                duration = 240 if int(action_host[slot]) == DO_ACTION else 90
                if bool(newly_done[slot]) or step == horizon:
                    duration = 900
                _append_frame(writer, frame, duration)
                if bool(newly_done[slot]) or step == horizon:
                    writer["closed"] = True

        if step % 25 == 0:
            print(
                f"[gallery] step {step}/{horizon}; "
                f"panel done={int(np.asarray(terminated).sum())}/{count}",
                flush=True,
            )
        if bool(np.asarray(terminated).all()):
            break

    succeeded_host = np.asarray(succeeded, dtype=bool)
    lengths_host = np.asarray(lengths, dtype=np.int32)
    final_action = np.asarray(timestep.state.world.action_map.map)
    final_target = np.asarray(timestep.state.world.target_map.map)

    gifs_dir = output_dir / "gifs"
    posters_dir = output_dir / "posters"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    posters_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for ordinal, meta in enumerate(selection, start=1):
        slot = int(meta["slot_index"]) - 1
        writer = writers[slot]
        replay_success = bool(succeeded_host[slot])
        replay_steps = int(lengths_host[slot])
        replay_dig = _completion(final_target[slot], final_action[slot])
        if replay_success != bool(meta["success"]):
            raise RuntimeError(
                f"slot {slot + 1} replay success disagrees with benchmark"
            )
        if replay_steps != int(meta["steps"]):
            raise RuntimeError(
                f"slot {slot + 1} replay length disagrees with benchmark"
            )
        if not math.isclose(replay_dig, float(meta["dig_fraction"]), abs_tol=1e-6):
            raise RuntimeError(
                f"slot {slot + 1} replay dig fraction disagrees with benchmark"
            )
        slug = (
            f"{ordinal:03d}-{safe_slug(meta['primary_cell'])}-"
            f"slot-{int(meta['slot_index']):03d}-{safe_slug(meta['gallery_role'])}"
        )
        gif_path = gifs_dir / f"{slug}.gif"
        poster_path = posters_dir / f"{slug}.png"
        frames = writer["frames"]
        durations = writer["durations"]
        if not frames:
            raise RuntimeError(f"slot {slot + 1} produced no frames")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
        frames[-1].convert("RGB").save(poster_path, optimize=True)
        enriched = dict(meta)
        enriched.update(
            {
                "gif": gif_path.relative_to(output_dir).as_posix(),
                "poster": poster_path.relative_to(output_dir).as_posix(),
                "gif_sha256": sha256_file(gif_path),
                "poster_sha256": sha256_file(poster_path),
                "frame_count": len(frames),
                "replay_verified": True,
            }
        )
        artifacts.append(enriched)
        print(f"[gallery] wrote {gif_path.name} ({len(frames)} frames)", flush=True)
    return artifacts


def _summary(per_map: list[dict]) -> dict:
    trench = [row for row in per_map if row.get("family") == "trench"]
    endpoint = [
        row for row in trench if not row["primary_cell"].startswith("trn-net4-")
    ]
    net4 = [row for row in trench if row["primary_cell"].startswith("trn-net4-")]

    def aggregate(rows: list[dict]) -> dict:
        return {
            "episodes": len(rows),
            "successes": int(sum(bool(row["success"]) for row in rows)),
            "success_rate": float(
                sum(bool(row["success"]) for row in rows) / max(len(rows), 1)
            ),
            "dig_fraction_mean": float(np.mean([row["dig_fraction"] for row in rows])),
            "steps_mean": float(np.mean([row["steps"] for row in rows])),
        }

    conditions = {}
    for condition in sorted({row["primary_cell"] for row in trench}):
        rows = [row for row in trench if row["primary_cell"] == condition]
        conditions[condition] = {
            **aggregate(rows),
            "geometry": trench_geometry(condition),
            "structural_exclusion": condition.startswith("trn-net4-"),
        }
    return {
        "endpoint_without_net4": aggregate(endpoint),
        "net4_diagnostic": aggregate(net4),
        "all_trenches": aggregate(trench),
        "conditions": conditions,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_html(summary: dict, artifacts: list[dict], provenance: dict) -> str:
    geometries = sorted({item["geometry"] for item in artifacts})
    conditions = sorted({item["primary_cell"] for item in artifacts})
    condition_rows = []
    for condition, stats in summary["conditions"].items():
        exclusion = (
            "<span class='tag warn'>diagnostic</span>"
            if stats["structural_exclusion"]
            else ""
        )
        condition_rows.append(
            "<tr>"
            f"<td><code>{html.escape(condition)}</code> {exclusion}</td>"
            f"<td>{html.escape(stats['geometry'])}</td>"
            f"<td>{stats['successes']}/{stats['episodes']}</td>"
            f"<td><div class='bar'><span style='width:{100 * stats['success_rate']:.3f}%'></span></div>{_percent(stats['success_rate'])}</td>"
            f"<td>{_percent(stats['dig_fraction_mean'])}</td>"
            f"<td>{stats['steps_mean']:.1f}</td>"
            "</tr>"
        )
    cards = []
    for item in artifacts:
        outcome = "success" if item["success"] else "stall"
        diagnostic = " diagnostic" if item["structural_exclusion"] else ""
        cards.append(
            f"<article class='trajectory-card{diagnostic}' data-geometry='{html.escape(item['geometry'])}' data-condition='{html.escape(item['primary_cell'])}' data-outcome='{outcome}'>"
            f"<img src='{html.escape(item['poster'])}' data-poster='{html.escape(item['poster'])}' data-gif='{html.escape(item['gif'])}' alt='{html.escape(item['primary_cell'])}, slot {item['slot_index']}, {outcome}' loading='lazy'>"
            "<div class='card-copy'>"
            f"<div class='card-title'><code>{html.escape(item['primary_cell'])}</code><span class='tag {outcome}'>{outcome}</span></div>"
            f"<p>{html.escape(item['gallery_role'])} · slot {item['slot_index']} · final dig {_percent(float(item['dig_fraction']))} · {item['steps']} steps</p>"
            "<div class='card-actions'>"
            "<button type='button' class='play'>Play GIF</button>"
            f"<a href='{html.escape(item['gif'])}'>Open</a>"
            "</div></div></article>"
        )
    endpoint = summary["endpoint_without_net4"]
    net4 = summary["net4_diagnostic"]
    all_trenches = summary["all_trenches"]
    geometry_options = "".join(
        f"<option value='{html.escape(value)}'>{html.escape(value)}</option>"
        for value in geometries
    )
    condition_options = "".join(
        f"<option value='{html.escape(value)}'>{html.escape(value)}</option>"
        for value in conditions
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Terra trench alignment gallery</title>
<style>
:root {{ color-scheme: light; --ink:#20242b; --muted:#686b73; --paper:#f6f3eb; --card:#fffdf8; --line:#d8d3c8; --blue:#367eb8; --green:#1e7e49; --red:#ae3935; --gold:#c69220; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:var(--paper); }}
header,main {{ width:min(1500px,calc(100% - 32px)); margin:auto; }}
header {{ padding:34px 0 18px; }}
h1 {{ margin:0 0 6px; font-size:clamp(26px,4vw,44px); letter-spacing:-.03em; }}
h2 {{ margin:30px 0 12px; }}
p {{ margin:5px 0; }}
.muted {{ color:var(--muted); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:20px 0; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:15px; }}
.stat strong {{ display:block; font-size:28px; }}
.controls {{ position:sticky; top:0; z-index:3; display:flex; flex-wrap:wrap; gap:10px; padding:12px; background:rgba(246,243,235,.96); border-block:1px solid var(--line); }}
label {{ display:grid; gap:3px; color:var(--muted); font-size:12px; }}
select,button {{ min-height:38px; border:1px solid #aaa59a; border-radius:7px; background:#fff; color:var(--ink); padding:7px 10px; font:inherit; }}
button {{ cursor:pointer; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(335px,1fr)); gap:16px; padding:18px 0 42px; }}
.trajectory-card {{ overflow:hidden; background:var(--card); border:1px solid var(--line); border-radius:11px; }}
.trajectory-card.diagnostic {{ border-color:#d3a63e; }}
.trajectory-card[hidden] {{ display:none; }}
.trajectory-card img {{ display:block; width:100%; aspect-ratio:574/468; object-fit:contain; background:#f7f5ef; }}
.card-copy {{ padding:12px 14px 14px; }}
.card-title,.card-actions {{ display:flex; gap:10px; align-items:center; justify-content:space-between; }}
.card-actions {{ margin-top:11px; justify-content:flex-start; }}
a {{ color:#275f91; }}
.tag {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; color:#fff; }}
.tag.success {{ background:var(--green); }} .tag.stall {{ background:var(--red); }} .tag.warn {{ background:var(--gold); color:#251d0b; }}
.table-wrap {{ overflow:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }}
table {{ width:100%; border-collapse:collapse; min-width:780px; }}
th,td {{ padding:9px 11px; text-align:left; border-bottom:1px solid #e7e2d8; }}
.bar {{ display:inline-block; width:95px; height:8px; margin-right:8px; vertical-align:middle; background:#e1ddd3; border-radius:5px; overflow:hidden; }}
.bar span {{ display:block; height:100%; background:var(--blue); }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.92em; }}
footer {{ padding:20px 0 45px; color:var(--muted); overflow-wrap:anywhere; }}
@media (max-width:620px) {{ header,main {{ width:min(100% - 18px,1500px); }} .gallery {{ grid-template-columns:1fr; }} .controls {{ position:static; }} }}
</style>
</head>
<body>
<header>
  <h1>Strict-alignment trench policy</h1>
  <p>Deterministic fixed-bank preview at update {int(provenance['checkpoint_update']):,}. GIFs are representative by a fixed outcome-balanced selection rule, not hand-picked.</p>
  <p class="muted">The 11-condition endpoint excludes net4, whose finite cover is structurally infeasible under the strict gate. Net4 remains visible as a diagnostic.</p>
  <div class="stats">
    <div class="stat"><span>11-condition endpoint</span><strong>{endpoint['successes']}/{endpoint['episodes']}</strong><span>{_percent(endpoint['success_rate'])} exact completion</span></div>
    <div class="stat"><span>Mean endpoint excavation</span><strong>{_percent(endpoint['dig_fraction_mean'])}</strong><span>{endpoint['steps_mean']:.1f} mean steps</span></div>
    <div class="stat"><span>Net4 diagnostic</span><strong>{net4['successes']}/{net4['episodes']}</strong><span>{_percent(net4['success_rate'])} exact completion</span></div>
    <div class="stat"><span>All trench context</span><strong>{all_trenches['successes']}/{all_trenches['episodes']}</strong><span>{_percent(all_trenches['success_rate'])} exact completion</span></div>
  </div>
</header>
<main>
  <h2>Per-condition benchmark</h2>
  <div class="table-wrap"><table><thead><tr><th>Condition</th><th>Geometry</th><th>Exact</th><th>Success rate</th><th>Mean excavation</th><th>Mean steps</th></tr></thead><tbody>{''.join(condition_rows)}</tbody></table></div>
  <h2>Trajectory gallery</h2>
  <div class="controls">
    <label>Geometry<select id="geometry"><option value="all">All</option>{geometry_options}</select></label>
    <label>Condition<select id="condition"><option value="all">All</option>{condition_options}</select></label>
    <label>Outcome<select id="outcome"><option value="all">All</option><option value="success">Success</option><option value="stall">Stall</option></select></label>
    <button type="button" id="play-visible">Play visible</button>
    <button type="button" id="stop-all">Stop all</button>
  </div>
  <div class="gallery">{''.join(cards)}</div>
  <footer>
    <p>Checkpoint SHA-256: <code>{html.escape(provenance['checkpoint_sha256'])}</code></p>
    <p>Benchmark SHA-256: <code>{html.escape(provenance['benchmark_sha256'])}</code></p>
    <p>Evaluation code: <code>{html.escape(provenance['evaluation_baselines_revision'])}</code> · Terra runtime: <code>{html.escape(provenance['evaluation_terra_revision'])}</code></p>
  </footer>
</main>
<script>
const geometry=document.getElementById('geometry');
const condition=document.getElementById('condition');
const outcome=document.getElementById('outcome');
const cards=[...document.querySelectorAll('.trajectory-card')];
function stop(card){{const img=card.querySelector('img');img.src=img.dataset.poster;card.querySelector('.play').textContent='Play GIF';}}
function filter(){{cards.forEach(card=>{{const visible=(geometry.value==='all'||card.dataset.geometry===geometry.value)&&(condition.value==='all'||card.dataset.condition===condition.value)&&(outcome.value==='all'||card.dataset.outcome===outcome.value);card.hidden=!visible;if(!visible)stop(card);}});}}
[geometry,condition,outcome].forEach(control=>control.addEventListener('change',filter));
cards.forEach(card=>card.querySelector('.play').addEventListener('click',()=>{{const img=card.querySelector('img');const playing=img.src.endsWith(img.dataset.gif);if(playing){{stop(card);}}else{{img.src=img.dataset.gif;card.querySelector('.play').textContent='Stop';}}}}));
document.getElementById('play-visible').addEventListener('click',()=>cards.filter(card=>!card.hidden).forEach(card=>{{card.querySelector('img').src=card.querySelector('img').dataset.gif;card.querySelector('.play').textContent='Stop';}}));
document.getElementById('stop-all').addEventListener('click',()=>cards.forEach(stop));
</script>
</body>
</html>
"""


def write_bundle(
    output_dir: Path,
    benchmark: dict,
    selection: list[dict],
    artifacts: list[dict],
    provenance: dict,
) -> None:
    summary = _summary(benchmark["per_map"])
    summary_payload = {
        "schema": SCHEMA,
        "summary": summary,
        "provenance": provenance,
        "gallery_items": artifacts,
    }
    summary_path = output_dir / "summary.json"
    selection_path = output_dir / "selection.json"
    csv_path = output_dir / "condition_summary.csv"
    index_path = output_dir / "index.html"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "geometry",
                "structural_exclusion",
                "successes",
                "episodes",
                "success_rate",
                "mean_dig_fraction",
                "mean_steps",
            ]
        )
        for condition, stats in summary["conditions"].items():
            writer.writerow(
                [
                    condition,
                    stats["geometry"],
                    stats["structural_exclusion"],
                    stats["successes"],
                    stats["episodes"],
                    stats["success_rate"],
                    stats["dig_fraction_mean"],
                    stats["steps_mean"],
                ]
            )
    index_path.write_text(build_html(summary, artifacts, provenance))

    files = [summary_path, selection_path, csv_path, index_path]
    files.extend(
        output_dir / item[key] for item in artifacts for key in ("gif", "poster")
    )
    manifest = {
        "schema": "terra_trench_gallery_file_manifest_v1",
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--panel-family", default="gate_main")
    parser.add_argument("--accepted-panel", default="development")
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--per-condition", type=int, default=3)
    parser.add_argument("--frame-every", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-baselines-revision", required=True)
    parser.add_argument("--training-terra-revision", required=True)
    parser.add_argument("--evaluation-baselines-revision", required=True)
    parser.add_argument("--evaluation-terra-revision", required=True)
    args = parser.parse_args()
    if args.per_condition < 1 or args.per_condition > 6:
        raise ValueError("--per-condition must be in [1, 6]")
    if args.frame_every < 1:
        raise ValueError("--frame-every must be positive")

    checkpoint = args.checkpoint.resolve()
    benchmark_path = args.benchmark_json.resolve()
    bank_root = args.bank_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark(
        benchmark_path,
        checkpoint,
        panel_family=args.panel_family,
        accepted_panel=args.accepted_panel,
        terra_revision=args.terra_revision,
    )
    if int(benchmark["horizon"]) != args.horizon or int(benchmark["seed"]) != args.seed:
        raise ValueError("gallery horizon/seed does not match benchmark")
    selection = select_representatives(benchmark["per_map"], args.per_condition)
    if not selection:
        raise ValueError("representative selection is empty")

    provenance = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_update": int(benchmark["checkpoint_update"]),
        "benchmark_json": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "bank_root": str(bank_root),
        "bank_dataset_sha256": sha256_file(bank_root / "dataset.json"),
        "panel_family": args.panel_family,
        "accepted_panel": args.accepted_panel,
        "panel_manifest_sha256": benchmark["manifest_sha256"],
        "seed": args.seed,
        "horizon": args.horizon,
        "deterministic": True,
        "per_condition": args.per_condition,
        "frame_every": args.frame_every,
        "training_baselines_revision": args.training_baselines_revision,
        "training_terra_revision": args.training_terra_revision,
        "evaluation_baselines_revision": args.evaluation_baselines_revision,
        "evaluation_terra_revision": args.evaluation_terra_revision,
        "terra_protocol_revision": args.terra_revision,
    }
    artifacts = replay_and_render(
        checkpoint_path=checkpoint,
        bank_root=bank_root,
        panel_family=args.panel_family,
        accepted_panel=args.accepted_panel,
        terra_revision=args.terra_revision,
        benchmark=benchmark,
        selection=selection,
        horizon=args.horizon,
        seed=args.seed,
        frame_every=args.frame_every,
        output_dir=output_dir,
    )
    write_bundle(output_dir, benchmark, selection, artifacts, provenance)
    print(json.dumps(_summary(benchmark["per_map"]), indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
