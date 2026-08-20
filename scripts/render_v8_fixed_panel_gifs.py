#!/usr/bin/env python3
"""Render parity-checked GIFs for selected V8 fixed-panel episodes.

Policy inference always runs over the complete 720-row panel with the canonical
chunking.  Only selected rows are copied to the renderer.  This preserves the
fixed-evaluation trajectory while keeping the qualitative artifact bounded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eval_mcts
from eval_fixed_bank import (
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
    sha256_file,
    verify_exact_reset,
)
from terra.actions import TrackedActionType
from terra.env import TerraEnv
from train_mixed import _validate_checkpoint_architecture, make_mixed_agent_states
from utils.accepted_bank import load_accepted_bank
from utils.helpers import load_pkl_object
from utils.utils_ppo import (
    initial_actor_hidden,
    is_recurrent_actor,
    obs_to_model_input,
    wrap_action,
)


SCHEMA = "terra_v8_fixed_panel_trace_v1"
FIXED_SCHEMA = "terra_fixed_bank_eval_v4"
PANEL_RELATIVE_PATH = "evaluation/main/promotion"
ACTION_NAMES = {int(action): action.name.lower() for action in TrackedActionType}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_manifest_reset_receipt(
    record: dict[str, Any],
    reset_verification: dict[str, Any],
    environment_state_keys,
) -> dict[str, Any]:
    """Match the renderer reset inputs to the canonical fixed-eval receipt."""
    state_keys_host = np.ascontiguousarray(np.asarray(environment_state_keys))
    observed = {
        **reset_verification,
        "manifest_episode_seeds": {
            "passed": True,
            "map_selection_decoupled": True,
            "sha256": hashlib.sha256(state_keys_host.tobytes()).hexdigest(),
        },
    }
    expected = record.get("reset_verification")
    if not isinstance(expected, dict) or observed != expected:
        raise RuntimeError(
            "renderer reset receipt does not match the fixed evaluation"
        )
    return observed


def verify_prepared_manifest_reset(
    env,
    env_params,
    directory: Path,
    count: int,
    timestep,
    environment_state_keys,
    record: dict[str, Any],
) -> dict[str, Any]:
    # reset_prepared consumes the supplied keys and advances timestep.state.key.
    # Canonical fixed evaluation therefore verifies the frozen input-key receipt,
    # not equality between the input keys and the post-reset state key.
    reset_verification = verify_exact_reset(
        env,
        env_params,
        None,
        directory,
        count,
        timestep=timestep,
    )
    return verify_manifest_reset_receipt(
        record,
        reset_verification,
        environment_state_keys,
    )


def load_fixed_record(path: Path, index: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("fixed evaluation must be a non-empty JSON list")
    try:
        record = payload[index]
    except IndexError as exc:
        raise ValueError(f"record index {index} is out of range") from exc
    if record.get("schema") != FIXED_SCHEMA:
        raise ValueError(f"expected fixed-evaluation schema {FIXED_SCHEMA}")
    if record.get("split") != "promotion" or record.get("stratum") != "all":
        raise ValueError("renderer supports only the V8 promotion panel")
    if record.get("deterministic") is not True or record.get("policy_mode") != "deterministic":
        raise ValueError("renderer supports only deterministic greedy fixed records")
    return record


def load_selection(path: Path, count: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("review selection must be a non-empty JSON list")
    slots = [int(row["slot"]) for row in rows]
    if len(set(slots)) != len(slots) or any(slot < 1 or slot > count for slot in slots):
        raise ValueError("review selection contains duplicate or invalid slots")
    return rows


def preserve_inactive(previous, candidate, active: jax.Array, count: int):
    if not hasattr(candidate, "shape"):
        return candidate
    if candidate.ndim == 0 or candidate.shape[0] != count:
        return candidate
    mask = active.reshape((count,) + (1,) * (candidate.ndim - 1))
    return jnp.where(mask, candidate, previous)


def _one_env_observation(timestep, index: int) -> dict[str, np.ndarray]:
    obs = {
        name: np.asarray(jax.device_get(value[index : index + 1]))
        for name, value in timestep.observation.items()
    }
    obs["action_map"] = np.asarray(
        jax.device_get(timestep.state.world.action_map.map[index : index + 1])
    )
    return obs


def _fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def capture_frame(
    renderer: TerraEnv,
    timestep,
    index: int,
    *,
    label: str,
    row: dict[str, Any],
    step: int,
    action: int | None,
    effect: bool | None,
    material: dict[str, float],
    hidden_norm: float,
    recurrent: bool,
) -> Image.Image:
    renderer.render_obs_pygame(_one_env_observation(timestep, index), generate_gif=True)
    raw = renderer.rendering_engine.frames.pop()
    map_image = Image.fromarray(raw).resize(
        (raw.shape[1] * 2, raw.shape[0] * 2), Image.Resampling.NEAREST
    )
    width, height = map_image.size
    header = 88
    footer = 27
    canvas = Image.new("RGB", (width, header + height + footer), "white")
    canvas.paste(map_image, (0, header))
    draw = ImageDraw.Draw(canvas)
    bold, regular = _fonts()
    draw.text(
        (8, 5),
        f"{label} | slot {row['slot_index']:04d} | step {step:03d}/450",
        font=bold,
        fill="#111111",
    )
    draw.text(
        (8, 27),
        f"{row['primary_cell']} | {row['map_id']}",
        font=regular,
        fill="#333333",
    )
    draw.text(
        (8, 47),
        "dig {dig:.3f} | terminal {terminal:.3f} | staged {off_zone:.3f} | loaded {loaded:.3f}".format(
            **material
        ),
        font=regular,
        fill="#333333",
    )
    draw.text(
        (8, 67),
        f"GRU hidden L2 {hidden_norm:.3f}" if recurrent else "feed-forward actor",
        font=regular,
        fill="#333333",
    )
    action_name = "reset" if action is None else ACTION_NAMES.get(action, str(action))
    effect_name = "n/a" if effect is None else ("yes" if effect else "no")
    draw.text(
        (8, header + height + 5),
        f"last action: {action_name} | physical effect: {effect_name}",
        font=regular,
        fill="#222222",
    )
    return canvas


def _material(info: dict[str, Any], source: np.ndarray) -> dict[str, np.ndarray]:
    components = info["reward_components"]
    dig = np.asarray(jax.device_get(components["dig_completion_total"]), dtype=np.float32)
    terminal = np.asarray(
        jax.device_get(components["accepted_dump_volume"]), dtype=np.float32
    ) / source
    off_zone = np.asarray(
        jax.device_get(components["illegal_dump_volume"]), dtype=np.float32
    ) / source
    loaded = np.maximum(dig - terminal - off_zone, 0.0)
    return {"dig": dig, "terminal": terminal, "off_zone": off_zone, "loaded": loaded}


def hash_model_input_rows(model_input, indices: np.ndarray) -> list[str]:
    leaves = [
        np.ascontiguousarray(np.asarray(jax.device_get(leaf[indices])))
        for leaf in jax.tree_util.tree_leaves(model_input)
    ]
    result = []
    for row_index in range(len(indices)):
        digest = hashlib.sha256()
        for leaf_index, leaf in enumerate(leaves):
            value = np.ascontiguousarray(leaf[row_index])
            header = f"{leaf_index}|{value.dtype.str}|{value.shape}|".encode("ascii")
            digest.update(len(header).to_bytes(4, "little"))
            digest.update(header)
            digest.update(value.tobytes())
        result.append(digest.hexdigest())
    return result


def longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def longest_equal_run(values: list[int]) -> int:
    best = current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        previous = value
        best = max(best, current)
    return best


def terminal_cycle(
    trace: list[dict[str, Any]],
    signature_fields: tuple[str, ...] = ("input_hash", "action"),
    maximum_period: int = 64,
) -> dict[str, int] | None:
    signatures = [tuple(step[field] for field in signature_fields) for step in trace]
    count = len(signatures)
    best = None
    for period in range(1, min(maximum_period, count // 3) + 1):
        mismatches = [
            index
            for index in range(period, count)
            if signatures[index] != signatures[index - period]
        ]
        start = max(mismatches) - period + 1 if mismatches else 0
        start = max(start, 0)
        suffix = count - start
        if suffix >= max(12, 3 * period):
            candidate = {"period": period, "start_step": start + 1, "decisions": suffix}
            if best is None or candidate["decisions"] > best["decisions"]:
                best = candidate
    return best


def summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [bool(step["action_had_effect"]) for step in trace]
    actions = [int(step["action"]) for step in trace]
    input_counts = Counter(step["input_hash"] for step in trace)
    material_steps = [
        step["step"]
        for step in trace
        if step["material_changed"]
    ]
    recurrent = bool(trace) and trace[0].get("hidden_hash") is not None
    return {
        "decisions": len(trace),
        "no_effect_actions": effects.count(False),
        "maximum_no_effect_streak": longest_true_run([not value for value in effects]),
        "maximum_repeated_action_streak": longest_equal_run(actions),
        "unique_policy_inputs": len(input_counts),
        "repeated_instantaneous_input_decisions": sum(
            value - 1 for value in input_counts.values()
        ),
        "last_material_change_step": max(material_steps, default=0),
        "terminal_observation_action_cycle": terminal_cycle(trace),
        "terminal_recurrent_state_action_cycle": (
            terminal_cycle(trace, ("input_hash", "hidden_hash", "action"))
            if recurrent
            else None
        ),
    }


def _save_gif(frames: list[Image.Image], path: Path) -> None:
    if not frames:
        raise RuntimeError(f"no frames captured for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=150,
        disposal=1,
        optimize=False,
    )


def run(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    fixed_path = args.fixed_json.resolve()
    bank_root = args.bank_root.resolve()
    record = load_fixed_record(fixed_path, args.record_index)
    if args.label != Path(args.label).name or not args.label:
        raise ValueError("--label must be one non-empty path component")
    label_dir = args.output_dir.resolve() / args.label
    if label_dir.exists():
        raise FileExistsError(label_dir)
    if eval_mcts.EVAL_FORWARD_CHUNK != 120:
        raise ValueError("fixed-panel media requires canonical forward chunk 120")
    if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match fixed-evaluation receipt")
    checkpoint = load_pkl_object(str(checkpoint_path))
    train_config = checkpoint["train_config"]
    _validate_checkpoint_architecture(checkpoint, train_config)
    if bool(_config_value(train_config, "action_logit_masking", False)):
        raise ValueError("this review path supports only the unmasked policy contract")

    directory = bank_root / PANEL_RELATIVE_PATH
    rows = load_manifest(directory)
    count = len(rows)
    if count != len(record["per_map"]):
        raise ValueError("bank and fixed evaluation contain different panel sizes")
    if sha256_file(directory / "manifest.jsonl") != record["manifest_sha256"]:
        raise ValueError("promotion manifest does not match fixed-evaluation receipt")
    selection = load_selection(args.selection.resolve(), count)
    selected_slots = [int(item["slot"]) for item in selection]
    selected_indices = np.asarray([slot - 1 for slot in selected_slots], dtype=np.int32)

    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(count)
    config = configure_for_bank(train_config, PANEL_RELATIVE_PATH, count)
    recurrent = is_recurrent_actor(config)
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    accepted_bank = load_accepted_bank(
        bank_root,
        "G-UNIFORM",
        args.terra_revision,
        curriculum_stage="full",
    )
    map_keys = exact_reset_keys(count)
    state_keys = manifest_environment_keys(
        rows, count, accepted_bank.environment_protocol_sha256
    )
    timestep, env_params, state_keys = prepare_manifest_episode_reset(
        env, env_params, map_keys, state_keys
    )
    reset_verification = verify_prepared_manifest_reset(
        env,
        env_params,
        directory,
        count,
        timestep,
        state_keys,
        record,
    )
    model = SimpleNamespace(apply=initialized_state.apply_fn)
    renderer = TerraEnv.new(
        maps_size_px=int(env.batch_cfg.maps_dims.maps_edge_length),
        rendering=True,
        n_envs_x=1,
        n_envs_y=1,
        display=False,
    )

    fixed_rows = record["per_map"]
    source = np.asarray([float(row["source_soil_volume"]) for row in fixed_rows])
    if np.any(source <= 0):
        raise ValueError("all promotion episodes must have positive source soil")
    frames: dict[int, list[Image.Image]] = {slot: [] for slot in selected_slots}
    traces: dict[int, list[dict[str, Any]]] = {slot: [] for slot in selected_slots}
    zero_material = {"dig": 0.0, "terminal": 0.0, "off_zone": 0.0, "loaded": 0.0}
    for slot in selected_slots:
        frames[slot].append(
            capture_frame(
                renderer,
                timestep,
                slot - 1,
                label=args.label,
                row=fixed_rows[slot - 1],
                step=0,
                action=None,
                effect=None,
                material=zero_material,
                hidden_norm=0.0,
                recurrent=recurrent,
            )
        )

    prev_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    actor_hidden = initial_actor_hidden(count, config)
    terminated = jnp.zeros(count, dtype=jnp.bool_)
    succeeded = jnp.zeros(count, dtype=jnp.bool_)
    lengths = jnp.zeros(count, dtype=jnp.int32)
    terminal_material = {
        name: jnp.zeros(count, dtype=jnp.float32)
        for name in ("dig", "terminal", "off_zone", "loaded")
    }
    no_effect_count = jnp.zeros(count, dtype=jnp.int32)
    previous_material = {name: np.zeros(len(selected_slots), dtype=np.float32) for name in terminal_material}
    last_actions = np.full(len(selected_slots), -1, dtype=np.int32)
    last_effects = np.zeros(len(selected_slots), dtype=bool)
    last_hidden_norms = np.zeros(len(selected_slots), dtype=np.float32)
    rng = jax.random.PRNGKey(int(record["seed"]))
    rng, _ = jax.random.split(rng)
    for step in range(1, int(record["horizon"]) + 1):
        active = ~terminated
        model_input = obs_to_model_input(timestep.observation, prev_actions, config)
        input_hashes = hash_model_input_rows(model_input, selected_indices)
        pre_hidden = np.asarray(jax.device_get(actor_hidden[selected_indices]))
        if recurrent:
            _, logits, next_hidden = eval_mcts._apply_recurrent_in_batch_chunks(
                model, checkpoint["model"], model_input, actor_hidden
            )
        else:
            _, logits = eval_mcts._apply_in_batch_chunks(model, checkpoint["model"], model_input)
            next_hidden = actor_hidden
        action = jnp.argmax(logits, axis=-1)
        selected_logits = np.asarray(jax.device_get(logits[selected_indices]), dtype=np.float32)
        selected_actions = np.asarray(jax.device_get(action[selected_indices]), dtype=np.int32)
        selected_next_hidden = np.asarray(jax.device_get(next_hidden[selected_indices]))
        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)
        rng, _, rng_step = jax.random.split(rng, 3)
        candidate = env.step_no_reset(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            jax.random.split(rng_step, count),
        )
        timestep = jax.tree_util.tree_map(
            lambda old, new: preserve_inactive(old, new, active, count),
            timestep,
            candidate,
        )
        prev_actions = jnp.where(timestep.done[:, None], jnp.zeros_like(prev_actions), prev_actions)
        actor_hidden = jnp.where(timestep.done[:, None], jnp.zeros_like(next_hidden), next_hidden)
        effect = np.asarray(
            jax.device_get(active & timestep.info["action_had_effect"]), dtype=bool
        )
        no_effect_count += active.astype(jnp.int32) * jnp.asarray(~effect, dtype=jnp.int32)
        material = _material(timestep.info, source)
        newly_done = active & timestep.done
        lengths += active.astype(jnp.int32)
        succeeded |= active & timestep.info["task_done"]
        for name in terminal_material:
            terminal_material[name] = jnp.where(
                newly_done, jnp.asarray(material[name]), terminal_material[name]
            )
        terminated |= newly_done
        active_host = np.asarray(jax.device_get(active), dtype=bool)

        for selected_offset, slot in enumerate(selected_slots):
            index = slot - 1
            if not active_host[index]:
                continue
            current_material = {
                name: float(values[index]) for name, values in material.items()
            }
            changed = any(
                abs(current_material[name] - float(previous_material[name][selected_offset])) > 1e-7
                for name in terminal_material
            )
            for name in terminal_material:
                previous_material[name][selected_offset] = current_material[name]
            action_value = int(selected_actions[selected_offset])
            probabilities = np.exp(
                selected_logits[selected_offset] - np.max(selected_logits[selected_offset])
            )
            probabilities /= probabilities.sum()
            sorted_probabilities = np.sort(probabilities)
            hidden_norm = float(np.linalg.norm(pre_hidden[selected_offset]))
            next_hidden_norm = float(np.linalg.norm(selected_next_hidden[selected_offset]))
            trace_row = {
                "step": step,
                "action": action_value,
                "action_name": ACTION_NAMES.get(action_value, str(action_value)),
                "action_had_effect": bool(effect[index]),
                "material_changed": changed,
                **current_material,
                "input_hash": input_hashes[selected_offset],
                "hidden_hash": (
                    _sha256_bytes(np.ascontiguousarray(pre_hidden[selected_offset]).tobytes())
                    if recurrent
                    else None
                ),
                "hidden_l2": hidden_norm,
                "hidden_delta_l2": float(
                    np.linalg.norm(selected_next_hidden[selected_offset] - pre_hidden[selected_offset])
                ),
                "top_action_probability": float(sorted_probabilities[-1]),
                "top_action_margin": float(sorted_probabilities[-1] - sorted_probabilities[-2]),
            }
            traces[slot].append(trace_row)
            last_actions[selected_offset] = action_value
            last_effects[selected_offset] = bool(effect[index])
            last_hidden_norms[selected_offset] = next_hidden_norm

        # Use one common time grid for every policy and keep completed states
        # frozen. Separate GIFs then contain the same simulator steps and frame
        # count, even when one policy terminates early.
        if step % args.frame_stride == 0 or step == int(record["horizon"]):
            for selected_offset, slot in enumerate(selected_slots):
                index = slot - 1
                current_material = {
                    name: float(previous_material[name][selected_offset])
                    for name in terminal_material
                }
                frames[slot].append(
                    capture_frame(
                        renderer,
                        timestep,
                        index,
                        label=args.label,
                        row=fixed_rows[index],
                        step=step,
                        action=(
                            int(last_actions[selected_offset])
                            if last_actions[selected_offset] >= 0
                            else None
                        ),
                        effect=(
                            bool(last_effects[selected_offset])
                            if last_actions[selected_offset] >= 0
                            else None
                        ),
                        material=current_material,
                        hidden_norm=float(last_hidden_norms[selected_offset]),
                        recurrent=recurrent,
                    )
                )

    success_host = np.asarray(jax.device_get(succeeded), dtype=bool)
    lengths_host = np.asarray(jax.device_get(lengths), dtype=np.int32)
    expected_success = np.asarray([bool(row["success"]) for row in fixed_rows])
    expected_lengths = np.asarray([int(row["steps"]) for row in fixed_rows], dtype=np.int32)
    np.testing.assert_array_equal(success_host, expected_success)
    np.testing.assert_array_equal(lengths_host, expected_lengths)
    expected_no_effect = np.asarray(
        [int(row["no_effect_action_count"]) for row in fixed_rows], dtype=np.int32
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(no_effect_count), dtype=np.int32),
        expected_no_effect,
    )
    field_map = {
        "dig": "dig_fraction",
        "terminal": "terminal_soil_fraction",
        "off_zone": "off_zone_staged_soil_fraction",
        "loaded": "loaded_soil_fraction",
    }
    for name, field in field_map.items():
        observed = np.asarray(jax.device_get(terminal_material[name]), dtype=np.float32)
        expected = np.asarray([float(row[field]) for row in fixed_rows], dtype=np.float32)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)

    (label_dir / "traces").mkdir(parents=True)
    summaries = []
    for slot in selected_slots:
        gif_path = label_dir / f"slot_{slot:04d}.gif"
        _save_gif(frames[slot], gif_path)
        summary = {
            "schema": SCHEMA,
            "label": args.label,
            "slot": slot,
            "episode_id": fixed_rows[slot - 1]["episode_id"],
            "map_id": fixed_rows[slot - 1]["map_id"],
            "condition": fixed_rows[slot - 1]["primary_cell"],
            "success": bool(success_host[slot - 1]),
            "steps": int(lengths_host[slot - 1]),
            "gif": gif_path.name,
            "trace_summary": summarize_trace(traces[slot]),
            "terminal_parity_verified": True,
        }
        (label_dir / "traces" / f"slot_{slot:04d}.json").write_text(
            json.dumps(
                {**summary, "steps_trace": traces[slot]},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
    receipt = {
        "schema": SCHEMA,
        "label": args.label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_update": record.get("checkpoint_update"),
        "fixed_json": str(fixed_path),
        "fixed_json_sha256": sha256_file(fixed_path),
        "manifest_sha256": record["manifest_sha256"],
        "panel_maps": count,
        "horizon": int(record["horizon"]),
        "seed": int(record["seed"]),
        "deterministic": True,
        "selected_slots": selected_slots,
        "canonical_forward_chunk": eval_mcts.EVAL_FORWARD_CHUNK,
        "reset_verification": reset_verification,
        "frame_cadence_steps": args.frame_stride,
        "frames_per_episode": len(next(iter(frames.values()))),
        "full_panel_terminal_parity_verified": True,
        "full_panel_no_effect_count_parity_verified": True,
        "episodes": summaries,
    }
    (label_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(label_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-json", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=-1)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    run(args)


if __name__ == "__main__":
    main()
