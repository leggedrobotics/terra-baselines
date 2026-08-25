#!/usr/bin/env python3
"""Paired GRU-carry probe on the twelve historical V8 failure-audit slots.

The only intervention is evaluation-time actor memory: the first 120-row
chunk carries the GRU state normally, while the second identical 120-row
chunk receives a zero carry before every decision.  Each chunk contains the
same twelve mechanism targets followed by the same 108 deterministic padding
slots.  This preserves the canonical policy-forward shape and chunk position.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import eval_mcts
from eval_fixed_bank import (
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
    verify_exact_reset,
)
from train_mixed import _validate_checkpoint_architecture, make_mixed_agent_states
from utils.accepted_bank import load_accepted_bank
from utils.helpers import load_pkl_object
from utils.utils_ppo import (
    _config_option,
    initial_actor_hidden,
    obs_to_model_input,
    wrap_action,
)


SCHEMA = "terra_v8_gru_recurrence_probe_v1"
HORIZON = 450
SEED = 20260807
FORWARD_CHUNK = 120
TARGET_COUNT = 12
PADDING_COUNT = FORWARD_CHUNK - TARGET_COUNT
TOTAL_COUNT = 2 * FORWARD_CHUNK
NUM_ACTIONS = 8
NUM_PREVIOUS_ACTIONS = 5
MIN_CYCLE_REPETITIONS = 3
PROMOTION_MANIFEST_SHA256 = (
    "dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826"
)

TARGET_SLOTS = (250, 100, 234, 17, 142, 68, 300, 338, 247, 177, 577, 210)
TARGET_ROLES = (
    "loaded_high_carry",
    "loaded_high_carry",
    "loaded_high_carry",
    "loaded_high_carry",
    "clean_high_dig",
    "clean_high_dig",
    "clean_high_dig",
    "clean_high_dig",
    "obstacle_stall",
    "obstacle_stall",
    "success_control",
    "success_control",
)
TARGET_IDENTITIES = (
    (250, "curriculum-diverse-320-17156", "fnd-slab-side1-obj", "foundation"),
    (100, "curriculum-diverse-320-8036", "fnd-slab-apron-d12", "foundation"),
    (234, "curriculum-diverse-320-10156", "fnd-slab-side1", "foundation"),
    (17, "curriculum-diverse-320-16008", "fnd-proc-side1-road", "foundation"),
    (142, "curriculum-diverse-320-3221", "fnd-slab-apron-near", "foundation"),
    (68, "curriculum-diverse-320-5036", "fnd-slab-apron-c2x", "foundation"),
    (300, "curriculum-diverse-320-27190", "trn-net3-side1-road", "trench"),
    (338, "curriculum-diverse-320-28009", "trn-net4-side1-road", "trench"),
    (247, "curriculum-diverse-320-17097", "fnd-slab-side1-obj", "foundation"),
    (177, "curriculum-diverse-320-14013", "fnd-slab-ring3x-obj", "foundation"),
    (
        577,
        "v8:promotion:v7-fnd-pads-adjacent:0000",
        "v7-fnd-pads-adjacent",
        "foundation",
    ),
    (210, "curriculum-diverse-320-15019", "fnd-slab-ring3x-road", "foundation"),
)

ACTION_NAMES = (
    "forward",
    "backward",
    "clock",
    "anticlock",
    "cabin_clock",
    "cabin_anticlock",
    "do",
    "do_nothing",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity(source_file: str) -> dict[str, Any]:
    """Read archive provenance first, then fall back to a live Git checkout."""
    source = Path(source_file).resolve()
    for candidate in (source.parent, *source.parents):
        marker = candidate / "REVISION"
        if not marker.is_file():
            continue
        revision = marker.read_text().strip()
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"invalid staged-source revision marker {marker}")
        return {
            "root": str(candidate),
            "revision": revision,
            "source_form": "git_archive_with_revision_marker",
            "dirty": False,
        }

    root = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    revision = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "root": root,
        "revision": revision,
        "source_form": "live_git_checkout",
        "dirty": dirty,
    }


def build_chunk_slots(panel_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep each target at its canonical position within a 120-row chunk."""
    if panel_count != 720:
        raise ValueError(f"expected the 720-slot promotion panel, got {panel_count}")
    if len(set(TARGET_SLOTS)) != TARGET_COUNT:
        raise ValueError("target slots must be unique")
    if min(TARGET_SLOTS) < 1 or max(TARGET_SLOTS) > panel_count:
        raise ValueError("target slot lies outside the promotion panel")

    target_positions = tuple((slot - 1) % FORWARD_CHUNK for slot in TARGET_SLOTS)
    if len(set(target_positions)) != TARGET_COUNT:
        raise ValueError("two historical targets require the same chunk position")
    chunk_values = list(range(1, FORWARD_CHUNK + 1))
    for slot, position in zip(TARGET_SLOTS, target_positions):
        chunk_values[position] = slot
    chunk = tuple(chunk_values)
    padding = tuple(
        slot for position, slot in enumerate(chunk) if position not in target_positions
    )
    if len(chunk) != FORWARD_CHUNK or len(set(chunk)) != FORWARD_CHUNK:
        raise ValueError("one probe chunk must contain 120 unique slots")
    if len(padding) != PADDING_COUNT:
        raise ValueError("one probe chunk must contain 108 padding slots")
    return chunk, padding


def validate_target_identities(manifest_rows: Sequence[dict[str, Any]]) -> None:
    observed = tuple(
        (
            slot,
            manifest_rows[slot - 1]["map_id"],
            manifest_rows[slot - 1]["primary_cell"],
            manifest_rows[slot - 1]["family"],
        )
        for slot in TARGET_SLOTS
    )
    if observed != TARGET_IDENTITIES:
        raise ValueError(
            "historical target identities differ from the pinned promotion panel"
        )


def load_fixed_record(
    path: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the canonical full-panel record for the probed checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("fixed evaluation must be a JSON list")
    matches = [
        record
        for record in records
        if record.get("checkpoint_sha256") == checkpoint_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one fixed record for checkpoint {checkpoint_sha256}, got "
            f"{len(matches)}"
        )
    record = matches[0]
    required = {
        "schema": "terra_fixed_bank_eval_v4",
        "horizon": HORIZON,
        "seed": SEED,
        "deterministic": True,
        "policy_mode": "deterministic",
        "manifest_sha256": PROMOTION_MANIFEST_SHA256,
        "exact_manifest_enumeration": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(
                f"fixed evaluation {key}={record.get(key)!r}, expected {expected!r}"
            )
    rows = record.get("per_map")
    if not isinstance(rows, list) or len(rows) != 720:
        raise ValueError("fixed evaluation must contain all 720 promotion rows")
    if [row.get("slot_index") for row in rows] != list(range(1, 721)):
        raise ValueError("fixed evaluation rows are not in contiguous slot order")
    if any(row.get("integrity_failure") for row in rows):
        raise ValueError("fixed evaluation contains an integrity failure")
    return record, rows


def slice_batch_tree(tree, indices: np.ndarray, batch_size: int):
    indices_jax = jnp.asarray(indices, dtype=jnp.int32)

    def take(value):
        if (
            hasattr(value, "shape")
            and value.ndim > 0
            and int(value.shape[0]) == batch_size
        ):
            return value[indices_jax]
        return value

    return jax.tree_util.tree_map(take, tree)


def preserve_active(previous, candidate, active: jax.Array):
    def choose(old, new):
        if not hasattr(new, "shape"):
            return new
        if new.ndim == 0 or int(new.shape[0]) != TOTAL_COUNT:
            return new
        mask = active.reshape((TOTAL_COUNT,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(choose, previous, candidate)


def assert_paired_batch_equal(tree, label: str) -> int:
    """Fail unless every 240-batched leaf has identical 120-row halves."""
    checked = 0
    for leaf_index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if not hasattr(leaf, "shape") or leaf.ndim == 0:
            continue
        if int(leaf.shape[0]) != TOTAL_COUNT:
            continue
        host = np.asarray(jax.device_get(leaf))
        if not np.array_equal(host[:FORWARD_CHUNK], host[FORWARD_CHUNK:]):
            raise ValueError(f"{label} differs between paired chunks at leaf {leaf_index}")
        checked += 1
    if checked == 0:
        raise ValueError(f"{label} had no 240-row leaves to compare")
    return checked


def digest_row(host_leaves: Sequence[np.ndarray], row: int) -> bytes:
    """Hash one exact post-preprocessing policy-input row."""
    digest = hashlib.sha256()
    for leaf_index, batched in enumerate(host_leaves):
        value = np.ascontiguousarray(batched[row])
        if value.dtype.hasobject:
            raise TypeError(f"object dtype at policy-input leaf {leaf_index}")
        header = (
            f"{leaf_index}|{value.dtype.str}|"
            f"{','.join(str(size) for size in value.shape)}|"
        ).encode("ascii")
        digest.update(len(header).to_bytes(4, "little"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
    return digest.digest()


def digest_array(value: np.ndarray) -> bytes:
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise TypeError("cannot hash an object array")
    digest = hashlib.sha256()
    header = f"{value.dtype.str}|{','.join(str(v) for v in value.shape)}|".encode(
        "ascii"
    )
    digest.update(len(header).to_bytes(4, "little"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.digest()


def _total_loaded(state) -> jax.Array:
    active = state.agent.agent_active.astype(jnp.float32)
    loaded = jnp.stack(
        [
            agent_state.loaded.astype(jnp.float32).reshape((TOTAL_COUNT, -1)).sum(
                axis=1
            )
            for agent_state in state.agent.agent_states
        ],
        axis=1,
    )
    return (loaded * active).sum(axis=1)


def material_snapshot(state) -> dict[str, jax.Array]:
    target = state.world.target_map.map.astype(jnp.float32).reshape((TOTAL_COUNT, -1))
    action = state.world.action_map.map.astype(jnp.float32).reshape((TOTAL_COUNT, -1))
    required = jnp.clip(-target, a_min=0.0)
    source = required.sum(axis=1)
    remaining = jnp.where(
        target < 0,
        jnp.clip(action - target, a_min=0.0, a_max=required),
        0.0,
    ).sum(axis=1)
    positive = jnp.clip(action, a_min=0.0)
    terminal = jnp.where(target > 0, positive, 0.0).sum(axis=1)
    staged = jnp.where(target <= 0, positive, 0.0).sum(axis=1)
    loaded = _total_loaded(state)
    inverse = 1.0 / jnp.maximum(source, 1.0)
    return {
        "source_volume": source,
        "dig_fraction": (source - remaining) * inverse,
        "terminal_soil_fraction": terminal * inverse,
        "off_zone_staged_soil_fraction": staged * inverse,
        "loaded_soil_fraction": loaded * inverse,
    }


def longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def longest_equal_run(values: Sequence[Any]) -> int:
    if not values:
        return 0
    longest = 1
    current = 1
    for previous, current_value in zip(values, values[1:]):
        if current_value == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def terminal_period(
    values: Sequence[Any], min_repetitions: int = MIN_CYCLE_REPETITIONS
) -> dict[str, int] | None:
    """Return the smallest exact period repeated at the end of a sequence."""
    values = list(values)
    for period in range(1, len(values) // min_repetitions + 1):
        final_block = values[-period:]
        repetitions = 1
        while len(values) >= (repetitions + 1) * period:
            start = -(repetitions + 1) * period
            stop = -repetitions * period
            if values[start:stop] != final_block:
                break
            repetitions += 1
        if repetitions >= min_repetitions:
            suffix = repetitions * period
            return {
                "period": period,
                "repetitions": repetitions,
                "suffix_decisions": suffix,
                "first_step": len(values) - suffix + 1,
            }
    return None


def first_paired_difference(
    left: np.ndarray,
    right: np.ndarray,
    left_active: np.ndarray,
    right_active: np.ndarray,
) -> int | None:
    common = left_active & right_active
    different = np.asarray(left != right)
    if different.ndim > 1:
        different = different.reshape((different.shape[0], -1)).any(axis=1)
    indices = np.flatnonzero(common & different)
    return int(indices[0] + 1) if indices.size else None


def analyze_lane(
    *,
    active: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    material_changed: np.ndarray,
    input_hashes: np.ndarray,
    hidden_hashes: np.ndarray,
    hidden_norms: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(active)
    action_values = [int(actions[index]) for index in indices]
    effect_values = [bool(effects[index]) for index in indices]
    material_values = [bool(material_changed[index]) for index in indices]
    input_values = [bytes(input_hashes[index]) for index in indices]
    hidden_values = [bytes(hidden_hashes[index]) for index in indices]
    logit_values = [
        np.ascontiguousarray(logits[index], dtype=np.float32).tobytes()
        for index in indices
    ]

    last_seen: dict[bytes, int] = {}
    first_state: dict[bytes, tuple[bytes, bytes, int]] = {}
    deterministic_logits: dict[tuple[bytes, bytes], bytes] = {}
    recurrence_lags: list[int] = []
    different_hidden = 0
    different_logits = 0
    different_actions = 0
    for local_step, (inp, hidden, logit, action) in enumerate(
        zip(input_values, hidden_values, logit_values, action_values)
    ):
        previous = last_seen.get(inp)
        if previous is not None:
            recurrence_lags.append(local_step - previous)
            first_hidden, first_logit, first_action = first_state[inp]
            different_hidden += hidden != first_hidden
            different_logits += logit != first_logit
            different_actions += action != first_action
        else:
            first_state[inp] = (hidden, logit, action)
        key = (inp, hidden)
        known_logits = deterministic_logits.get(key)
        if known_logits is not None and known_logits != logit:
            raise ValueError("identical input and hidden produced different logits")
        deterministic_logits[key] = logit
        last_seen[inp] = local_step

    no_effect = [not value for value in effect_values]
    lag_counts = Counter(recurrence_lags)
    input_action = list(zip(input_values, action_values))
    full_policy_state = list(zip(input_values, hidden_values, action_values))
    changed_steps = [index + 1 for index, changed in enumerate(material_values) if changed]
    active_norms = np.asarray(hidden_norms[indices], dtype=np.float32)
    return {
        "active_decisions": len(indices),
        "unique_instantaneous_input_count": len(set(input_values)),
        "repeated_instantaneous_input_decisions": len(recurrence_lags),
        "same_input_different_hidden_decisions": int(different_hidden),
        "same_input_different_logits_decisions": int(different_logits),
        "same_input_different_action_decisions": int(different_actions),
        "top_instantaneous_input_recurrence_lags": [
            {"lag": int(lag), "count": int(count)}
            for lag, count in lag_counts.most_common(5)
        ],
        "no_effect_action_count": int(sum(no_effect)),
        "longest_no_effect_streak": longest_true_run(no_effect),
        "longest_repeated_action_streak": longest_equal_run(action_values),
        "last_material_change_step": changed_steps[-1] if changed_steps else 0,
        "terminal_action_cycle": terminal_period(action_values),
        "terminal_input_action_cycle": terminal_period(input_action),
        "terminal_full_policy_state_cycle": terminal_period(full_policy_state),
        "maximum_pre_action_hidden_norm": (
            float(np.max(active_norms)) if active_norms.size else 0.0
        ),
        "terminal_pre_action_hidden_norm": (
            float(active_norms[-1]) if active_norms.size else 0.0
        ),
    }


def _pair_outcome(normal_success: bool, zero_success: bool) -> str:
    if normal_success and not zero_success:
        return "normal_only_success"
    if zero_success and not normal_success:
        return "zero_only_success"
    return "both_success" if normal_success else "both_fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-eval", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    fixed_eval_path = args.fixed_eval.resolve()
    bank_root = args.bank_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not bank_root.is_dir():
        raise FileNotFoundError(bank_root)

    import terra.env as terra_env_module

    baselines_source = source_identity(eval_mcts.__file__)
    terra_source = source_identity(terra_env_module.__file__)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    fixed_record, fixed_rows = load_fixed_record(
        fixed_eval_path,
        checkpoint_sha256,
    )

    checkpoint = load_pkl_object(str(checkpoint_path))
    train_config = checkpoint["train_config"]
    _validate_checkpoint_architecture(checkpoint, train_config)
    if _config_option(train_config, "actor_core", "mlp") != "gru":
        raise ValueError("the recurrence probe requires actor_core='gru'")
    if int(_config_option(train_config, "num_prev_actions", 0)) != NUM_PREVIOUS_ACTIONS:
        raise ValueError("the recurrence probe requires the five-action history")
    if bool(_config_option(train_config, "action_logit_masking", False)):
        raise ValueError("action masking is outside the paired GRU probe contract")

    accepted_bank = load_accepted_bank(
        bank_root,
        "G-UNIFORM",
        args.terra_revision,
        curriculum_stage="full",
    )
    panel = next(
        panel for panel in accepted_bank.evaluation_panels if panel.name == "promotion"
    )
    directory = bank_root / panel.maps_path
    manifest_rows = load_manifest(directory)
    panel_count = len(manifest_rows)
    manifest_sha256 = sha256_file(directory / "manifest.jsonl")
    if manifest_sha256 != PROMOTION_MANIFEST_SHA256:
        raise ValueError(
            f"promotion manifest {manifest_sha256}, expected "
            f"{PROMOTION_MANIFEST_SHA256}"
        )
    validate_target_identities(manifest_rows)
    chunk_slots, padding_slots = build_chunk_slots(panel_count)

    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(panel_count)
    config = configure_for_bank(train_config, panel.maps_path, panel_count)
    if tuple(config.agent_types_override) != (0,) or tuple(
        config.action_types_override
    ) != (0,):
        raise ValueError("the recurrence probe supports one tracked excavator")
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)

    reset_keys = exact_reset_keys(panel_count)
    state_keys = manifest_environment_keys(
        manifest_rows,
        panel_count,
        accepted_bank.environment_protocol_sha256,
    )
    full_timestep, env_params, _ = prepare_manifest_episode_reset(
        env, env_params, reset_keys, state_keys
    )
    reset_receipt = verify_exact_reset(
        env, env_params, None, directory, panel_count, timestep=full_timestep
    )
    if not reset_receipt["passed"]:
        raise ValueError("promotion-panel exact reset verification failed")

    one_chunk_indices = np.asarray(chunk_slots, dtype=np.int32) - 1
    paired_indices = np.concatenate([one_chunk_indices, one_chunk_indices])
    timestep = slice_batch_tree(full_timestep, paired_indices, panel_count)
    env_params = slice_batch_tree(env_params, paired_indices, panel_count)
    config.num_envs_per_device = TOTAL_COUNT
    config.num_test_rollouts = TOTAL_COUNT
    config.num_minibatches = TOTAL_COUNT
    initial_timestep_pair_leaves = assert_paired_batch_equal(
        timestep, "initial timestep"
    )
    initial_env_param_pair_leaves = assert_paired_batch_equal(
        env_params, "environment parameters"
    )

    model = SimpleNamespace(apply=initialized_state.apply_fn)
    if eval_mcts.EVAL_FORWARD_CHUNK != FORWARD_CHUNK:
        raise ValueError(
            f"canonical evaluator chunk is {eval_mcts.EVAL_FORWARD_CHUNK}, "
            f"expected {FORWARD_CHUNK}"
        )
    prev_actions = jnp.zeros((TOTAL_COUNT, NUM_PREVIOUS_ACTIONS), dtype=jnp.int32)
    actor_hidden = initial_actor_hidden(TOTAL_COUNT, config)
    if actor_hidden.shape != (
        TOTAL_COUNT,
        int(_config_option(config, "actor_gru_hidden_dim", 64)),
    ):
        raise ValueError(f"unexpected initial actor-hidden shape {actor_hidden.shape}")

    target_positions = np.asarray(
        [chunk_slots.index(slot) for slot in TARGET_SLOTS], dtype=np.int32
    )
    padding_positions = np.asarray(
        [position for position in range(FORWARD_CHUNK) if position not in target_positions],
        dtype=np.int32,
    )
    if len(padding_positions) != PADDING_COUNT:
        raise ValueError("probe chunk does not have exactly 108 padding positions")
    target_lanes = np.concatenate(
        [target_positions, FORWARD_CHUNK + target_positions]
    ).astype(np.int32)
    target_lanes_jax = jnp.asarray(target_lanes)
    initial_material = material_snapshot(timestep.state)
    source_volume = initial_material["source_volume"]
    if bool(jnp.any(source_volume <= 0)):
        raise ValueError("every probe reset must have positive source-soil volume")
    previous_target_material = np.stack(
        [
            np.asarray(jax.device_get(initial_material[name][target_lanes_jax]))
            for name in (
                "dig_fraction",
                "terminal_soil_fraction",
                "off_zone_staged_soil_fraction",
                "loaded_soil_fraction",
            )
        ],
        axis=-1,
    )

    initial_target = timestep.state.world.target_map.map
    initial_padding = timestep.state.world.padding_mask.map
    initial_world_mass = timestep.state.world.action_map.map.astype(jnp.int32).sum(
        axis=tuple(range(1, timestep.state.world.action_map.map.ndim))
    )
    initial_loaded_mass = _total_loaded(timestep.state).astype(jnp.int32)
    initial_mass = initial_world_mass + initial_loaded_mass
    maximum_mass_residual = jnp.zeros(TOTAL_COUNT, dtype=jnp.int32)
    target_mutation = jnp.zeros(TOTAL_COUNT, dtype=jnp.bool_)
    obstacle_mutation = jnp.zeros(TOTAL_COUNT, dtype=jnp.bool_)
    nonfinite_state = jnp.zeros(TOTAL_COUNT, dtype=jnp.bool_)

    terminated = jnp.zeros(TOTAL_COUNT, dtype=jnp.bool_)
    succeeded = jnp.zeros(TOTAL_COUNT, dtype=jnp.bool_)
    episode_length = jnp.zeros(TOTAL_COUNT, dtype=jnp.int32)
    terminal_absolute = jnp.zeros(TOTAL_COUNT, dtype=jnp.float32)

    active_trace: list[np.ndarray] = []
    action_trace: list[np.ndarray] = []
    effect_trace: list[np.ndarray] = []
    material_trace: list[np.ndarray] = []
    material_changed_trace: list[np.ndarray] = []
    input_hash_trace: list[np.ndarray] = []
    hidden_hash_trace: list[np.ndarray] = []
    hidden_norm_trace: list[np.ndarray] = []
    logit_trace: list[np.ndarray] = []
    rng = jrandom.PRNGKey(SEED)
    rng, _ = jrandom.split(rng)
    input_schema = None
    first_step_pair_gate = None
    post_first_transition_pair_leaves = None

    for step in range(HORIZON):
        active = ~terminated
        active_host = np.asarray(jax.device_get(active), dtype=bool)
        model_input = obs_to_model_input(timestep.observation, prev_actions, config)

        target_input = jax.tree_util.tree_map(
            lambda value: value[target_lanes_jax], model_input
        )
        host_input = [np.asarray(value) for value in jax.device_get(target_input)]
        current_schema = [
            {"index": index, "dtype": str(value.dtype), "shape": list(value.shape[1:])}
            for index, value in enumerate(host_input)
        ]
        if input_schema is None:
            input_schema = current_schema
        elif input_schema != current_schema:
            raise ValueError(f"policy-input schema changed at step {step + 1}")
        input_hashes = np.stack(
            [
                np.frombuffer(digest_row(host_input, row), dtype=np.uint8)
                for row in range(2 * TARGET_COUNT)
            ]
        )

        hidden_used = actor_hidden.at[FORWARD_CHUNK:].set(0.0)
        host_hidden = np.asarray(
            jax.device_get(hidden_used[target_lanes_jax]), dtype=np.float32
        )
        hidden_hashes = np.stack(
            [np.frombuffer(digest_array(row), dtype=np.uint8) for row in host_hidden]
        )
        hidden_norms = np.linalg.norm(host_hidden, axis=1).astype(np.float32)

        _, logits, next_hidden = eval_mcts._apply_recurrent_in_batch_chunks(
            model, checkpoint["model"], model_input, hidden_used
        )
        host_logits = np.asarray(jax.device_get(logits), dtype=np.float32)
        if host_logits.shape != (TOTAL_COUNT, NUM_ACTIONS):
            raise ValueError(f"unexpected policy-logit shape {host_logits.shape}")
        if not np.all(np.isfinite(host_logits)):
            raise FloatingPointError(f"nonfinite logits at step {step + 1}")
        action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        host_action = np.asarray(jax.device_get(action), dtype=np.int32)

        if step == 0:
            full_input_pair_leaves = assert_paired_batch_equal(
                model_input, "initial model input"
            )
            if not np.array_equal(
                host_logits[:FORWARD_CHUNK], host_logits[FORWARD_CHUNK:]
            ):
                raise ValueError("paired chunks produced different initial logits")
            if not np.array_equal(
                host_action[:FORWARD_CHUNK], host_action[FORWARD_CHUNK:]
            ):
                raise ValueError("paired chunks chose different initial actions")
            first_step_pair_gate = {
                "model_input_pair_leaves": full_input_pair_leaves,
                "logits_bit_exact": True,
                "actions_exact": True,
            }

        active_trace.append(active_host[target_lanes])
        action_trace.append(host_action)
        input_hash_trace.append(input_hashes)
        hidden_hash_trace.append(hidden_hashes)
        hidden_norm_trace.append(hidden_norms)
        logit_trace.append(host_logits[target_lanes])

        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)
        rng, _, rng_step = jrandom.split(rng, 3)
        canonical_panel_keys = jrandom.split(rng_step, panel_count)
        one_chunk_keys = canonical_panel_keys[jnp.asarray(one_chunk_indices)]
        step_keys = jnp.concatenate([one_chunk_keys, one_chunk_keys], axis=0)
        candidate = env.step_no_reset(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            step_keys,
        )
        timestep = preserve_active(timestep, candidate, active)
        if step == 0:
            post_first_transition_pair_leaves = assert_paired_batch_equal(
                timestep, "post-first-transition timestep"
            )
        effect = active & jnp.asarray(timestep.info["action_had_effect"], dtype=jnp.bool_)
        effect_trace.append(np.asarray(jax.device_get(effect), dtype=bool))

        current_material = material_snapshot(timestep.state)
        target_material = np.stack(
            [
                np.asarray(jax.device_get(current_material[name][target_lanes_jax]))
                for name in (
                    "dig_fraction",
                    "terminal_soil_fraction",
                    "off_zone_staged_soil_fraction",
                    "loaded_soil_fraction",
                )
            ],
            axis=-1,
        ).astype(np.float32)
        target_active = active_host[target_lanes]
        material_changed = target_active & np.any(
            np.abs(target_material - previous_target_material) > 1e-7, axis=1
        )
        material_trace.append(target_material)
        material_changed_trace.append(material_changed)
        previous_target_material = target_material

        world_mass = timestep.state.world.action_map.map.astype(jnp.int32).sum(
            axis=tuple(range(1, timestep.state.world.action_map.map.ndim))
        )
        loaded_mass = _total_loaded(timestep.state).astype(jnp.int32)
        maximum_mass_residual = jnp.maximum(
            maximum_mass_residual,
            jnp.where(active, jnp.abs(world_mass + loaded_mass - initial_mass), 0),
        )
        target_mutation |= active & jnp.any(
            timestep.state.world.target_map.map != initial_target,
            axis=tuple(range(1, initial_target.ndim)),
        )
        obstacle_mutation |= active & jnp.any(
            timestep.state.world.padding_mask.map != initial_padding,
            axis=tuple(range(1, initial_padding.ndim)),
        )
        finite_per_leaf = []
        for leaf in jax.tree_util.tree_leaves(timestep.state):
            if (
                not hasattr(leaf, "shape")
                or leaf.ndim == 0
                or int(leaf.shape[0]) != TOTAL_COUNT
            ):
                continue
            finite_per_leaf.append(
                jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, leaf.ndim)))
            )
        if not finite_per_leaf:
            raise ValueError("state integrity check found no batched leaves")
        nonfinite_state |= active & ~jnp.all(jnp.stack(finite_per_leaf), axis=0)

        step_done = jnp.asarray(timestep.done, dtype=jnp.bool_)
        step_success = jnp.asarray(timestep.info["task_done"], dtype=jnp.bool_)
        components = timestep.info["reward_components"]
        if "absolute_completion" not in components:
            raise KeyError("reward components lack absolute_completion")
        terminal_event = active & step_done
        terminal_absolute = jnp.where(
            terminal_event,
            jnp.asarray(components["absolute_completion"], dtype=jnp.float32),
            terminal_absolute,
        )
        episode_length += active.astype(jnp.int32)
        succeeded |= active & step_success
        terminated |= terminal_event
        prev_actions = jnp.where(
            step_done[:, None], jnp.zeros_like(prev_actions), prev_actions
        )
        actor_hidden = jnp.where(
            step_done[:, None], jnp.zeros_like(next_hidden), next_hidden
        ).at[FORWARD_CHUNK:].set(0.0)

        if bool(jnp.all(terminated)):
            break

    if not bool(jnp.all(terminated)):
        missing = (np.flatnonzero(~np.asarray(jax.device_get(terminated))) + 1).tolist()
        raise ValueError(f"probe lanes did not terminate by horizon: {missing}")
    if first_step_pair_gate is None:
        raise RuntimeError("the first-step paired gate did not run")
    if post_first_transition_pair_leaves is None:
        raise RuntimeError("the first-transition paired gate did not run")

    active_array = np.stack(active_trace)
    all_actions = np.stack(action_trace)
    all_effects = np.stack(effect_trace)
    material_array = np.stack(material_trace)
    material_changed_array = np.stack(material_changed_trace)
    input_hash_array = np.stack(input_hash_trace)
    hidden_hash_array = np.stack(hidden_hash_trace)
    hidden_norm_array = np.stack(hidden_norm_trace)
    logits_array = np.stack(logit_trace)
    target_actions = all_actions[:, target_lanes]
    target_effects = all_effects[:, target_lanes]
    if np.any(hidden_norm_array[:, TARGET_COUNT:] != 0.0):
        raise ValueError("zero-each-decision target lanes received nonzero carry")

    success_host = np.asarray(jax.device_get(succeeded), dtype=bool)
    terminated_host = np.asarray(jax.device_get(terminated), dtype=bool)
    lengths_host = np.asarray(jax.device_get(episode_length), dtype=np.int32)
    absolute_host = np.asarray(jax.device_get(terminal_absolute), dtype=np.float32)
    mass_residual_host = np.asarray(jax.device_get(maximum_mass_residual), dtype=np.int32)
    target_mutation_host = np.asarray(jax.device_get(target_mutation), dtype=bool)
    obstacle_mutation_host = np.asarray(jax.device_get(obstacle_mutation), dtype=bool)
    nonfinite_host = np.asarray(jax.device_get(nonfinite_state), dtype=bool)
    final_material = {
        key: np.asarray(jax.device_get(value), dtype=np.float32)
        for key, value in material_snapshot(timestep.state).items()
    }
    partition_residual = np.abs(
        final_material["dig_fraction"]
        - final_material["terminal_soil_fraction"]
        - final_material["off_zone_staged_soil_fraction"]
        - final_material["loaded_soil_fraction"]
    )

    if not np.array_equal(terminated_host, np.ones(TOTAL_COUNT, dtype=bool)):
        raise ValueError("terminal integrity failed")
    if not np.array_equal(success_host, np.isclose(absolute_host, 1.0, atol=1e-6)):
        raise ValueError("task_done is not equivalent to absolute_completion == 1")
    if np.any(mass_residual_host != 0):
        raise ValueError("mass conservation failed")
    if np.any(target_mutation_host) or np.any(obstacle_mutation_host):
        raise ValueError("immutable target or obstacle map changed")
    if np.any(nonfinite_host):
        raise FloatingPointError("nonfinite simulator state encountered")
    if np.max(partition_residual) > 1e-6:
        raise ValueError(f"material partition residual {np.max(partition_residual)}")

    target_rows = []
    total_same_input_hidden = 0
    total_same_input_logits = 0
    total_same_input_actions = 0
    for target_index, (slot, role) in enumerate(zip(TARGET_SLOTS, TARGET_ROLES)):
        normal_lane = target_index
        zero_lane = TARGET_COUNT + target_index
        normal_full_lane = int(target_positions[target_index])
        zero_full_lane = FORWARD_CHUNK + normal_full_lane
        normal_analysis = analyze_lane(
            active=active_array[:, normal_lane],
            actions=target_actions[:, normal_lane],
            effects=target_effects[:, normal_lane],
            material_changed=material_changed_array[:, normal_lane],
            input_hashes=input_hash_array[:, normal_lane],
            hidden_hashes=hidden_hash_array[:, normal_lane],
            hidden_norms=hidden_norm_array[:, normal_lane],
            logits=logits_array[:, normal_lane],
        )
        zero_analysis = analyze_lane(
            active=active_array[:, zero_lane],
            actions=target_actions[:, zero_lane],
            effects=target_effects[:, zero_lane],
            material_changed=material_changed_array[:, zero_lane],
            input_hashes=input_hash_array[:, zero_lane],
            hidden_hashes=hidden_hash_array[:, zero_lane],
            hidden_norms=hidden_norm_array[:, zero_lane],
            logits=logits_array[:, zero_lane],
        )
        total_same_input_hidden += normal_analysis["same_input_different_hidden_decisions"]
        total_same_input_logits += normal_analysis["same_input_different_logits_decisions"]
        total_same_input_actions += normal_analysis["same_input_different_action_decisions"]
        manifest = manifest_rows[slot - 1]
        expected = fixed_rows[slot - 1]
        if (
            expected["map_id"] != manifest["map_id"]
            or expected["primary_cell"] != manifest["primary_cell"]
            or expected["family"] != manifest["family"]
        ):
            raise ValueError(f"fixed-record identity mismatch at slot {slot}")
        normal_observed = {
            "success": bool(success_host[normal_full_lane]),
            "steps": int(lengths_host[normal_full_lane]),
            "terminal_absolute": float(absolute_host[normal_full_lane]),
            "dig_fraction": float(final_material["dig_fraction"][normal_full_lane]),
            "terminal_soil_fraction": float(
                final_material["terminal_soil_fraction"][normal_full_lane]
            ),
            "off_zone_staged_soil_fraction": float(
                final_material["off_zone_staged_soil_fraction"][normal_full_lane]
            ),
            "loaded_soil_fraction": float(
                final_material["loaded_soil_fraction"][normal_full_lane]
            ),
            "no_effect_action_count": normal_analysis["no_effect_action_count"],
        }
        normal_expected = {
            key: expected[key]
            for key in (
                "success",
                "steps",
                "terminal_absolute",
                "dig_fraction",
                "terminal_soil_fraction",
                "off_zone_staged_soil_fraction",
                "loaded_soil_fraction",
                "no_effect_action_count",
            )
        }
        for key, expected_value in normal_expected.items():
            observed_value = normal_observed[key]
            if isinstance(expected_value, float):
                matches = bool(np.isclose(observed_value, expected_value, atol=1e-6))
            else:
                matches = observed_value == expected_value
            if not matches:
                raise ValueError(
                    f"normal-arm canonical parity mismatch at slot {slot}, {key}: "
                    f"{observed_value!r} != {expected_value!r}"
                )

        zero_terminal = {
            "terminal_absolute": float(absolute_host[zero_full_lane]),
            "dig_fraction": float(final_material["dig_fraction"][zero_full_lane]),
            "terminal_soil_fraction": float(
                final_material["terminal_soil_fraction"][zero_full_lane]
            ),
            "off_zone_staged_soil_fraction": float(
                final_material["off_zone_staged_soil_fraction"][zero_full_lane]
            ),
            "loaded_soil_fraction": float(
                final_material["loaded_soil_fraction"][zero_full_lane]
            ),
        }
        common = active_array[:, normal_lane] & active_array[:, zero_lane]
        same_input = np.all(
            input_hash_array[:, normal_lane] == input_hash_array[:, zero_lane],
            axis=1,
        )
        different_hidden = np.any(
            hidden_hash_array[:, normal_lane] != hidden_hash_array[:, zero_lane],
            axis=1,
        )
        different_logits = np.any(
            logits_array[:, normal_lane] != logits_array[:, zero_lane], axis=1
        )
        different_actions = (
            target_actions[:, normal_lane] != target_actions[:, zero_lane]
        )
        same_input_hidden_steps = np.flatnonzero(
            common & same_input & different_hidden
        )
        same_input_logit_steps = np.flatnonzero(
            common & same_input & different_logits
        )
        same_input_action_steps = np.flatnonzero(
            common & same_input & different_actions
        )
        target_rows.append(
            {
                "slot_index": slot,
                "map_id": manifest["map_id"],
                "condition": manifest["primary_cell"],
                "family": manifest["family"],
                "diagnostic_role": role,
                "pair_outcome": _pair_outcome(
                    bool(success_host[normal_full_lane]),
                    bool(success_host[zero_full_lane]),
                ),
                "normal_carry": {
                    **normal_observed,
                    **normal_analysis,
                },
                "zero_each_decision": {
                    "success": bool(success_host[zero_full_lane]),
                    "steps": int(lengths_host[zero_full_lane]),
                    **zero_terminal,
                    **zero_analysis,
                },
                "canonical_normal_arm_parity": {
                    "passed": True,
                    "fixed_expected": normal_expected,
                },
                "paired": {
                    "first_action_divergence_step": first_paired_difference(
                        target_actions[:, normal_lane],
                        target_actions[:, zero_lane],
                        active_array[:, normal_lane],
                        active_array[:, zero_lane],
                    ),
                    "first_instantaneous_input_divergence_step": first_paired_difference(
                        input_hash_array[:, normal_lane],
                        input_hash_array[:, zero_lane],
                        active_array[:, normal_lane],
                        active_array[:, zero_lane],
                    ),
                    "first_logit_divergence_step": first_paired_difference(
                        logits_array[:, normal_lane],
                        logits_array[:, zero_lane],
                        active_array[:, normal_lane],
                        active_array[:, zero_lane],
                    ),
                    "same_input_different_hidden_decisions": int(
                        same_input_hidden_steps.size
                    ),
                    "same_input_different_logits_decisions": int(
                        same_input_logit_steps.size
                    ),
                    "same_input_different_action_decisions": int(
                        same_input_action_steps.size
                    ),
                    "first_same_input_hidden_divergence_step": (
                        int(same_input_hidden_steps[0] + 1)
                        if same_input_hidden_steps.size
                        else None
                    ),
                    "first_same_input_logit_divergence_step": (
                        int(same_input_logit_steps[0] + 1)
                        if same_input_logit_steps.size
                        else None
                    ),
                    "first_same_input_action_divergence_step": (
                        int(same_input_action_steps[0] + 1)
                        if same_input_action_steps.size
                        else None
                    ),
                },
            }
        )

    padding_rows = []
    for padding_position, slot in zip(padding_positions, padding_slots):
        normal_lane = int(padding_position)
        zero_lane = FORWARD_CHUNK + normal_lane
        common_active = np.arange(all_actions.shape[0]) < np.minimum(
            lengths_host[normal_lane], lengths_host[zero_lane]
        )
        divergence = np.flatnonzero(
            common_active & (all_actions[:, normal_lane] != all_actions[:, zero_lane])
        )
        manifest = manifest_rows[slot - 1]
        padding_rows.append(
            {
                "slot_index": slot,
                "map_id": manifest["map_id"],
                "condition": manifest["primary_cell"],
                "family": manifest["family"],
                "normal_success": bool(success_host[normal_lane]),
                "zero_success": bool(success_host[zero_lane]),
                "normal_steps": int(lengths_host[normal_lane]),
                "zero_steps": int(lengths_host[zero_lane]),
                "pair_outcome": _pair_outcome(
                    bool(success_host[normal_lane]), bool(success_host[zero_lane])
                ),
                "first_action_divergence_step": (
                    int(divergence[0] + 1) if divergence.size else None
                ),
                "normal_no_effect_action_count": int(
                    np.sum(~all_effects[: lengths_host[normal_lane], normal_lane])
                ),
                "zero_no_effect_action_count": int(
                    np.sum(~all_effects[: lengths_host[zero_lane], zero_lane])
                ),
            }
        )

    normal_target_success = int(success_host[target_positions].sum())
    zero_target_success = int(
        success_host[FORWARD_CHUNK + target_positions].sum()
    )
    outcome_counts = Counter(row["pair_outcome"] for row in target_rows)
    paired_same_input_logit_maps = sum(
        row["paired"]["same_input_different_logits_decisions"] > 0
        for row in target_rows
    )
    paired_same_input_action_maps = sum(
        row["paired"]["same_input_different_action_decisions"] > 0
        for row in target_rows
    )
    summary = {
        "target_episodes_per_arm": TARGET_COUNT,
        "padding_controls_per_arm": PADDING_COUNT,
        "normal_target_exact": normal_target_success,
        "zero_each_decision_target_exact": zero_target_success,
        "target_pair_outcomes": dict(sorted(outcome_counts.items())),
        "normal_target_same_input_different_hidden_decisions": total_same_input_hidden,
        "normal_target_same_input_different_logits_decisions": total_same_input_logits,
        "normal_target_same_input_different_action_decisions": total_same_input_actions,
        "target_maps_with_paired_same_input_logit_divergence": (
            paired_same_input_logit_maps
        ),
        "target_maps_with_paired_same_input_action_divergence": (
            paired_same_input_action_maps
        ),
        "normal_padding_exact": int(success_host[padding_positions].sum()),
        "zero_each_decision_padding_exact": int(
            success_host[FORWARD_CHUNK + padding_positions].sum()
        ),
        "maximum_mass_residual": int(mass_residual_host.max()),
        "maximum_material_partition_residual": float(partition_residual.max()),
    }

    output.mkdir(parents=True)
    arrays_path = output / "target_trace_arrays.npz"
    np.savez_compressed(
        arrays_path,
        target_slots=np.asarray(TARGET_SLOTS, dtype=np.int32),
        target_lane_order=np.asarray(
            [f"normal:{slot}" for slot in TARGET_SLOTS]
            + [f"zero_each_decision:{slot}" for slot in TARGET_SLOTS]
        ),
        active_before_step=active_array,
        actions=target_actions.astype(np.int8),
        action_had_effect=target_effects,
        material_fractions=material_array,
        material_changed=material_changed_array,
        instantaneous_input_sha256=input_hash_array,
        pre_action_hidden_sha256=hidden_hash_array,
        pre_action_hidden_norm=hidden_norm_array,
        logits=logits_array,
    )

    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_update": int(checkpoint.get("next_update", 0)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "baselines_source": baselines_source,
        "terra_source": terra_source,
        "canonical_fixed_evaluation": {
            "path": str(fixed_eval_path),
            "sha256": sha256_file(fixed_eval_path),
            "checkpoint_sha256": fixed_record["checkpoint_sha256"],
            "normal_target_parity_passed": True,
            "parity_fields": [
                "success",
                "steps",
                "terminal_absolute",
                "dig_fraction",
                "terminal_soil_fraction",
                "off_zone_staged_soil_fraction",
                "loaded_soil_fraction",
                "no_effect_action_count",
            ],
        },
        "bank_root": str(bank_root),
        "bank_dataset_sha256": sha256_file(bank_root / "dataset.json"),
        "manifest": str(directory / "manifest.jsonl"),
        "manifest_sha256": manifest_sha256,
        "bank_terra_revision": args.terra_revision,
        "environment_protocol_sha256": accepted_bank.environment_protocol_sha256,
        "horizon": HORIZON,
        "seed": SEED,
        "deterministic_greedy": True,
        "action_logit_masking": False,
        "num_previous_actions": NUM_PREVIOUS_ACTIONS,
        "action_names_by_id": list(ACTION_NAMES),
        "intervention": "zero_actor_gru_carry_before_every_decision",
        "paired_execution": {
            "total_rows": TOTAL_COUNT,
            "normal_chunk_rows": [0, FORWARD_CHUNK - 1],
            "zero_each_decision_chunk_rows": [FORWARD_CHUNK, TOTAL_COUNT - 1],
            "policy_forward_chunk": FORWARD_CHUNK,
            "same_order_and_chunk_position": True,
            "paired_transition_rng_keys": True,
            "target_slots": list(TARGET_SLOTS),
            "target_chunk_positions_zero_based": target_positions.tolist(),
            "target_position_contract": "canonical_(slot_minus_1)_mod_120",
            "padding_selection": "canonical_slots_1_to_120_with_target_substitution",
            "padding_slots": list(padding_slots),
            "initial_timestep_pair_leaves": initial_timestep_pair_leaves,
            "initial_env_param_pair_leaves": initial_env_param_pair_leaves,
            "first_step_gate": first_step_pair_gate,
            "post_first_transition_pair_leaves": (
                post_first_transition_pair_leaves
            ),
        },
        "policy_input_schema": input_schema,
        "hash_contract": {
            "instantaneous_input": "sha256 over exact post-obs_to_model_input row leaves in order",
            "pre_action_hidden": "sha256 over exact float32 GRU carry row",
        },
        "material_fraction_order": [
            "dig_fraction",
            "terminal_soil_fraction",
            "off_zone_staged_soil_fraction",
            "loaded_soil_fraction",
        ],
        "integrity": {
            "exact_reset": reset_receipt,
            "all_lanes_terminated": True,
            "task_done_iff_absolute_completion_one": True,
            "maximum_mass_residual": int(mass_residual_host.max()),
            "target_mutation_count": int(target_mutation_host.sum()),
            "obstacle_mutation_count": int(obstacle_mutation_host.sum()),
            "nonfinite_state_count": int(nonfinite_host.sum()),
            "maximum_material_partition_residual": float(partition_residual.max()),
        },
        "interpretation_limit": (
            "This same-process intervention isolates evaluation-time use of recurrent carry. "
            "It does not isolate recurrence from training-time capacity or sequence batching. "
            "The 108 padding rows preserve execution shape and are not the mechanism panel."
        ),
        "summary": summary,
        "target_rows": target_rows,
        "padding_control_rows": padding_rows,
        "target_trace_arrays": str(arrays_path),
        "target_trace_arrays_sha256": sha256_file(arrays_path),
    }
    write_json(output / "receipt.json", receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
