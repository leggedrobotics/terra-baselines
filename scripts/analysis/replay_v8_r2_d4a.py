#!/usr/bin/env python3
"""Replay the frozen compact-u20 policy and emit the durable D4a receipt.

Run this script with PYTHONPATH pointing at the exact baseline and Terra source
revisions named on the command line.  It first calls the existing fixed-bank
rollout, requires full 720-map parity with the frozen evaluator JSON, then
replays that exact action tensor through ``step_no_reset`` to audit the hidden
material-work ledger.  It never trains or submits a job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from eval_fixed_bank import (
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
    verify_exact_reset,
)
from eval_mcts import rollout_episode
from train_mixed import _validate_checkpoint_architecture, make_mixed_agent_states
from utils.accepted_bank import load_accepted_bank
from utils.helpers import load_pkl_object
from utils.utils_ppo import wrap_action

from d4a_ledger import (
    LIFT_ABSOLUTE_FLOOR,
    LIFT_ULP_MULTIPLIER,
    lift_conservation_diagnostic,
)

SCHEMA = "terra_v8_r2_d4a_replay_v1"
LIFT_DIAGNOSTICS_SCHEMA = "terra_v8_r2_d4a_lift_diagnostics_v1"
LIFT_DIAGNOSTICS_TOP_K = 32
EXPECTED_CHECKPOINT_SHA256 = (
    "0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834"
)
EXPECTED_EVAL_SHA256 = (
    "dd8c3b381e57889827462222c81f29003a8b19f6285abd87247db5e60a2fea26"
)
EXPECTED_BANK_DATASET_SHA256 = (
    "715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798"
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "8b49bd848c542e30b9e4d45639e4678905244b09a50482ff1c5c2f1d979dff19"
)
EXPECTED_ENVIRONMENT_PROTOCOL_SHA256 = (
    "9917b9238e9e6e844377e6d4a8ca18d1f0defbbacf887642743e579243109367"
)
TARGET_TRACES = {
    97: "d12_success",
    100: "d12_illegal_soil_timeout",
    109: "d12_fully_dug_loaded_timeout",
    112: "d12_no_accepted_dump_timeout",
    113: "d16_no_soil_touch_timeout",
    119: "d16_success",
    124: "d16_near_finished_rehandling_loop",
    129: "nearby_success",
    177: "obstacle_no_effect_loop",
}
COMPLETION_TO_FROZEN = {
    "absolute": "terminal_absolute",
    "dig": "terminal_dig",
    "dump_purity": "terminal_dump_purity",
    "dump_volume": "terminal_dump_volume",
    "unloaded": "terminal_unloaded",
    "accepted_dump_volume": "terminal_accepted_dump_volume",
    "illegal_dump_volume": "terminal_illegal_dump_volume",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    return sha256_file(path)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_is_clean(path: Path) -> bool:
    return not subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()


def _map2(array: jax.Array) -> jax.Array:
    return jnp.reshape(array, (array.shape[0], array.shape[-2], array.shape[-1]))


@jax.jit
def _material_snapshot(
    target_raw: jax.Array,
    action_raw: jax.Array,
    distance_raw: jax.Array,
    agent_active: jax.Array,
    carry_credit: jax.Array,
    loaded: jax.Array,
) -> tuple[jax.Array, ...]:
    target = _map2(target_raw).astype(jnp.float32)
    action = _map2(action_raw).astype(jnp.float32)
    distance = _map2(distance_raw).astype(jnp.float32)
    required = jnp.clip(-target, a_min=0.0)
    remaining = jnp.where(
        target < 0,
        jnp.clip(action - target, a_min=0.0, a_max=required),
        0.0,
    )
    accepted = target > 0
    offzone = jnp.where(~accepted, jnp.clip(action, a_min=0.0), 0.0)
    active = agent_active.astype(jnp.float32)
    total_carry = jnp.sum(carry_credit * active, axis=1)
    total_loaded = jnp.sum(loaded * active, axis=1)
    h_remaining = jnp.sum(remaining * distance, axis=(-2, -1))
    h_offzone = jnp.sum(offzone * distance, axis=(-2, -1))
    v0 = jnp.sum(required, axis=(-2, -1))
    q = (v0 - jnp.sum(remaining, axis=(-2, -1))) / jnp.maximum(v0, 1.0)
    return (
        h_remaining + h_offzone + total_carry,
        h_remaining,
        h_offzone,
        total_carry,
        total_loaded,
        q,
        v0,
    )


@jax.jit
def _map_transition(
    target_raw: jax.Array,
    before_raw: jax.Array,
    after_raw: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    target = _map2(target_raw).astype(jnp.float32)
    before = _map2(before_raw).astype(jnp.float32)
    after = _map2(after_raw).astype(jnp.float32)
    required = jnp.clip(-target, a_min=0.0)
    before_remaining = jnp.where(
        target < 0,
        jnp.clip(before - target, a_min=0.0, a_max=required),
        0.0,
    )
    after_remaining = jnp.where(
        target < 0,
        jnp.clip(after - target, a_min=0.0, a_max=required),
        0.0,
    )
    fresh_lift = jnp.sum(
        jnp.clip(before_remaining - after_remaining, a_min=0.0), axis=(-2, -1)
    )
    rehandled_lift = jnp.sum(
        jnp.clip(
            jnp.clip(before, a_min=0.0) - jnp.clip(after, a_min=0.0),
            a_min=0.0,
        ),
        axis=(-2, -1),
    )
    deposited = jnp.sum(
        jnp.clip(
            jnp.clip(after, a_min=0.0) - jnp.clip(before, a_min=0.0),
            a_min=0.0,
        ),
        axis=(-2, -1),
    )
    return fresh_lift, rehandled_lift, deposited


def material_snapshot(state) -> tuple[np.ndarray, ...]:
    carry = jnp.stack(
        [
            jnp.reshape(
                agent_state.carry_relocation_credit, (state.env_steps.shape[0], -1)
            ).sum(axis=1)
            for agent_state in state.agent.agent_states
        ],
        axis=1,
    )
    loaded = jnp.stack(
        [
            jnp.reshape(agent_state.loaded, (state.env_steps.shape[0], -1)).sum(axis=1)
            for agent_state in state.agent.agent_states
        ],
        axis=1,
    )
    values = _material_snapshot(
        state.world.target_map.map,
        state.world.action_map.map,
        state.world.relocation_distance_map,
        state.agent.agent_active,
        carry,
        loaded,
    )
    return tuple(np.asarray(value) for value in values)


def selected_terminal_state_sha256(state, index: int) -> str:
    digest = hashlib.sha256()
    fields = {
        "target_map": state.world.target_map.map,
        "action_map": state.world.action_map.map,
        "padding_mask": state.world.padding_mask.map,
        "dumpability_mask_init": state.world.dumpability_mask_init.map,
        "distance": state.world.relocation_distance_map,
        "agent_active": state.agent.agent_active,
        "current_agent": state.agent.current_agent,
        "env_steps": state.env_steps,
    }
    for agent_index, agent_state in enumerate(state.agent.agent_states):
        for field in agent_state._fields:
            fields[f"agent_{agent_index}_{field}"] = getattr(agent_state, field)
    for name, value in sorted(fields.items()):
        array = np.asarray(jax.device_get(value))
        if array.ndim and array.shape[0] == state.env_steps.shape[0]:
            array = array[index]
        array = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_frozen_parity(
    stats: dict[str, Any],
    frozen: dict[str, Any],
    targeted_slots: list[int],
) -> dict[str, Any]:
    rows = sorted(frozen["per_map"], key=lambda row: row["slot_index"])
    if [row["slot_index"] for row in rows] != list(range(1, 721)):
        raise ValueError(
            "frozen development panel is not an exact 720-slot enumeration"
        )
    success = np.asarray(stats["episode_done_once"], dtype=bool)
    lengths = np.asarray(stats["episode_length"], dtype=np.int32)
    expected_success = np.asarray([row["success"] for row in rows], dtype=bool)
    expected_lengths = np.asarray([row["steps"] for row in rows], dtype=np.int32)
    success_mismatches = np.flatnonzero(success != expected_success) + 1
    length_mismatches = np.flatnonzero(lengths != expected_lengths) + 1
    max_completion_error = 0.0
    completion_errors: dict[str, np.ndarray] = {}
    for name, frozen_name in COMPLETION_TO_FROZEN.items():
        actual = np.asarray(stats["terminal_completion"][name], dtype=np.float64)
        expected = np.asarray([row[frozen_name] for row in rows], dtype=np.float64)
        errors = np.abs(actual - expected)
        completion_errors[name] = errors
        error = float(np.max(errors))
        max_completion_error = max(max_completion_error, error)
    no_effect = np.asarray(stats["integrity"]["no_effect_action_count"], dtype=np.int32)
    expected_no_effect = np.asarray(
        [row["no_effect_action_count"] for row in rows], dtype=np.int32
    )
    no_effect_mismatches = np.flatnonzero(no_effect != expected_no_effect) + 1
    targeted_indices = np.asarray(targeted_slots, dtype=np.int32) - 1
    targeted_checks = {
        "success": bool(
            np.array_equal(
                success[targeted_indices], expected_success[targeted_indices]
            )
        ),
        "episode_length": bool(
            np.array_equal(
                lengths[targeted_indices], expected_lengths[targeted_indices]
            )
        ),
        "no_effect_count": bool(
            np.array_equal(
                no_effect[targeted_indices], expected_no_effect[targeted_indices]
            )
        ),
        **{
            f"completion_{name}": bool(np.max(errors[targeted_indices]) <= 1e-6)
            for name, errors in completion_errors.items()
        },
    }
    if not all(targeted_checks.values()):
        raise ValueError(f"one or more targeted traces differ: {targeted_checks}")
    full_panel_equal = bool(
        not success_mismatches.size
        and not length_mismatches.size
        and not no_effect_mismatches.size
        and max_completion_error <= 1e-6
    )
    return {
        "episodes": 720,
        "frozen_exact_successes": int(expected_success.sum()),
        "local_exact_successes": int(success.sum()),
        "full_panel_equal": full_panel_equal,
        "success_mismatch_slots": success_mismatches.tolist(),
        "episode_length_mismatch_slots": length_mismatches.tolist(),
        "no_effect_mismatch_slots": no_effect_mismatches.tolist(),
        "maximum_terminal_completion_error": max_completion_error,
        "cross_hardware_policy_drift_is_non_gating": True,
        "targeted_slots": targeted_slots,
        "targeted_checks": targeted_checks,
        "all_nine_equal": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--frozen-development-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baselines-source", type=Path, required=True)
    parser.add_argument("--terra-source", type=Path, required=True)
    parser.add_argument("--baselines-revision", required=True)
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--bank-terra-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--gamma", type=float, default=0.9984)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    bank_root = args.bank_root.resolve()
    frozen_eval_path = args.frozen_development_eval.resolve()
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash mismatch")
    if sha256_file(frozen_eval_path) != EXPECTED_EVAL_SHA256:
        raise ValueError("frozen development-eval hash mismatch")
    if sha256_file(bank_root / "dataset.json") != EXPECTED_BANK_DATASET_SHA256:
        raise ValueError("accepted bank dataset hash mismatch")
    bank_index = json.loads((bank_root / "dataset.json").read_text())
    if bank_index["source_registry_sha256"] != EXPECTED_SOURCE_REGISTRY_SHA256:
        raise ValueError("accepted bank source-registry hash mismatch")
    if (
        bank_index["environment_protocol_sha256"]
        != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
    ):
        raise ValueError("accepted bank environment-protocol hash mismatch")
    if git_head(args.baselines_source) != args.baselines_revision:
        raise ValueError("baselines source is not at the requested revision")
    if git_head(args.terra_source) != args.terra_revision:
        raise ValueError("Terra source is not at the requested revision")
    if not git_is_clean(args.baselines_source) or not git_is_clean(args.terra_source):
        raise ValueError("D4a requires clean exact source trees")

    frozen_history = json.loads(frozen_eval_path.read_text())
    frozen = frozen_history[-1]
    if frozen["checkpoint_update"] != 20_000:
        raise ValueError("frozen evaluation does not end at update 20,000")
    if frozen["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen evaluation checkpoint hash mismatch")

    checkpoint_data = load_pkl_object(str(checkpoint))
    train_config = checkpoint_data["train_config"]
    _validate_checkpoint_architecture(checkpoint_data, train_config)
    accepted_bank = load_accepted_bank(
        bank_root,
        "G-UNIFORM",
        args.bank_terra_revision,
        curriculum_stage="full",
    )
    panel = next(
        panel
        for panel in accepted_bank.evaluation_panels
        if panel.name == "development"
    )
    relative_path = "evaluation/main/development"
    if panel.maps_path != relative_path:
        raise ValueError(f"unexpected development path: {panel.maps_path}")
    directory = bank_root / relative_path
    rows = load_manifest(directory)
    count = len(rows)
    if count != 720:
        raise ValueError(f"expected 720 development maps, got {count}")
    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(count)
    config = configure_for_bank(train_config, relative_path, count)
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    map_reset_keys = exact_reset_keys(count)
    state_keys = manifest_environment_keys(
        rows, count, accepted_bank.environment_protocol_sha256
    )
    initial_timestep, env_params, state_keys = prepare_manifest_episode_reset(
        env, env_params, map_reset_keys, state_keys
    )
    reset_verification = verify_exact_reset(
        env,
        env_params,
        None,
        directory,
        count,
        timestep=initial_timestep,
    )
    if not reset_verification["passed"]:
        raise ValueError("fixed reset verification failed")

    model = SimpleNamespace(apply=initialized_state.apply_fn)
    _, stats, _ = rollout_episode(
        env,
        model,
        checkpoint_data["model"],
        env_params,
        config,
        max_frames=450,
        deterministic=True,
        seed=args.seed,
        use_mcts=False,
        reset_keys=None,
        record_observations=False,
        record_actions=True,
        preserve_terminal_states=True,
        expected_slot_indices=np.arange(count, dtype=np.int32),
        record_completion=True,
        initial_timestep=initial_timestep,
    )
    parity = require_frozen_parity(stats, frozen, sorted(TARGET_TRACES))
    actions = np.asarray(stats["action_sequence"], dtype=np.int32)
    effects = np.asarray(stats["action_had_effect_sequence"], dtype=bool)
    if actions.shape != (450, 720):
        raise ValueError(f"unexpected action tensor shape: {actions.shape}")
    action_sha = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()

    frozen_rows = sorted(frozen["per_map"], key=lambda row: row["slot_index"])
    target_indices = {slot - 1: label for slot, label in TARGET_TRACES.items()}
    trace_rows: list[dict[str, Any]] = []
    trace_accumulator = {
        index: {
            "dump_progress": 0.0,
            "undiscounted_relocation_return": 0.0,
            "discounted_relocation_return": 0.0,
            "lift_events": 0,
            "rehandle_events": 0,
            "rehandled_volume": 0.0,
            "dump_events": 0,
            "zero_progress_dump_events": 0,
            "max_lift_conservation_error": 0.0,
            "max_carry_credit": 0.0,
            "longest_h_stagnation_steps": 0,
            "current_h_stagnation_steps": 0,
            "last_100_h": [],
        }
        for index in target_indices
    }

    timestep = initial_timestep
    initial_snapshot = material_snapshot(timestep.state)
    initial_h = initial_snapshot[0]
    v0 = initial_snapshot[6]
    target_scale = np.clip(170.0 / np.maximum(v0, 1.0), 2.0, 5.0) / 2.0
    active = np.ones(count, dtype=bool)
    dump_progress_all = np.zeros(count, dtype=np.float64)
    max_lift_error = 0.0
    max_lift_location = None
    max_inert_error = 0.0
    lift_diagnostics: list[dict[str, Any]] = []
    rng = jrandom.PRNGKey(args.seed)
    rng, _ = jrandom.split(rng)
    for step in range(450):
        before = timestep
        before_snapshot = material_snapshot(before.state)
        rng, _, rng_step = jrandom.split(rng, 3)
        rng_step = jrandom.split(rng_step, count)
        wrapped = wrap_action(jnp.asarray(actions[step]), env.batch_cfg.action_type)
        candidate = env.step_no_reset(before, wrapped, rng_step)

        active_jax = jnp.asarray(active)

        def preserve(previous, proposed):
            if not hasattr(proposed, "shape"):
                return proposed
            if proposed.ndim == 0 or proposed.shape[0] != count:
                return proposed
            mask = active_jax.reshape((count,) + (1,) * (proposed.ndim - 1))
            return jnp.where(mask, proposed, previous)

        timestep = jax.tree_util.tree_map(preserve, before, candidate)
        after_snapshot = material_snapshot(timestep.state)
        h_before, _, _, carry_before, load_before, q_before, _ = before_snapshot
        h_after, _, _, carry_after, load_after, q_after, _ = after_snapshot
        fresh, rehandled, deposited = (
            np.asarray(value)
            for value in _map_transition(
                before.state.world.target_map.map,
                before.state.world.action_map.map,
                timestep.state.world.action_map.map,
            )
        )
        h_progress = h_before.astype(np.float64) - h_after.astype(np.float64)
        lifted = (load_after > load_before) & active
        dumped = (load_after < load_before) & active
        inert = active & ~lifted & ~dumped
        if np.any(lifted):
            if h_before.dtype != np.float32 or h_after.dtype != np.float32:
                raise TypeError(
                    "D4a lift diagnostics require float32 H values, got "
                    f"{h_before.dtype} and {h_after.dtype}"
                )
            active_indices = np.flatnonzero(lifted)
            local = np.abs(h_progress[active_indices])
            for index in active_indices:
                lift_diagnostics.append(
                    lift_conservation_diagnostic(
                        h_before[index],
                        h_after[index],
                        slot_index=int(index + 1),
                        step=step + 1,
                        targeted_label=TARGET_TRACES.get(int(index + 1)),
                    )
                )
            observed = float(local.max())
            if observed > max_lift_error:
                local_index = int(np.argmax(local))
                max_lift_error = observed
                max_lift_location = {
                    "step": step + 1,
                    "slot_index": int(active_indices[local_index] + 1),
                }
        if np.any(inert):
            max_inert_error = max(
                max_inert_error, float(np.max(np.abs(h_progress[inert])))
            )
        dump_progress_all += np.where(dumped, h_progress, 0.0)

        completion = stats["completion_sequence"]
        for index, label in target_indices.items():
            if step >= int(stats["episode_length"][index]):
                continue
            relocation_reward = (
                h_progress[index] * 1.5 * target_scale[index] / 70.0
                if dumped[index]
                else 0.0
            )
            acc = trace_accumulator[index]
            if lifted[index]:
                acc["lift_events"] += 1
                acc["max_lift_conservation_error"] = max(
                    acc["max_lift_conservation_error"], abs(float(h_progress[index]))
                )
            if rehandled[index] > 0:
                acc["rehandle_events"] += 1
                acc["rehandled_volume"] += float(rehandled[index])
            if dumped[index]:
                acc["dump_events"] += 1
                acc["dump_progress"] += float(h_progress[index])
                if abs(float(h_progress[index])) <= 1e-6:
                    acc["zero_progress_dump_events"] += 1
            acc["undiscounted_relocation_return"] += float(relocation_reward)
            acc["discounted_relocation_return"] += float(
                (args.gamma**step) * relocation_reward
            )
            acc["max_carry_credit"] = max(
                acc["max_carry_credit"], float(carry_after[index])
            )
            if abs(float(h_progress[index])) <= 1e-6:
                acc["current_h_stagnation_steps"] += 1
                acc["longest_h_stagnation_steps"] = max(
                    acc["longest_h_stagnation_steps"],
                    acc["current_h_stagnation_steps"],
                )
            else:
                acc["current_h_stagnation_steps"] = 0
            episode_length = int(stats["episode_length"][index])
            if step >= max(0, episode_length - 100):
                acc["last_100_h"].append(float(h_after[index]))
            trace_rows.append(
                {
                    "label": label,
                    "slot_index": index + 1,
                    "step": step + 1,
                    "action": int(actions[step, index]),
                    "action_had_effect": bool(effects[step, index]),
                    "h_before": float(h_before[index]),
                    "h_after": float(h_after[index]),
                    "h_progress": float(h_progress[index]),
                    "q_before": float(q_before[index]),
                    "q_after": float(q_after[index]),
                    "carry_before": float(carry_before[index]),
                    "carry_after": float(carry_after[index]),
                    "load_before": float(load_before[index]),
                    "load_after": float(load_after[index]),
                    "fresh_lift_volume": float(fresh[index]),
                    "rehandled_lift_volume": float(rehandled[index]),
                    "deposited_volume": float(deposited[index]),
                    "legacy_relocation_reward": float(relocation_reward),
                    "absolute_completion": float(completion["absolute"][step, index]),
                    "illegal_dump_volume": float(
                        completion["illegal_dump_volume"][step, index]
                    ),
                }
            )
        active &= ~np.asarray(timestep.done, dtype=bool)

    final_snapshot = material_snapshot(timestep.state)
    final_h = final_snapshot[0]
    telescope_error = dump_progress_all - (initial_h.astype(np.float64) - final_h)
    max_telescope_error = float(np.max(np.abs(telescope_error)))
    failed_lifts = [row for row in lift_diagnostics if not row["passed"]]
    top_by_absolute_error = sorted(
        lift_diagnostics,
        key=lambda row: (-row["absolute_residual"], row["slot_index"], row["step"]),
    )[:LIFT_DIAGNOSTICS_TOP_K]
    top_by_ulp_error = sorted(
        lift_diagnostics,
        key=lambda row: (-row["ulp_residual"], row["slot_index"], row["step"]),
    )[:LIFT_DIAGNOSTICS_TOP_K]
    top_failed_events = sorted(
        failed_lifts,
        key=lambda row: (
            -(row["absolute_residual"] / row["tolerance"]),
            row["slot_index"],
            row["step"],
        ),
    )[:LIFT_DIAGNOSTICS_TOP_K]
    diagnostics_path = output / "d4a_lift_diagnostics.json"
    write_json(
        diagnostics_path,
        {
            "schema": LIFT_DIAGNOSTICS_SCHEMA,
            "status": (
                "passed"
                if not failed_lifts
                and max_inert_error <= 1e-6
                and max_telescope_error <= 1e-4
                else "failed"
            ),
            "gate": {
                "rule": "max(absolute_floor, ulp_multiplier * max_float32_spacing)",
                "absolute_floor": LIFT_ABSOLUTE_FLOOR,
                "ulp_multiplier": LIFT_ULP_MULTIPLIER,
                "inert_absolute": 1e-6,
                "telescope_absolute": 1e-4,
            },
            "lift_event_count": len(lift_diagnostics),
            "failed_lift_event_count": len(failed_lifts),
            "max_lift_conservation_error": max_lift_error,
            "max_lift_error_location": max_lift_location,
            "max_inert_transition_error": max_inert_error,
            "max_dump_progress_telescope_error": max_telescope_error,
            "top_by_absolute_error": top_by_absolute_error,
            "top_by_ulp_error": top_by_ulp_error,
            "top_failed_events": top_failed_events,
            "written_before_gate_raise": True,
        },
    )
    if failed_lifts or max_inert_error > 1e-6 or max_telescope_error > 1e-4:
        raise ValueError(
            "material ledger parity failed: "
            f"failed_lifts={len(failed_lifts)}, max_lift={max_lift_error}, "
            f"inert={max_inert_error}, telescope={max_telescope_error}, "
            f"diagnostics={diagnostics_path}"
        )

    trace_summaries = []
    for index, label in target_indices.items():
        frozen_row = frozen_rows[index]
        acc = trace_accumulator[index]
        last_h = acc.pop("last_100_h")
        acc.pop("current_h_stagnation_steps")
        trace_summaries.append(
            {
                "label": label,
                "slot_index": index + 1,
                "scenario_id": frozen_row["scenario_id"],
                "map_id": frozen_row["map_id"],
                "source_id": frozen_row["source_id"],
                "condition_id": frozen_row["primary_cell"],
                "success": bool(frozen_row["success"]),
                "steps": int(frozen_row["steps"]),
                "terminal_absolute": float(frozen_row["terminal_absolute"]),
                "terminal_dig": float(frozen_row["terminal_dig"]),
                "terminal_dump_purity": float(frozen_row["terminal_dump_purity"]),
                "terminal_illegal_dump_volume": float(
                    frozen_row["terminal_illegal_dump_volume"]
                ),
                "terminal_unloaded": bool(frozen_row["terminal_unloaded"]),
                "h_reset": float(initial_h[index]),
                "h_terminal": float(final_h[index]),
                "h_terminal_over_reset": float(final_h[index] / initial_h[index]),
                "dump_progress_telescope_error": float(telescope_error[index]),
                "last_100_h_min": min(last_h),
                "last_100_h_max": max(last_h),
                "terminal_state_sha256": selected_terminal_state_sha256(
                    timestep.state, index
                ),
                **acc,
            }
        )
    trace_summaries.sort(key=lambda row: row["slot_index"])
    trace_rows.sort(key=lambda row: (row["slot_index"], row["step"]))
    trace_rows_sha = write_jsonl(output / "d4a_trace_rows.jsonl", trace_rows)
    write_json(output / "d4a_trace_summaries.json", trace_summaries)

    source_files = {
        "eval_fixed_bank.py": args.baselines_source / "eval_fixed_bank.py",
        "eval_mcts.py": args.baselines_source / "eval_mcts.py",
        "train_mixed.py": args.baselines_source / "train_mixed.py",
        "utils/models.py": args.baselines_source / "utils/models.py",
        "scripts/analysis/d4a_ledger.py": Path(__file__)
        .resolve()
        .with_name("d4a_ledger.py"),
        "terra/env.py": args.terra_source / "terra" / "env.py",
        "terra/state.py": args.terra_source / "terra" / "state.py",
    }
    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "command_contract": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "bank_root": str(bank_root),
            "bank_dataset_sha256": EXPECTED_BANK_DATASET_SHA256,
            "source_registry_sha256": EXPECTED_SOURCE_REGISTRY_SHA256,
            "environment_protocol_sha256": EXPECTED_ENVIRONMENT_PROTOCOL_SHA256,
            "frozen_development_eval": str(frozen_eval_path),
            "frozen_development_eval_sha256": EXPECTED_EVAL_SHA256,
            "baselines_revision": args.baselines_revision,
            "terra_revision": args.terra_revision,
            "bank_declared_terra_revision": args.bank_terra_revision,
            "seed": args.seed,
            "horizon": 450,
            "deterministic": True,
            "wandb": "disabled",
        },
        "source_file_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "execution": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "reset_verification": reset_verification,
        "frozen_parity": parity,
        "action_tensor": {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "raw_c_order_sha256": action_sha,
        },
        "ledger": {
            "max_lift_conservation_error": max_lift_error,
            "max_lift_error_location": max_lift_location,
            "max_inert_transition_error": max_inert_error,
            "max_dump_progress_telescope_error": max_telescope_error,
            "lift_gate_passed": not failed_lifts,
            "failed_lift_event_count": len(failed_lifts),
            "lift_event_count": len(lift_diagnostics),
            "tolerance": {
                "lift": {
                    "rule": "max(absolute_floor, ulp_multiplier * max_float32_spacing)",
                    "absolute_floor": LIFT_ABSOLUTE_FLOOR,
                    "ulp_multiplier": LIFT_ULP_MULTIPLIER,
                },
                "inert": 1e-6,
                "telescope": 1e-4,
            },
        },
        "lift_diagnostics": diagnostics_path.name,
        "lift_diagnostics_sha256": sha256_file(diagnostics_path),
        "targeted_trace_count": len(trace_summaries),
        "targeted_slots": sorted(TARGET_TRACES),
        "all_targeted_traces_match_frozen_rows": True,
        "analysis_support_sha256": sha256_file(
            Path(__file__).resolve().with_name("d4a_ledger.py")
        ),
        "trace_rows": "d4a_trace_rows.jsonl",
        "trace_rows_sha256": trace_rows_sha,
        "trace_summaries": "d4a_trace_summaries.json",
        "trace_summaries_sha256": sha256_file(output / "d4a_trace_summaries.json"),
        "runtime_seconds": time.time() - started,
    }
    write_json(output / "d4a_receipt.json", receipt)
    manifest = {
        "schema": SCHEMA,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    write_json(output / "receipt_manifest.json", manifest)
    print(output)


if __name__ == "__main__":
    main()
