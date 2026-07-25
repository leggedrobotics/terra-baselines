#!/usr/bin/env python3
"""Read-only D1/D2 audit for the frozen 2026-07-24 map-curriculum screen.

This script intentionally targets one historical experiment.  It runs the
recorded policy against the recorded environment without changing actions,
transitions, reward, termination, or reset behavior.  Extra values are pure
observers over the historical state and proposed action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tensorflow_probability.substrates import jax as tfp  # noqa: E402

from eval_fixed_bank import (  # noqa: E402
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    sha256_file,
)
from train import TrainConfig  # noqa: E402
from train_mixed import (  # noqa: E402
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.helpers import load_pkl_object  # noqa: E402
from utils.utils_ppo import obs_to_model_input, wrap_action  # noqa: E402

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

EXPECTED_TERRA_REVISION = "d37e780480c0fae64a4b9e4ba6638b4499748761"
EXPECTED_BASELINES_REVISION = "2722d832c8381a68d594d8bf8298ba3aec7f4c6a"
HISTORICAL_COMPLETION_CONTRACT = "legacy_implicit_buffer_v0"
AUDIT_SCHEMA = "terra_historical_curriculum_audit_v1"
D1_MAX_TRANSITIONS = 259_200
DO_ACTION = 6
TERMINAL_REWARD_RECONSTRUCTION_ATOL = 1e-5


def terminal_reward_reconstruction_is_valid(error: float) -> bool:
    """Accept only float32-scale observer reordering error."""
    return abs(float(error)) <= TERMINAL_REWARD_RECONSTRUCTION_ATOL


def parse_labelled_values(values: list[str], *, option: str) -> dict[str, str]:
    """Parse repeated LABEL=VALUE arguments and reject ambiguous labels."""
    parsed: dict[str, str] = {}
    for value in values:
        label, separator, payload = value.partition("=")
        if not separator or not label or not payload:
            raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
        if label in parsed:
            raise ValueError(f"{option} repeats label {label!r}")
        parsed[label] = payload
    return parsed


def read_frozen_revisions(source_root: Path) -> dict[str, str]:
    revisions = {
        "terra": (source_root / "terra" / "REVISION").read_text().strip(),
        "terra_baselines": (source_root / "terra-baselines" / "REVISION")
        .read_text()
        .strip(),
    }
    expected = {
        "terra": EXPECTED_TERRA_REVISION,
        "terra_baselines": EXPECTED_BASELINES_REVISION,
    }
    if revisions != expected:
        raise RuntimeError(
            f"historical source mismatch: observed={revisions}, expected={expected}"
        )
    return revisions


def verify_unique_training_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove that a training view contains each identity exactly once."""
    source_ids = [str(row["source_id"]) for row in rows]
    map_ids = [str(row["map_id"]) for row in rows]
    duplicate_sources = sorted(
        source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
    )
    duplicate_maps = sorted(
        map_id for map_id in set(map_ids) if map_ids.count(map_id) > 1
    )
    if duplicate_sources or duplicate_maps:
        raise RuntimeError(
            "training manifest is not identity-unique: "
            f"duplicate_sources={duplicate_sources[:8]}, "
            f"duplicate_maps={duplicate_maps[:8]}"
        )
    return {
        "passed": True,
        "slots": len(rows),
        "unique_source_ids": len(set(source_ids)),
        "unique_map_ids": len(set(map_ids)),
    }


def _layer_digest(directory: Path, subdirectory: str, filenames: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in filenames:
        digest.update((directory / subdirectory / filename).read_bytes())
    return digest.hexdigest()


def verify_exact_historical_reset(
    env,
    env_params,
    reset_keys: jax.Array,
    directory: Path,
    count: int,
) -> dict[str, Any]:
    """Verify every reset layer used by the historical environment."""
    timestep = env.reset(env_params, reset_keys)
    state = timestep.state
    observed_fields = {
        "target": np.asarray(state.world.target_map.map),
        "initial_action": np.asarray(state.world.action_map.map),
        "occupancy": np.asarray(state.world.padding_mask.map),
        "dumpability": np.asarray(state.world.dumpability_mask_init.map),
        "distance": np.asarray(state.world.relocation_distance_map),
    }
    source_directories = {
        "target": "images",
        "initial_action": "actions",
        "occupancy": "occupancy",
        "dumpability": "dumpability",
        "distance": "distance",
    }
    if observed_fields["target"].shape[0] != count:
        raise RuntimeError(
            f"reset produced {observed_fields['target'].shape[0]} maps, "
            f"expected {count}"
        )

    target_obstacle_overlaps = 0
    for index in range(count):
        for field, subdirectory in source_directories.items():
            expected = np.load(directory / subdirectory / f"img_{index + 1}.npy")
            observed = np.squeeze(observed_fields[field][index])
            equal = (
                np.allclose(observed, expected, rtol=0.0, atol=1e-7)
                if field == "distance"
                else np.array_equal(observed, expected)
            )
            if not equal:
                raise RuntimeError(f"exact reset {field} mismatch at slot {index + 1}")
        target_obstacle_overlaps += int(
            np.logical_and(
                np.squeeze(observed_fields["target"][index]) > 0,
                np.squeeze(observed_fields["occupancy"][index]) == 1,
            ).sum()
        )

        expected_metadata = {
            "trench_axes": np.asarray(env.maps_buffer.trench_axes[0, index]),
            "trench_type": np.asarray(env.maps_buffer.trench_types[0, index]),
            "foundation_border_axes": np.asarray(
                env.maps_buffer.foundation_border_axes[0, index]
            ),
            "foundation_border_type": np.asarray(
                env.maps_buffer.foundation_border_types[0, index]
            ),
        }
        observed_metadata = {
            "trench_axes": np.asarray(state.world.trench_axes[index]),
            "trench_type": np.asarray(state.world.trench_type[index]),
            "foundation_border_axes": np.asarray(
                state.world.foundation_border_axes[index]
            ),
            "foundation_border_type": np.asarray(
                state.world.foundation_border_type[index]
            ),
        }
        for field, expected in expected_metadata.items():
            if not np.array_equal(observed_metadata[field], expected):
                raise RuntimeError(f"exact reset {field} mismatch at slot {index + 1}")

    env_steps = np.asarray(state.env_steps)
    if env_steps.shape != (count,) or np.any(env_steps != 0):
        raise RuntimeError("historical fixed evaluation must reset at env_steps == 0")
    if target_obstacle_overlaps:
        raise RuntimeError(
            "historical audit assumes visible dump targets exclude obstacles; "
            f"found {target_obstacle_overlaps} overlapping cells"
        )

    layer_hashes = {
        field: _layer_digest(
            directory,
            subdirectory,
            [f"img_{index}.npy" for index in range(1, count + 1)],
        )
        for field, subdirectory in source_directories.items()
    }
    layer_hashes["metadata"] = _layer_digest(
        directory,
        "metadata",
        [f"trench_{index}.json" for index in range(1, count + 1)],
    )
    return {
        "passed": True,
        "slots": count,
        "env_steps_min": int(env_steps.min()),
        "env_steps_max": int(env_steps.max()),
        "target_obstacle_overlap_cells": target_obstacle_overlaps,
        "layer_sha256": layer_hashes,
    }


def _batched_dilate(mask: jax.Array) -> jax.Array:
    return (
        jax.lax.reduce_window(
            mask.astype(jnp.float32),
            jnp.float32(0.0),
            jax.lax.add,
            window_dimensions=(1, 3, 3),
            window_strides=(1, 1, 1),
            padding="SAME",
        )
        > 0
    )


def completion_observer(timestep) -> dict[str, jax.Array]:
    """Compute exact-visible and legacy-buffer completion from one batch state."""
    state = timestep.state
    target = state.world.target_map.map
    action = state.world.action_map.map
    obstacle = state.world.padding_mask.map == 1
    exact_mask = jnp.logical_and(target > 0, ~obstacle)
    accepted_mask = jnp.logical_and(_batched_dilate(target > 0), ~obstacle)
    buffer_only_mask = jnp.logical_and(accepted_mask, ~exact_mask)
    positive = jnp.clip(action.astype(jnp.float32), a_min=0.0)
    exact_volume = jnp.where(exact_mask, positive, 0.0).sum(axis=(-2, -1))
    accepted_volume = jnp.where(accepted_mask, positive, 0.0).sum(axis=(-2, -1))
    buffer_only_volume = jnp.where(buffer_only_mask, positive, 0.0).sum(axis=(-2, -1))
    positive_volume = positive.sum(axis=(-2, -1))
    illegal_volume = positive_volume - accepted_volume
    undug_volume = jnp.logical_and(target < 0, action >= 0).sum(axis=(-2, -1))
    loaded_volume = sum(
        agent_state.loaded.astype(jnp.float32).sum(axis=-1)
        for agent_state in state.agent.agent_states
    )
    total_volume = positive_volume + undug_volume + loaded_volume
    exact_completion = jnp.where(
        total_volume > 0,
        exact_volume / jnp.maximum(total_volume, 1.0),
        0.0,
    )
    accepted_completion = jnp.where(
        total_volume > 0,
        accepted_volume / jnp.maximum(total_volume, 1.0),
        0.0,
    )
    return {
        "exact_completion": exact_completion,
        "accepted_buffer_completion": accepted_completion,
        "completion_delta": accepted_completion - exact_completion,
        "exact_dump_volume": exact_volume,
        "accepted_dump_volume": accepted_volume,
        "buffer_only_positive_volume": buffer_only_volume,
        "illegal_positive_volume": illegal_volume,
        "total_task_soil_volume": total_volume,
    }


def _historical_terminal_reward(
    state,
    completion: jax.Array,
    task_done: jax.Array,
    done: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reproduce the historical terminal reward before and after normalization."""
    base_reward = state.env_cfg.rewards.terminal.astype(jnp.float32)
    minimum = jnp.asarray(
        getattr(state.env_cfg, "terminal_completion_min_threshold", 0.6),
        dtype=jnp.float32,
    )
    scaled = (completion - minimum) / jnp.maximum(1.0 - minimum, 1e-6)
    success_reward = jnp.where(
        completion < minimum,
        0.0,
        base_reward * scaled**2
        + jnp.where(completion >= 0.999, base_reward * 0.2, 0.0),
    )
    timeout_reward = base_reward * 0.1 * jnp.clip(completion, 0.0, 1.0) ** 2
    raw = jnp.where(task_done, success_reward, jnp.where(done, timeout_reward, 0.0))
    active_count = state.agent.agent_active.astype(jnp.float32).sum(axis=-1)
    normalized = (
        raw
        * 2.0
        / jnp.maximum(active_count, 1.0)
        / state.env_cfg.rewards.normalizer.astype(jnp.float32)
    )
    full_terminal_normalized = (
        base_reward
        * 2.0
        / jnp.maximum(active_count, 1.0)
        / state.env_cfg.rewards.normalizer.astype(jnp.float32)
    )
    return normalized, full_terminal_normalized


def counterfactual_terminal_observer(
    timestep,
    completion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Replace only exact dump volume with legacy accepted-buffer volume."""
    state = timestep.state
    target = state.world.target_map.map
    required_dig = (
        jnp.where(target < 0, -target, 0).astype(jnp.float32).sum(axis=(-2, -1))
    )
    has_dig = jnp.any(target < 0, axis=(-2, -1))
    has_dump = jnp.any(target > 0, axis=(-2, -1))
    components = timestep.info["reward_components"]
    dig_completion = components["dig_completion_total"].astype(jnp.float32)
    exact_dig_dump = jnp.clip(
        completion["exact_dump_volume"] / jnp.maximum(required_dig, 1.0),
        0.0,
        1.0,
    )
    accepted_dig_dump = jnp.clip(
        completion["accepted_dump_volume"] / jnp.maximum(required_dig, 1.0),
        0.0,
        1.0,
    )
    combined = jnp.logical_and(has_dig, has_dump)
    relocation_only = jnp.logical_and(~has_dig, has_dump)
    current_gated = jnp.where(
        combined,
        0.6 * exact_dig_dump + 0.4 * dig_completion,
        jnp.where(
            relocation_only,
            completion["exact_completion"],
            dig_completion,
        ),
    )
    counterfactual_gated = jnp.where(
        combined,
        0.6 * accepted_dig_dump + 0.4 * dig_completion,
        jnp.where(
            relocation_only,
            completion["accepted_buffer_completion"],
            dig_completion,
        ),
    )
    task_done = timestep.info["task_done"]
    counterfactual_reward, full_terminal = _historical_terminal_reward(
        state,
        counterfactual_gated,
        task_done,
        timestep.done,
    )
    reconstructed_current_reward, _ = _historical_terminal_reward(
        state,
        current_gated,
        task_done,
        timestep.done,
    )
    current_reward = components["terminal"].astype(jnp.float32)
    return {
        "current_gated_completion": current_gated,
        "counterfactual_gated_completion": counterfactual_gated,
        "current_terminal_reward": current_reward,
        "reconstructed_current_terminal_reward": reconstructed_current_reward,
        "counterfactual_terminal_reward": counterfactual_reward,
        "terminal_reward_delta": counterfactual_reward - current_reward,
        "full_terminal_reward_scale": full_terminal,
    }


def _collapse_with_boundary_flow(
    state,
    action_map: jax.Array,
    affected_mask: jax.Array,
    exact_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reproduce the old three-pass relaxation and count exact-mask crossings."""
    mask = state._expand_mask_for_soil_mechanics(affected_mask.astype(jnp.bool_))
    result = action_map
    outward = jnp.int32(0)
    inward = jnp.int32(0)
    for _ in range(3):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = jnp.roll(result, shift=(dy, dx), axis=(0, 1))
            diff = shifted - result
            neighbor_mask = jnp.roll(mask, shift=(dy, dx), axis=(0, 1))
            move = (diff >= 2) & mask & neighbor_mask
            source_exact = jnp.roll(exact_mask, shift=(dy, dx), axis=(0, 1))
            outward += jnp.sum((move & source_exact & ~exact_mask).astype(jnp.int32))
            inward += jnp.sum((move & ~source_exact & exact_mask).astype(jnp.int32))
            result = result + move.astype(result.dtype)
            result = jnp.roll(result, shift=(dy, dx), axis=(0, 1))
            result = result - move.astype(result.dtype)
            result = jnp.roll(result, shift=(-dy, -dx), axis=(0, 1))
    return result.astype(action_map.dtype), outward, inward


def historical_dump_observer(state, action: jax.Array) -> dict[str, jax.Array]:
    """Observe the old excavator dump proposal before its potential veto."""
    current = state._get_current_agent_state()
    loaded = current.loaded[0].astype(jnp.int32)
    dump_mask = state._build_dig_dump_cone()
    dump_mask = state._exclude_dig_tiles_from_dump_mask(dump_mask)
    dump_mask = state._exclude_dumpability_mask_tiles_from_dump_mask(dump_mask)
    dump_mask = state._exclude_traversability_mask_tiles_from_dump_mask(dump_mask)
    dump_mask = state._exclude_just_moved_tiles_from_dump_mask(dump_mask)
    lacks_free_space = state._dump_cone_lacks_free_space(dump_mask)
    dump_mask = jnp.where(lacks_free_space, jnp.zeros_like(dump_mask), dump_mask)
    dump_volume = dump_mask.astype(jnp.int32).sum()
    safe_dump_volume = jnp.maximum(dump_volume, 1)
    remaining = loaded % safe_dump_volume
    even = (loaded - remaining) // safe_dump_volume

    map_2d = state.world.action_map.map
    target_2d = state.world.target_map.map
    height, width = map_2d.shape
    dump_mask_2d = dump_mask.reshape((height, width)).astype(jnp.bool_)
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    centroid_y = jnp.sum(yy * dump_mask_2d) / jnp.maximum(jnp.sum(dump_mask_2d), 1)
    centroid_x = jnp.sum(xx * dump_mask_2d) / jnp.maximum(jnp.sum(dump_mask_2d), 1)
    distance = jnp.sqrt((yy - centroid_y) ** 2 + (xx - centroid_x) ** 2)
    concentrated = dump_mask_2d & (distance <= 2.0)

    distances_in_mask = jnp.where(dump_mask_2d, distance, jnp.inf)
    closest_flat = jnp.argmin(distances_in_mask)
    closest_y, closest_x = jnp.unravel_index(closest_flat, (height, width))
    closest_distance = jnp.sqrt((yy - closest_y) ** 2 + (xx - closest_x) ** 2)
    fallback = dump_mask_2d & (closest_distance <= 2.0)
    concentrated = jnp.where(jnp.any(concentrated), concentrated, fallback)
    concentrated_count = jnp.maximum(jnp.sum(concentrated), 1)
    even_concentrated = loaded // concentrated_count
    remaining_concentrated = loaded % concentrated_count
    volume_per_tile = (even_concentrated * concentrated).astype(map_2d.dtype)
    concentrated_flat = concentrated.reshape(-1)
    bonus_indices = jnp.where(
        concentrated_flat,
        jnp.cumsum(concentrated_flat.astype(jnp.int32)),
        concentrated_flat.size + 1,
    )
    bonus = (
        (bonus_indices <= remaining_concentrated)
        .reshape((height, width))
        .astype(map_2d.dtype)
    )
    pre_collapse = map_2d + volume_per_tile + bonus
    exact_mask = target_2d > 0
    collapsed, outward, inward = _collapse_with_boundary_flow(
        state,
        pre_collapse,
        concentrated,
        exact_mask,
    )
    historical_candidate = state._apply_dump_mask(
        map_2d.reshape(-1),
        dump_mask,
        even,
        remaining,
        target_2d,
        use_condensed_dump=True,
    ).reshape((height, width))

    current_potential = state._compute_relocation_potential(map_2d)
    predicted_potential = state._compute_relocation_potential(historical_candidate)
    baseline_effective = current.carry_baseline_potential + (
        current_potential - current.carry_potential_after_lift
    )
    attempt = (
        (jnp.asarray(action).reshape(()) == DO_ACTION)
        & (loaded > 0)
        & ~state._workspace_intersects_obstacle()
        & (dump_volume > 0)
    )
    candidate_changes_map = jnp.any(historical_candidate != map_2d)
    would_increase = predicted_potential > baseline_effective
    return {
        "attempt": attempt,
        "potential_veto_condition": attempt & candidate_changes_map & would_increase,
        "candidate_map": historical_candidate,
        "candidate_internal_mismatch": attempt
        & jnp.any(collapsed != historical_candidate),
        "outward_boundary_volume": jnp.where(attempt, outward, 0),
        "inward_boundary_volume": jnp.where(attempt, inward, 0),
    }


def _preserve_inactive(previous, candidate, active: jax.Array):
    if not hasattr(candidate, "shape"):
        return candidate
    if candidate.ndim == 0 or candidate.shape[0] != active.shape[0]:
        return candidate
    mask = active.reshape((active.shape[0],) + (1,) * (candidate.ndim - 1))
    return jnp.where(mask, candidate, previous)


def _finite_state_per_environment(state, count: int) -> jax.Array:
    finite_per_leaf = []
    for leaf in jax.tree_util.tree_leaves(state):
        if not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != count:
            continue
        finite = jnp.isfinite(leaf)
        if leaf.ndim > 1:
            finite = jnp.all(finite, axis=tuple(range(1, leaf.ndim)))
        finite_per_leaf.append(finite)
    if not finite_per_leaf:
        return jnp.ones((count,), dtype=jnp.bool_)
    return jnp.all(jnp.stack(finite_per_leaf), axis=0)


def rollout_historical_audit(
    env,
    model,
    model_params,
    env_params,
    config,
    *,
    horizon: int,
    mode: str,
    seed: int,
    reset_keys: jax.Array,
    observe_dump_semantics: bool,
) -> dict[str, np.ndarray]:
    """Run one frozen policy/bank pair and return per-episode audit arrays."""
    count = int(config.num_test_rollouts)
    timestep = env.reset(env_params, reset_keys)
    prev_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    active = jnp.ones((count,), dtype=jnp.bool_)
    succeeded = jnp.zeros((count,), dtype=jnp.bool_)
    terminated = jnp.zeros((count,), dtype=jnp.bool_)
    lengths = jnp.zeros((count,), dtype=jnp.int32)
    initial_target = timestep.state.world.target_map.map
    initial_obstacle = timestep.state.world.padding_mask.map
    initial_mass = timestep.state.world.action_map.map.astype(jnp.int32).sum(
        axis=(-2, -1)
    ) + sum(
        agent_state.loaded.astype(jnp.int32).sum(axis=-1)
        for agent_state in timestep.state.agent.agent_states
    )
    maximum_mass_residual = jnp.zeros((count,), dtype=jnp.int32)
    target_mutation = jnp.zeros((count,), dtype=jnp.bool_)
    obstacle_mutation = jnp.zeros((count,), dtype=jnp.bool_)
    nonfinite_state = jnp.zeros((count,), dtype=jnp.bool_)
    entropy_sum = jnp.zeros((count,), dtype=jnp.float32)
    logit_margin_sum = jnp.zeros((count,), dtype=jnp.float32)
    sampled_disagreement_count = jnp.zeros((count,), dtype=jnp.int32)
    decision_count = jnp.zeros((count,), dtype=jnp.int32)
    potential_veto_attempts = jnp.zeros((count,), dtype=jnp.int32)
    dump_attempts = jnp.zeros((count,), dtype=jnp.int32)
    executed_dump_attempts = jnp.zeros((count,), dtype=jnp.int32)
    outward_boundary_volume = jnp.zeros((count,), dtype=jnp.int32)
    inward_boundary_volume = jnp.zeros((count,), dtype=jnp.int32)
    observer_internal_mismatch = jnp.zeros((count,), dtype=jnp.bool_)
    observer_transition_mismatch = jnp.zeros((count,), dtype=jnp.bool_)
    terminal_values: dict[str, jax.Array] = {}
    rng = jax.random.PRNGKey(seed)
    dump_observer = (
        jax.jit(jax.vmap(historical_dump_observer)) if observe_dump_semantics else None
    )

    for _ in range(horizon):
        rng, action_key, step_key = jax.random.split(rng, 3)
        model_input = obs_to_model_input(timestep.observation, prev_actions, config)
        _, logits = model.apply(model_params, model_input)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probabilities * log_probs, axis=-1)
        top_two = jax.lax.top_k(logits, 2)[0]
        margin = top_two[:, 0] - top_two[:, 1]
        greedy_action = jnp.argmax(logits, axis=-1)
        if mode == "deterministic":
            action = greedy_action
        else:
            action = tfp.distributions.Categorical(logits=logits).sample(
                seed=action_key
            )

        entropy_sum += jnp.where(active, entropy, 0.0)
        logit_margin_sum += jnp.where(active, margin, 0.0)
        sampled_disagreement_count += (active & (action != greedy_action)).astype(
            jnp.int32
        )
        decision_count += active.astype(jnp.int32)
        dump_diagnostics = (
            dump_observer(timestep.state, action) if dump_observer is not None else None
        )

        previous_timestep = timestep
        previous_map = previous_timestep.state.world.action_map.map
        previous_loaded = sum(
            agent_state.loaded.astype(jnp.int32).sum(axis=-1)
            for agent_state in previous_timestep.state.agent.agent_states
        )
        step_keys = jax.random.split(step_key, count)
        candidate_timestep = env.step_no_reset(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            step_keys,
        )
        timestep = jax.tree_util.tree_map(
            lambda previous, candidate: _preserve_inactive(previous, candidate, active),
            timestep,
            candidate_timestep,
        )
        current_map = timestep.state.world.action_map.map
        current_loaded = sum(
            agent_state.loaded.astype(jnp.int32).sum(axis=-1)
            for agent_state in timestep.state.agent.agent_states
        )
        if dump_diagnostics is not None:
            terrain_changed = jnp.any(current_map != previous_map, axis=(-2, -1))
            load_changed = current_loaded != previous_loaded
            candidate_match = jnp.all(
                current_map == dump_diagnostics["candidate_map"],
                axis=(-2, -1),
            )
            veto_condition = dump_diagnostics["potential_veto_condition"]
            confirmed_veto = active & veto_condition & ~terrain_changed & ~load_changed
            executed_dump = (
                active
                & dump_diagnostics["attempt"]
                & ~veto_condition
                & candidate_match
                & load_changed
            )
            potential_veto_attempts += confirmed_veto.astype(jnp.int32)
            dump_attempts += (active & dump_diagnostics["attempt"]).astype(jnp.int32)
            executed_dump_attempts += executed_dump.astype(jnp.int32)
            outward_boundary_volume += jnp.where(
                executed_dump,
                dump_diagnostics["outward_boundary_volume"],
                0,
            )
            inward_boundary_volume += jnp.where(
                executed_dump,
                dump_diagnostics["inward_boundary_volume"],
                0,
            )
            observer_internal_mismatch |= (
                active & dump_diagnostics["candidate_internal_mismatch"]
            )
            observer_transition_mismatch |= active & (
                (
                    dump_diagnostics["attempt"]
                    & ~veto_condition
                    & (~candidate_match | ~load_changed)
                )
                | (veto_condition & (terrain_changed | load_changed))
            )

        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)
        prev_actions = jnp.where(
            timestep.done[:, None], jnp.zeros_like(prev_actions), prev_actions
        )
        lengths += active.astype(jnp.int32)

        world_mass = timestep.state.world.action_map.map.astype(jnp.int32).sum(
            axis=(-2, -1)
        )
        loaded_mass = sum(
            agent_state.loaded.astype(jnp.int32).sum(axis=-1)
            for agent_state in timestep.state.agent.agent_states
        )
        mass_residual = jnp.abs(world_mass + loaded_mass - initial_mass)
        maximum_mass_residual = jnp.maximum(
            maximum_mass_residual, jnp.where(active, mass_residual, 0)
        )
        target_mutation |= active & jnp.any(
            timestep.state.world.target_map.map != initial_target,
            axis=(-2, -1),
        )
        obstacle_mutation |= active & jnp.any(
            timestep.state.world.padding_mask.map != initial_obstacle,
            axis=(-2, -1),
        )
        nonfinite_state |= active & ~_finite_state_per_environment(
            timestep.state, count
        )

        completion = completion_observer(timestep)
        counterfactual = counterfactual_terminal_observer(timestep, completion)
        step_done = timestep.done
        newly_done = active & step_done
        for key, value in {**completion, **counterfactual}.items():
            if key not in terminal_values:
                terminal_values[key] = jnp.zeros_like(value)
            terminal_values[key] = jnp.where(newly_done, value, terminal_values[key])
        succeeded |= active & timestep.info["task_done"]
        terminated |= newly_done
        active &= ~step_done
        if bool(jnp.all(~active).item()):
            break

    if bool(jnp.any(active).item()):
        raise RuntimeError(
            f"{int(active.sum())} audit episodes did not terminate by horizon {horizon}"
        )

    reconstructed_error = jnp.abs(
        terminal_values["current_terminal_reward"]
        - terminal_values["reconstructed_current_terminal_reward"]
    )
    output = {
        "success": succeeded,
        "terminated": terminated,
        "steps": lengths,
        "policy_entropy": entropy_sum / jnp.maximum(decision_count, 1),
        "action_logit_margin": logit_margin_sum / jnp.maximum(decision_count, 1),
        "sampled_argmax_disagreement_rate": sampled_disagreement_count
        / jnp.maximum(decision_count, 1),
        "policy_decisions": decision_count,
        "potential_veto_attempts": potential_veto_attempts,
        "dump_attempts": dump_attempts,
        "executed_dump_attempts": executed_dump_attempts,
        "outward_boundary_relaxation_volume": outward_boundary_volume,
        "inward_boundary_relaxation_volume": inward_boundary_volume,
        "maximum_mass_residual": maximum_mass_residual,
        "target_mutation": target_mutation,
        "obstacle_mutation": obstacle_mutation,
        "nonfinite_state": nonfinite_state,
        "observer_internal_mismatch": observer_internal_mismatch,
        "observer_transition_mismatch": observer_transition_mismatch,
        "terminal_reward_reconstruction_error": reconstructed_error,
        **terminal_values,
    }
    return {key: np.asarray(value) for key, value in output.items()}


def _termination_reason(
    *, success: bool, terminated: bool, steps: int, horizon: int
) -> str:
    timeout = terminated and steps >= horizon
    if success and timeout:
        return "task_done_and_timeout"
    if success:
        return "task_done"
    if timeout:
        return "timeout"
    if terminated:
        return "other_termination"
    return "horizon_censored"


def build_per_map(
    rows: list[dict[str, Any]],
    metrics: dict[str, np.ndarray],
    *,
    horizon: int,
) -> list[dict[str, Any]]:
    per_map = []
    for index, row in enumerate(rows):
        scalar_metrics = {
            key: np.asarray(value[index]).item() for key, value in metrics.items()
        }
        success = bool(scalar_metrics.pop("success"))
        terminated = bool(scalar_metrics.pop("terminated"))
        steps = int(scalar_metrics.pop("steps"))
        termination_reason = _termination_reason(
            success=success,
            terminated=terminated,
            steps=steps,
            horizon=horizon,
        )
        integrity_failure = bool(
            int(scalar_metrics["maximum_mass_residual"]) != 0
            or bool(scalar_metrics["target_mutation"])
            or bool(scalar_metrics["obstacle_mutation"])
            or bool(scalar_metrics["nonfinite_state"])
            or bool(scalar_metrics["observer_internal_mismatch"])
            or bool(scalar_metrics["observer_transition_mismatch"])
            or not terminal_reward_reconstruction_is_valid(
                scalar_metrics["terminal_reward_reconstruction_error"]
            )
        )
        per_map.append(
            {
                **row,
                "success": success,
                "terminated": terminated,
                "steps": steps,
                "termination_reason": termination_reason,
                **scalar_metrics,
                "task_done_with_inexact_visible_completion": bool(
                    success and float(scalar_metrics["exact_completion"]) < 1.0 - 1e-6
                ),
                "integrity_failure": integrity_failure,
            }
        )
    return per_map


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    timeouts = [row for row in rows if row["termination_reason"] == "timeout"]
    top_timeout_count = math.ceil(len(timeouts) * 0.25)
    top_timeouts = sorted(
        timeouts,
        key=lambda row: float(row["exact_completion"]),
        reverse=True,
    )[:top_timeout_count]

    def materially_changed(row: dict[str, Any]) -> bool:
        return bool(
            float(row["completion_delta"]) >= 0.05
            or abs(float(row["terminal_reward_delta"]))
            >= 0.1 * float(row["full_terminal_reward_scale"])
        )

    success_material = sum(int(materially_changed(row)) for row in successes)
    timeout_material = sum(int(materially_changed(row)) for row in top_timeouts)
    veto_attempts = sum(int(row["potential_veto_attempts"]) for row in rows)
    dump_attempts = sum(int(row["dump_attempts"]) for row in rows)
    executed_dumps = sum(int(row["executed_dump_attempts"]) for row in rows)
    success_count = len(successes)
    summary = {
        "episodes": len(rows),
        "successes": success_count,
        "success_rate": success_count / max(len(rows), 1),
        "termination_reasons": {
            reason: sum(int(row["termination_reason"] == reason) for row in rows)
            for reason in (
                "task_done",
                "timeout",
                "task_done_and_timeout",
                "other_termination",
                "horizon_censored",
            )
        },
        "task_done_with_inexact_visible_completion": sum(
            int(row["task_done_with_inexact_visible_completion"]) for row in rows
        ),
        "semantic_materiality": {
            "successes_changed": success_material,
            "successes_denominator": success_count,
            "success_fraction": success_material / max(success_count, 1),
            "top_quartile_timeouts_changed": timeout_material,
            "top_quartile_timeouts_denominator": len(top_timeouts),
            "top_quartile_timeout_fraction": timeout_material
            / max(len(top_timeouts), 1),
            "material_contributor": bool(
                (success_count > 0 and success_material / success_count >= 0.1)
                or (top_timeouts and timeout_material / len(top_timeouts) >= 0.1)
            ),
        },
        "means": {
            field: _mean(rows, field)
            for field in (
                "exact_completion",
                "accepted_buffer_completion",
                "completion_delta",
                "current_terminal_reward",
                "counterfactual_terminal_reward",
                "terminal_reward_delta",
                "buffer_only_positive_volume",
                "illegal_positive_volume",
                "policy_entropy",
                "action_logit_margin",
                "sampled_argmax_disagreement_rate",
            )
        },
        "potential_veto": {
            "attempts": veto_attempts,
            "dump_attempts": dump_attempts,
            "rate_per_dump_attempt": veto_attempts / max(dump_attempts, 1),
        },
        "boundary_relaxation": {
            "executed_dumps": executed_dumps,
            "outward_volume": sum(
                int(row["outward_boundary_relaxation_volume"]) for row in rows
            ),
            "inward_volume": sum(
                int(row["inward_boundary_relaxation_volume"]) for row in rows
            ),
        },
        "integrity": {
            "passed": not any(row["integrity_failure"] for row in rows),
            "failures": sum(int(row["integrity_failure"]) for row in rows),
            "maximum_mass_residual": max(
                (int(row["maximum_mass_residual"]) for row in rows),
                default=0,
            ),
        },
    }
    return summary


def grouped_summary(per_map: list[dict[str, Any]]) -> dict[str, Any]:
    def group(field: str) -> dict[str, Any]:
        return {
            value: summarize_rows([row for row in per_map if row[field] == value])
            for value in sorted({str(row[field]) for row in per_map})
        }

    return {
        "overall": summarize_rows(per_map),
        "by_family": group("family"),
        "by_primary_cell": group("primary_cell"),
    }


def checkpoint_specs(args) -> list[tuple[str, Path, dict[str, Any]]]:
    checkpoint_values = parse_labelled_values(args.checkpoint, option="--checkpoint")
    expected_hashes = parse_labelled_values(
        args.expected_checkpoint_sha256,
        option="--expected-checkpoint-sha256",
    )
    if set(checkpoint_values) != set(expected_hashes):
        raise ValueError(
            "checkpoint labels and expected hash labels must match exactly"
        )
    specs = []
    for label, value in checkpoint_values.items():
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hashes[label]:
            raise RuntimeError(
                f"{label}: checkpoint hash mismatch: "
                f"observed={observed_hash}, expected={expected_hashes[label]}"
            )
        specs.append((label, path, load_pkl_object(str(path))))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
    )
    parser.add_argument("--semantic-label", action="append", default=[])
    parser.add_argument("--semantic-dataset", action="append", default=[])
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--unique-training-dataset")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument(
        "--mode", choices=("deterministic", "sampled"), default="deterministic"
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.horizon != 450:
        raise ValueError("historical audit requires the recorded 450-step horizon")
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = args.seed or [2026072500]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"evaluation seeds must be unique, got {seeds}")
    if args.mode == "deterministic" and len(seeds) != 1:
        raise ValueError("deterministic audit requires exactly one seed")
    revisions = read_frozen_revisions(args.source_root.resolve())
    specs = checkpoint_specs(args)
    labels = {label for label, _, _ in specs}
    semantic_labels = set(args.semantic_label)
    semantic_datasets = set(args.semantic_dataset)
    if not semantic_labels <= labels:
        raise ValueError(f"unknown semantic labels: {sorted(semantic_labels - labels)}")
    if args.mode != "deterministic" and semantic_labels:
        raise ValueError("D1 semantic attribution is deterministic only")
    if bool(semantic_labels) != bool(semantic_datasets):
        raise ValueError(
            "--semantic-label and --semantic-dataset must be used together"
        )

    bank_root = args.bank_root.resolve()
    datasets: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for relative in args.dataset:
        directory = bank_root / relative
        rows = load_manifest(directory)
        datasets.append((relative, directory, rows))
    unknown_semantic_datasets = semantic_datasets - {
        relative for relative, _, _ in datasets
    }
    if unknown_semantic_datasets:
        raise ValueError(
            f"unknown semantic datasets: {sorted(unknown_semantic_datasets)}"
        )
    semantic_transitions = sum(
        len(rows) * args.horizon * len(semantic_labels)
        for relative, _, rows in datasets
        if relative in semantic_datasets
    )
    if semantic_transitions > D1_MAX_TRANSITIONS:
        raise RuntimeError(
            "D1 transition budget exceeded: "
            f"{semantic_transitions} > {D1_MAX_TRANSITIONS}"
        )

    unique_receipt = None
    if args.unique_training_dataset is not None:
        matching = [
            rows
            for relative, _, rows in datasets
            if relative == args.unique_training_dataset
        ]
        if len(matching) != 1:
            raise ValueError(
                "--unique-training-dataset must name exactly one --dataset"
            )
        unique_receipt = verify_unique_training_manifest(matching[0])

    reference_config = specs[0][2]["train_config"]
    for label, _, checkpoint in specs:
        if "model" not in checkpoint:
            raise KeyError(f"{label}: checkpoint has no model")
        _validate_checkpoint_architecture(checkpoint, reference_config)

    records = []
    reset_receipts = {}
    for relative, directory, rows in datasets:
        count = len(rows)
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(count)
        config = configure_for_bank(reference_config, relative, count)
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        if bool(
            jnp.any(
                env_params.enforce_foundation_border_alignment.astype(jnp.bool_)
            ).item()
        ):
            raise RuntimeError(
                "historical D1 reward reconstruction is registered for the "
                "recorded edge-alignment-disabled configuration only"
            )
        initial_timestep = env.reset(env_params, exact_reset_keys(count))
        active_agents = np.asarray(initial_timestep.state.agent.agent_active).sum(
            axis=-1
        )
        active_types = np.stack(
            [
                np.asarray(agent_state.agent_type)[:, 0]
                for agent_state in initial_timestep.state.agent.agent_states
            ],
            axis=-1,
        )
        if np.any(active_agents != 1) or np.any(active_types[:, 0] != 0):
            raise RuntimeError(
                "historical dump observer requires the recorded one-excavator "
                "evaluation configuration"
            )
        reset_keys = exact_reset_keys(count)
        reset_receipts[relative] = verify_exact_historical_reset(
            env, env_params, reset_keys, directory, count
        )
        model = SimpleNamespace(apply=initialized_state.apply_fn)

        for label, checkpoint_path, checkpoint in specs:
            for seed in seeds:
                d1_semantic_attribution = (
                    label in semantic_labels and relative in semantic_datasets
                )
                metrics = rollout_historical_audit(
                    env,
                    model,
                    checkpoint["model"],
                    env_params,
                    config,
                    horizon=args.horizon,
                    mode=args.mode,
                    seed=seed,
                    reset_keys=reset_keys,
                    observe_dump_semantics=d1_semantic_attribution,
                )
                per_map = build_per_map(rows, metrics, horizon=args.horizon)
                summary = grouped_summary(per_map)
                record = {
                    "checkpoint_label": label,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_update": int(checkpoint.get("next_update", 0)),
                    "dataset": relative,
                    "manifest": str(directory / "manifest.jsonl"),
                    "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                    "mode": args.mode,
                    "seed": seed,
                    "horizon": args.horizon,
                    "maximum_transitions": count * args.horizon,
                    "d1_semantic_attribution": d1_semantic_attribution,
                    "summary": summary,
                    "per_map": per_map,
                }
                records.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema": AUDIT_SCHEMA,
                    "completion_contract": HISTORICAL_COMPLETION_CONTRACT,
                    "observer_only": True,
                    "source_revisions": revisions,
                    "bank_root": str(bank_root),
                    "mode": args.mode,
                    "seeds": seeds,
                    "horizon": args.horizon,
                    "d1_semantic_labels": sorted(semantic_labels),
                    "d1_semantic_datasets": sorted(semantic_datasets),
                    "d1_maximum_transitions": semantic_transitions,
                    "d1_transition_budget": D1_MAX_TRANSITIONS,
                    "numerical_tolerances": {
                        "terminal_reward_reconstruction_atol": (
                            TERMINAL_REWARD_RECONSTRUCTION_ATOL
                        )
                    },
                    "unique_training_manifest": unique_receipt,
                    "reset_integrity": reset_receipts,
                    "records": records,
                }
                args.output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                overall = summary["overall"]
                print(
                    f"{label} {relative} {args.mode} seed={seed}: "
                    f"{overall['successes']}/{overall['episodes']} success",
                    flush=True,
                )


if __name__ == "__main__":
    main()
