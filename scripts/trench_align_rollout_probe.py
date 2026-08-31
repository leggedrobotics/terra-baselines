#!/usr/bin/env python3
"""Instrumented fresh-trench dig-alignment rollout probe.

Terra never logs the fresh-trench dig-admissibility mechanism, so the
pilot's mechanism endpoint has to be measured offline.  For one checkpoint
this rolls out a fixed subset of a frozen accepted-bank evaluation panel and
records, per step and per episode slot:

* the chosen action and whether it was ``DO``;
* the three exported alignment scalars **before** the step
  (``fresh_trench_dig_alignment_valid`` / ``..._yaw_error`` /
  ``..._standoff_error``), i.e. the values the policy actually saw;
* whether that prospective ``DO`` was fresh-trench *applicable* -- the gate
  exports ``valid=1`` both for a pose-valid dig and for an inapplicable
  action, so applicability is needed to read the export correctly;
* whether the step had an effect, and whether it mutated a fresh trench
  target cell (the research note's code-stop predicate).

With ``--differential-gate`` every step is additionally re-executed from the
same state with ``enforce_trench_dig_alignment`` flipped, and the successor
states are compared.  Divergences are classified: the contract requires them
only at applicable, invalid, empty-excavator ``DO`` steps.

Rollout mechanics (deterministic argmax, ``prev_actions`` history, frozen
manifest map/episode seeds, ``step_no_reset`` with inactive-slot
preservation) mirror ``eval_mcts.rollout_episode`` as used by
``eval_fixed_bank.py`` so the numbers are matched to the primary endpoint.

No run directory, bank, or Slurm job is written to; the checkpoint is opened
read-only.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_fixed_bank import (  # noqa: E402
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
)
from eval_mcts import _apply_in_batch_chunks  # noqa: E402
from train import TrainConfig  # noqa: E402
from train_mixed import (  # noqa: E402
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.accepted_bank import load_accepted_bank  # noqa: E402
from utils.helpers import load_pkl_object  # noqa: E402
from utils.models import validate_model_params_match  # noqa: E402
from utils.utils_ppo import obs_to_model_input, wrap_action  # noqa: E402

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

DO_ACTION = 6  # TrackedActionType.DO / WheeledActionType.DO
DO_NOTHING_ACTION = 7
SCHEMA = "terra_trench_align_rollout_probe_v1"


# ---------------------------------------------------------------------------
# alignment applicability probe
# ---------------------------------------------------------------------------
#
# ``State._get_fresh_trench_dig_alignment_details`` returns ``(valid, yaw,
# standoff, admitted_mask)`` but not its own ``applicable`` predicate, and an
# inapplicable action is reported as ``(1, 0, 0)``.  The predicate below is a
# verbatim transcription of the ``applicable`` term in
# ``terra/state.py::_get_fresh_trench_dig_alignment_details`` (including its
# fail-closed metadata branch) so an exported ``valid`` can be read as
# "pose-valid" rather than "valid or irrelevant".


def _make_alignment_probe():
    from terra.state import _as_2d_map, _as_axes_table, _as_scalar_int

    def probe(state):
        cur = state._get_current_agent_state()
        target = _as_2d_map(state.world.target_map.map)
        action_map = _as_2d_map(state.world.action_map.map)
        records = _as_axes_table(state.world.trench_axes).astype(jnp.float32)
        max_axes = records.shape[0]
        trench_type = jnp.clip(_as_scalar_int(state.world.trench_type), 0, max_axes)

        dig_mask = state._mask_out_wrong_dig_tiles(state._build_dig_dump_cone())
        dig_mask = jnp.asarray(dig_mask, dtype=jnp.bool_).reshape(-1)
        dig_mask_2d = dig_mask.reshape(target.shape)

        fresh_target = jnp.logical_and(
            dig_mask_2d, jnp.logical_and(target < 0, action_map == 0)
        )
        valid_axes = jnp.arange(max_axes) < trench_type
        if records.shape[1] >= 8:
            segment_vectors = records[:, 5:7] - records[:, 3:5]
            finite_section_metadata = jnp.logical_and(
                jnp.all(records[:, 3:8] > jnp.float32(-96.0), axis=1),
                jnp.logical_and(
                    records[:, 7] > jnp.float32(0.0),
                    jnp.linalg.norm(segment_vectors, axis=1) > jnp.float32(1e-6),
                ),
            )
        else:
            finite_section_metadata = jnp.zeros((max_axes,), dtype=jnp.bool_)
        declared_metadata_valid = jnp.all(
            jnp.logical_or(~valid_axes, finite_section_metadata)
        )
        fail_closed_metadata = jnp.logical_and(
            trench_type > 0, ~declared_metadata_valid
        )
        fresh_trench_target = jnp.logical_and(
            fresh_target,
            jnp.logical_or(
                state.world.trench_axis_membership != jnp.uint8(0),
                fail_closed_metadata,
            ),
        )
        applicable = jnp.logical_and(
            cur.agent_type[0] == 0,
            jnp.logical_and(
                cur.loaded[0] == 0,
                jnp.logical_and(trench_type > 0, jnp.any(fresh_trench_target)),
            ),
        )
        valid, yaw, standoff, _ = state._get_fresh_trench_dig_alignment_details(
            dig_mask
        )

        # The exported errors are normalized, and the standoff error is 0 by
        # construction whenever the pose is in band -- useless as the note's
        # "raw successful fresh-dig yaw/standoff" endpoint.  Recover the raw
        # physical quantities of the SAME diagnostic axis the export selects.
        # Cross-checked against the export every step by the caller.
        axes = records[:, :3]
        line_denominators = jnp.maximum(
            jnp.linalg.norm(axes[:, :2], axis=1), jnp.float32(1e-6)
        )
        bit_values = jnp.left_shift(
            jnp.ones((max_axes,), dtype=jnp.uint8),
            jnp.arange(max_axes, dtype=jnp.uint8),
        )
        section_membership = jnp.logical_and(
            valid_axes[:, None, None],
            jnp.bitwise_and(
                state.world.trench_axis_membership[None, :, :], bit_values[:, None, None]
            )
            != 0,
        )
        axis_has_fresh = jnp.any(
            jnp.logical_and(section_membership, fresh_trench_target[None, :, :]),
            axis=(1, 2),
        )
        base_angle = jnp.ravel(state._get_base_angle_rad())[0]
        base_forward = jnp.array(
            [-jnp.sin(base_angle), jnp.cos(base_angle)], dtype=jnp.float32
        )
        trench_tangents = jnp.stack([-axes[:, 0], axes[:, 1]], axis=1)
        tangent_norms = jnp.maximum(
            jnp.linalg.norm(trench_tangents, axis=1), jnp.float32(1e-6)
        )
        parallel_cosines = jnp.clip(
            jnp.abs(trench_tangents @ base_forward) / tangent_norms, 0.0, 1.0
        )
        yaw_errors = jnp.arccos(parallel_cosines)
        yaw_errors_normalized = jnp.clip(
            yaw_errors / (jnp.pi / jnp.float32(2.0)), 0.0, 1.0
        )
        base_row = cur.pos_base[0].astype(jnp.float32)
        base_col = cur.pos_base[1].astype(jnp.float32)
        standoffs_m = (
            jnp.abs(axes[:, 0] * base_col + axes[:, 1] * base_row + axes[:, 2])
            / line_denominators
            * state.env_cfg.tile_size
        )
        standoff_min = jnp.float32(state.env_cfg.trench_dig_standoff_min_m)
        standoff_max = jnp.float32(state.env_cfg.trench_dig_standoff_max_m)
        standoff_errors_normalized = jnp.where(
            standoffs_m < standoff_min,
            (standoffs_m - standoff_min) / jnp.maximum(standoff_min, jnp.float32(1e-6)),
            jnp.where(
                standoffs_m > standoff_max,
                (standoffs_m - standoff_max)
                / jnp.maximum(standoff_max, jnp.float32(1e-6)),
                jnp.float32(0.0),
            ),
        )
        standoff_errors_normalized = jnp.clip(standoff_errors_normalized, -1.0, 1.0)
        axis_pose_valid = jnp.logical_and(
            valid_axes,
            jnp.logical_and(
                finite_section_metadata,
                jnp.logical_and(
                    axis_has_fresh,
                    jnp.logical_and(
                        yaw_errors
                        <= jnp.float32(state.env_cfg.trench_dig_yaw_tolerance_rad),
                        jnp.logical_and(
                            standoffs_m >= standoff_min, standoffs_m <= standoff_max
                        ),
                    ),
                ),
            ),
        )
        diagnostic_pool = jnp.where(
            valid, axis_pose_valid, jnp.logical_and(axis_has_fresh, ~axis_pose_valid)
        )
        diagnostic_score = yaw_errors_normalized + jnp.abs(standoff_errors_normalized)
        diagnostic_axis = jnp.argmin(
            jnp.where(diagnostic_pool, diagnostic_score, jnp.float32(jnp.inf))
        )
        raw_yaw_rad = jnp.where(
            applicable, yaw_errors[diagnostic_axis], jnp.float32(jnp.nan)
        )
        raw_standoff_m = jnp.where(
            applicable, standoffs_m[diagnostic_axis], jnp.float32(jnp.nan)
        )
        # Faithfulness cross-check of the transcription above.
        replica_yaw = jnp.where(
            applicable, yaw_errors_normalized[diagnostic_axis], jnp.float32(0.0)
        )
        replica_standoff = jnp.where(
            applicable, standoff_errors_normalized[diagnostic_axis], jnp.float32(0.0)
        )
        return (
            applicable,
            valid,
            yaw,
            standoff,
            fresh_trench_target,
            jnp.sum(fresh_trench_target).astype(jnp.int32),
            jnp.sum(dig_mask.astype(jnp.int32)),
            cur.loaded[0].astype(jnp.int32),
            fail_closed_metadata,
            raw_yaw_rad,
            raw_standoff_m,
            replica_yaw,
            replica_standoff,
            trench_type.astype(jnp.int32),
            jnp.sum(axis_pose_valid.astype(jnp.int32)),
            jnp.sum(axis_has_fresh.astype(jnp.int32)),
        )

    return jax.jit(jax.vmap(probe))


def _leaf_diff(state_a, state_b, count):
    """Per-slot boolean: does any leaf of the two state pytrees differ?

    ``State`` carries its own ``env_cfg``, so the flipped-gate successor
    trivially differs in ``enforce_trench_dig_alignment``; that subtree is
    the treatment knob itself and is excluded from the comparison.
    """
    paths_a, _ = jax.tree_util.tree_flatten_with_path(state_a)
    paths_b, _ = jax.tree_util.tree_flatten_with_path(state_b)
    diffs = []
    for (path, a), (_, b) in zip(paths_a, paths_b):
        if any("env_cfg" in str(entry) for entry in path):
            continue
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        if a.ndim == 0 or a.shape != b.shape or a.shape[0] != count:
            continue
        axes = tuple(range(1, a.ndim))
        diffs.append(jnp.any(a != b, axis=axes) if axes else (a != b))
    if not diffs:
        raise RuntimeError("differential gate comparison found no comparable leaves")
    return jnp.any(jnp.stack(diffs), axis=0)


def select_slots(rows, include_families, include_cells, exclude_cells):
    selected = []
    for row in rows:
        family = row["family"]
        cell = row["primary_cell"]
        if include_families and family not in include_families:
            continue
        if include_cells and not any(
            fnmatch.fnmatch(cell, pattern) for pattern in include_cells
        ):
            continue
        if exclude_cells and any(
            fnmatch.fnmatch(cell, pattern) for pattern in exclude_cells
        ):
            continue
        selected.append(int(row["slot_index"]) - 1)
    return selected


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--accepted-panel", default="development")
    parser.add_argument("--panel-family", default="gate_main")
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--include-family", nargs="*", default=["trench"])
    parser.add_argument("--include-cell", nargs="*", default=[])
    parser.add_argument("--exclude-cell", nargs="*", default=["trn-net4-*"])
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--differential-gate",
        action="store_true",
        help=(
            "also step every state with enforce_trench_dig_alignment flipped "
            "and classify successor-state divergences (code-stop check)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank_root = args.bank_root.resolve()

    from utils.accepted_bank import V8_RELEASE_ID

    release_id = json.loads((bank_root / "dataset.json").read_text()).get("release_id")
    accepted_bank = load_accepted_bank(
        bank_root,
        "G-UNIFORM",
        args.terra_revision,
        curriculum_stage="full" if release_id == V8_RELEASE_ID else None,
        evaluation_panel_family=args.panel_family,
    )
    panel = next(
        item
        for item in accepted_bank.evaluation_panels
        if item.name == args.accepted_panel
    )
    directory = bank_root / panel.maps_path
    rows = load_manifest(directory)
    panel_count = len(rows)

    slot_indices = select_slots(
        rows,
        set(args.include_family) if args.include_family else set(),
        list(args.include_cell),
        list(args.exclude_cell),
    )
    if not slot_indices:
        raise ValueError("slot selection is empty")
    count = len(slot_indices)
    slot_rows = [rows[index] for index in slot_indices]

    checkpoint = load_pkl_object(str(args.checkpoint))
    train_config = checkpoint["train_config"]
    _validate_checkpoint_architecture(checkpoint, train_config)

    # DATASET_SIZE stays at the full panel size so exact_reset_keys keeps its
    # slot->map identity; only the rollout batch is subset.
    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(panel_count)
    config = configure_for_bank(train_config, panel.maps_path, count)
    _validate_checkpoint_architecture(checkpoint, config)

    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    validate_model_params_match(
        initialized_state.params, checkpoint["model"], str(args.checkpoint)
    )
    model = SimpleNamespace(apply=initialized_state.apply_fn)
    model_params = checkpoint["model"]

    expected_gate = bool(getattr(config, "enforce_trench_dig_alignment", None) or False)
    raw_gate = getattr(env_params, "enforce_trench_dig_alignment", None)
    if raw_gate is None:
        raise RuntimeError("this Terra runtime has no fresh-trench dig gate")
    effective_gate = bool(np.ravel(np.asarray(raw_gate))[0])
    if effective_gate != expected_gate:
        raise RuntimeError(
            f"gate mismatch: checkpoint={expected_gate} env={effective_gate}"
        )

    all_map_keys = np.asarray(exact_reset_keys(panel_count))
    all_state_keys = np.asarray(
        manifest_environment_keys(
            rows, panel_count, accepted_bank.environment_protocol_sha256
        )
    )
    map_reset_keys = jnp.asarray(all_map_keys[np.asarray(slot_indices)])
    state_keys = jnp.asarray(all_state_keys[np.asarray(slot_indices)])

    timestep, env_params, state_keys = prepare_manifest_episode_reset(
        env, env_params, map_reset_keys, state_keys
    )

    probe = _make_alignment_probe()

    rng = jrandom.PRNGKey(args.seed)
    rng, _rng = jrandom.split(rng)
    prev_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)

    obs = timestep.observation
    target_maps_init = obs["target_map"].copy()

    terminated_once = jnp.zeros(count, dtype=jnp.bool_)
    succeeded_once = jnp.zeros(count, dtype=jnp.bool_)
    episode_length = jnp.zeros(count, dtype=jnp.int32)

    flipped_env_cfg = jax.tree_util.tree_map(lambda x: x, timestep.env_cfg)
    if args.differential_gate:
        gate_field = jnp.asarray(timestep.env_cfg.enforce_trench_dig_alignment)
        flipped_env_cfg = timestep.env_cfg._replace(
            enforce_trench_dig_alignment=jnp.logical_not(gate_field.astype(jnp.bool_))
        )

    records = {
        key: []
        for key in (
            "action",
            "active",
            "align_valid",
            "align_applicable",
            "yaw_error",
            "standoff_error",
            "fresh_trench_cells",
            "dig_cone_cells",
            "loaded",
            "action_had_effect",
            "fresh_trench_cells_dug",
            "target_map_mutated",
            "gate_divergence",
            "raw_yaw_rad",
            "raw_standoff_m",
            "pose_valid_axis_count",
            "fresh_axis_count",
        )
    }
    slot_trench_type = np.zeros(count, dtype=np.int32)

    start = time.time()
    for step in range(args.horizon):
        active = ~terminated_once
        (
            applicable,
            valid,
            yaw,
            standoff,
            fresh_trench_target,
            fresh_cells,
            cone_cells,
            loaded,
            fail_closed,
            raw_yaw_rad,
            raw_standoff_m,
            replica_yaw,
            replica_standoff,
            trench_type,
            pose_valid_axis_count,
            fresh_axis_count,
        ) = probe(timestep.state)
        if step == 0:
            slot_trench_type = np.asarray(trench_type, dtype=np.int32)
        if bool(np.asarray(fail_closed).any()):
            raise RuntimeError(
                "fresh-trench alignment metadata failed closed during rollout"
            )
        obs_valid = np.asarray(
            timestep.observation["fresh_trench_dig_alignment_valid"]
        ).reshape(-1)
        obs_yaw = np.asarray(
            timestep.observation["fresh_trench_dig_yaw_error"]
        ).reshape(-1)
        obs_standoff = np.asarray(
            timestep.observation["fresh_trench_dig_standoff_error"]
        ).reshape(-1)
        if not np.array_equal(obs_valid > 0.5, np.asarray(valid).astype(bool)):
            raise RuntimeError("probe disagrees with the exported alignment validity")
        if not (
            np.allclose(obs_yaw, np.asarray(replica_yaw), atol=1e-5)
            and np.allclose(obs_standoff, np.asarray(replica_standoff), atol=1e-5)
        ):
            raise RuntimeError(
                "raw-quantity transcription disagrees with the exported "
                "normalized alignment errors"
            )

        rng, rng_act, rng_step = jrandom.split(rng, 3)
        obs_model = obs_to_model_input(timestep.observation, prev_actions, config)
        _, logits = _apply_in_batch_chunks(model, model_params, obs_model)
        if args.stochastic:
            from tensorflow_probability.substrates import jax as tfp

            action = tfp.distributions.Categorical(logits=logits).sample(seed=rng_act)
        else:
            action = jnp.argmax(logits, axis=-1)
        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)

        action_map_before = timestep.state.world.action_map.map
        state_before = timestep.state

        rng_step_keys = jrandom.split(rng_step, count)
        wrapped = wrap_action(action, env.batch_cfg.action_type)
        candidate = env.step_no_reset(timestep, wrapped, rng_step_keys)

        divergence = jnp.zeros(count, dtype=jnp.bool_)
        if args.differential_gate:
            flipped_timestep = timestep._replace(env_cfg=flipped_env_cfg)
            flipped = env.step_no_reset(flipped_timestep, wrapped, rng_step_keys)
            divergence = _leaf_diff(candidate.state, flipped.state, count)

        def _preserve(previous, cand):
            if not hasattr(cand, "shape"):
                return cand
            if cand.ndim == 0 or cand.shape[0] != count:
                return cand
            mask = active.reshape((count,) + (1,) * (cand.ndim - 1))
            return jnp.where(mask, cand, previous)

        timestep = jax.tree_util.tree_map(_preserve, timestep, candidate)
        prev_actions = jnp.where(
            timestep.done[:, None], jnp.zeros_like(prev_actions), prev_actions
        )

        action_map_after = timestep.state.world.action_map.map
        fresh_dug = jnp.sum(
            jnp.logical_and(
                fresh_trench_target.reshape(action_map_after.shape[0], -1),
                (action_map_before != action_map_after).reshape(
                    action_map_after.shape[0], -1
                ),
            ),
            axis=-1,
        ).astype(jnp.int32)
        target_mutated = jnp.any(
            timestep.state.world.target_map.map != target_maps_init,
            axis=tuple(range(1, target_maps_init.ndim)),
        )

        step_done = timestep.done
        step_succeeded = timestep.info["task_done"]
        episode_length += active.astype(jnp.int32)
        succeeded_once |= active & step_succeeded.astype(jnp.bool_)
        terminated_once |= active & step_done.astype(jnp.bool_)

        records["action"].append(np.asarray(action, dtype=np.int8))
        records["active"].append(np.asarray(active, dtype=bool))
        records["align_valid"].append(np.asarray(valid, dtype=bool))
        records["align_applicable"].append(np.asarray(applicable, dtype=bool))
        records["yaw_error"].append(np.asarray(yaw, dtype=np.float32))
        records["standoff_error"].append(np.asarray(standoff, dtype=np.float32))
        records["fresh_trench_cells"].append(np.asarray(fresh_cells, dtype=np.int32))
        records["dig_cone_cells"].append(np.asarray(cone_cells, dtype=np.int32))
        records["loaded"].append(np.asarray(loaded, dtype=np.int32))
        records["action_had_effect"].append(
            np.asarray(timestep.info["action_had_effect"], dtype=bool)
        )
        records["fresh_trench_cells_dug"].append(np.asarray(fresh_dug, dtype=np.int32))
        records["target_map_mutated"].append(np.asarray(target_mutated, dtype=bool))
        records["gate_divergence"].append(np.asarray(divergence, dtype=bool))
        records["raw_yaw_rad"].append(np.asarray(raw_yaw_rad, dtype=np.float32))
        records["raw_standoff_m"].append(np.asarray(raw_standoff_m, dtype=np.float32))
        records["pose_valid_axis_count"].append(
            np.asarray(pose_valid_axis_count, dtype=np.int32)
        )
        records["fresh_axis_count"].append(np.asarray(fresh_axis_count, dtype=np.int32))

        if bool(np.asarray(terminated_once).all()):
            break
        if step % 50 == 0:
            done = int(np.asarray(terminated_once).sum())
            print(
                f"[probe] step {step:4d} terminated={done}/{count} "
                f"{time.time() - start:6.1f}s",
                flush=True,
            )

    # Per-finite-section completion at the terminal state.  Under the gate,
    # continuing a trench past the stretch the current pose admits requires
    # relocating and re-yawing into a fresh admissible lane.  If stalled
    # episodes show one section near-complete and its siblings near-zero, the
    # deficit is an unlearned re-approach maneuver; if every section sits at a
    # similar partial value, it is general slowness instead.
    final = timestep.state
    final_action = np.asarray(final.world.action_map.map).reshape(count, -1)
    final_target = np.asarray(final.world.target_map.map).reshape(count, -1)
    membership = np.asarray(final.world.trench_axis_membership).reshape(count, -1)
    final_trench_type = np.asarray(final.world.trench_type).reshape(-1)
    section_completion = []
    for episode in range(count):
        sections = []
        for bit in range(int(final_trench_type[episode])):
            owned = (membership[episode] >> bit) & 1
            target_cells = (final_target[episode] < 0) & (owned > 0)
            total = int(target_cells.sum())
            if total == 0:
                sections.append(None)
                continue
            dug = int((target_cells & (final_action[episode] < 0)).sum())
            sections.append({"target_cells": total, "dug": dug, "fraction": dug / total})
        section_completion.append(sections)

    def _section_shape(indices):
        rows = []
        for episode in indices:
            values = [
                section["fraction"]
                for section in section_completion[episode]
                if section is not None
            ]
            if len(values) < 2:
                continue
            values = sorted(values, reverse=True)
            rows.append(
                {
                    "sections": len(values),
                    "best": values[0],
                    "worst": values[-1],
                    "spread": values[0] - values[-1],
                    "mean": float(np.mean(values)),
                }
            )
        if not rows:
            return {"episodes": 0}
        return {
            "episodes": len(rows),
            "mean_best_section": float(np.mean([r["best"] for r in rows])),
            "mean_worst_section": float(np.mean([r["worst"] for r in rows])),
            "mean_spread": float(np.mean([r["spread"] for r in rows])),
            "median_spread": float(np.median([r["spread"] for r in rows])),
            "episodes_with_spread_ge_0p5": int(
                sum(1 for r in rows if r["spread"] >= 0.5)
            ),
            "episodes_with_a_section_ge_0p9_and_another_le_0p1": int(
                sum(1 for r in rows if r["best"] >= 0.9 and r["worst"] <= 0.1)
            ),
        }

    arrays = {key: np.stack(value) for key, value in records.items()}
    steps = arrays["action"].shape[0]
    active = arrays["active"]
    is_do = (arrays["action"] == DO_ACTION) & active
    valid = arrays["align_valid"]
    applicable = arrays["align_applicable"]

    do_steps = int(is_do.sum())
    invalid_do = is_do & ~valid
    applicable_do = is_do & applicable
    invalid_do_count = int(invalid_do.sum())
    applicable_do_count = int(applicable_do.sum())

    # A successful fresh dig: an applicable, pose-valid DO that actually
    # removed fresh trench target soil.
    successful_fresh = is_do & applicable & valid & (arrays["fresh_trench_cells_dug"] > 0)
    attempted_fresh_valid = is_do & applicable & valid

    summary = {
        "schema": SCHEMA,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_next_update": int(checkpoint.get("next_update", -1)),
        "gate_enabled": effective_gate,
        "bank_root": str(bank_root),
        "panel": f"evaluation/{args.panel_family}/{args.accepted_panel}",
        "panel_slot_count": panel_count,
        "terra_revision": args.terra_revision,
        "episodes": count,
        "slot_indices_1based": [index + 1 for index in slot_indices],
        "conditions": sorted({row["primary_cell"] for row in slot_rows}),
        "horizon": args.horizon,
        "steps_executed": steps,
        "seed": args.seed,
        "deterministic": not args.stochastic,
        "differential_gate": bool(args.differential_gate),
        "wallclock_s": round(time.time() - start, 1),
        "episode": {
            "terminated": int(np.asarray(terminated_once).sum()),
            "succeeded": int(np.asarray(succeeded_once).sum()),
            "mean_length": float(np.asarray(episode_length).mean()),
        },
        "action_counts": {
            str(index): int(((arrays["action"] == index) & active).sum())
            for index in range(8)
        },
        "active_steps": int(active.sum()),
        "mechanism": {
            "do_steps": do_steps,
            "invalid_do_steps": invalid_do_count,
            # preregistered denominator: every DO the policy chose
            "invalid_fresh_do_attempt_fraction": (
                invalid_do_count / do_steps if do_steps else None
            ),
            "fresh_applicable_do_steps": applicable_do_count,
            # applicability-conditioned denominator: only DOs that would
            # actually have removed fresh trench soil
            "invalid_fresh_do_fraction_of_applicable": (
                invalid_do_count / applicable_do_count if applicable_do_count else None
            ),
            "applicable_do_fraction_of_do": (
                applicable_do_count / do_steps if do_steps else None
            ),
            "invalid_do_steps_with_effect": int(
                (invalid_do & arrays["action_had_effect"]).sum()
            ),
            "invalid_do_steps_that_dug_fresh_trench_cells": int(
                (invalid_do & (arrays["fresh_trench_cells_dug"] > 0)).sum()
            ),
            "successful_fresh_dig_steps": int(successful_fresh.sum()),
            "attempted_valid_fresh_dig_steps": int(attempted_fresh_valid.sum()),
            # raw physical quantities of the diagnostic section (note's
            # "raw successful fresh-dig yaw/standoff" endpoint)
            "raw_yaw_rad_successful_fresh_dig": _stats(
                arrays["raw_yaw_rad"][successful_fresh]
            ),
            "raw_yaw_deg_successful_fresh_dig": _stats(
                np.degrees(arrays["raw_yaw_rad"][successful_fresh])
            ),
            "raw_standoff_m_successful_fresh_dig": _stats(
                arrays["raw_standoff_m"][successful_fresh]
            ),
            "raw_yaw_deg_invalid_do": _stats(
                np.degrees(arrays["raw_yaw_rad"][invalid_do])
            ),
            "raw_standoff_m_invalid_do": _stats(arrays["raw_standoff_m"][invalid_do]),
            "raw_yaw_deg_all_applicable_do": _stats(
                np.degrees(arrays["raw_yaw_rad"][applicable_do])
            ),
            "raw_standoff_m_all_applicable_do": _stats(
                arrays["raw_standoff_m"][applicable_do]
            ),
            # exported normalized errors, for completeness (the standoff
            # error is 0 by construction whenever the pose is in band)
            "normalized_yaw_error_successful_fresh_dig": _stats(
                arrays["yaw_error"][successful_fresh]
            ),
            "normalized_standoff_error_successful_fresh_dig": _stats(
                arrays["standoff_error"][successful_fresh]
            ),
            "normalized_yaw_error_invalid_do": _stats(arrays["yaw_error"][invalid_do]),
            "normalized_standoff_error_invalid_do": _stats(
                arrays["standoff_error"][invalid_do]
            ),
        },
        "code_stop": {
            "differential_gate_measured": bool(args.differential_gate),
            "invalid_fresh_do_mutated_a_trench_cell": int(
                (invalid_do & (arrays["fresh_trench_cells_dug"] > 0)).sum()
            ),
            "target_map_mutated_any_slot": bool(arrays["target_map_mutated"].any()),
            "gate_divergence_steps": int(arrays["gate_divergence"].sum()),
            "gate_divergence_at_non_do_steps": int(
                (arrays["gate_divergence"] & active & ~is_do).sum()
            ),
            "gate_divergence_at_valid_do_steps": int(
                (arrays["gate_divergence"] & is_do & valid).sum()
            ),
            "gate_divergence_at_loaded_do_steps": int(
                (arrays["gate_divergence"] & is_do & (arrays["loaded"] > 0)).sum()
            ),
            "gate_divergence_at_invalid_applicable_do_steps": int(
                (arrays["gate_divergence"] & invalid_do & applicable).sum()
            ),
        },
    }

    per_condition = {}
    cells = np.asarray([row["primary_cell"] for row in slot_rows])
    for cell in sorted(set(cells.tolist())):
        mask = np.zeros_like(active)
        mask[:, cells == cell] = True
        cell_do = is_do & mask
        cell_invalid = invalid_do & mask
        cell_applicable = applicable_do & mask
        per_condition[cell] = {
            "episodes": int((cells == cell).sum()),
            "do_steps": int(cell_do.sum()),
            "invalid_do_steps": int(cell_invalid.sum()),
            "invalid_fresh_do_attempt_fraction": (
                float(cell_invalid.sum() / cell_do.sum()) if cell_do.sum() else None
            ),
            "fresh_applicable_do_steps": int(cell_applicable.sum()),
            "invalid_fresh_do_fraction_of_applicable": (
                float(cell_invalid.sum() / cell_applicable.sum())
                if cell_applicable.sum()
                else None
            ),
            "succeeded": int(np.asarray(succeeded_once)[cells == cell].sum()),
        }
    summary["per_condition"] = per_condition

    # Axis-count class: the generated number of finite trench sections on the
    # map (State.world.trench_type).  1-axis is a single straight run; >=3 is a
    # multi-branch junction where a macro cone can straddle perpendicular
    # sections.  This split separates "aligning, easy geometry first" from
    # "structurally blocked at junctions".
    per_axis_class = {}
    for axis_count in sorted(set(slot_trench_type.tolist())):
        mask = np.zeros_like(active)
        member = slot_trench_type == axis_count
        mask[:, member] = True
        class_do = is_do & mask
        class_invalid = invalid_do & mask
        class_applicable = applicable_do & mask
        class_success = successful_fresh & mask
        per_axis_class[str(int(axis_count))] = {
            "episodes": int(member.sum()),
            "conditions": sorted(
                {
                    row["primary_cell"]
                    for row, keep in zip(slot_rows, member.tolist())
                    if keep
                }
            ),
            "do_steps": int(class_do.sum()),
            "invalid_do_steps": int(class_invalid.sum()),
            "invalid_fresh_do_attempt_fraction": (
                float(class_invalid.sum() / class_do.sum())
                if class_do.sum()
                else None
            ),
            "fresh_applicable_do_steps": int(class_applicable.sum()),
            "invalid_fresh_do_fraction_of_applicable": (
                float(class_invalid.sum() / class_applicable.sum())
                if class_applicable.sum()
                else None
            ),
            "successful_fresh_dig_steps": int(class_success.sum()),
            "raw_yaw_deg_successful_fresh_dig": _stats(
                np.degrees(arrays["raw_yaw_rad"][class_success])
            ),
            "raw_standoff_m_successful_fresh_dig": _stats(
                arrays["raw_standoff_m"][class_success]
            ),
            "episodes_succeeded": int(np.asarray(succeeded_once)[member].sum()),
        }
    summary["per_axis_class"] = per_axis_class
    summary["slot_trench_type"] = slot_trench_type.tolist()

    succeeded_host = np.asarray(succeeded_once).astype(bool)
    summary["section_completion"] = {
        "definition": (
            "per generated finite trench section at the terminal state: dug = "
            "target<0 cells owned by that section whose action map went "
            "negative"
        ),
        "stalled_episodes": _section_shape(np.flatnonzero(~succeeded_host).tolist()),
        "successful_episodes": _section_shape(
            np.flatnonzero(succeeded_host).tolist()
        ),
        "all_episodes": _section_shape(list(range(count))),
        "per_episode": [
            {
                "slot_index": int(slot_rows[episode]["slot_index"]),
                "primary_cell": slot_rows[episode]["primary_cell"],
                "succeeded": bool(succeeded_host[episode]),
                "sections": [
                    None if section is None else round(section["fraction"], 4)
                    for section in section_completion[episode]
                ],
            }
            for episode in range(count)
        ],
    }

    # How often is a pose-valid section even available at an applicable DO?
    # If this is near zero at junctions the gate is structurally blocking
    # rather than the policy failing to aim.
    summary["mechanism"]["pose_valid_axis_available_at_applicable_do"] = {
        "steps": int(applicable_do.sum()),
        "with_at_least_one_pose_valid_axis": int(
            (applicable_do & (arrays["pose_valid_axis_count"] > 0)).sum()
        ),
        "fraction": (
            float(
                (applicable_do & (arrays["pose_valid_axis_count"] > 0)).sum()
                / applicable_do.sum()
            )
            if applicable_do.sum()
            else None
        ),
        "mean_fresh_axis_count": (
            float(arrays["fresh_axis_count"][applicable_do].mean())
            if applicable_do.sum()
            else None
        ),
    }

    per_slot = []
    for position, row in enumerate(slot_rows):
        slot_do = is_do[:, position]
        slot_invalid = invalid_do[:, position]
        per_slot.append(
            {
                "slot_index": int(row["slot_index"]),
                "primary_cell": row["primary_cell"],
                "family": row["family"],
                "do_steps": int(slot_do.sum()),
                "invalid_do_steps": int(slot_invalid.sum()),
                "fresh_applicable_do_steps": int(applicable_do[:, position].sum()),
                "successful_fresh_dig_steps": int(successful_fresh[:, position].sum()),
                "episode_length": int(np.asarray(episode_length)[position]),
                "succeeded": bool(np.asarray(succeeded_once)[position]),
            }
        )
    summary["per_slot"] = per_slot

    output.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    npz_path = output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        **arrays,
        slot_index=np.asarray(slot_indices) + 1,
        slot_trench_type=slot_trench_type,
    )
    print(json.dumps(summary["mechanism"], indent=1))
    print(json.dumps(summary["code_stop"], indent=1))
    print(f"wrote {output}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
