#!/usr/bin/env python3
"""Evaluate every frozen manifest slot exactly once at a fixed horizon."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from eval_mcts import rollout_episode
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.accepted_bank import load_accepted_bank
from utils.helpers import load_pkl_object

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

LEGACY_COMPLETION_CONTRACT = "legacy_implicit_buffer_v0"


def environment_completion_contract() -> str:
    try:
        from terra.state import CORRECTED_DENSE_CONTRACT
    except ImportError:
        return LEGACY_COMPLETION_CONTRACT
    return str(CORRECTED_DENSE_CONTRACT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def checkpoint_treatment_fingerprint(checkpoint: dict) -> dict:
    """Return the path-independent treatment contract for one checkpoint."""
    config = checkpoint.get("train_config")
    if config is None:
        raise ValueError("checkpoint has no train_config")
    bank = _field(config, "accepted_bank")
    curriculum = _field(config, "curriculum_levels_override")
    contract = {
        "schema": "terra_fixed_bank_treatment_v1",
        "run": {
            "name": _field(config, "name"),
            "seed": _field(config, "seed"),
            "config_name": _field(config, "config_name"),
            "accepted_bank_arm": _field(bank, "arm"),
        },
        "bank": {
            "terra_revision": _field(bank, "terra_revision"),
            "environment_protocol_sha256": _field(
                bank, "environment_protocol_sha256"
            ),
            "source_registry_sha256": _field(bank, "source_registry_sha256"),
        },
        "ppo": {
            name: _jsonable(_field(config, name))
            for name in (
                "num_devices",
                "num_envs_per_device",
                "num_steps",
                "update_epochs",
                "num_minibatches",
                "lr",
                "gamma",
                "gae_lambda",
                "clip_eps",
                "vf_coef",
                "max_grad_norm",
                "ent_schedule_start",
                "ent_schedule_end",
                "ent_schedule_steps",
                "use_value_clip",
                "flat_minibatch_shuffle",
            )
        },
        "reward_action": {
            "agent_types": _jsonable(_field(config, "agent_types_override")),
            "action_types": _jsonable(_field(config, "action_types_override")),
            "relocation_progress_mult": _field(
                config, "relocation_progress_mult"
            ),
            "curriculum_levels": _jsonable(curriculum),
        },
        "sampler": _jsonable(_field(config, "pooled_sampler")),
        "architecture": {
            name: _jsonable(_field(config, name))
            for name in (
                "model_size",
                "model_core",
                "map_encoder",
                "encoder_compute_dtype",
                "attention_compute_dtype",
                "token_mixer_residual_init_scale",
                "critic_hidden_dims",
                "loaded_max",
            )
        },
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "contract": contract,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def selected_map_indices(keys: jax.Array, count: int) -> np.ndarray:
    def selected(key):
        _, subkey = jax.random.split(key)
        return jax.random.randint(subkey, (), 0, count)

    return np.asarray(jax.vmap(selected)(keys))


def exact_reset_keys(count: int) -> jax.Array:
    found: list[np.ndarray | None] = [None] * count
    start = 0
    while any(key is None for key in found):
        seeds = jnp.arange(start, start + 4096, dtype=jnp.uint32)
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        indices = selected_map_indices(keys, count)
        keys_host = np.asarray(keys)
        for key, index in zip(keys_host, indices):
            if found[int(index)] is None:
                found[int(index)] = key
        start += 4096
        if start > 1_000_000:
            raise RuntimeError(f"could not construct exact reset keys for {count} maps")
    result = jnp.asarray(np.stack(found))
    actual = selected_map_indices(result, count)
    np.testing.assert_array_equal(actual, np.arange(count))
    return result


def manifest_reset_keys(
    rows: list[dict],
    count: int,
    environment_protocol_sha256: str,
) -> jax.Array:
    """Load the frozen episode reset seeds and verify exact slot selection."""
    if len(rows) != count:
        raise ValueError("manifest row count does not match the panel slot count")
    seeds = []
    for row in rows:
        seed = row.get("reset_seed")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ValueError(
                f"manifest slot {row.get('slot_index')} has invalid reset_seed"
            )
        if row.get("environment_protocol_sha256") != (
            environment_protocol_sha256
        ):
            raise ValueError(
                f"manifest slot {row.get('slot_index')} has a stale protocol"
            )
        seeds.append(seed)
    keys = jax.vmap(jax.random.PRNGKey)(
        jnp.asarray(seeds, dtype=jnp.uint32)
    )
    actual = selected_map_indices(keys, count)
    np.testing.assert_array_equal(actual, np.arange(count))
    return keys


def load_manifest(directory: Path) -> list[dict]:
    path = directory / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: int(row["slot_index"]))
    expected = list(range(1, len(rows) + 1))
    actual = [int(row["slot_index"]) for row in rows]
    if actual != expected:
        raise RuntimeError(f"{path} does not enumerate contiguous slots: {actual[:8]}")
    return rows


def configure_for_bank(train_config, relative_path: str, count: int):
    config = copy.deepcopy(train_config)
    config.num_devices = 1
    config.num_envs_per_device = count
    config.num_test_rollouts = count
    # num_minibatches is unused by the fixed-bank rollout but must still divide
    # num_envs_per_device for MixedAgentTrainConfig to construct. Take the
    # largest divisor of the slot count that does not exceed the configured
    # value (identical to the old min() whenever that already divided count).
    requested = min(int(getattr(config, "num_minibatches", 32)), count)
    config.num_minibatches = next(
        candidate for candidate in range(requested, 0, -1) if count % candidate == 0
    )
    config.agent_types_override = (0,)
    config.action_types_override = (0,)
    config.curriculum_levels_override = [
        {
            "maps_path": relative_path,
            "max_steps_in_episode": 450,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
    ]
    config.curriculum_increase_level_threshold = 3
    config.curriculum_decrease_level_threshold = 3
    config.curriculum_last_level_type = "none"
    config.pooled_sampler = None
    config.accepted_bank = None
    config.single_map_path = None
    config.replay_map_count = 0
    config.target_map_repeat = 0
    config.teacher_checkpoint = None
    config.teacher_obs_downsample = 1
    config.resume_from = None
    config.warm_start_from = None
    config.resume_update = None
    config.load_env_from_checkpoint = False
    return config


def verify_exact_reset(
    env,
    env_params,
    reset_keys,
    directory: Path,
    count: int,
) -> dict:
    timestep = env.reset(env_params, reset_keys)
    state = timestep.state
    observed_fields = {
        "target": np.asarray(state.world.target_map.map),
        "initial_action": np.asarray(state.world.action_map.map),
        "occupancy": np.asarray(state.world.padding_mask.map),
        "dumpability": np.asarray(state.world.dumpability_mask_init.map),
        "distance": np.asarray(state.world.relocation_distance_map),
    }
    if observed_fields["target"].shape[0] != count:
        raise RuntimeError(
            f"reset produced {observed_fields['target'].shape[0]} maps, "
            f"expected {count}"
        )
    source_directories = {
        "target": "images",
        "initial_action": "actions",
        "occupancy": "occupancy",
        "dumpability": "dumpability",
        "distance": "distance",
    }
    for index in range(count):
        for field, subdirectory in source_directories.items():
            expected = np.load(
                directory / subdirectory / f"img_{index + 1}.npy"
            )
            observed = np.squeeze(observed_fields[field][index])
            equal = (
                np.allclose(observed, expected, rtol=0.0, atol=1e-7)
                if field == "distance"
                else np.array_equal(observed, expected)
            )
            if not equal:
                raise RuntimeError(
                    f"exact reset {field} mismatch at slot {index + 1}"
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
                raise RuntimeError(
                    f"exact reset {field} mismatch at slot {index + 1}"
                )

    env_steps = np.asarray(state.env_steps)
    if env_steps.shape != (count,) or np.any(env_steps != 0):
        raise RuntimeError(
            "fixed evaluation reset must start every slot with env_steps == 0"
        )

    layer_hashes = {}
    for field, subdirectory in {
        **source_directories,
        "metadata": "metadata",
    }.items():
        digest = hashlib.sha256()
        for index in range(1, count + 1):
            filename = (
                f"trench_{index}.json"
                if field == "metadata"
                else f"img_{index}.npy"
            )
            digest.update(
                (directory / subdirectory / filename).read_bytes()
            )
        layer_hashes[field] = digest.hexdigest()
    return {
        "passed": True,
        "slots": count,
        "env_steps_min": int(env_steps.min()),
        "env_steps_max": int(env_steps.max()),
        "verified_fields": [
            "target",
            "initial_action",
            "occupancy",
            "dumpability",
            "distance",
            "trench_axes",
            "trench_type",
            "foundation_border_axes",
            "foundation_border_type",
        ],
        "layer_sha256": layer_hashes,
    }


GRADED_FIELD = "terminal_absolute"
PROMOTION_MACRO_GAIN = 0.01
PROMOTION_GUARD_TOLERANCE = 0.05


def _completion_stats(values: np.ndarray) -> dict:
    return {
        "episodes": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def graded_summary(per_map: list[dict]) -> dict:
    """Condition-balanced terminal completion and lower-tail guards."""
    if not per_map or any(GRADED_FIELD not in row for row in per_map):
        return {
            "available": False,
            "reason": f"{GRADED_FIELD} is absent from at least one rollout",
        }
    completion = np.asarray(
        [float(row[GRADED_FIELD]) for row in per_map],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(completion)):
        raise ValueError("terminal absolute completion contains non-finite values")
    if np.any(completion < -1e-6) or np.any(completion > 1.0 + 1e-6):
        raise ValueError("terminal absolute completion must be in [0, 1]")

    by_condition: dict[str, list[float]] = {}
    condition_family: dict[str, str] = {}
    by_family_values: dict[str, list[float]] = {}
    for row, value in zip(per_map, completion):
        condition = row["primary_cell"]
        family = row["family"]
        previous_family = condition_family.setdefault(condition, family)
        if previous_family != family:
            raise ValueError(
                f"condition {condition!r} appears in multiple families"
            )
        by_condition.setdefault(condition, []).append(float(value))
        by_family_values.setdefault(family, []).append(float(value))
    condition_stats = {
        condition: _completion_stats(np.asarray(values, dtype=np.float64))
        for condition, values in sorted(by_condition.items())
    }
    condition_means = np.asarray(
        [statistics["mean"] for statistics in condition_stats.values()],
        dtype=np.float64,
    )
    worst_condition, worst_statistics = min(
        condition_stats.items(),
        key=lambda item: item[1]["mean"],
    )
    by_family = {}
    for family, values in sorted(by_family_values.items()):
        family_conditions = [
            condition
            for condition, condition_value in condition_family.items()
            if condition_value == family
        ]
        family_condition_means = np.asarray(
            [
                condition_stats[condition]["mean"]
                for condition in family_conditions
            ],
            dtype=np.float64,
        )
        family_worst = min(
            family_conditions,
            key=lambda condition: condition_stats[condition]["mean"],
        )
        by_family[family] = {
            **_completion_stats(np.asarray(values, dtype=np.float64)),
            "condition_count": len(family_conditions),
            "macro_completion": float(family_condition_means.mean()),
            "worst_condition": family_worst,
            "worst_condition_completion": float(
                condition_stats[family_worst]["mean"]
            ),
        }
    family_macro_completion = float(
        np.mean(
            [
                family_statistics["macro_completion"]
                for family_statistics in by_family.values()
            ]
        )
    )
    return {
        "available": True,
        "field": GRADED_FIELD,
        "macro_completion": float(condition_means.mean()),
        "family_macro_completion": family_macro_completion,
        "condition_count": len(condition_stats),
        "worst_condition": worst_condition,
        "worst_condition_completion": float(worst_statistics["mean"]),
        "micro": _completion_stats(completion),
        "by_family": by_family,
        "by_primary_cell": condition_stats,
    }


def comparison_gate(reference: dict, candidate: dict) -> dict:
    """Compare two evaluations of the same bank without panel-size constants."""
    reference_overall = reference["overall"]
    candidate_overall = candidate["overall"]
    reference_episodes = int(reference_overall["episodes"])
    candidate_episodes = int(candidate_overall["episodes"])
    if reference_episodes != candidate_episodes:
        raise ValueError(
            "fixed-bank comparison requires equal episode counts; "
            f"got {reference_episodes} and {candidate_episodes}"
        )
    if set(reference["by_primary_cell"]) != set(candidate["by_primary_cell"]):
        raise ValueError(
            "fixed-bank comparison requires identical condition identities"
        )

    reference_graded = reference["graded"]
    candidate_graded = candidate["graded"]
    graded_available = bool(
        reference_graded.get("available")
        and candidate_graded.get("available")
    )
    exact_map_gain = (
        int(candidate_overall["successes"])
        - int(reference_overall["successes"])
    )
    exact_rate_quantum = 1.0 / candidate_episodes
    macro_gain = (
        float(candidate_graded["macro_completion"])
        - float(reference_graded["macro_completion"])
        if graded_available
        else None
    )
    micro_p10_delta = (
        float(candidate_graded["micro"]["p10"])
        - float(reference_graded["micro"]["p10"])
        if graded_available
        else None
    )
    worst_condition_delta = (
        float(candidate_graded["worst_condition_completion"])
        - float(reference_graded["worst_condition_completion"])
        if graded_available
        else None
    )
    progress_passed = exact_map_gain >= 1 or (
        graded_available and macro_gain >= PROMOTION_MACRO_GAIN
    )
    guards_passed = bool(
        graded_available
        and micro_p10_delta >= -PROMOTION_GUARD_TOLERANCE
        and worst_condition_delta >= -PROMOTION_GUARD_TOLERANCE
    )
    reference_integrity_passed = bool(reference["integrity"]["passed"])
    candidate_integrity_passed = bool(candidate["integrity"]["passed"])
    integrity_passed = (
        reference_integrity_passed and candidate_integrity_passed
    )
    return {
        "schema": "terra_fixed_bank_comparison_gate_v1",
        "reference_episodes": reference_episodes,
        "candidate_episodes": candidate_episodes,
        "exact_map_gain": exact_map_gain,
        "exact_rate_gain": exact_map_gain * exact_rate_quantum,
        "exact_rate_quantum": exact_rate_quantum,
        "required_exact_map_gain": 1,
        "macro_completion_gain": macro_gain,
        "required_macro_completion_gain": PROMOTION_MACRO_GAIN,
        "micro_p10_delta": micro_p10_delta,
        "worst_condition_delta": worst_condition_delta,
        "guard_tolerance": PROMOTION_GUARD_TOLERANCE,
        "progress_passed": bool(progress_passed),
        "guards_passed": guards_passed,
        "reference_integrity_passed": reference_integrity_passed,
        "candidate_integrity_passed": candidate_integrity_passed,
        "integrity_passed": integrity_passed,
        "passed": bool(progress_passed and guards_passed and integrity_passed),
    }


def grouped_results(
    rows: list[dict],
    successes: np.ndarray,
    terminations: np.ndarray,
    lengths: np.ndarray,
    *,
    horizon: int | None = None,
    completion_metrics: dict[str, np.ndarray] | None = None,
    integrity_metrics: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    completion_metrics = completion_metrics or {}
    integrity_metrics = integrity_metrics or {}
    per_map = []
    for index, (row, success, terminated, length) in enumerate(
        zip(rows, successes, terminations, lengths)
    ):
        timed_out = bool(
            terminated
            and horizon is not None
            and int(length) >= int(horizon)
        )
        if bool(success) and timed_out:
            termination_reason = "task_done_and_timeout"
        elif bool(success):
            termination_reason = "task_done"
        elif timed_out:
            termination_reason = "timeout"
        elif bool(terminated):
            termination_reason = "other_termination"
        else:
            termination_reason = "horizon_censored"
        metric_values = {
            key: float(np.asarray(values)[index])
            for key, values in completion_metrics.items()
        }
        integrity_values = {
            key: np.asarray(values)[index].item()
            for key, values in integrity_metrics.items()
        }
        integrity_failure = bool(
            int(integrity_values.get("maximum_mass_residual", 0)) != 0
            or bool(integrity_values.get("target_mutation", False))
            or bool(integrity_values.get("obstacle_mutation", False))
            or bool(integrity_values.get("nonfinite_state", False))
            or bool(integrity_values.get("termination_disagreement", False))
            or bool(integrity_values.get("slot_index_disagreement", False))
            or bool(integrity_values.get("integrity_unavailable", False))
        )
        per_map.append(
            {
                **row,
                "success": bool(success),
                "terminated": bool(terminated),
                "timeout": timed_out,
                "termination_reason": termination_reason,
                "steps": int(length),
                **metric_values,
                **integrity_values,
                "integrity_failure": integrity_failure,
            }
        )

    def summarize(field: str) -> dict:
        values = sorted({row[field] for row in per_map})
        result = {}
        for value in values:
            selected = [row for row in per_map if row[field] == value]
            successes_here = sum(int(row["success"]) for row in selected)
            result[value] = {
                "successes": successes_here,
                "episodes": len(selected),
                "success_rate": successes_here / len(selected),
                "terminations": sum(int(row["terminated"]) for row in selected),
            }
        return result

    total_successes = int(successes.sum())
    summary = {
        "overall": {
            "successes": total_successes,
            "episodes": len(rows),
            "success_rate": total_successes / len(rows),
            "terminations": int(terminations.sum()),
        },
        "by_family": summarize("family"),
        "by_primary_cell": summarize("primary_cell"),
    }
    integrity_failure_count = sum(
        int(row["integrity_failure"]) for row in per_map
    )
    summary["integrity"] = {
        "passed": integrity_failure_count == 0,
        "failure_count": integrity_failure_count,
        "mass_residual_failures": sum(
            int(int(row.get("maximum_mass_residual", 0)) != 0)
            for row in per_map
        ),
        "target_mutations": sum(
            int(bool(row.get("target_mutation", False))) for row in per_map
        ),
        "obstacle_mutations": sum(
            int(bool(row.get("obstacle_mutation", False))) for row in per_map
        ),
        "nonfinite_states": sum(
            int(bool(row.get("nonfinite_state", False))) for row in per_map
        ),
        "termination_disagreements": sum(
            int(bool(row.get("termination_disagreement", False)))
            for row in per_map
        ),
        "slot_index_disagreements": sum(
            int(bool(row.get("slot_index_disagreement", False)))
            for row in per_map
        ),
        "unavailable": sum(
            int(bool(row.get("integrity_unavailable", False)))
            for row in per_map
        ),
    }
    summary["graded"] = graded_summary(per_map)
    summary["comparison_gate_contract"] = {
        "requires_reference_evaluation": True,
        "exact_progress": "at least one additional map",
        "macro_completion_gain": PROMOTION_MACRO_GAIN,
        "micro_p10_max_regression": PROMOTION_GUARD_TOLERANCE,
        "worst_condition_max_regression": PROMOTION_GUARD_TOLERANCE,
        "integrity_required": True,
    }
    summary["termination_reasons"] = {
        reason: sum(
            int(row["termination_reason"] == reason) for row in per_map
        )
        for reason in (
            "task_done",
            "timeout",
            "task_done_and_timeout",
            "other_termination",
            "horizon_censored",
        )
    }
    return per_map, summary


def checkpoint_paths(args) -> list[Path]:
    paths = [Path(path).resolve() for path in args.checkpoint]
    for pattern in args.checkpoint_glob:
        paths.extend(Path(path).resolve() for path in glob.glob(pattern))
    if not paths:
        raise ValueError("provide --checkpoint or --checkpoint-glob")
    if len(paths) != len(set(paths)):
        raise ValueError("fixed-bank evaluation received a duplicate checkpoint path")
    paths = sorted(paths)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return paths


def validate_checkpoint_sequence(
    checkpoints: list[tuple[Path, dict]],
) -> dict:
    """Require one strictly ordered treatment before any environment setup."""
    updates = [
        int(checkpoint.get("next_update", 0))
        for _, checkpoint in checkpoints
    ]
    if any(update <= 0 for update in updates):
        raise ValueError("checkpoint next_update must be a positive integer")
    if any(current >= following for current, following in zip(updates, updates[1:])):
        raise ValueError(
            "fixed-bank checkpoint updates must be strictly increasing and unique"
        )
    treatment_fingerprints = [
        checkpoint_treatment_fingerprint(checkpoint)
        for _, checkpoint in checkpoints
    ]
    reference_treatment = treatment_fingerprints[0]
    for (path, _), fingerprint in zip(
        checkpoints[1:],
        treatment_fingerprints[1:],
    ):
        if fingerprint["sha256"] != reference_treatment["sha256"]:
            raise ValueError(
                "fixed-bank checkpoint list mixes treatment contracts: "
                f"{path} has {fingerprint['sha256']}, expected "
                f"{reference_treatment['sha256']}"
            )
    return reference_treatment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="development",
        help="bank subdirectory (e.g. 'development', 'sealed', 'held_out')",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        default=None,
        help="stratum directories under --split (e.g. 'M0 M1 M2', or 'all')",
    )
    parser.add_argument(
        "--accepted-panel",
        choices=("promotion", "development", "sealed"),
        help=(
            "Evaluate one panel directly from a "
            "terra_curriculum_loader_bank_v1 root. Uses the manifest's frozen "
            "reset_seed and episode_id; incompatible with --strata."
        ),
    )
    parser.add_argument(
        "--terra-revision",
        help=(
            "Exact immutable Terra revision bound into an accepted bank. "
            "Required with --accepted-panel; no Git metadata is consulted."
        ),
    )
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--expect-completion-contract",
        choices=("exact_visible_dump_v1", LEGACY_COMPLETION_CONTRACT),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon != 450:
        raise ValueError("training-design-v1 fixed evaluation requires horizon 450")
    bank_root = args.bank_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completion_contract = environment_completion_contract()
    if (
        args.expect_completion_contract is not None
        and completion_contract != args.expect_completion_contract
    ):
        raise RuntimeError(
            "completion contract mismatch: expected "
            f"{args.expect_completion_contract}, imported {completion_contract}"
        )

    accepted_bank = None
    if args.accepted_panel is not None:
        if args.strata is not None:
            raise ValueError("--accepted-panel is incompatible with --strata")
        if args.terra_revision is None:
            raise ValueError("--accepted-panel requires --terra-revision")
        accepted_bank = load_accepted_bank(
            bank_root,
            "G-UNIFORM",
            args.terra_revision,
        )
        panel = next(
            panel
            for panel in accepted_bank.evaluation_panels
            if panel.name == args.accepted_panel
        )
        targets = [
            (
                args.accepted_panel,
                "all",
                panel.maps_path,
            )
        ]
    else:
        if args.terra_revision is not None:
            raise ValueError("--terra-revision requires --accepted-panel")
        strata = args.strata or ("M0", "M1", "M2")
        targets = [
            (args.split, stratum, f"{args.split}/{stratum}")
            for stratum in strata
        ]

    paths = checkpoint_paths(args)
    checkpoints = [(path, load_pkl_object(str(path))) for path in paths]
    checkpoints.sort(
        key=lambda item: (
            int(item[1].get("next_update", 0)),
            str(item[0]),
        )
    )
    reference_treatment = validate_checkpoint_sequence(checkpoints)
    reference_train_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        if "model" not in checkpoint:
            raise KeyError("checkpoint has no model parameters")
        _validate_checkpoint_architecture(checkpoint, reference_train_config)

    records = []
    for split_name, stratum, relative_path in targets:
        directory = bank_root / relative_path
        rows = load_manifest(directory)
        count = len(rows)
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(count)
        config = configure_for_bank(reference_train_config, relative_path, count)
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        reset_keys = (
            manifest_reset_keys(
                rows,
                count,
                accepted_bank.environment_protocol_sha256,
            )
            if accepted_bank is not None
            else exact_reset_keys(count)
        )
        reset_verification = verify_exact_reset(
            env,
            env_params,
            reset_keys,
            directory,
            count,
        )
        model = SimpleNamespace(apply=initialized_state.apply_fn)
        previous_summary = None
        previous_checkpoint = None

        for checkpoint_path, checkpoint in checkpoints:
            _, stats, _ = rollout_episode(
                env,
                model,
                checkpoint["model"],
                env_params,
                config,
                max_frames=args.horizon,
                deterministic=not args.stochastic,
                seed=args.seed,
                use_mcts=False,
                reset_keys=reset_keys,
                record_observations=False,
                preserve_terminal_states=True,
                expected_slot_indices=np.arange(count, dtype=np.int32),
            )
            successes = np.asarray(stats["episode_done_once"], dtype=bool)
            terminations = np.asarray(stats["episode_terminated_once"], dtype=bool)
            lengths = np.asarray(stats["episode_length"], dtype=np.int32)
            terminal_completion = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.get("terminal_completion", {}).items()
            }
            raw_integrity = stats.get("integrity", {})
            integrity_supported = bool(raw_integrity.get("supported", False))
            if (
                completion_contract == "exact_visible_dump_v1"
                and not integrity_supported
            ):
                raise RuntimeError(
                    "fixed evaluation did not return supported state integrity metrics"
                )
            integrity_metrics = {
                key: np.asarray(value)
                for key, value in raw_integrity.items()
                if key != "supported"
            }
            integrity_metrics["integrity_unavailable"] = np.full(
                count,
                not integrity_supported,
                dtype=bool,
            )
            if (
                not np.all(np.isfinite(lengths))
                or successes.shape != (count,)
                or terminations.shape != (count,)
            ):
                raise RuntimeError("fixed evaluation returned invalid arrays")
            if completion_contract == "exact_visible_dump_v1":
                absolute = terminal_completion.get("absolute")
                if absolute is None or absolute.shape != (count,):
                    raise RuntimeError(
                        "exact_visible_dump_v1 evaluation did not return "
                        "per-map absolute completion"
                    )
                completion_one = np.isclose(absolute, 1.0, atol=1e-6)
                if not np.array_equal(successes, completion_one):
                    mismatches = np.flatnonzero(successes != completion_one)
                    raise RuntimeError(
                        "task_done <=> absolute_completion == 1 violated at "
                        f"fixed-bank slots {(mismatches + 1).tolist()}"
                    )
            expected_slots = np.arange(count, dtype=np.int32)
            observed_slots = np.asarray(
                integrity_metrics["slot_index_zero_based"],
                dtype=np.int32,
            )
            if observed_slots.shape != (count,):
                raise RuntimeError(
                    "fixed evaluation returned invalid terminal slot indices"
                )
            integrity_metrics["slot_index_disagreement"] = (
                observed_slots != expected_slots
            )
            expected_termination = successes | (lengths >= args.horizon)
            integrity_metrics["termination_disagreement"] = (
                terminations != expected_termination
            )
            per_map, summary = grouped_results(
                rows,
                successes,
                terminations,
                lengths,
                horizon=args.horizon,
                completion_metrics={
                    f"terminal_{key}": values
                    for key, values in terminal_completion.items()
                },
                integrity_metrics=integrity_metrics,
            )
            comparison_to_previous = (
                None
                if previous_summary is None
                else {
                    "reference_checkpoint": str(previous_checkpoint),
                    **comparison_gate(previous_summary, summary),
                }
            )
            record = {
                "schema": "terra_fixed_bank_eval_v4",
                "completion_contract": completion_contract,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_update": int(checkpoint.get("next_update", 0)),
                "treatment_fingerprint": reference_treatment,
                "bank_root": str(bank_root),
                "accepted_bank": (
                    None
                    if accepted_bank is None
                    else {
                        "schema": "terra_curriculum_loader_bank_v1",
                        "terra_revision": accepted_bank.terra_revision,
                        "environment_protocol_sha256": (
                            accepted_bank.environment_protocol_sha256
                        ),
                        "source_registry_sha256": (
                            accepted_bank.source_registry_sha256
                        ),
                    }
                ),
                "split": split_name,
                "stratum": stratum,
                "manifest": str(directory / "manifest.jsonl"),
                "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                "horizon": args.horizon,
                "deterministic": not args.stochastic,
                "policy_mode": (
                    "sampled" if args.stochastic else "deterministic"
                ),
                "seed": args.seed,
                "exact_manifest_enumeration": True,
                "reset_verification": reset_verification,
                "summary": summary,
                "comparison_to_previous": comparison_to_previous,
                "per_map": per_map,
            }
            records.append(record)
            with output.open("w") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)
                handle.write("\n")
            overall = summary["overall"]
            graded = summary["graded"]
            if graded.get("available"):
                print(
                    f"{checkpoint_path.name} {split_name}/{stratum}: "
                    f"macro={graded['macro_completion']:.3f}, "
                    f"micro_p10={graded['micro']['p10']:.3f}, "
                    f"worst={graded['worst_condition_completion']:.3f}, "
                    f"exact={overall['successes']}/{overall['episodes']}"
                )
            else:
                print(
                    f"{checkpoint_path.name} {split_name}/{stratum}: "
                    f"exact={overall['successes']}/{overall['episodes']} "
                    f"({graded['reason']})"
                )
            previous_summary = summary
            previous_checkpoint = checkpoint_path


if __name__ == "__main__":
    main()
