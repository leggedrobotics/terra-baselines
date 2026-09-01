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
    PARTIAL_RESET_CURRICULUM_SCHEMA,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.accepted_bank import (
    EVALUATION_PANEL_FAMILY_DEFAULT,
    V8_RELEASE_ID,
    load_accepted_bank,
)
from utils.models import validate_model_params_match
from utils.explicit_episode_bank import ExplicitEpisodePanel
from utils.explicit_episode_bank import load_explicit_episode_panel
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
            "environment_protocol_sha256": _field(bank, "environment_protocol_sha256"),
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
            "relocation_progress_mult": _field(config, "relocation_progress_mult"),
            "reward_stage": _field(config, "reward_stage"),
            "carry_work_observation": _field(config, "carry_work_observation"),
            "distance_protocol_id": _field(config, "distance_protocol_id"),
            "distance_sidecar_sha256": _field(config, "distance_sidecar_sha256"),
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
                "resnet_stage_channels",
                "resnet_blocks_per_stage",
                "loaded_max",
            )
        },
    }
    if bool(_field(config, "reward_v2_reset_context_observation", False)):
        # Conditional inclusion keeps historical fingerprints unchanged while
        # binding both the partial arm and its matched full-start control.
        contract["architecture"]["reward_v2_reset_context_observation"] = True
    if bool(_field(config, "trench_alignment_observation", False)):
        # Same conditional-inclusion rule: the width-3 alignment vector adds two
        # (3, 704) embeddings, so it is architecture, and it must bind the C0
        # control to its matched T1 treatment.
        contract["architecture"]["trench_alignment_observation"] = True
    trench_gate = _field(config, "enforce_trench_dig_alignment")
    if trench_gate is not None or bool(
        _field(config, "require_trench_alignment_metadata", False)
    ):
        # The C0/T1 pilot's only intended divergence. Recorded outside
        # "architecture" because the gate changes the environment, not shapes.
        contract["trench_dig_alignment"] = {
            "enforce_trench_dig_alignment": (
                None if trench_gate is None else bool(trench_gate)
            ),
            "require_trench_alignment_metadata": bool(
                _field(config, "require_trench_alignment_metadata", False)
            ),
        }
        # Gate semantics (v1 band vs v2 yaw-only) is an env treatment too.
        # Included only when the checkpoint's config carries the field, so the
        # pilot's pre-v2 fingerprints stay byte-identical.
        standoff_enforced = _field(config, "trench_dig_standoff_enforced")
        if standoff_enforced is not None:
            contract["trench_dig_alignment"]["trench_dig_standoff_enforced"] = bool(
                standoff_enforced
            )
    partial_reset_digest = _field(config, "partial_reset_bank_sha256")
    if partial_reset_digest is not None:
        raw_partial_receipt = checkpoint.get("partial_reset_curriculum")
        if not isinstance(raw_partial_receipt, dict) or raw_partial_receipt.get(
            "schema"
        ) != PARTIAL_RESET_CURRICULUM_SCHEMA:
            raise ValueError("partial-reset checkpoint lacks its curriculum receipt")
        partial_reset_receipt = dict(raw_partial_receipt)
        partial_reset_receipt.pop("partial_reset_root", None)
        for dynamic_field in (
            "next_update",
            "last_applied_tiers",
            "last_applied_share",
        ):
            partial_reset_receipt.pop(dynamic_field, None)
        contract["partial_reset"] = {
            "bank_sha256": partial_reset_digest,
            "observation": bool(
                _field(config, "reward_v2_reset_context_observation", False)
            ),
            "curriculum": _jsonable(partial_reset_receipt),
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


def _manifest_environment_keys(
    rows: list[dict],
    count: int,
    environment_protocol_sha256: str,
) -> jax.Array:
    """Load frozen episode seeds after validating their runtime contract."""
    from terra.benchmark_protocol import BENCHMARK_JAX_DEFAULT_PRNG_IMPL
    from terra.benchmark_protocol import BENCHMARK_JAX_THREEFRY_PARTITIONABLE

    runtime_contract = {
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
    }
    expected_contract = {
        "jax_default_prng_impl": BENCHMARK_JAX_DEFAULT_PRNG_IMPL,
        "jax_threefry_partitionable": (BENCHMARK_JAX_THREEFRY_PARTITIONABLE),
    }
    if runtime_contract != expected_contract:
        raise RuntimeError(
            "fixed-bank reset PRNG contract mismatch: "
            f"runtime={runtime_contract!r}, expected={expected_contract!r}"
        )
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
        if row.get("environment_protocol_sha256") != (environment_protocol_sha256):
            raise ValueError(
                f"manifest slot {row.get('slot_index')} has a stale protocol"
            )
        seeds.append(seed)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, dtype=jnp.uint32))
    return keys


def manifest_environment_keys(
    rows: list[dict],
    count: int,
    environment_protocol_sha256: str,
) -> jax.Array:
    """Return frozen environment-state keys without treating them as map selectors."""
    return _manifest_environment_keys(rows, count, environment_protocol_sha256)


def manifest_reset_keys(
    rows: list[dict],
    count: int,
    environment_protocol_sha256: str,
) -> jax.Array:
    """Load legacy seeds whose PRNG keys also select their ordered map slots."""
    keys = _manifest_environment_keys(rows, count, environment_protocol_sha256)
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
    # The policy keeps its two-scalar architecture, but held-out evaluation
    # never loads or samples the partial sidecar bank.
    config.partial_reset_root = None
    config.partial_reset_bank_sha256 = None
    config.single_map_path = None
    config.replay_map_count = 0
    config.target_map_repeat = 0
    config.teacher_checkpoint = None
    config.teacher_obs_downsample = 1
    config.resume_from = None
    config.warm_start_from = None
    config.resume_update = None
    config.load_env_from_checkpoint = False
    # Deliberately NOT reset here: trench_alignment_observation (architecture),
    # enforce_trench_dig_alignment (the arm's env treatment) and
    # require_trench_alignment_metadata (fail-closed bank contract) are
    # properties of the trained policy, so they ride along from the checkpoint's
    # own train_config and make the eval env auto-match the trained one.
    return config


def _prepare_ordered_map_slots(env, env_params, map_reset_keys):
    count = np.asarray(map_reset_keys).shape[0]
    if np.asarray(map_reset_keys).shape != (count, 2):
        raise ValueError("map_reset_keys must contain one PRNG key per episode")

    # prepare_reset owns a leading device axis; fixed evaluation operates on
    # its sole device after materializing every ordered map slot exactly once.
    device_env_params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[None], env_params
    )
    prepared_device = env.prepare_reset(
        device_env_params,
        jnp.asarray(map_reset_keys)[None],
    )
    return jax.tree_util.tree_map(lambda value: value[0], prepared_device)


def prepare_manifest_episode_reset(
    env,
    env_params,
    map_reset_keys,
    environment_state_keys,
):
    """Materialize ordered maps while retaining each manifest episode seed."""
    prepared = _prepare_ordered_map_slots(env, env_params, map_reset_keys)
    (
        prepared_env_params,
        target_maps,
        padding_masks,
        trench_axes,
        trench_type,
        foundation_border_axes,
        foundation_border_type,
        dumpability_mask_init,
        action_maps,
        distance_maps,
    ) = prepared
    if np.asarray(environment_state_keys).shape != np.asarray(map_reset_keys).shape:
        raise ValueError("environment_state_keys must match map_reset_keys")
    timestep = env.reset_prepared(
        prepared_env_params,
        environment_state_keys,
        target_maps,
        padding_masks,
        trench_axes,
        trench_type,
        foundation_border_axes,
        foundation_border_type,
        dumpability_mask_init,
        action_maps,
        distance_maps,
    )
    return timestep, prepared_env_params, environment_state_keys


def prepare_explicit_episode_reset(
    env,
    env_params,
    map_reset_keys,
    panel: ExplicitEpisodePanel,
):
    """Reset one ordered map panel from its complete frozen Agent states."""
    from terra.benchmark_state import validate_benchmark_initial_agent

    count = panel.slot_count
    prepared = _prepare_ordered_map_slots(env, env_params, map_reset_keys)
    (
        prepared_env_params,
        target_maps,
        padding_masks,
        trench_axes,
        trench_type,
        foundation_border_axes,
        foundation_border_type,
        dumpability_mask_init,
        action_maps,
        distance_maps,
    ) = prepared

    for index, agent in enumerate(panel.initial_agents):
        env_cfg = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value[index])),
            prepared_env_params,
        )
        validate_benchmark_initial_agent(
            agent,
            env_cfg=env_cfg,
            padding_mask=padding_masks[index],
            action_map=action_maps[index],
            dumpability_mask=dumpability_mask_init[index],
        )

    initial_agents = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *panel.initial_agents,
    )
    state_keys = jax.vmap(jax.random.PRNGKey)(
        jnp.asarray(panel.environment_reset_seeds, dtype=jnp.uint32)
    )
    timestep = env.reset_prepared(
        prepared_env_params,
        state_keys,
        target_maps,
        padding_masks,
        trench_axes,
        trench_type,
        foundation_border_axes,
        foundation_border_type,
        dumpability_mask_init,
        action_maps,
        distance_maps,
        initial_agents,
    )
    return timestep, prepared_env_params, state_keys


def verify_exact_reset(
    env,
    env_params,
    reset_keys,
    directory: Path,
    count: int,
    *,
    timestep=None,
    expected_initial_state_sha256: tuple[str, ...] | None = None,
    expected_state_keys=None,
) -> dict:
    precomputed_reset = timestep is not None
    if timestep is None:
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
            expected = np.load(directory / subdirectory / f"img_{index + 1}.npy")
            observed = np.squeeze(observed_fields[field][index])
            equal = (
                np.allclose(observed, expected, rtol=0.0, atol=1e-7)
                if field == "distance"
                else np.array_equal(observed, expected)
            )
            if not equal:
                raise RuntimeError(f"exact reset {field} mismatch at slot {index + 1}")

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
                f"trench_{index}.json" if field == "metadata" else f"img_{index}.npy"
            )
            digest.update((directory / subdirectory / filename).read_bytes())
        layer_hashes[field] = digest.hexdigest()
    result = {
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
    if expected_initial_state_sha256 is not None:
        from terra.benchmark_state import agent_state_sha256

        if expected_state_keys is None:
            raise ValueError("explicit initial-state verification requires state keys")
        expected_hashes = tuple(expected_initial_state_sha256)
        if len(expected_hashes) != count:
            raise ValueError("expected state-hash count does not match panel")
        observed_hashes = []
        for index in range(count):
            agent = jax.tree_util.tree_map(
                lambda value: value[index],
                state.agent,
            )
            observed_hashes.append(agent_state_sha256(agent))
        if tuple(observed_hashes) != expected_hashes:
            raise RuntimeError("explicit reset changed an initial Agent state")
        result["explicit_initial_state"] = {
            "passed": True,
            "slots": count,
            "ordered_agent_state_sha256": hashlib.sha256(
                json.dumps(
                    expected_hashes,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    if expected_state_keys is not None:
        if not precomputed_reset:
            raise ValueError("state-key verification requires a precomputed reset")
        observed_state_keys = np.asarray(state.key)
        expected_state_keys_host = np.asarray(expected_state_keys)
        if not np.array_equal(observed_state_keys, expected_state_keys_host):
            raise RuntimeError("precomputed reset changed an environment state key")
        result["environment_state_keys"] = {
            "passed": True,
            "sha256": hashlib.sha256(
                np.ascontiguousarray(expected_state_keys_host).tobytes()
            ).hexdigest(),
        }
        if expected_initial_state_sha256 is not None:
            result["explicit_initial_state"]["environment_state_keys_sha256"] = result[
                "environment_state_keys"
            ]["sha256"]
    return result


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
            raise ValueError(f"condition {condition!r} appears in multiple families")
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
            [condition_stats[condition]["mean"] for condition in family_conditions],
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
            "worst_condition_completion": float(condition_stats[family_worst]["mean"]),
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
    """Return a historical diagnostic screen, not an automatic promotion verdict."""
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
        reference_graded.get("available") and candidate_graded.get("available")
    )
    exact_map_gain = int(candidate_overall["successes"]) - int(
        reference_overall["successes"]
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
    integrity_passed = reference_integrity_passed and candidate_integrity_passed
    return {
        "schema": "terra_fixed_bank_comparison_gate_v1",
        "decision_authority": "advisory_diagnostics_only",
        "passed_is_promotion_decision": False,
        "promotion_requires": "exact-success gain plus failure-mechanism review",
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


def validate_progress_diagnostics(
    material: dict[str, np.ndarray],
    stall_age: dict[str, np.ndarray],
    episode_lengths: np.ndarray,
) -> None:
    """Fail closed on malformed per-map material and stall diagnostics."""
    lengths = np.asarray(episode_lengths)
    if lengths.ndim != 1 or not np.all(np.isfinite(lengths)):
        raise RuntimeError("episode lengths must be a finite one-dimensional array")
    count = len(lengths)

    def require(name: str, source: dict[str, np.ndarray]) -> np.ndarray:
        if name not in source:
            raise RuntimeError(f"fixed evaluation omitted diagnostic {name}")
        value = np.asarray(source[name])
        if value.shape != (count,):
            raise RuntimeError(
                f"fixed evaluation diagnostic {name} has shape {value.shape}, "
                f"expected {(count,)}"
            )
        return value

    supported = require("material_progress_supported", material).astype(bool)
    measured = require("loaded_soil_measured", material).astype(bool)
    if not bool(supported.all() and measured.all()):
        raise RuntimeError(
            "fixed evaluation requires supported material progress and measured load"
        )

    source_volume = require("source_soil_volume", material).astype(np.float64)
    fraction_names = (
        "dig_fraction",
        "terminal_soil_fraction",
        "off_zone_staged_soil_fraction",
        "loaded_soil_fraction",
    )
    fractions = {
        name: require(name, material).astype(np.float64) for name in fraction_names
    }
    numeric = [source_volume, *fractions.values()]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise RuntimeError("fixed evaluation returned nonfinite material progress")
    if np.any(source_volume <= 0):
        raise RuntimeError("fixed evaluation returned a nonpositive source-soil volume")
    tolerance = 1e-5
    if any(
        np.any((value < -tolerance) | (value > 1.0 + tolerance))
        for value in fractions.values()
    ):
        raise RuntimeError("fixed evaluation material fractions fall outside [0, 1]")
    partition_error = np.abs(
        fractions["terminal_soil_fraction"]
        + fractions["off_zone_staged_soil_fraction"]
        + fractions["loaded_soil_fraction"]
        - fractions["dig_fraction"]
    )
    if np.any(partition_error > tolerance):
        raise RuntimeError("fixed evaluation material partition does not conserve soil")

    available = require("stall_age_available", stall_age).astype(bool)
    stall_numeric_names = (
        "stall_age_decision_mean",
        "maximum_stall_age",
        "stall_age_saturated_decision_fraction",
    )
    stall_numeric = {
        name: require(name, stall_age).astype(np.float64)
        for name in stall_numeric_names
    }
    raw_saturation_count = require(
        "stall_age_saturated_decision_count", stall_age
    ).astype(np.float64)
    if not np.all(np.isfinite(raw_saturation_count)) or np.any(
        raw_saturation_count != np.floor(raw_saturation_count)
    ):
        raise RuntimeError("fixed evaluation stall saturation count is not integral")
    saturation_count = raw_saturation_count.astype(np.int64)
    if not all(np.all(np.isfinite(value)) for value in stall_numeric.values()):
        raise RuntimeError("fixed evaluation returned nonfinite stall-age diagnostics")
    if any(
        np.any((value < -tolerance) | (value > 1.0 + tolerance))
        for value in stall_numeric.values()
    ):
        raise RuntimeError("fixed evaluation normalized stall age falls outside [0, 1]")
    if np.any(
        stall_numeric["stall_age_decision_mean"]
        > stall_numeric["maximum_stall_age"] + tolerance
    ):
        raise RuntimeError("fixed evaluation mean stall age exceeds its maximum")
    if np.any(saturation_count < 0) or np.any(saturation_count > lengths):
        raise RuntimeError(
            "fixed evaluation returned an invalid stall saturation count"
        )
    expected_fraction = saturation_count / np.maximum(lengths, 1)
    if np.any(
        available
        & (
            np.abs(
                stall_numeric["stall_age_saturated_decision_fraction"]
                - expected_fraction
            )
            > tolerance
        )
    ):
        raise RuntimeError("fixed evaluation stall saturation fraction is inconsistent")


def grouped_results(
    rows: list[dict],
    successes: np.ndarray,
    terminations: np.ndarray,
    lengths: np.ndarray,
    *,
    horizon: int | None = None,
    completion_metrics: dict[str, np.ndarray] | None = None,
    integrity_metrics: dict[str, np.ndarray] | None = None,
    productive_workspace_cycles: np.ndarray | None = None,
    productive_workspace_cycles_available: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    completion_metrics = completion_metrics or {}
    integrity_metrics = integrity_metrics or {}
    count = len(rows)
    if productive_workspace_cycles is None:
        productive_workspace_cycles = np.full(count, -1, dtype=np.int32)
    if productive_workspace_cycles_available is None:
        productive_workspace_cycles_available = np.zeros(count, dtype=bool)
    productive_workspace_cycles = np.asarray(
        productive_workspace_cycles, dtype=np.int32
    )
    productive_workspace_cycles_available = np.asarray(
        productive_workspace_cycles_available, dtype=bool
    )
    if productive_workspace_cycles.shape != (count,):
        raise ValueError("productive_workspace_cycles must have one value per map")
    if productive_workspace_cycles_available.shape != (count,):
        raise ValueError(
            "productive_workspace_cycles_available must have one value per map"
        )
    per_map = []
    for index, (row, success, terminated, length) in enumerate(
        zip(rows, successes, terminations, lengths)
    ):
        timed_out = bool(
            terminated and horizon is not None and int(length) >= int(horizon)
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
            key: np.asarray(values)[index].item()
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
                "productive_workspace_cycles": (
                    int(productive_workspace_cycles[index])
                    if productive_workspace_cycles_available[index]
                    else None
                ),
                "productive_workspace_cycles_available": bool(
                    productive_workspace_cycles_available[index]
                ),
                **metric_values,
                **integrity_values,
                "integrity_failure": integrity_failure,
            }
        )

    def successful_efficiency(selected: list[dict]) -> dict:
        successful = [row for row in selected if row["success"]]
        cycles_available = bool(successful) and all(
            row["productive_workspace_cycles_available"] for row in successful
        )
        cycles_total = (
            sum(int(row["productive_workspace_cycles"]) for row in successful)
            if cycles_available
            else None
        )
        steps_total = sum(int(row["steps"]) for row in successful)
        return {
            "objective_order": [
                "successes_desc",
                "productive_workspace_cycles_on_successes_asc",
                "steps_on_successes_asc",
            ],
            "uses_reward_return": False,
            "successful_episodes": len(successful),
            "productive_workspace_cycles_available": cycles_available,
            "productive_workspace_cycles_total": cycles_total,
            "productive_workspace_cycles_mean": (
                cycles_total / len(successful) if cycles_available else None
            ),
            "steps_total": steps_total,
            "steps_mean": steps_total / len(successful) if successful else None,
            "lexicographic_key": [
                len(successful),
                -cycles_total if cycles_available else None,
                -steps_total,
            ],
        }

    def carry_work_summary(selected: list[dict]) -> dict:
        if not selected or "terminal_carry_work_normalized" not in selected[0]:
            return {"available": False}
        terminal = np.asarray(
            [row["terminal_carry_work_normalized"] for row in selected],
            dtype=np.float64,
        )
        maximum = np.asarray(
            [row["maximum_carry_work_normalized"] for row in selected],
            dtype=np.float64,
        )
        return {
            "available": True,
            "terminal_mean": float(terminal.mean()),
            "terminal_max": float(terminal.max()),
            "trajectory_max": float(maximum.max()),
        }

    def material_progress_summary(selected: list[dict]) -> dict:
        fields = (
            "dig_fraction",
            "terminal_soil_fraction",
            "off_zone_staged_soil_fraction",
            "loaded_soil_fraction",
        )
        if not selected or any(field not in selected[0] for field in fields):
            return {"available": False}
        available = np.asarray(
            [bool(row.get("material_progress_supported", False)) for row in selected],
            dtype=bool,
        )
        measured_load = np.asarray(
            [bool(row.get("loaded_soil_measured", False)) for row in selected],
            dtype=bool,
        )
        if not bool(available.all() and measured_load.all()):
            return {
                "available": False,
                "available_episodes": int(available.sum()),
                "measured_load_episodes": int(measured_load.sum()),
                "episodes": len(selected),
            }
        values = {
            field: np.asarray([row[field] for row in selected], dtype=np.float64)
            for field in fields
        }
        partition_error = np.abs(
            values["terminal_soil_fraction"]
            + values["off_zone_staged_soil_fraction"]
            + values["loaded_soil_fraction"]
            - values["dig_fraction"]
        )
        return {
            "available": True,
            "source_soil_volume_mean": float(
                np.mean([row["source_soil_volume"] for row in selected])
            ),
            **{f"{field}_mean": float(value.mean()) for field, value in values.items()},
            "maximum_partition_error": float(partition_error.max()),
        }

    def stall_age_summary(selected: list[dict]) -> dict:
        if not selected or "stall_age_available" not in selected[0]:
            return {"available": False}
        available_rows = [
            row for row in selected if bool(row.get("stall_age_available", False))
        ]
        if not available_rows:
            return {
                "available": False,
                "available_episodes": 0,
                "episodes": len(selected),
            }
        return {
            "available": len(available_rows) == len(selected),
            "available_episodes": len(available_rows),
            "episodes": len(selected),
            "decision_mean": float(
                np.mean([row["stall_age_decision_mean"] for row in available_rows])
            ),
            "maximum": float(
                np.max([row["maximum_stall_age"] for row in available_rows])
            ),
            "saturated_decisions": int(
                np.sum(
                    [
                        row["stall_age_saturated_decision_count"]
                        for row in available_rows
                    ]
                )
            ),
            "episodes_with_saturation": int(
                np.sum(
                    [
                        row["stall_age_saturated_decision_count"] > 0
                        for row in available_rows
                    ]
                )
            ),
            "saturated_decision_fraction_mean": float(
                np.mean(
                    [
                        row["stall_age_saturated_decision_fraction"]
                        for row in available_rows
                    ]
                )
            ),
        }

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
                "successful_efficiency": successful_efficiency(selected),
                "carry_work": carry_work_summary(selected),
                "material_progress": material_progress_summary(selected),
                "stall_age": stall_age_summary(selected),
            }
        return result

    total_successes = int(successes.sum())
    summary = {
        "overall": {
            "successes": total_successes,
            "episodes": len(rows),
            "success_rate": total_successes / len(rows),
            "terminations": int(terminations.sum()),
            "successful_efficiency": successful_efficiency(per_map),
            "carry_work": carry_work_summary(per_map),
            "material_progress": material_progress_summary(per_map),
            "stall_age": stall_age_summary(per_map),
        },
        "by_family": summarize("family"),
        "by_primary_cell": summarize("primary_cell"),
    }
    integrity_failure_count = sum(int(row["integrity_failure"]) for row in per_map)
    summary["integrity"] = {
        "passed": integrity_failure_count == 0,
        "failure_count": integrity_failure_count,
        "mass_residual_failures": sum(
            int(int(row.get("maximum_mass_residual", 0)) != 0) for row in per_map
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
            int(bool(row.get("termination_disagreement", False))) for row in per_map
        ),
        "slot_index_disagreements": sum(
            int(bool(row.get("slot_index_disagreement", False))) for row in per_map
        ),
        "unavailable": sum(
            int(bool(row.get("integrity_unavailable", False))) for row in per_map
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
        reason: sum(int(row["termination_reason"] == reason) for row in per_map)
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
    if any("next_update" not in checkpoint for _, checkpoint in checkpoints):
        raise ValueError("checkpoint must declare next_update")
    updates = [int(checkpoint["next_update"]) for _, checkpoint in checkpoints]
    if any(update < 0 for update in updates):
        raise ValueError("checkpoint next_update must be a nonnegative integer")
    if any(current >= following for current, following in zip(updates, updates[1:])):
        raise ValueError(
            "fixed-bank checkpoint updates must be strictly increasing and unique"
        )
    treatment_fingerprints = [
        checkpoint_treatment_fingerprint(checkpoint) for _, checkpoint in checkpoints
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
        "--diagnostic-panel",
        choices=("promotion", "development", "sealed"),
        help=(
            "Evaluate one explicitly non-admitted diagnostic control panel. "
            "This preserves frozen reset seeds but keeps the scores out of the "
            "accepted-bank macro."
        ),
    )
    parser.add_argument(
        "--capability-panel",
        choices=("promotion", "development", "sealed"),
        help=(
            "Evaluate the named release's separate capability-floor panel. "
            "These scores never enter the main benchmark macro."
        ),
    )
    parser.add_argument(
        "--explicit-episode-panel",
        choices=("train", "promotion", "development", "sealed"),
        help=(
            "Evaluate one diagnostic explicit-episode panel. Each episode "
            "uses its frozen map, complete initial Agent state, environment "
            "reset seed, and protocol hash. Incompatible with --strata and "
            "the accepted/diagnostic map-panel options."
        ),
    )
    parser.add_argument(
        "--terra-revision",
        help=(
            "Exact immutable Terra revision bound into an accepted bank. "
            "Required with --accepted-panel; no Git metadata is consulted."
        ),
    )
    parser.add_argument(
        "--gate-v1",
        action="store_true",
        help=(
            "Evaluate under the v1 gate semantics (perpendicular standoff band "
            "enforced). Required to evaluate checkpoints TRAINED under v1 whose "
            "config predates the selector, e.g. the C0/T1 pilot; without it a "
            "pre-v2 checkpoint is silently evaluated under v2 (yaw-only)."
        ),
    )
    parser.add_argument(
        "--panel-family",
        default=EVALUATION_PANEL_FAMILY_DEFAULT,
        help=(
            "Evaluate 'evaluation/<family>/<panel>' instead of the declared "
            "'evaluation/main/<panel>'. Use 'gate_main' on the fresh-trench "
            "finite-metadata enriched bank, whose root dataset.json still "
            "declares the frozen main panels on purpose. Only affects "
            "--accepted-panel/--diagnostic-panel."
        ),
    )
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--expect-completion-contract",
        choices=("exact_visible_dump_v1", LEGACY_COMPLETION_CONTRACT),
    )
    parser.add_argument(
        "--require-productive-workspace-cycles",
        action="store_true",
        help=(
            "Fail unless every evaluated terminal episode exposes Terra's raw "
            "productive_workspace_cycles counter."
        ),
    )
    parser.add_argument("--expect-reward-protocol-id")
    parser.add_argument("--expect-distance-protocol-id")
    parser.add_argument("--expect-distance-sidecar-sha256")
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

    selected_panel_modes = sum(
        option is not None
        for option in (
            args.accepted_panel,
            args.diagnostic_panel,
            args.capability_panel,
            args.explicit_episode_panel,
        )
    )
    if selected_panel_modes > 1:
        raise ValueError(
            "choose only one of --accepted-panel, --diagnostic-panel, "
            "--capability-panel, and --explicit-episode-panel"
        )
    panel_name = args.accepted_panel or args.diagnostic_panel or args.capability_panel
    if args.panel_family != EVALUATION_PANEL_FAMILY_DEFAULT and panel_name is None:
        raise ValueError(
            "--panel-family requires --accepted-panel or --diagnostic-panel"
        )
    accepted_bank = None
    explicit_episode_panel = None
    if args.explicit_episode_panel is not None:
        if args.strata is not None:
            raise ValueError("explicit episode panels are incompatible with --strata")
        if args.terra_revision is None:
            raise ValueError("explicit episode panels require --terra-revision")
        explicit_episode_panel = load_explicit_episode_panel(
            bank_root,
            args.explicit_episode_panel,
            args.terra_revision,
        )
        targets = [
            (
                args.explicit_episode_panel,
                "legacy_easy_capability_floor",
                explicit_episode_panel.maps_path,
            )
        ]
    elif panel_name is not None:
        if args.strata is not None:
            raise ValueError("fixed manifest panels are incompatible with --strata")
        if args.terra_revision is None:
            raise ValueError("fixed manifest panels require --terra-revision")
        release_id = json.loads((bank_root / "dataset.json").read_text()).get(
            "release_id"
        )
        if args.capability_panel is not None and (
            args.panel_family != EVALUATION_PANEL_FAMILY_DEFAULT
        ):
            raise ValueError(
                "--panel-family applies to the main evaluation panels, not the "
                "capability-floor panels"
            )
        accepted_bank = load_accepted_bank(
            bank_root,
            "G-UNIFORM",
            args.terra_revision,
            allow_diagnostic_control=args.diagnostic_panel is not None,
            curriculum_stage="full" if release_id == V8_RELEASE_ID else None,
            evaluation_panel_family=args.panel_family,
        )
        available_panels = (
            accepted_bank.capability_floor_evaluation_panels
            if args.capability_panel is not None
            else accepted_bank.evaluation_panels
        )
        panel = next(panel for panel in available_panels if panel.name == panel_name)
        targets = [
            (
                panel_name,
                "capability" if args.capability_panel is not None else "all",
                panel.maps_path,
            )
        ]
    else:
        if args.terra_revision is not None:
            raise ValueError("--terra-revision requires a fixed manifest panel option")
        strata = args.strata or ("M0", "M1", "M2")
        targets = [
            (args.split, stratum, f"{args.split}/{stratum}") for stratum in strata
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
    expected_protocol = {
        "reward_protocol_id": args.expect_reward_protocol_id,
        "distance_protocol_id": args.expect_distance_protocol_id,
        "distance_sidecar_sha256": args.expect_distance_sidecar_sha256,
    }
    if any(value is not None for value in expected_protocol.values()):
        if any(value is None for value in expected_protocol.values()):
            raise ValueError("all three R2 protocol expectations are required together")
        for path, checkpoint in checkpoints:
            receipt = checkpoint.get("r2_protocol_receipt")
            if not isinstance(receipt, dict):
                raise ValueError(f"{path}: missing R2 reward protocol receipt")
            observed = {key: receipt.get(key) for key in expected_protocol}
            if observed != expected_protocol:
                raise ValueError(f"{path}: R2 reward protocol mismatch: {observed!r}")
    reference_train_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        if "model" not in checkpoint:
            raise KeyError("checkpoint has no model parameters")
        _validate_checkpoint_architecture(checkpoint, reference_train_config)

    records = []
    for split_name, stratum, relative_path in targets:
        directory = bank_root / relative_path
        rows = (
            list(explicit_episode_panel.manifest_rows)
            if explicit_episode_panel is not None
            else load_manifest(directory)
        )
        count = len(rows)
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(count)
        config = configure_for_bank(reference_train_config, relative_path, count)
        # Gate semantics: the evaluator rebuilds the env from train_config, not
        # from the checkpoint's env_config, so a checkpoint TRAINED under the v1
        # band but lacking the selector would otherwise be evaluated under v2.
        _ckpt_standoff = getattr(config, "trench_dig_standoff_enforced", None)
        if args.gate_v1:
            config.trench_dig_standoff_enforced = True
        if (
            bool(getattr(config, "enforce_trench_dig_alignment", None))
            and _ckpt_standoff is None
        ):
            print(
                "⚠️  gate-on checkpoint predates the v2 semantics selector; "
                + (
                    "evaluating under v1 (--gate-v1)."
                    if args.gate_v1
                    else "evaluating under v2 (yaw-only). Pass --gate-v1 to "
                    "replay it as trained."
                ),
                flush=True,
            )
        # The eval config is a rewrite of the checkpoint's own train_config, so
        # re-run the architecture contract against what actually builds the
        # model: any field the rewrite dropped fails here instead of silently
        # evaluating a different network.
        for _, checkpoint in checkpoints:
            _validate_checkpoint_architecture(checkpoint, config)
        if explicit_episode_panel is None:
            env_config_override = None
        else:
            from terra.benchmark_protocol import frozen_benchmark_protocol

            env_config_override, _ = frozen_benchmark_protocol()
            if bool(getattr(config, "enforce_trench_dig_alignment", None)):
                raise ValueError(
                    "explicit-episode panels replace the env config with the "
                    "frozen benchmark protocol, which cannot carry this "
                    "checkpoint's enforce_trench_dig_alignment=True treatment; "
                    "use --accepted-panel on a gate-capable panel family"
                )
        _, env, env_params, initialized_state = make_mixed_agent_states(
            config,
            env_params=env_config_override,
        )
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        expected_trench_gate = bool(
            getattr(config, "enforce_trench_dig_alignment", None) or False
        )
        raw_trench_gate = getattr(env_params, "enforce_trench_dig_alignment", None)
        if raw_trench_gate is None:
            if expected_trench_gate:
                raise RuntimeError(
                    "the checkpoint was trained with enforce_trench_dig_alignment"
                    "=True but this Terra runtime has no fresh-trench dig gate"
                )
        else:
            effective_trench_gate = bool(np.ravel(np.asarray(raw_trench_gate))[0])
            if effective_trench_gate != expected_trench_gate:
                raise RuntimeError(
                    "fresh-trench dig gate mismatch between the checkpoint "
                    f"config ({expected_trench_gate}) and the eval env "
                    f"({effective_trench_gate})"
                )
            print(
                "🧭 eval env fresh-trench dig gate (effective): "
                f"{effective_trench_gate}",
                flush=True,
            )
            _eff_standoff = getattr(env_params, "trench_dig_standoff_enforced", None)
            if _eff_standoff is not None:
                _eff_standoff = bool(np.ravel(np.asarray(_eff_standoff))[0])
                _want_standoff = getattr(config, "trench_dig_standoff_enforced", None)
                if _want_standoff is not None and bool(_want_standoff) != _eff_standoff:
                    raise RuntimeError(
                        "gate semantics mismatch: requested "
                        f"standoff_enforced={bool(_want_standoff)} but the eval env "
                        f"has {_eff_standoff}"
                    )
                print(
                    "🧭 eval env gate semantics (effective): "
                    + ("v1 (standoff band enforced)" if _eff_standoff else "v2 (yaw-only)"),
                    flush=True,
                )
        if explicit_episode_panel is not None:
            selection_rows = [
                {
                    "slot_index": row["slot_index"],
                    "reset_seed": row["slot_selection_seed"],
                    "environment_protocol_sha256": row["environment_protocol_sha256"],
                }
                for row in rows
            ]
            map_reset_keys = manifest_reset_keys(
                selection_rows,
                count,
                explicit_episode_panel.environment_protocol_sha256,
            )
            initial_timestep, env_params, state_keys = prepare_explicit_episode_reset(
                env,
                env_params,
                map_reset_keys,
                explicit_episode_panel,
            )
            reset_keys = None
            reset_verification = verify_exact_reset(
                env,
                env_params,
                None,
                directory,
                count,
                timestep=initial_timestep,
                expected_initial_state_sha256=(
                    explicit_episode_panel.initial_agent_state_sha256
                ),
                expected_state_keys=state_keys,
            )
        elif accepted_bank is not None:
            map_reset_keys = exact_reset_keys(count)
            state_keys = manifest_environment_keys(
                rows,
                count,
                accepted_bank.environment_protocol_sha256,
            )
            initial_timestep, env_params, state_keys = prepare_manifest_episode_reset(
                env,
                env_params,
                map_reset_keys,
                state_keys,
            )
            reset_keys = None
            reset_verification = verify_exact_reset(
                env,
                env_params,
                None,
                directory,
                count,
                timestep=initial_timestep,
            )
            reset_verification["manifest_episode_seeds"] = {
                "passed": True,
                "map_selection_decoupled": True,
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(np.asarray(state_keys)).tobytes()
                ).hexdigest(),
            }
        else:
            initial_timestep = None
            reset_keys = exact_reset_keys(count)
            reset_verification = verify_exact_reset(
                env,
                env_params,
                reset_keys,
                directory,
                count,
            )
        model = SimpleNamespace(apply=initialized_state.apply_fn)
        for checkpoint_path, checkpoint in checkpoints:
            validate_model_params_match(
                initialized_state.params,
                checkpoint["model"],
                f"{checkpoint_path}",
            )
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
                initial_timestep=initial_timestep,
            )
            successes = np.asarray(stats["episode_done_once"], dtype=bool)
            terminations = np.asarray(stats["episode_terminated_once"], dtype=bool)
            lengths = np.asarray(stats["episode_length"], dtype=np.int32)
            workspace_cycles = np.asarray(
                stats.get(
                    "productive_workspace_cycles",
                    np.full(count, -1, dtype=np.int32),
                ),
                dtype=np.int32,
            )
            workspace_cycles_available = np.asarray(
                stats.get(
                    "productive_workspace_cycles_available",
                    np.zeros(count, dtype=bool),
                ),
                dtype=bool,
            )
            if workspace_cycles.shape != (count,) or (
                workspace_cycles_available.shape != (count,)
            ):
                raise RuntimeError(
                    "fixed evaluation returned invalid productive workspace arrays"
                )
            if args.require_productive_workspace_cycles and not np.all(
                workspace_cycles_available
            ):
                missing = (np.flatnonzero(~workspace_cycles_available) + 1).tolist()
                raise RuntimeError(
                    "productive_workspace_cycles unavailable at fixed-bank slots "
                    f"{missing}"
                )
            terminal_completion = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.get("terminal_completion", {}).items()
            }
            carry_work = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.get("carry_work", {}).items()
            }
            require_carry_work = bool(_field(config, "carry_work_observation", False))
            if require_carry_work:
                for key in ("terminal_normalized", "maximum_normalized"):
                    if key not in carry_work or carry_work[key].shape != (count,):
                        raise RuntimeError(
                            f"fixed evaluation lacks R2 carry diagnostic {key!r}"
                        )
            carry_metrics = (
                {
                    "terminal_carry_work_normalized": carry_work["terminal_normalized"],
                    "maximum_carry_work_normalized": carry_work["maximum_normalized"],
                }
                if require_carry_work
                else {}
            )
            if require_carry_work and np.any(
                carry_work["terminal_normalized"][successes] > 1e-6
            ):
                raise RuntimeError("successful R2 episode retained carry work")
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
            raw_material_progress = stats.get("material_progress", {})
            material_progress_supported = np.asarray(
                raw_material_progress.get("supported", np.zeros(count, dtype=bool)),
                dtype=bool,
            )
            if material_progress_supported.shape != (count,) or not bool(
                material_progress_supported.all()
            ):
                raise RuntimeError(
                    "fixed evaluation did not return a supported per-map material "
                    "progress ledger"
                )
            loaded_soil_measured = np.asarray(
                raw_material_progress.get(
                    "loaded_soil_measured", np.zeros(count, dtype=bool)
                ),
                dtype=bool,
            )
            if loaded_soil_measured.shape != (count,) or not bool(
                loaded_soil_measured.all()
            ):
                raise RuntimeError(
                    "fixed evaluation did not measure per-map terminal carried soil"
                )
            material_progress_metrics = {
                "material_progress_supported": material_progress_supported,
                "loaded_soil_measured": loaded_soil_measured,
                **{
                    key: np.asarray(raw_material_progress[key])
                    for key in (
                        "source_soil_volume",
                        "dig_fraction",
                        "terminal_soil_fraction",
                        "off_zone_staged_soil_fraction",
                        "loaded_soil_fraction",
                    )
                },
            }
            raw_stall_age = stats.get("stall_age", {})
            stall_age_metrics = {
                "stall_age_available": np.asarray(
                    raw_stall_age.get("available", np.zeros(count, dtype=bool)),
                    dtype=bool,
                ),
                "stall_age_decision_mean": np.asarray(
                    raw_stall_age.get("decision_mean", np.zeros(count)),
                    dtype=np.float32,
                ),
                "maximum_stall_age": np.asarray(
                    raw_stall_age.get("maximum", np.zeros(count)),
                    dtype=np.float32,
                ),
                "stall_age_saturated_decision_count": np.asarray(
                    raw_stall_age.get(
                        "saturated_decision_count", np.zeros(count, dtype=np.int32)
                    ),
                    dtype=np.int32,
                ),
                "stall_age_saturated_decision_fraction": np.asarray(
                    raw_stall_age.get("saturated_decision_fraction", np.zeros(count)),
                    dtype=np.float32,
                ),
            }
            if (
                not np.all(np.isfinite(lengths))
                or successes.shape != (count,)
                or terminations.shape != (count,)
            ):
                raise RuntimeError("fixed evaluation returned invalid arrays")
            validate_progress_diagnostics(
                material_progress_metrics,
                stall_age_metrics,
                lengths,
            )
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
                }
                | carry_metrics
                | material_progress_metrics
                | stall_age_metrics,
                integrity_metrics=integrity_metrics,
                productive_workspace_cycles=workspace_cycles,
                productive_workspace_cycles_available=workspace_cycles_available,
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
                "material_progress_contract": {
                    "name": "source_soil_partition_v1",
                    "denominator": "required negative-target volume",
                    "loaded_soil_source": "measured preserved terminal agent state",
                    "identity": (
                        "terminal_soil_fraction + off_zone_staged_soil_fraction + "
                        "loaded_soil_fraction == dig_fraction"
                    ),
                },
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_update": int(checkpoint.get("next_update", 0)),
                "treatment_fingerprint": reference_treatment,
                "r2_protocol_receipt": checkpoints[0][1].get("r2_protocol_receipt"),
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
                        "diagnostic_control": (
                            accepted_bank.diagnostic_contract_sha256 is not None
                        ),
                        "diagnostic_contract_sha256": (
                            accepted_bank.diagnostic_contract_sha256
                        ),
                        # Conditional: recording it unconditionally would change
                        # every historical record's bank identity dict, which
                        # downstream tooling compares verbatim.
                        **(
                            {}
                            if accepted_bank.evaluation_panel_family
                            == EVALUATION_PANEL_FAMILY_DEFAULT
                            else {
                                "evaluation_panel_family": (
                                    accepted_bank.evaluation_panel_family
                                )
                            }
                        ),
                    }
                ),
                "split": split_name,
                "stratum": stratum,
                "manifest": str(directory / "manifest.jsonl"),
                "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                "horizon": args.horizon,
                "deterministic": not args.stochastic,
                "policy_mode": ("sampled" if args.stochastic else "deterministic"),
                "seed": args.seed,
                "exact_manifest_enumeration": True,
                "reset_verification": reset_verification,
                "summary": summary,
                "comparison_to_previous": comparison_to_previous,
                "per_map": per_map,
            }
            if explicit_episode_panel is not None:
                record["explicit_episode_bank"] = explicit_episode_panel.receipt()
            records.append(record)
            with output.open("w") as handle:
                json.dump(records, handle, indent=2, sort_keys=True, allow_nan=False)
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
