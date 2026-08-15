#!/usr/bin/env python3
"""Add zero stall-age embeddings and Adam moments to the v6.1 u14k checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from terra.config import BatchConfig, MapsDimsConfig

from scripts.grow_checkpoint import _derive_action_type
from utils import helpers
from utils.models import get_model_ready
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

SCHEMA = "terra_v8_v61_stall_age_v3_prepared_v1"
SOURCE_SHA256 = "79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609"
SOURCE_UPDATE = 14_000
SOURCE_PARAMETER_COUNT = 2_303_421
FUSED_WIDTH = 704
TARGET_PARAMETER_COUNT = SOURCE_PARAMETER_COUNT + 2 * FUSED_WIDTH
PARAMETER_NAMES = (
    "stall_age_actor_embedding",
    "stall_age_critic_embedding",
)
SOURCE_SAMPLER_RULE = "continuous_banded_v2"
TARGET_SAMPLER_RULE = "continuous_banded_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def set_config_value(config, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def parameter_count(params) -> int:
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree.leaves(params))


def add_zero_embeddings(model):
    mutable = (
        model.unfreeze() if isinstance(model, FrozenDict) else copy.deepcopy(model)
    )
    params = mutable["params"]
    if any(name in params for name in PARAMETER_NAMES):
        raise ValueError("source model already contains stall-age embeddings")
    for name in PARAMETER_NAMES:
        params[name] = jnp.zeros((FUSED_WIDTH,), dtype=jnp.float32)
    return FrozenDict(mutable) if isinstance(model, FrozenDict) else mutable


def add_zero_adam_moments(optimizer_state):
    if len(optimizer_state) != 2 or len(optimizer_state[1]) != 2:
        raise ValueError("unexpected optimizer state structure")
    adam = optimizer_state[1][0]
    mu = copy.deepcopy(adam.mu)
    nu = copy.deepcopy(adam.nu)
    for moments in (mu, nu):
        params = moments["params"]
        if any(name in params for name in PARAMETER_NAMES):
            raise ValueError("optimizer already contains stall-age moments")
        for name in PARAMETER_NAMES:
            params[name] = jnp.zeros((FUSED_WIDTH,), dtype=jnp.float32)
    grown_adam = adam._replace(mu=mu, nu=nu)
    return (optimizer_state[0], (grown_adam, optimizer_state[1][1]))


def migrate_sampler_state(source_state: dict) -> tuple[dict, dict]:
    """Move the u14 sampler to family-free v3 and discard its partial window."""
    source_settings = dict(source_state.get("settings", {}))
    if source_settings.get("rule") != SOURCE_SAMPLER_RULE:
        raise ValueError("source sampler is not continuous_banded_v2")
    if source_state["current_window"].get("updates") != 50:
        raise ValueError(
            "source sampler does not contain the expected 50-update window"
        )
    sampler = PooledConditionSampler(
        list(source_state["conditions"]),
        SamplerSettings(**{**source_settings, "rule": TARGET_SAMPLER_RULE}),
        maps_per_condition=list(source_state["maps_per_condition"]),
        labels=copy.deepcopy(source_state["labels"]),
    )
    sampler.restore_state_dict(
        copy.deepcopy(source_state), clear_window_on_migration=True
    )
    migrated = sampler.state_dict()
    preserved = (
        "conditions",
        "maps_per_condition",
        "labels",
        "competence",
        "closed_window",
        "refresh",
        "numpy_rng",
        "mastery",
    )
    if any(migrated[key] != source_state[key] for key in preserved):
        raise ValueError("v3 migration changed retained sampler history")
    current = migrated["current_window"]
    if current["updates"] != 0 or any(
        any(current[key])
        for key in (
            "completed_episode_count",
            "task_done_count",
            "sampled_assignment_count",
            "reset_exposure_count",
            "transition_exposure_count",
        )
    ):
        raise ValueError("v3 migration did not clear the partial sampler window")
    probabilities = np.asarray(migrated["probabilities"], dtype=np.float64)
    mastered = np.asarray(migrated["mastery"]["mastered"], dtype=bool)
    receipt = {
        "source_rule": SOURCE_SAMPLER_RULE,
        "target_rule": TARGET_SAMPLER_RULE,
        "discarded_partial_window_updates": 50,
        "discarded_assignments": int(
            sum(source_state["current_window"]["sampled_assignment_count"])
        ),
        "discarded_completed_episodes": int(
            sum(source_state["current_window"]["completed_episode_count"])
        ),
        "discarded_reset_exposures": int(
            sum(source_state["current_window"]["reset_exposure_count"])
        ),
        "discarded_transition_exposures": int(
            sum(source_state["current_window"]["transition_exposure_count"])
        ),
        "discarded_task_done_count": int(
            sum(source_state["current_window"]["task_done_count"])
        ),
        "mastered_conditions": int(mastered.sum()),
        "open_mass": float(probabilities[~mastered].sum()),
        "mastered_replay_mass": float(probabilities[mastered].sum()),
        "max_condition_mass": float(probabilities.max()),
        "preserved_history": list(preserved),
        "next_refresh_update": 14_100,
    }
    return migrated, receipt


def parity_observation(config, *, stall_age: bool):
    batch = 3
    edge = 64
    angles = 12
    obs = [
        jnp.zeros((batch, 4, 9), dtype=jnp.float32),
        jnp.ones((batch, 4), dtype=jnp.int8),
        jnp.ones((batch,), dtype=jnp.int32),
    ]
    obs += [jnp.zeros((batch, angles), dtype=jnp.float32) for _ in range(9)]
    obs += [jnp.zeros((batch, edge, edge), dtype=jnp.float32) for _ in range(4)]
    obs += [jnp.zeros((batch,), dtype=jnp.int32) for _ in range(2)]
    obs += [jnp.zeros((batch, edge, edge), dtype=jnp.float32) for _ in range(3)]
    obs += [
        jnp.zeros(
            (batch, int(config_value(config, "num_prev_actions"))), dtype=jnp.int32
        )
    ]
    if stall_age:
        obs += [jnp.asarray([[0.0], [0.5], [1.0]], dtype=jnp.float32)]
    return obs


def validate_source(checkpoint: dict) -> None:
    if int(checkpoint.get("next_update", -1)) != SOURCE_UPDATE:
        raise ValueError("source checkpoint is not v6.1 update 14000")
    if parameter_count(checkpoint["model"]) != SOURCE_PARAMETER_COUNT:
        raise ValueError("source parameter count changed")
    if int(np.asarray(checkpoint["train_state_step"]).reshape(())) != 896_000:
        raise ValueError("source optimizer clock is not 896000")
    adam = checkpoint["optimizer_state"][1][0]
    if int(np.asarray(adam.count).reshape(())) != 896_000:
        raise ValueError("source Adam clock is not 896000")
    config = checkpoint["train_config"]
    expected = {
        "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
        "carry_work_observation": True,
        "action_logit_masking": False,
        "reward_stage": "reward_v2",
        "reward_v2_timing_variant": 0,
    }
    observed = {name: config_value(config, name) for name in expected}
    if observed != expected:
        raise ValueError(f"source v6.1 contract changed: {observed!r}")
    if config_value(config, "stall_age_observation", False):
        raise ValueError("source already consumes stall age")
    sampler = checkpoint.get("pooled_sampler_state")
    if sampler.get("settings", {}).get("rule") != SOURCE_SAMPLER_RULE:
        raise ValueError(f"source sampler must be {SOURCE_SAMPLER_RULE}")
    window = sampler.get("current_window", {})
    expected_window_totals = {
        "sampled_assignment_count": 102_400,
        "completed_episode_count": 13_725,
        "reset_exposure_count": 13_725,
        "transition_exposure_count": 3_276_800,
        "task_done_count": 8_810,
    }
    observed_window_totals = {
        key: int(sum(window.get(key, ()))) for key in expected_window_totals
    }
    if window.get("updates") != 50 or observed_window_totals != expected_window_totals:
        raise ValueError(
            "source sampler partial window changed: "
            f"updates={window.get('updates')!r}, totals={observed_window_totals!r}"
        )


def validate_prepared(source: dict, prepared: dict) -> None:
    receipt = prepared.get("stall_age_prepared_continuation")
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("prepared checkpoint receipt is missing")
    if receipt.get("source_checkpoint_sha256") != SOURCE_SHA256:
        raise ValueError("prepared checkpoint names the wrong source")
    if parameter_count(prepared["model"]) != TARGET_PARAMETER_COUNT:
        raise ValueError("prepared parameter count changed")
    if not config_value(prepared["train_config"], "stall_age_observation", False):
        raise ValueError("prepared checkpoint does not enable stall age")
    for field in ("next_update", "train_state_step"):
        source_value = source[field]
        prepared_value = prepared[field]
        if not np.array_equal(np.asarray(source_value), np.asarray(prepared_value)):
            raise ValueError(f"prepared checkpoint changed {field}")

    expected_sampler, expected_migration = migrate_sampler_state(
        source["pooled_sampler_state"]
    )
    if prepared["pooled_sampler_state"] != expected_sampler:
        raise ValueError("prepared checkpoint has the wrong v3 sampler migration")
    if receipt.get("sampler_migration") != expected_migration:
        raise ValueError("prepared checkpoint sampler migration receipt changed")
    prepared_config = prepared["train_config"]
    if config_value(prepared_config, "config_name") != "G-V8-CONTINUOUS-V3":
        raise ValueError("prepared checkpoint does not select the v3 preset")
    if config_value(prepared_config, "pooled_sampler", {}).get("rule") != (
        TARGET_SAMPLER_RULE
    ):
        raise ValueError("prepared checkpoint config does not select v3")
    if config_value(prepared_config, "accepted_bank").sampler_profile != (
        TARGET_SAMPLER_RULE
    ):
        raise ValueError("prepared checkpoint bank does not identify v3")

    source_model = (
        source["model"].unfreeze()
        if isinstance(source["model"], FrozenDict)
        else copy.deepcopy(source["model"])
    )
    prepared_model = (
        prepared["model"].unfreeze()
        if isinstance(prepared["model"], FrozenDict)
        else copy.deepcopy(prepared["model"])
    )
    if type(source["model"]) is not type(prepared["model"]):
        raise ValueError("prepared model changed container type")
    source_params = source_model["params"]
    prepared_params = prepared_model["params"]
    for name in PARAMETER_NAMES:
        embedding = np.asarray(prepared_params.pop(name))
        if embedding.shape != (FUSED_WIDTH,) or np.any(embedding != 0):
            raise ValueError(f"{name} is not a zero fused embedding")
    if (
        jax.tree_util.tree_all(
            jax.tree.map(
                lambda left, right: np.array_equal(np.asarray(left), np.asarray(right)),
                source_params,
                prepared_params,
            )
        )
        is not True
    ):
        raise ValueError("prepared checkpoint changed existing model parameters")

    source_adam = source["optimizer_state"][1][0]
    prepared_adam = prepared["optimizer_state"][1][0]
    if not (
        jax.tree.structure(prepared["model"])
        == jax.tree.structure(prepared_adam.mu)
        == jax.tree.structure(prepared_adam.nu)
    ):
        raise ValueError("prepared params and Adam moments have different trees")
    if not np.array_equal(
        np.asarray(source_adam.count), np.asarray(prepared_adam.count)
    ):
        raise ValueError("prepared checkpoint changed Adam clock")
    for field in ("mu", "nu"):
        source_moments = copy.deepcopy(getattr(source_adam, field))["params"]
        prepared_moments = copy.deepcopy(getattr(prepared_adam, field))["params"]
        for name in PARAMETER_NAMES:
            moment = np.asarray(prepared_moments.pop(name))
            if moment.shape != (FUSED_WIDTH,) or np.any(moment != 0):
                raise ValueError(f"{field}/{name} is not zero")
        if (
            jax.tree_util.tree_all(
                jax.tree.map(
                    lambda left, right: np.array_equal(
                        np.asarray(left), np.asarray(right)
                    ),
                    source_moments,
                    prepared_moments,
                )
            )
            is not True
        ):
            raise ValueError(f"prepared checkpoint changed existing Adam {field}")

    env = type("ModelEnv", (), {})()
    env.batch_cfg = BatchConfig(
        action_type=_derive_action_type(source),
        maps_dims=MapsDimsConfig(maps_edge_length=64),
    )
    source_config = copy.deepcopy(source["train_config"])
    set_config_value(source_config, "stall_age_observation", False)
    source_model, _ = get_model_ready(jax.random.PRNGKey(1), source_config, env)
    target_model, _ = get_model_ready(
        jax.random.PRNGKey(2), prepared["train_config"], env
    )
    source_value, source_logits = source_model.apply(
        source["model"], parity_observation(source_config, stall_age=False)
    )
    target_value, target_logits = target_model.apply(
        prepared["model"],
        parity_observation(prepared["train_config"], stall_age=True),
    )
    if not np.array_equal(np.asarray(source_value), np.asarray(target_value)):
        raise ValueError("zero stall-age embeddings changed source values")
    if not np.array_equal(np.asarray(source_logits), np.asarray(target_logits)):
        raise ValueError("zero stall-age embeddings changed source logits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("source checkpoint SHA-256 changed")

    helpers.register_checkpoint_config_classes()
    source = helpers.load_pkl_object(str(args.source))
    validate_source(source)
    if args.verify is not None:
        prepared = helpers.load_pkl_object(str(args.verify))
        validate_prepared(source, prepared)
        print(f"prepared_sha256={sha256_file(args.verify)}")
        return
    prepared = copy.deepcopy(source)
    prepared["model"] = add_zero_embeddings(source["model"])
    prepared["optimizer_state"] = add_zero_adam_moments(source["optimizer_state"])
    set_config_value(prepared["train_config"], "stall_age_observation", True)
    set_config_value(prepared["train_config"], "config_name", "G-V8-CONTINUOUS-V3")
    sampler_config = copy.deepcopy(
        config_value(prepared["train_config"], "pooled_sampler")
    )
    sampler_config["rule"] = TARGET_SAMPLER_RULE
    set_config_value(prepared["train_config"], "pooled_sampler", sampler_config)
    source_bank = config_value(prepared["train_config"], "accepted_bank")
    set_config_value(
        prepared["train_config"],
        "accepted_bank",
        replace(source_bank, sampler_profile=TARGET_SAMPLER_RULE),
    )
    prepared["pooled_sampler_state"], sampler_migration = migrate_sampler_state(
        source["pooled_sampler_state"]
    )
    prepared["stall_age_prepared_continuation"] = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_next_update": SOURCE_UPDATE,
        "source_parameter_count": SOURCE_PARAMETER_COUNT,
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "fused_width": FUSED_WIDTH,
        "zero_parameter_names": list(PARAMETER_NAMES),
        "existing_head_shapes_unchanged": True,
        "existing_model_parameters_preserved": True,
        "existing_adam_moments_preserved": True,
        "parameter_and_adam_trees_match": True,
        "optimizer_clock_preserved": True,
        "sampler_migration": sampler_migration,
        "z0_value_and_logits_exact": True,
        "source_contract": {
            "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
            "reward_stage": "reward_v2",
            "reward_v2_timing_variant": 0,
            "carry_work_observation": True,
            "action_logit_masking": False,
            "source_sampler_rule": SOURCE_SAMPLER_RULE,
            "target_sampler_rule": TARGET_SAMPLER_RULE,
        },
    }
    if parameter_count(prepared["model"]) != TARGET_PARAMETER_COUNT:
        raise ValueError("prepared parameter count changed")
    validate_prepared(source, prepared)
    helpers.save_pkl_object(prepared, str(args.output))
    print(f"prepared_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
