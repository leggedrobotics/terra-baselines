#!/usr/bin/env python3
"""Verify the source and prepared v6.1 stall-age/v3 continuation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state
from terra.config import (
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from utils import helpers
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

# The frozen source: v6.1 phase1's terminal periodic checkpoint.
SOURCE_UPDATE = 14_000
UPDATE_EPOCHS = 2
NUM_MINIBATCHES = 32
OPTIMIZER_STEPS_PER_UPDATE = UPDATE_EPOCHS * NUM_MINIBATCHES
PARAMETER_COUNT = 2_303_421
PREPARED_PARAMETER_COUNT = 2_304_829
CONDITION_COUNT = 47
# 14000 % 150 == 50: the source stopped 50 updates into a window.
SOURCE_WINDOW_UPDATES = 50
SOURCE_LAST_REFRESH_UPDATE = 13_950
# The direct continuation redistributes the same global env count over 8 GPUs.
PHASE2_NUM_DEVICES = 8
PHASE2_NUM_ENVS_PER_DEVICE = 256
PHASE2_TARGET_UPDATE = 40_000

ARCHITECTURE = {
    "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
    "token_mixer_residual_init_scale": 0.1,
    "flatten_reduce_channels": 32,
    "attn_latent_queries": 8,
    "resnet_blocks_per_stage": (2, 2, 3, 3),
    "resnet_stage_channels": (24, 48, 64, 96),
    "critic_hidden_dims": (512, 256),
    "aux_coef": 0,
    "vf_coef": 2.0,
    "action_logit_masking": False,
    "model_size": "medium",
    "model_core": "mlp",
    "encoder_compute_dtype": "bfloat16",
    "attention_compute_dtype": "float32",
}
CONTRACT = {
    "seed": 20260807,
    "num_steps": 32,
    "num_minibatches": NUM_MINIBATCHES,
    "update_epochs": UPDATE_EPOCHS,
    "lr": 3e-4,
    "gamma": float(REWARD_V2_POTENTIAL_GAMMA),
    "reward_stage": "reward_v2",
    "reward_v2_timing_variant": 0,
    "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
    "carry_work_observation": True,
    "ent_schedule_start": 0.15,
    "ent_schedule_end": 0.02,
    "ent_schedule_steps": 20_000,
    "use_value_clip": False,
    "flat_minibatch_shuffle": True,
    "warm_start_from": None,
    "teacher_checkpoint": None,
    **ARCHITECTURE,
}


def config_value(config: object, name: str, default: object = "<absent>") -> object:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def check_config(path: Path, config: object, expected: dict) -> None:
    for name, want in expected.items():
        observed = config_value(config, name)
        if isinstance(observed, list):
            observed = tuple(observed)
        if observed != want:
            raise ValueError(f"{path}: {name}={observed!r}, expected {want!r}")


def finite(tree: object, label: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{label}: non-finite leaf {index}")


def parameter_count(model: object) -> int:
    return sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(model))


def verify_legacy_source_sampler(state: object) -> dict:
    """Validate the one frozen v2 source without exposing v2 to training."""
    if not isinstance(state, dict):
        raise ValueError("source checkpoint lacks pooled_sampler_state")
    legacy_settings = state.get("settings")
    if not isinstance(legacy_settings, dict) or legacy_settings.get("rule") != (
        "continuous_banded_v2"
    ):
        raise ValueError("source checkpoint has the wrong legacy sampler rule")
    conditions = state.get("conditions")
    labels = state.get("labels")
    maps_per_condition = state.get("maps_per_condition")
    if not isinstance(conditions, list) or len(conditions) != CONDITION_COUNT:
        raise ValueError("source checkpoint must contain 47 sampler conditions")
    if not isinstance(labels, dict) or set(labels) != set(conditions):
        raise ValueError("source sampler labels do not match its conditions")

    current = PooledConditionSampler(
        conditions,
        SamplerSettings(**{**legacy_settings, "rule": "continuous_banded_v3"}),
        maps_per_condition=maps_per_condition,
        labels=labels,
    )
    probabilities = np.asarray(state.get("probabilities"), dtype=np.float64)
    mastered = np.asarray(state.get("mastery", {}).get("mastered"), dtype=bool)
    if probabilities.shape != (CONDITION_COUNT,) or mastered.shape != (
        CONDITION_COUNT,
    ):
        raise ValueError("source sampler arrays have the wrong shape")
    np.testing.assert_allclose(
        probabilities,
        current._continuous_distribution_v2(mastered),
        rtol=0.0,
        atol=1e-12,
    )
    if not np.all(probabilities > 0.0):
        raise ValueError("source sampler lost positive support")
    depths = [labels[name].get("curriculum_depth") for name in conditions]
    families = [labels[name].get("family") for name in conditions]
    return {
        "schema": "terra_continuous_banded_source_validation_v1",
        "passed": True,
        "condition_count": CONDITION_COUNT,
        "family_counts": {
            family: families.count(family) for family in ("foundation", "trench")
        },
        "depth_counts": {str(depth): depths.count(depth) for depth in (0, 1, 2)},
        "minimum_probability": float(probabilities.min()),
        "probability_sum": float(probabilities.sum()),
        "sampler_state_schema": state.get("schema"),
        "sampler_rule": legacy_settings["rule"],
    }


def check_reward_protocol(path: Path, checkpoint: dict, sidecar_sha: str) -> dict:
    protocol = checkpoint.get("r2_protocol_receipt")
    expected = {
        "schema": "terra_v8_r2_reward_protocol_v1",
        "reward_stage": "reward_v2",
        "reward_protocol_id": REWARD_V2_PROTOCOL_ID,
        "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
        "distance_sidecar_sha256": sidecar_sha,
        "distance_artifact_kind": "canonical_distance_sidecar_dataset_json",
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{path}: reward protocol receipt changed")
    constants = protocol.get("constants", {})
    if constants.get("potential_gamma") != float(REWARD_V2_POTENTIAL_GAMMA):
        raise ValueError(f"{path}: reward-v2 potential gamma changed")
    if constants.get("step_cost_total") != 1.0:
        raise ValueError(f"{path}: step cost is not the frozen reward_v2 pace")
    # A receipt written before the v2.1 selector existed carries no timing
    # fields at all; either way this must be baseline timing.
    timing = protocol.get("reward_v2_timing", "baseline")
    if timing != "baseline" or protocol.get("reward_v2_timing_variant", 0) != 0:
        raise ValueError(f"{path}: continuation must stay on baseline v2 timing")
    if float(constants.get("shaping_gamma", REWARD_V2_POTENTIAL_GAMMA)) != float(
        REWARD_V2_POTENTIAL_GAMMA
    ):
        raise ValueError(f"{path}: shaping gamma is not the frozen reward_v2 value")
    env_stage = np.asarray(checkpoint["env_config"].reward_stage)
    if not np.all(env_stage == int(RewardStage.REWARD_V2)):
        raise ValueError(f"{path}: environment reward selector changed")
    return protocol


def verify_source(path: Path, expected_sha: str, sidecar_sha: str) -> dict:
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha != expected_sha:
        raise ValueError(
            f"{path}: source checkpoint SHA-256 is {observed_sha}, "
            f"expected {expected_sha}"
        )
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("next_update") != SOURCE_UPDATE:
        raise ValueError(
            f"{path}: source next_update must be {SOURCE_UPDATE}, "
            f"got {checkpoint.get('next_update')}"
        )
    optimizer_step = int(np.asarray(checkpoint["train_state_step"]).reshape(()))
    expected_step = SOURCE_UPDATE * OPTIMIZER_STEPS_PER_UPDATE
    if optimizer_step != expected_step:
        raise ValueError(
            f"{path}: optimizer clock is {optimizer_step}, expected {expected_step}"
        )
    count = parameter_count(checkpoint["model"])
    if count != PARAMETER_COUNT:
        raise ValueError(f"{path}: {count} parameters, expected {PARAMETER_COUNT}")
    finite(checkpoint["model"], f"{path}.model")
    finite(checkpoint["optimizer_state"], f"{path}.optimizer_state")

    config = checkpoint["train_config"]
    check_config(
        path,
        config,
        {
            **CONTRACT,
            "distance_sidecar_sha256": sidecar_sha,
            "prepared_fork_from": None,
        },
    )
    if config_value(config, "stall_age_observation", False):
        raise ValueError(f"{path}: source already consumes stall age")
    # Phase1's per-device shape; phase2 redistributes the same 2,048 envs over
    # eight devices while preserving the global transitions per update.
    check_config(path, config, {"num_devices": 4, "num_envs_per_device": 512})
    if config_value(config, "config_name") != "G-V8-CONTINUOUS-V2":
        raise ValueError(f"{path}: source must be the v2 preset")
    protocol = check_reward_protocol(path, checkpoint, sidecar_sha)

    sampler = verify_legacy_source_sampler(checkpoint.get("pooled_sampler_state"))
    if sampler["sampler_rule"] != "continuous_banded_v2":
        raise ValueError(f"{path}: source sampler must be continuous_banded_v2")
    state = checkpoint["pooled_sampler_state"]
    refresh = state["refresh"]
    if refresh["last_refresh_update"] != SOURCE_LAST_REFRESH_UPDATE:
        raise ValueError(
            f"{path}: last sampler refresh is {refresh['last_refresh_update']}, "
            f"expected {SOURCE_LAST_REFRESH_UPDATE}"
        )
    if state["current_window"]["updates"] != SOURCE_WINDOW_UPDATES:
        raise ValueError(
            f"{path}: source window holds {state['current_window']['updates']} "
            f"updates, expected {SOURCE_WINDOW_UPDATES}"
        )
    if not refresh["has_closed_window"]:
        raise ValueError(f"{path}: source has no closed sampler window")
    probabilities = np.asarray(state["probabilities"], dtype=np.float64)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": observed_sha,
        "next_update": SOURCE_UPDATE,
        "optimizer_step": optimizer_step,
        "parameter_count": count,
        "sampler": sampler,
        "sampler_window_updates": SOURCE_WINDOW_UPDATES,
        "sampler_last_refresh_update": SOURCE_LAST_REFRESH_UPDATE,
        "sampler_refreshes": int(refresh["refreshes"]),
        "sampler_mastered": int(sum(state["mastery"]["mastered"])),
        "sampler_max_condition_mass": float(probabilities.max()),
        "reward_protocol": protocol,
    }


def accepted_sampler_profile(config: object) -> object:
    bank = config_value(config, "accepted_bank", None)
    return config_value(bank, "sampler_profile", None)


def verify_prepared(
    path: Path,
    expected_sha: str,
    sidecar_sha: str,
    source_path: Path,
) -> dict:
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha != expected_sha:
        raise ValueError(
            f"{path}: prepared checkpoint SHA-256 is {observed_sha}, "
            f"expected {expected_sha}"
        )
    source = helpers.load_pkl_object(str(source_path))
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("next_update") != SOURCE_UPDATE:
        raise ValueError(f"{path}: prepared checkpoint changed next_update")
    optimizer_step = int(np.asarray(checkpoint["train_state_step"]).reshape(()))
    if optimizer_step != SOURCE_UPDATE * OPTIMIZER_STEPS_PER_UPDATE:
        raise ValueError(f"{path}: prepared checkpoint changed optimizer clock")
    count = parameter_count(checkpoint["model"])
    if count != PREPARED_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: {count} prepared parameters, expected {PREPARED_PARAMETER_COUNT}"
        )
    finite(checkpoint["model"], f"{path}.model")
    finite(checkpoint["optimizer_state"], f"{path}.optimizer_state")
    check_reward_protocol(path, checkpoint, sidecar_sha)

    config = checkpoint["train_config"]
    check_config(
        path,
        config,
        {
            **CONTRACT,
            "distance_sidecar_sha256": sidecar_sha,
            "stall_age_observation": True,
        },
    )
    if config_value(config, "config_name") != "G-V8-CONTINUOUS-V3":
        raise ValueError(f"{path}: prepared checkpoint must name the v3 preset")
    pooled = config_value(config, "pooled_sampler", None)
    if config_value(pooled, "rule", None) != "continuous_banded_v3":
        raise ValueError(f"{path}: prepared config must select continuous_banded_v3")
    if accepted_sampler_profile(config) != "continuous_banded_v3":
        raise ValueError(f"{path}: prepared bank must select continuous_banded_v3")

    receipt = checkpoint.get("stall_age_prepared_continuation")
    if not isinstance(receipt, dict) or receipt.get("schema") != (
        "terra_v8_v61_stall_age_v3_prepared_v1"
    ):
        raise ValueError(f"{path}: stall-age preparation receipt is missing")

    source_state = source["pooled_sampler_state"]
    state = checkpoint["pooled_sampler_state"]
    sampler = verify_sampler_state(state)
    if sampler["sampler_rule"] != "continuous_banded_v3":
        raise ValueError(f"{path}: prepared sampler must be continuous_banded_v3")
    if source_state["settings"]["rule"] != "continuous_banded_v2":
        raise ValueError(f"{source_path}: source sampler must be continuous_banded_v2")
    if source_state["current_window"]["updates"] != SOURCE_WINDOW_UPDATES:
        raise ValueError(f"{source_path}: source partial sampler window changed")
    if state["current_window"]["updates"] != 0 or any(
        any(values)
        for key, values in state["current_window"].items()
        if key != "updates"
    ):
        raise ValueError(f"{path}: prepared sampler partial window was not cleared")
    for key in (
        "conditions",
        "maps_per_condition",
        "labels",
        "competence",
        "closed_window",
        "refresh",
        "mastery",
        "numpy_rng",
    ):
        if state[key] != source_state[key]:
            raise ValueError(f"{path}: sampler migration changed {key}")

    probabilities = np.asarray(state["probabilities"], dtype=np.float64)
    mastered = np.asarray(state["mastery"]["mastered"], dtype=bool)
    if int(mastered.sum()) != 29 or int((~mastered).sum()) != 18:
        raise ValueError(f"{path}: prepared mastery population changed")
    np.testing.assert_allclose(
        probabilities[~mastered].sum(), 0.80, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        probabilities[mastered].sum(), 0.20, rtol=0.0, atol=1e-12
    )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": observed_sha,
        "next_update": SOURCE_UPDATE,
        "optimizer_step": optimizer_step,
        "parameter_count": count,
        "sampler": sampler,
        "discarded_partial_window_updates": SOURCE_WINDOW_UPDATES,
        "sampler_last_refresh_update": int(state["refresh"]["last_refresh_update"]),
        "sampler_refreshes": int(state["refresh"]["refreshes"]),
        "sampler_mastered": int(mastered.sum()),
        "sampler_open_mass": float(probabilities[~mastered].sum()),
        "sampler_mastered_mass": float(probabilities[mastered].sum()),
        "sampler_max_condition_mass": float(probabilities.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--distance-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for sha in (
        args.source_sha256,
        args.prepared_sha256,
        args.distance_artifact_sha256,
    ):
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError("SHA-256 arguments must be lowercase 64-hex")

    helpers.register_checkpoint_config_classes()
    receipt = {
        "schema": "terra_v8_v61_stall_age_resume_validation_v2",
        "passed": True,
        "arm": "v6_1_rv2_stall_age_v3",
        "absolute_start_update": SOURCE_UPDATE,
        "absolute_target_update": PHASE2_TARGET_UPDATE,
        "num_devices": PHASE2_NUM_DEVICES,
        "num_envs_per_device": PHASE2_NUM_ENVS_PER_DEVICE,
        "source_sampler_rule": "continuous_banded_v2",
        "sampler_rule": "continuous_banded_v3",
        "sampler_migration": "materialized_before_resume",
        "sampler_partial_window": "discarded_50_updates_before_resume",
        "source": verify_source(
            args.source, args.source_sha256, args.distance_artifact_sha256
        ),
        "prepared": verify_prepared(
            args.prepared,
            args.prepared_sha256,
            args.distance_artifact_sha256,
            args.source,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
