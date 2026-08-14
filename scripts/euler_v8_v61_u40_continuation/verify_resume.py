#!/usr/bin/env python3
"""Validate the exact evaluated u40000 checkpoint before phase-3 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state
from terra.config import REWARD_V2_POTENTIAL_GAMMA, REWARD_V2_PROTOCOL_ID, RewardStage
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from utils import helpers


SOURCE_UPDATE = 40_000
TARGET_UPDATE = 70_000
OPTIMIZER_STEPS_PER_UPDATE = 64
PARAMETER_COUNT = 2_304_829
DISTANCE_SHA256 = "f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980"

EXPECTED_CONFIG = {
    "config_name": "G-V8-CONTINUOUS-V3",
    "seed": 20260807,
    "num_devices": 8,
    "num_envs_per_device": 256,
    "num_steps": 32,
    "num_minibatches": 32,
    "update_epochs": 2,
    "lr": 3e-4,
    "gamma": float(REWARD_V2_POTENTIAL_GAMMA),
    "reward_stage": "reward_v2",
    "reward_v2_timing_variant": 0,
    "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
    "distance_sidecar_sha256": DISTANCE_SHA256,
    "carry_work_observation": True,
    "stall_age_observation": True,
    "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
    "model_size": "medium",
    "model_core": "mlp",
    "encoder_compute_dtype": "bfloat16",
    "attention_compute_dtype": "float32",
    "critic_hidden_dims": (512, 256),
    "resnet_stage_channels": (24, 48, 64, 96),
    "resnet_blocks_per_stage": (2, 2, 3, 3),
    "token_mixer_residual_init_scale": 0.1,
    "flatten_reduce_channels": 32,
    "attn_latent_queries": 8,
    "aux_coef": 0.0,
    "vf_coef": 2.0,
    "ent_schedule_start": 0.15,
    "ent_schedule_end": 0.02,
    "ent_schedule_steps": 20_000,
    "use_value_clip": False,
    "flat_minibatch_shuffle": True,
    "action_logit_masking": False,
    "warm_start_from": None,
    "teacher_checkpoint": None,
}


def config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name, "<absent>")
    return getattr(config, name, "<absent>")


def check_config(path: Path, config: object) -> None:
    for name, expected in EXPECTED_CONFIG.items():
        observed = config_value(config, name)
        if isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"{path}: {name}={observed!r}, expected {expected!r}")

    accepted_bank = config_value(config, "accepted_bank")
    pooled_sampler = config_value(config, "pooled_sampler")
    if config_value(accepted_bank, "sampler_profile") != "continuous_banded_v3":
        raise ValueError(f"{path}: accepted bank is not on continuous_banded_v3")
    if config_value(pooled_sampler, "rule") != "continuous_banded_v3":
        raise ValueError(f"{path}: pooled sampler is not on continuous_banded_v3")


def finite(tree: object, label: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{label}: non-finite leaf {index}")


def check_reward_protocol(path: Path, checkpoint: dict) -> None:
    protocol = checkpoint.get("r2_protocol_receipt")
    expected = {
        "schema": "terra_v8_r2_reward_protocol_v1",
        "reward_stage": "reward_v2",
        "reward_protocol_id": REWARD_V2_PROTOCOL_ID,
        "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
        "distance_sidecar_sha256": DISTANCE_SHA256,
        "reward_v2_timing": "baseline",
        "reward_v2_timing_variant": 0,
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{path}: reward-v2 protocol receipt changed")
    if protocol.get("constants", {}).get("potential_gamma") != float(
        REWARD_V2_POTENTIAL_GAMMA
    ):
        raise ValueError(f"{path}: reward-v2 potential gamma changed")
    if not np.all(np.asarray(checkpoint["env_config"].reward_stage) == int(RewardStage.REWARD_V2)):
        raise ValueError(f"{path}: environment reward stage changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.sha256) != 64 or any(c not in "0123456789abcdef" for c in args.sha256):
        raise ValueError("--sha256 must be lowercase 64-hex")
    payload = args.checkpoint.read_bytes()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != args.sha256:
        raise ValueError(
            f"{args.checkpoint}: SHA-256 is {observed_sha}, expected {args.sha256}"
        )

    helpers.register_checkpoint_config_classes()
    checkpoint = pickle.loads(payload)
    if checkpoint.get("update") != SOURCE_UPDATE - 1:
        raise ValueError(f"{args.checkpoint}: terminal update is not 39999")
    if checkpoint.get("next_update") != SOURCE_UPDATE:
        raise ValueError(f"{args.checkpoint}: next_update is not {SOURCE_UPDATE}")
    optimizer_step = int(np.asarray(checkpoint["train_state_step"]).reshape(()))
    expected_optimizer_step = SOURCE_UPDATE * OPTIMIZER_STEPS_PER_UPDATE
    if optimizer_step != expected_optimizer_step:
        raise ValueError(
            f"{args.checkpoint}: optimizer step is {optimizer_step}, "
            f"expected {expected_optimizer_step}"
        )

    parameter_count = sum(
        int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(checkpoint["model"])
    )
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(
            f"{args.checkpoint}: {parameter_count} parameters, expected {PARAMETER_COUNT}"
        )
    finite(checkpoint["model"], "model")
    finite(checkpoint["optimizer_state"], "optimizer_state")
    finite(checkpoint["loss_info"], "loss_info")
    check_config(args.checkpoint, checkpoint["train_config"])
    check_reward_protocol(args.checkpoint, checkpoint)

    state = checkpoint.get("pooled_sampler_state")
    sampler = verify_sampler_state(state)
    if sampler["sampler_rule"] != "continuous_banded_v3":
        raise ValueError(f"{args.checkpoint}: sampler rule changed")
    if state["current_window"]["updates"] != 100:
        raise ValueError(f"{args.checkpoint}: sampler partial window is not 100 updates")
    if state["refresh"]["last_refresh_update"] != 39_900:
        raise ValueError(f"{args.checkpoint}: sampler refresh clock changed")
    mastered = int(np.asarray(state["mastery"]["mastered"], dtype=bool).sum())
    if mastered != 46:
        raise ValueError(f"{args.checkpoint}: mastered count is {mastered}, expected 46")

    integrity = checkpoint.get("transition_integrity")
    if integrity != {
        "maximum_mass_residual": 0,
        "obstacle_mutation_count": 0,
        "target_mutation_count": 0,
    }:
        raise ValueError(f"{args.checkpoint}: transition-integrity receipt changed")

    receipt = {
        "schema": "terra_v8_v61_u40000_resume_validation_v1",
        "passed": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha,
        "source_update": SOURCE_UPDATE,
        "target_update": TARGET_UPDATE,
        "optimizer_step": optimizer_step,
        "parameter_count": parameter_count,
        "sampler_rule": sampler["sampler_rule"],
        "sampler_mastered": mastered,
        "sampler_window_updates": state["current_window"]["updates"],
        "sampler_last_refresh_update": state["refresh"]["last_refresh_update"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
