#!/usr/bin/env python3
"""Fail closed on one independently executed R2 update-1 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state
from terra.config import (
    DENSE_REWARD_PROTOCOL_ID,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.maps_buffer import LEGACY_DISTANCE_PROTOCOL_ID
from utils import helpers

PREPARED_SCHEMA = "terra_v8_r2_prepared_fork_v1"
PROTOCOL_SCHEMA = "terra_v8_r2_reward_protocol_v1"
PARENT_SHA256 = "0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834"
PREPARED_SHA256 = "8e01ebd3dfd99b36cea90a251dfe4a4e305228abeb2f5ecba633a9fc6805b1d0"
PREPARED_RECEIPT_SHA256 = (
    "d119f443613d4959d5f63918971c50c5ad204e4b6c1d65ec985c3fc31b005185"
)
PARAMETER_COUNT = 2_856_701


def config_value(config: object, name: str) -> object:
    return config[name] if isinstance(config, dict) else getattr(config, name)


def finite(tree: object, label: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{label}: non-finite leaf {index}")


def one(
    path: Path,
    *,
    arm: str,
    distance_artifact_sha256: str,
) -> dict[str, object]:
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("update") != 20_000 or checkpoint.get("next_update") != 20_001:
        raise ValueError(
            f"{path}: smoke must advance absolute update 20000 exactly once"
        )
    optimizer_step = int(np.asarray(checkpoint.get("train_state_step")).reshape(()))
    if optimizer_step != 64:
        raise ValueError(
            f"{path}: fresh optimizer-local step must be 64, got {optimizer_step}"
        )
    count = sum(
        int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(checkpoint["model"])
    )
    if count != PARAMETER_COUNT:
        raise ValueError(f"{path}: prepared model has {count} parameters")
    finite(checkpoint["model"], f"{path}.model")
    finite(checkpoint["optimizer_state"], f"{path}.optimizer_state")
    transition_integrity = checkpoint.get("transition_integrity")
    if not isinstance(transition_integrity, dict) or any(
        int(value) != 0 for value in transition_integrity.values()
    ):
        raise ValueError(f"{path}: transition-integrity counters are nonzero")

    reward_stage, reward_protocol, distance_protocol, env_stage, artifact_kind = {
        "control": (
            "dense_skill",
            DENSE_REWARD_PROTOCOL_ID,
            LEGACY_DISTANCE_PROTOCOL_ID,
            RewardStage.DENSE_SKILL,
            "accepted_bank_dataset_json",
        ),
        "reward_v2": (
            "reward_v2",
            REWARD_V2_PROTOCOL_ID,
            REWARD_V2_DISTANCE_PROTOCOL_ID,
            RewardStage.REWARD_V2,
            "canonical_distance_sidecar_dataset_json",
        ),
    }[arm]
    config = checkpoint["train_config"]
    expected_config = {
        "config_name": "G-V8-CONTINUOUS-V2",
        "seed": 20260807,
        "num_devices": 4,
        "num_envs_per_device": 512,
        "num_steps": 32,
        "num_minibatches": 32,
        "update_epochs": 2,
        "lr": 3e-4,
        "gamma": float(REWARD_V2_POTENTIAL_GAMMA),
        "reward_stage": reward_stage,
        "distance_protocol_id": distance_protocol,
        "distance_sidecar_sha256": distance_artifact_sha256,
        "carry_work_observation": True,
        "ent_schedule_start": 0.02,
        "ent_schedule_end": 0.02,
        "ent_schedule_steps": 1,
        "kickstart_lr_warmup_updates": 100,
        "map_encoder": "resnet_spatial_8x8_se_xattn",
        "model_size": "medium",
        "model_core": "mlp",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "float32",
        "critic_hidden_dims": (512, 256),
        "resnet_stage_channels": (24, 48, 64, 96),
        "resnet_blocks_per_stage": (2, 2, 3, 3),
        "use_value_clip": False,
        "flat_minibatch_shuffle": True,
    }
    for name, expected in expected_config.items():
        observed = config_value(config, name)
        if isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"{path}: {name}={observed!r}, expected {expected!r}")
    sampler_config = config_value(config, "pooled_sampler")
    accepted_bank = config_value(config, "accepted_bank")
    if sampler_config.get("rule") != "continuous_banded_v2" or (
        accepted_bank.sampler_profile != "continuous_banded_v2"
    ):
        raise ValueError(f"{path}: saved config does not consistently select v2")

    env_stage_value = np.asarray(checkpoint["env_config"].reward_stage)
    if not np.all(env_stage_value == int(env_stage)):
        raise ValueError(f"{path}: environment reward selector changed")
    prepared = checkpoint.get("r2_prepared_fork")
    if not isinstance(prepared, dict) or prepared.get("schema") != PREPARED_SCHEMA:
        raise ValueError(f"{path}: prepared-fork receipt missing")
    if prepared.get("source_checkpoint_sha256") != PARENT_SHA256 or (
        prepared.get("source_next_update") != 20_000
        or prepared.get("prepared_parameter_count") != PARAMETER_COUNT
        or prepared.get("output_preserving") is not True
        or prepared.get("target_sampler_rule") != "continuous_banded_v2"
        or prepared.get("target_config_name") != "G-V8-CONTINUOUS-V2"
        or prepared.get("target_bank_sampler_profile") != "continuous_banded_v2"
    ):
        raise ValueError(f"{path}: prepared-fork contract changed")
    protocol = checkpoint.get("r2_protocol_receipt")
    expected_protocol = {
        "schema": PROTOCOL_SCHEMA,
        "reward_stage": reward_stage,
        "reward_protocol_id": reward_protocol,
        "distance_protocol_id": distance_protocol,
        "distance_sidecar_sha256": distance_artifact_sha256,
        "distance_artifact_kind": artifact_kind,
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected_protocol.items()
    ):
        raise ValueError(f"{path}: reward protocol receipt changed")
    constants = protocol.get("constants", {})
    if constants.get("potential_gamma") != float(REWARD_V2_POTENTIAL_GAMMA):
        raise ValueError(f"{path}: reward-v2 constant receipt changed")
    sampler = verify_sampler_state(checkpoint.get("pooled_sampler_state"))
    if sampler.get("sampler_rule") != "continuous_banded_v2":
        raise ValueError(f"{path}: sampler did not remain on v2")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameter_count": count,
        "optimizer_local_step": optimizer_step,
        "sampler": sampler,
        "reward_protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "reward_v2"), required=True)
    parser.add_argument("--distance-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.distance_artifact_sha256) != 64:
        raise ValueError("distance artifact SHA-256 must contain 64 characters")
    periodic = list((args.run / "checkpoints").glob("*_update_020001.pkl"))
    final = list((args.run / "checkpoints").glob("*_FINAL.pkl"))
    if len(periodic) != 1 or len(final) != 1:
        raise ValueError("smoke must emit one update-20001 and one FINAL checkpoint")
    helpers.register_checkpoint_config_classes()
    contract = {}
    for line in (args.run / "run_contract.env").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            contract[key] = value
    expected_contract = {
        "arm": args.arm,
        "absolute_start_update": "20000",
        "absolute_target_update": "20001",
        "additional_updates": "1",
        "prepared_fork_sha256": PREPARED_SHA256,
        "prepared_fork_receipt_sha256": PREPARED_RECEIPT_SHA256,
        "optimizer_initial_local_step": "0",
        "lr_warmup_updates": "100",
        "entropy_fixed": "0.02",
        "distance_artifact_sha256": args.distance_artifact_sha256,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("run contract does not match the prepared update-1 smoke")
    receipt = {
        "schema": "terra_v8_r2_update1_smoke_v1",
        "passed": True,
        "arm": args.arm,
        "run_contract": expected_contract,
        "checkpoints": {
            "periodic": one(
                periodic[0],
                arm=args.arm,
                distance_artifact_sha256=args.distance_artifact_sha256,
            ),
            "final": one(
                final[0],
                arm=args.arm,
                distance_artifact_sha256=args.distance_artifact_sha256,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
