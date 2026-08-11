#!/usr/bin/env python3
"""Fail closed on the v6 yolo / v6.1 update-1 smoke.

This is euler_v8_r2_reward_v2/verify_smoke.py's reward-v2 contract plus the V6
architecture receipt: the reward stage, carry-work channel, distance protocol,
sidecar SHA, sampler state and entropy schedule must match the paired baseline
exactly, while the architecture/optimization flags must be the treatment's.

The arm is selected by the ARM_NAME environment variable (run.sbatch passes the
same value it wrote into run_contract.env), and every arm-dependent expectation
lives in ARMS. Parameter counts are per arm because the stage rebalance moves
them: 2,134,771 at blocks (3,3,2,2), 2,328,225 at the baseline's (2,2,3,3).
Those are the carry-work counts; the 16 fewer weights of the without-carry tree
are the same +16 that separates the compact baseline's 2,856,685 from 2,856,701.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

PROTOCOL_SCHEMA = "terra_v8_r2_reward_protocol_v1"
AUX_DECODER_LEAVES = 6

# The V6 readout flags every arm shares.
ARCHITECTURE = {
    "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
    "token_mixer_residual_init_scale": 0.1,
    "flatten_reduce_channels": 32,
    "attn_latent_queries": 8,
}
# The flags that separate the arms, plus the parameter count each one implies.
ARMS = {
    "v6_3m_yolo_rv2": {
        "resnet_blocks_per_stage": (3, 3, 2, 2),
        "aux_coef": 0.25,
        "vf_coef": 0.5,
        "action_logit_masking": True,
        "parameter_count": 2_134_771,
        "parameter_count_without_carry_work": 2_134_755,
    },
    "v6_3m_yolo_rv2_nomask": {
        "resnet_blocks_per_stage": (3, 3, 2, 2),
        "aux_coef": 0.25,
        "vf_coef": 0.5,
        "action_logit_masking": False,
        "parameter_count": 2_134_771,
        "parameter_count_without_carry_work": 2_134_755,
    },
    # v6.1: the full-res rebalance, vf_coef 0.5 and D3 masking are all reverted
    # to the baseline; aux drops 0.25 -> 0.1. vf_coef 2.0 is the trainer default
    # the launcher reaches by not passing the flag at all.
    "v6_1_rv2": {
        "resnet_blocks_per_stage": (2, 2, 3, 3),
        "aux_coef": 0.1,
        "vf_coef": 2.0,
        "action_logit_masking": False,
        "parameter_count": 2_328_225,
        "parameter_count_without_carry_work": 2_328_209,
    },
}


def config_value(config: object, name: str) -> object:
    return config[name] if isinstance(config, dict) else getattr(config, name)


def finite(tree: object, label: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{label}: non-finite leaf {index}")


def one(path: Path, *, arm: dict[str, object], distance_artifact_sha256: str) -> dict:
    expected_count = arm["parameter_count"]
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("update") != 0 or checkpoint.get("next_update") != 1:
        raise ValueError(f"{path}: scratch smoke must advance update 0 exactly once")
    optimizer_step = int(np.asarray(checkpoint.get("train_state_step")).reshape(()))
    if optimizer_step != 64:
        raise ValueError(f"{path}: first PPO update must take 64 optimizer steps")
    model = checkpoint["model"]
    count = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(model))
    if count != expected_count:
        raise ValueError(
            f"{path}: this arm under carry-work must have {expected_count} "
            f"parameters, got {count}"
        )
    # The aux decode head must be trained, not merely allocated.
    aux_leaves = [
        jax.tree_util.keystr(keys)
        for keys, _ in jax.tree_util.tree_flatten_with_path(model)[0]
        if "aux_decoder" in jax.tree_util.keystr(keys)
    ]
    if len(aux_leaves) != AUX_DECODER_LEAVES:
        raise ValueError(
            f"{path}: expected {AUX_DECODER_LEAVES} aux decoder leaves, "
            f"got {len(aux_leaves)}"
        )
    finite(model, f"{path}.model")
    finite(checkpoint["optimizer_state"], f"{path}.optimizer_state")
    transition_integrity = checkpoint.get("transition_integrity")
    if not isinstance(transition_integrity, dict) or any(
        int(value) != 0 for value in transition_integrity.values()
    ):
        raise ValueError(f"{path}: transition-integrity counters are nonzero")

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
        "reward_stage": "reward_v2",
        "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
        "distance_sidecar_sha256": distance_artifact_sha256,
        "carry_work_observation": True,
        "ent_schedule_start": 0.15,
        "ent_schedule_end": 0.02,
        "ent_schedule_steps": 20_000,
        "model_size": "medium",
        "model_core": "mlp",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "float32",
        "critic_hidden_dims": (512, 256),
        "resnet_stage_channels": (24, 48, 64, 96),
        "use_value_clip": False,
        "flat_minibatch_shuffle": True,
        "prepared_fork_from": None,
        "warm_start_from": None,
        "resume_from": None,
        "teacher_checkpoint": None,
        **ARCHITECTURE,
        **{
            name: arm[name]
            for name in (
                "resnet_blocks_per_stage",
                "aux_coef",
                "vf_coef",
                "action_logit_masking",
            )
        },
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
    if not np.all(env_stage_value == int(RewardStage.REWARD_V2)):
        raise ValueError(f"{path}: environment reward selector changed")
    if "r2_prepared_fork" in checkpoint:
        raise ValueError(f"{path}: scratch checkpoint claims a prepared fork")
    protocol = checkpoint.get("r2_protocol_receipt")
    expected_protocol = {
        "schema": PROTOCOL_SCHEMA,
        "reward_stage": "reward_v2",
        "reward_protocol_id": REWARD_V2_PROTOCOL_ID,
        "distance_protocol_id": REWARD_V2_DISTANCE_PROTOCOL_ID,
        "distance_sidecar_sha256": distance_artifact_sha256,
        "distance_artifact_kind": "canonical_distance_sidecar_dataset_json",
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
        raise ValueError(f"{path}: sampler did not start on v2")
    sampler_state = checkpoint["pooled_sampler_state"]
    conditions = sampler_state["conditions"]
    labels = sampler_state["labels"]
    probabilities = np.asarray(sampler_state["probabilities"], dtype=np.float64)
    family_mass = {
        family: float(
            probabilities[
                [labels[name]["family"] == family for name in conditions]
            ].sum()
        )
        for family in ("foundation", "trench")
    }
    depth_mass = {
        str(depth): float(
            probabilities[
                [labels[name]["curriculum_depth"] == depth for name in conditions]
            ].sum()
        )
        for depth in (0, 1, 2)
    }
    np.testing.assert_allclose(
        [family_mass["foundation"], family_mass["trench"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [depth_mass["0"], depth_mass["1"], depth_mass["2"]],
        [
            0.11346390374331551,
            0.3836076203208556,
            0.5029284759358292,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    sampler["family_mass"] = family_mass
    sampler["depth_mass"] = depth_mass
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameter_count": count,
        "aux_decoder_leaves": len(aux_leaves),
        "optimizer_step": optimizer_step,
        "sampler": sampler,
        "reward_protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--distance-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.distance_artifact_sha256) != 64:
        raise ValueError("distance artifact SHA-256 must contain 64 characters")
    arm_name = os.environ.get("ARM_NAME", "")
    if arm_name not in ARMS:
        raise ValueError(f"ARM_NAME must be one of {sorted(ARMS)}, got {arm_name!r}")
    arm = ARMS[arm_name]
    periodic = list((args.run / "checkpoints").glob("*_update_000001.pkl"))
    final = list((args.run / "checkpoints").glob("*_FINAL.pkl"))
    if len(periodic) != 1 or len(final) != 1:
        raise ValueError("smoke must emit one update-1 and one FINAL checkpoint")
    helpers.register_checkpoint_config_classes()
    contract = {}
    for line in (args.run / "run_contract.env").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            contract[key] = value
    expected_contract = {
        "experiment": "v8_v6_3m_yolo_rv2_architecture_screen",
        "arm": arm_name,
        "paired_baseline_arm": "reward_v2_scratch",
        "absolute_start_update": "0",
        "absolute_target_update": "1",
        "updates": "1",
        "initialization": "random_no_teacher",
        "prepared_fork_from": "none",
        "entropy_schedule": "0.15_to_0.02_over_20000",
        "sampler_profile": "continuous_banded_v2",
        "initial_sampler_depth_mass_d0": "0.11346390374331551",
        "initial_sampler_depth_mass_d1": "0.3836076203208556",
        "initial_sampler_depth_mass_d2": "0.5029284759358292",
        "reward_protocol_id": "material_potential_v2",
        "distance_artifact_sha256": args.distance_artifact_sha256,
        "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
        "resnet_blocks_per_stage": ",".join(
            str(x) for x in arm["resnet_blocks_per_stage"]
        ),
        "token_mixer_residual_init_scale": "0.1",
        "flatten_reduce_channels": "32",
        "attn_latent_queries": "8",
        "aux_coef": str(arm["aux_coef"]),
        "vf_coef": str(arm["vf_coef"]),
        "action_logit_masking": "true" if arm["action_logit_masking"] else "false",
        "model_parameter_count": str(arm["parameter_count"]),
        "model_parameter_count_without_carry_work": str(
            arm["parameter_count_without_carry_work"]
        ),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("run contract does not match the v6 yolo update-1 smoke")
    receipt = {
        "schema": "terra_v8_v6_yolo_rv2_update1_smoke_v1",
        "passed": True,
        "arm": arm_name,
        "run_contract": expected_contract,
        "checkpoints": {
            "periodic": one(
                periodic[0],
                arm=arm,
                distance_artifact_sha256=args.distance_artifact_sha256,
            ),
            "final": one(
                final[0],
                arm=arm,
                distance_artifact_sha256=args.distance_artifact_sha256,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
