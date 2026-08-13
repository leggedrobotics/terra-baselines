#!/usr/bin/env python3
"""Fail closed on the v6.1 phase2 continuation: source checkpoint, then resume.

Two jobs, one script.

``--source`` alone is the phase2 launch gate. It pins the u14000 checkpoint by
SHA-256 and asserts everything a continuation inherits and must not change: the
optimizer clock (14000 x 2 x 32 = 896,000 steps), the 2,303,421-parameter
v6.1 readout tree, the reward-v2 protocol receipt at baseline timing, and the
47-condition continuous_banded_v2 sampler with its refresh grid intact.

Adding ``--run`` verifies what a resume actually produced: the update advanced
by exactly one from 14000, the optimizer clock advanced by exactly 64 steps, the
parameter tree is unchanged in shape and finite, the batch shape is phase2's
8 x 512 x 32 / 32, and the sampler migrated to continuous_banded_v3 keeping its
mastery and closed window while the partial window it was saved mid-way through
was discarded (that is the migration contract, not a loss of state).
"""

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

# The frozen source: v6.1 phase1's terminal periodic checkpoint.
SOURCE_UPDATE = 14_000
UPDATE_EPOCHS = 2
NUM_MINIBATCHES = 32
OPTIMIZER_STEPS_PER_UPDATE = UPDATE_EPOCHS * NUM_MINIBATCHES
PARAMETER_COUNT = 2_303_421
CONDITION_COUNT = 47
# 14000 % 150 == 50: the source stopped 50 updates into a window.
SOURCE_WINDOW_UPDATES = 50
SOURCE_LAST_REFRESH_UPDATE = 13_950
# phase2's shape. Not the phase1 shape: comparability is deliberately dropped.
PHASE2_NUM_DEVICES = 8
PHASE2_NUM_ENVS_PER_DEVICE = 512
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
    "prepared_fork_from": None,
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
    check_config(path, config, {**CONTRACT, "distance_sidecar_sha256": sidecar_sha})
    # phase1's shape, which phase2 deliberately leaves behind.
    check_config(path, config, {"num_devices": 4, "num_envs_per_device": 512})
    if config_value(config, "config_name") != "G-V8-CONTINUOUS-V2":
        raise ValueError(f"{path}: source must be the v2 preset")
    protocol = check_reward_protocol(path, checkpoint, sidecar_sha)

    sampler = verify_sampler_state(checkpoint.get("pooled_sampler_state"))
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


def verify_resumed(
    path: Path, source: dict, source_path: Path, sidecar_sha: str
) -> dict:
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("update") != SOURCE_UPDATE or (
        checkpoint.get("next_update") != SOURCE_UPDATE + 1
    ):
        raise ValueError(
            f"{path}: resume smoke must advance update {SOURCE_UPDATE} exactly once"
        )
    optimizer_step = int(np.asarray(checkpoint["train_state_step"]).reshape(()))
    expected_step = source["optimizer_step"] + OPTIMIZER_STEPS_PER_UPDATE
    if optimizer_step != expected_step:
        raise ValueError(
            f"{path}: optimizer clock is {optimizer_step}, expected {expected_step}"
        )
    count = parameter_count(checkpoint["model"])
    if count != PARAMETER_COUNT:
        raise ValueError(f"{path}: {count} parameters, expected {PARAMETER_COUNT}")
    finite(checkpoint["model"], f"{path}.model")
    finite(checkpoint["optimizer_state"], f"{path}.optimizer_state")
    integrity = checkpoint.get("transition_integrity")
    if not isinstance(integrity, dict) or any(
        int(value) != 0 for value in integrity.values()
    ):
        raise ValueError(f"{path}: transition-integrity counters are nonzero")

    config = checkpoint["train_config"]
    check_config(
        path,
        config,
        {
            **CONTRACT,
            "distance_sidecar_sha256": sidecar_sha,
            "num_devices": PHASE2_NUM_DEVICES,
            "num_envs_per_device": PHASE2_NUM_ENVS_PER_DEVICE,
            "config_name": "G-V8-CONTINUOUS-V3",
            "resume_from": str(source_path),
            "load_env_from_checkpoint": False,
            "sampler_migration_clear_window": True,
        },
    )
    check_reward_protocol(path, checkpoint, sidecar_sha)

    sampler = verify_sampler_state(checkpoint.get("pooled_sampler_state"))
    if sampler["sampler_rule"] != "continuous_banded_v3":
        raise ValueError(f"{path}: resumed sampler must be continuous_banded_v3")
    state = checkpoint["pooled_sampler_state"]
    refresh = state["refresh"]
    # Migration keeps the refresh grid and the closed window, and discards the
    # partial window: after one resumed update the new window holds exactly 1.
    if refresh["last_refresh_update"] != SOURCE_LAST_REFRESH_UPDATE:
        raise ValueError(f"{path}: sampler refresh grid moved across the migration")
    if refresh["refreshes"] != source["sampler_refreshes"]:
        raise ValueError(f"{path}: sampler refresh count changed across the migration")
    if state["current_window"]["updates"] != 1:
        raise ValueError(
            f"{path}: migrated window holds {state['current_window']['updates']} "
            "updates, expected 1 (the discarded partial window plus one)"
        )
    if int(sum(state["mastery"]["mastered"])) != source["sampler_mastered"]:
        raise ValueError(f"{path}: sampler mastery changed across the migration")
    probabilities = np.asarray(state["probabilities"], dtype=np.float64)
    if probabilities.max() > 0.15 + 1e-12:
        raise ValueError(f"{path}: v3 cap is not holding: {probabilities.max()}")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "update": SOURCE_UPDATE,
        "next_update": SOURCE_UPDATE + 1,
        "optimizer_step": optimizer_step,
        "optimizer_steps_taken": optimizer_step - source["optimizer_step"],
        "parameter_count": count,
        "sampler": sampler,
        "sampler_window_updates": int(state["current_window"]["updates"]),
        "sampler_mastered": int(sum(state["mastery"]["mastered"])),
        "sampler_max_condition_mass": float(probabilities.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--distance-artifact-sha256", required=True)
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Resume-smoke run directory; when given, verify what it produced.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for sha in (args.source_sha256, args.distance_artifact_sha256):
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError("SHA-256 arguments must be lowercase 64-hex")

    helpers.register_checkpoint_config_classes()
    receipt = {
        "schema": "terra_v8_v6_yolo_rv2_resume_validation_v1",
        "passed": True,
        "arm": "v6_1_rv2",
        "absolute_start_update": SOURCE_UPDATE,
        "absolute_target_update": PHASE2_TARGET_UPDATE,
        "num_devices": PHASE2_NUM_DEVICES,
        "num_envs_per_device": PHASE2_NUM_ENVS_PER_DEVICE,
        "sampler_migration": "continuous_banded_v2->continuous_banded_v3",
        "source": verify_source(
            args.source, args.source_sha256, args.distance_artifact_sha256
        ),
    }
    if args.run is not None:
        checkpoints = args.run / "checkpoints"
        token = f"_update_{SOURCE_UPDATE + 1:06d}.pkl"
        periodic = [p for p in checkpoints.glob(f"*{token}")]
        final = list(checkpoints.glob("*_FINAL.pkl"))
        if len(periodic) != 1 or len(final) != 1:
            raise ValueError(
                f"resume smoke must emit one {token} and one FINAL checkpoint"
            )
        receipt["resumed"] = {
            "periodic": verify_resumed(
                periodic[0],
                receipt["source"],
                args.source,
                args.distance_artifact_sha256,
            ),
            "final": verify_resumed(
                final[0],
                receipt["source"],
                args.source,
                args.distance_artifact_sha256,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
