#!/usr/bin/env python3
"""Verify one V8 architecture-control update-1 checkpoint pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state
from terra.config import RewardStage
from utils import helpers


ARCHITECTURES = {
    "compact_xattn": {
        "parameter_count": 2_856_685,
        "model_size": "medium",
        "model_core": "mlp",
        "map_encoder": "resnet_spatial_8x8_se_xattn",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "float32",
        "critic_hidden_dims": (512, 256),
        "resnet_stage_channels": (24, 48, 64, 96),
        "resnet_blocks_per_stage": (2, 2, 3, 3),
    },
    "atari_base": {
        "parameter_count": 480_137,
        "model_size": "base",
        "model_core": "mlp",
        "map_encoder": "atari",
        "encoder_compute_dtype": "float32",
        "attention_compute_dtype": "encoder",
        "critic_hidden_dims": None,
        "resnet_stage_channels": None,
        "resnet_blocks_per_stage": None,
    },
}

COMMON_TRAINING = {
    "num_devices": 4,
    "num_envs_per_device": 512,
    "num_steps": 32,
    "num_minibatches": 32,
    "update_epochs": 2,
    "lr": 3e-4,
    "seed": 20260807,
    "reward_stage": "dense_skill",
    "use_value_clip": False,
    "flat_minibatch_shuffle": True,
    "ent_schedule_start": 0.15,
    "ent_schedule_end": 0.02,
    "ent_schedule_steps": 1,
}


def config_value(config: object, name: str) -> object:
    return config[name] if isinstance(config, dict) else getattr(config, name)


def normalized(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def require_config(config: object, expected: dict[str, object], label: str) -> None:
    for name, wanted in expected.items():
        observed = normalized(config_value(config, name))
        if observed != wanted:
            raise ValueError(f"{label}: {name}={observed!r}, expected {wanted!r}")


def require_finite_tree(tree: object, label: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"{label}: non-finite leaf {index}")


def one(path: Path, arm: str) -> dict[str, object]:
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("next_update") != 1:
        raise ValueError(f"{path}: next_update must be 1")

    model = checkpoint["model"]
    parameter_count = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(model))
    expected_architecture = ARCHITECTURES[arm]
    if parameter_count != expected_architecture["parameter_count"]:
        raise ValueError(f"{path}: unexpected parameter count {parameter_count}")
    require_finite_tree(model, f"{path}.model")
    require_finite_tree(checkpoint["optimizer_state"], f"{path}.optimizer_state")
    require_config(checkpoint["train_config"], COMMON_TRAINING, str(path))
    require_config(
        checkpoint["train_config"],
        {k: v for k, v in expected_architecture.items() if k != "parameter_count"},
        str(path),
    )

    env_config = checkpoint["env_config"]
    env_stage = np.asarray(env_config.reward_stage)
    env_mix = np.asarray(env_config.terminal_reward_mix, dtype=np.float64)
    if not np.all(env_stage == int(RewardStage.DENSE_SKILL)):
        raise ValueError(f"{path}: environment is not dense_skill")
    if not np.allclose(env_mix, 0.0, atol=0.0, rtol=0.0):
        raise ValueError(f"{path}: terminal reward mix is not zero")
    if checkpoint.get("reward_anneal_state") is not None:
        raise ValueError(f"{path}: dense control must not store anneal state")

    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameter_count": parameter_count,
        "sampler": verify_sampler_state(checkpoint.get("pooled_sampler_state")),
    }


def checkpoint_pair(run: Path) -> tuple[Path, Path]:
    periodic = list((run / "checkpoints").glob("*_update_000001.pkl"))
    final = list((run / "checkpoints").glob("*_FINAL.pkl"))
    if len(periodic) != 1 or len(final) != 1:
        raise ValueError(
            f"{run}: expected one update-1 and one FINAL checkpoint, "
            f"found {len(periodic)} and {len(final)}"
        )
    return periodic[0], final[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=sorted(ARCHITECTURES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    helpers.register_checkpoint_config_classes()
    periodic, final = checkpoint_pair(args.run)
    output = {
        "schema": "terra_v8_architecture_smoke_v1",
        "passed": True,
        "arm": args.arm,
        "architecture": ARCHITECTURES[args.arm],
        "common_training_contract": COMMON_TRAINING,
        "checkpoints": {
            "periodic": one(periodic, args.arm),
            "final": one(final, args.arm),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
