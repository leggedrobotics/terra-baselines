#!/usr/bin/env python3
"""Verify both update-1 checkpoints before either long reward arm is submitted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from terra.config import RewardStage
from utils import helpers
from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state

EXPECTED_PARAMETERS = 2_856_685


def array_exact(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def assert_tree_exact(left: object, right: object, label: str) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        raise ValueError(f"{label}: tree structure differs between reward arms")
    for index, (left_leaf, right_leaf) in enumerate(
        zip(left_leaves, right_leaves, strict=True)
    ):
        if not array_exact(left_leaf, right_leaf):
            raise ValueError(f"{label}: leaf {index} differs between reward arms")


def assert_nested_exact(left: object, right: object, label: str) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"{label}: dictionary fields differ")
        for key in sorted(left):
            assert_nested_exact(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise ValueError(f"{label}: sequence lengths differ")
        for index, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            assert_nested_exact(left_value, right_value, f"{label}[{index}]")
        return
    if not array_exact(left, right):
        raise ValueError(f"{label}: values differ")


def one(path: Path, reward_stage: str) -> dict:
    checkpoint = helpers.load_pkl_object(str(path))
    if checkpoint.get("next_update") != 1:
        raise ValueError(f"{path}: next_update must be 1")
    parameter_count = sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(checkpoint["model"])
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"{path}: unexpected parameter count {parameter_count}")

    train_config = checkpoint["train_config"]
    observed_stage = (
        train_config["reward_stage"]
        if isinstance(train_config, dict)
        else train_config.reward_stage
    )
    if observed_stage != reward_stage:
        raise ValueError(
            f"{path}: train reward stage {observed_stage!r} != {reward_stage!r}"
        )

    env_config = checkpoint["env_config"]
    expected_stage = {
        "dense_skill": int(RewardStage.DENSE_SKILL),
        "annealed_objective": int(RewardStage.ANNEALED_OBJECTIVE),
    }[reward_stage]
    env_stage = np.asarray(env_config.reward_stage)
    env_mix = np.asarray(env_config.terminal_reward_mix, dtype=np.float64)
    if not np.all(env_stage == expected_stage):
        raise ValueError(f"{path}: env reward stage is not {expected_stage}")
    if not np.allclose(env_mix, 0.0, atol=0.0, rtol=0.0):
        raise ValueError(f"{path}: update-1 terminal mix is not zero")

    anneal_state = checkpoint.get("reward_anneal_state")
    if reward_stage == "dense_skill":
        if anneal_state is not None:
            raise ValueError(f"{path}: dense arm unexpectedly stores anneal state")
    else:
        expected = {
            "schema": "terra_reward_anneal_v1",
            "started_update": None,
            "duration_updates": 5000,
            "last_applied_mix": 0.0,
        }
        if anneal_state != expected:
            raise ValueError(f"{path}: unexpected anneal state {anneal_state!r}")

    sampler = verify_sampler_state(checkpoint.get("pooled_sampler_state"))
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reward_stage": reward_stage,
        "terminal_reward_mix": float(np.asarray(env_mix).reshape(-1)[0]),
        "reward_anneal_state": anneal_state,
        "parameter_count": parameter_count,
        "sampler": sampler,
    }


def checkpoint_pair(directory: Path) -> tuple[Path, Path]:
    periodic = list((directory / "checkpoints").glob("*_update_000001.pkl"))
    final = list((directory / "checkpoints").glob("*_FINAL.pkl"))
    if len(periodic) != 1 or len(final) != 1:
        raise ValueError(
            f"{directory}: expected one update-1 and one FINAL checkpoint, "
            f"found {len(periodic)} and {len(final)}"
        )
    return periodic[0], final[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True, type=Path)
    parser.add_argument("--annealed-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    helpers.register_checkpoint_config_classes()
    records = {}
    checkpoints = {}
    for arm, directory, reward_stage in (
        ("constant_dense", args.dense_run, "dense_skill"),
        ("dense_to_terminal", args.annealed_run, "annealed_objective"),
    ):
        periodic, final = checkpoint_pair(directory)
        checkpoints[arm] = {
            "periodic": helpers.load_pkl_object(str(periodic)),
            "final": helpers.load_pkl_object(str(final)),
        }
        records[arm] = {
            "periodic": one(periodic, reward_stage),
            "final": one(final, reward_stage),
        }

    for checkpoint_kind in ("periodic", "final"):
        dense = checkpoints["constant_dense"][checkpoint_kind]
        annealed = checkpoints["dense_to_terminal"][checkpoint_kind]
        assert_tree_exact(dense["model"], annealed["model"], f"{checkpoint_kind}.model")
        assert_tree_exact(
            dense["optimizer_state"],
            annealed["optimizer_state"],
            f"{checkpoint_kind}.optimizer_state",
        )
        assert_tree_exact(
            dense["loss_info"], annealed["loss_info"], f"{checkpoint_kind}.loss_info"
        )
        assert_nested_exact(
            dense["transition_integrity"],
            annealed["transition_integrity"],
            f"{checkpoint_kind}.transition_integrity",
        )
        assert_nested_exact(
            dense["pooled_sampler_state"],
            annealed["pooled_sampler_state"],
            f"{checkpoint_kind}.pooled_sampler_state",
        )

    output = {
        "schema": "terra_continuous_reward_pair_smoke_v1",
        "passed": True,
        "pretrigger_dense_parity": True,
        "parity_fields": [
            "model",
            "optimizer_state",
            "loss_info",
            "transition_integrity",
            "pooled_sampler_state",
        ],
        "arms": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
