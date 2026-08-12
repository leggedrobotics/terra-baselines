#!/usr/bin/env python3
"""Verify the exact saved update-1 F0 checkpoint and training receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from eval_f0_identity import F0_TREATMENTS, IDENTITIES, treatment_spec
from utils import helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_value(config, name):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def assert_finite_tree(tree, label: str) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise RuntimeError(f"{label} has no leaves")
    for index, leaf in enumerate(leaves):
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise RuntimeError(f"{label} leaf {index} is non-finite")
    return len(leaves)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument(
        "--identity",
        choices=tuple(IDENTITIES),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--treatment",
        choices=tuple(F0_TREATMENTS),
        default="corrected_dense_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.checkpoint, args.aggregate):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    helpers.register_checkpoint_config_classes()
    checkpoint = helpers.load_pkl_object(str(args.checkpoint))
    if int(checkpoint["next_update"]) != 1:
        raise RuntimeError("F0 smoke checkpoint is not update 1")
    if "optimizer_state" not in checkpoint:
        raise RuntimeError("F0 smoke checkpoint lacks optimizer state")

    config = checkpoint["train_config"]
    reward_spec = treatment_spec(args.identity, args.treatment)
    expected_preset = reward_spec["config_name"]
    expected = {
        "config_name": expected_preset,
        "seed": args.seed,
        "num_devices": 4,
        "num_envs_per_device": 1024,
        "num_steps": 32,
        "num_updates": 1,
        "actual_total_timesteps": 131072,
        "lr": 3e-4,
        "update_epochs": 2,
        "num_minibatches": 32,
        "ent_schedule_start": 0.15,
        "ent_schedule_end": 0.005,
        "ent_schedule_steps": 950,
        "model_size": "base",
        "model_core": "mlp",
        "map_encoder": "resnet_spatial_8x8",
        "encoder_compute_dtype": "float32",
        "use_value_clip": False,
        "flat_minibatch_shuffle": True,
        "fail_on_nonfinite": True,
        "finite_check_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoint_history": True,
        "log_eval_interval": 0,
        "resume_from": None,
        "warm_start_from": None,
        "teacher_checkpoint": None,
    }
    observed = {name: config_value(config, name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"F0 smoke config mismatch: observed={observed}, expected={expected}"
        )
    levels = config_value(config, "curriculum_levels_override")
    if len(levels) != 1:
        raise RuntimeError("F0 smoke must have exactly one map level")
    level = levels[0]
    if (
        level["maps_path"] != args.identity
        or int(level["max_steps_in_episode"]) != 450
        or bool(level["apply_trench_rewards"]) != reward_spec["apply_trench_rewards"]
    ):
        raise RuntimeError(f"F0 smoke map level mismatch: {level}")

    model_leaf_count = assert_finite_tree(
        checkpoint["model"],
        "model",
    )
    optimizer_leaf_count = assert_finite_tree(
        checkpoint["optimizer_state"],
        "optimizer",
    )
    transition_integrity = checkpoint.get("transition_integrity")
    if transition_integrity is None or any(
        int(value) != 0 for value in transition_integrity.values()
    ):
        raise RuntimeError(
            f"F0 smoke transition integrity failed: {transition_integrity}"
        )

    aggregate = json.loads(args.aggregate.read_text())
    if (
        aggregate["schema"] != "terra_training_episode_aggregate_v2"
        or aggregate["contract"] != "exact_visible_dump_v1"
        or int(aggregate["update"]) != 1
    ):
        raise RuntimeError("F0 smoke aggregate contract mismatch")
    totals = aggregate["totals"]
    for field in (
        "mass_residual_violation_count",
        "target_mutation_count",
        "obstacle_mutation_count",
        "step_reward_residual_violation_count",
    ):
        if int(totals[field]) != 0:
            raise RuntimeError(f"F0 smoke aggregate {field} is nonzero")
    expected_family = IDENTITIES[args.identity]["family"]
    expected_cell = IDENTITIES[args.identity]["primary_cell"]
    if expected_family not in aggregate["family_names"]:
        raise RuntimeError("F0 smoke aggregate lacks family provenance")
    if expected_cell not in aggregate["primary_cell_names"]:
        raise RuntimeError("F0 smoke aggregate lacks primary-cell provenance")

    receipt = {
        "schema": "terra_f0_update1_smoke_v1",
        "status": "passed",
        "identity": args.identity,
        "reward_contract": args.treatment,
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "aggregate": str(args.aggregate.resolve()),
        "aggregate_sha256": sha256_file(args.aggregate),
        "model_leaf_count": model_leaf_count,
        "optimizer_leaf_count": optimizer_leaf_count,
        "transition_integrity": transition_integrity,
        "checked_config": observed,
    }
    with args.output.open("w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"F0_{args.identity.upper()}_UPDATE1_SMOKE_PASSED")


if __name__ == "__main__":
    main()
