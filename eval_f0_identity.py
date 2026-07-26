#!/usr/bin/env python3
"""Evaluate one F0 map identity on 32 frozen reset seeds."""

from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from eval_fixed_bank import (
    environment_completion_contract,
    sha256_file,
)
from eval_mcts import rollout_episode
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.helpers import load_pkl_object

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

RESET_SEEDS = tuple(range(2026072600, 2026072632))
HORIZON = 450
CHECKPOINT_CADENCE = 100
PRODUCTION_UPDATES = 1_000
PRODUCTION_TIMESTEPS = 131_072_000
REQUIRED_SUCCESSES = 29
NUM_TRACKED_ACTIONS = 8
IDENTITIES = {
    "foundation": {
        "apply_trench_rewards": False,
        "family": "foundation",
        "primary_cell": "all_around_low_volume",
        "train_seed": 2026072601,
    },
    "trench": {
        "apply_trench_rewards": True,
        "family": "trench",
        "primary_cell": "straight_both_side_low_volume",
        "train_seed": 2026072602,
    },
}
F0_TREATMENTS = {
    "corrected_dense_v1": {
        "foundation": {
            "config_name": "f0_foundation_identity_v1",
            "apply_trench_rewards": False,
        },
        "trench": {
            "config_name": "f0_trench_identity_v1",
            "apply_trench_rewards": True,
        },
    },
    "corrected_dense_v1_trench_absolute_off": {
        "trench": {
            "config_name": "f0_trench_identity_shaping_off_v1",
            "apply_trench_rewards": False,
        },
    },
}


def treatment_spec(identity: str, treatment: str) -> dict:
    try:
        return F0_TREATMENTS[treatment][identity]
    except KeyError as error:
        raise ValueError(
            f"unsupported F0 identity/treatment pair: {identity}/{treatment}"
        ) from error


def load_single_manifest(directory: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (directory / "manifest.jsonl").read_text().splitlines()
    ]
    if len(rows) != 1 or int(rows[0]["slot_index"]) != 1:
        raise RuntimeError(f"{directory} must contain exactly manifest slot 1")
    return rows[0]


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


def verify_production_checkpoint(
    checkpoint: dict,
    identity: str,
    expected_update: int,
    treatment: str = "corrected_dense_v1",
) -> dict:
    identity_spec = IDENTITIES[identity]
    reward_spec = treatment_spec(identity, treatment)
    if int(checkpoint["next_update"]) != expected_update:
        raise RuntimeError(
            "F0 checkpoint update mismatch: "
            f"{checkpoint['next_update']} != {expected_update}"
        )
    if "optimizer_state" not in checkpoint:
        raise RuntimeError("F0 checkpoint lacks optimizer state")

    expected = {
        "config_name": reward_spec["config_name"],
        "seed": identity_spec["train_seed"],
        "num_devices": 4,
        "num_envs_per_device": 1024,
        "num_steps": 32,
        "num_updates": PRODUCTION_UPDATES,
        "actual_total_timesteps": PRODUCTION_TIMESTEPS,
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
        "resume_from": None,
        "warm_start_from": None,
        "teacher_checkpoint": None,
    }
    config = checkpoint["train_config"]
    observed = {name: config_value(config, name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            "F0 production checkpoint config mismatch: "
            f"observed={observed}, expected={expected}"
        )

    levels = config_value(config, "curriculum_levels_override")
    if len(levels) != 1:
        raise RuntimeError("F0 checkpoint must have exactly one map level")
    level = levels[0]
    if (
        level["maps_path"] != identity
        or int(level["max_steps_in_episode"]) != HORIZON
        or bool(level["apply_trench_rewards"]) != reward_spec["apply_trench_rewards"]
    ):
        raise RuntimeError(f"F0 checkpoint map level mismatch: {level}")

    transition_integrity = checkpoint.get("transition_integrity")
    if transition_integrity is None or any(
        int(value) != 0 for value in transition_integrity.values()
    ):
        raise RuntimeError(
            f"F0 checkpoint transition integrity failed: {transition_integrity}"
        )
    return {
        "passed": True,
        "checked_config": observed,
        "transition_integrity": transition_integrity,
        "model_leaf_count": assert_finite_tree(
            checkpoint["model"],
            "F0 model",
        ),
        "optimizer_leaf_count": assert_finite_tree(
            checkpoint["optimizer_state"],
            "F0 optimizer",
        ),
    }


def configure_for_identity(
    train_config,
    identity: str,
    treatment: str = "corrected_dense_v1",
):
    reward_spec = treatment_spec(identity, treatment)
    config = copy.deepcopy(train_config)
    config.num_devices = 1
    config.num_envs_per_device = len(RESET_SEEDS)
    config.num_envs = len(RESET_SEEDS)
    config.num_test_rollouts = len(RESET_SEEDS)
    config.eval_episodes = len(RESET_SEEDS)
    config.eval_episodes_per_device = len(RESET_SEEDS)
    config.num_minibatches = min(
        int(getattr(config, "num_minibatches", 16)),
        len(RESET_SEEDS),
    )
    if len(RESET_SEEDS) % config.num_minibatches != 0:
        raise RuntimeError("F0 reset count must be divisible by evaluator minibatches")
    config.agent_types_override = (0,)
    config.action_types_override = (0,)
    config.curriculum_levels_override = [
        {
            "maps_path": identity,
            "max_steps_in_episode": HORIZON,
            "rewards_type": 0,
            "apply_trench_rewards": reward_spec["apply_trench_rewards"],
        }
    ]
    config.curriculum_increase_level_threshold = 20
    config.curriculum_decrease_level_threshold = 80
    config.curriculum_last_level_type = "none"
    config.single_map_path = None
    config.replay_map_count = 0
    config.target_map_repeat = 0
    config.teacher_checkpoint = None
    config.teacher_obs_downsample = 1
    config.resume_from = None
    config.warm_start_from = None
    config.resume_update = None
    config.load_env_from_checkpoint = False
    return config


def declared_reset_keys() -> jax.Array:
    seeds = jnp.asarray(RESET_SEEDS, dtype=jnp.uint32)
    return jax.vmap(jax.random.PRNGKey)(seeds)


def verify_single_identity_reset(
    env,
    env_params,
    reset_keys: jax.Array,
    directory: Path,
    manifest: dict,
) -> dict:
    timestep = env.reset(env_params, reset_keys)
    count = len(RESET_SEEDS)
    observed = {
        "target": np.asarray(timestep.state.world.target_map.map),
        "initial_action": np.asarray(timestep.state.world.action_map.map),
        "occupancy": np.asarray(timestep.state.world.padding_mask.map),
        "dumpability": np.asarray(timestep.state.world.dumpability_mask_init.map),
        "distance": np.asarray(timestep.state.world.relocation_distance_map),
    }
    source_directories = {
        "target": "images",
        "initial_action": "actions",
        "occupancy": "occupancy",
        "dumpability": "dumpability",
        "distance": "distance",
    }
    for field, subdirectory in source_directories.items():
        expected = np.load(directory / subdirectory / "img_1.npy")
        expected_batch = np.broadcast_to(
            np.squeeze(expected),
            (count,) + np.squeeze(expected).shape,
        )
        equal = (
            np.allclose(observed[field], expected_batch, rtol=0.0, atol=1e-7)
            if field == "distance"
            else np.array_equal(observed[field], expected_batch)
        )
        if not equal:
            raise RuntimeError(f"F0 exact reset {field} mismatch")

    expected_metadata = {
        "trench_axes": np.asarray(env.maps_buffer.trench_axes[0, 0]),
        "trench_type": np.asarray(env.maps_buffer.trench_types[0, 0]),
        "foundation_border_axes": np.asarray(
            env.maps_buffer.foundation_border_axes[0, 0]
        ),
        "foundation_border_type": np.asarray(
            env.maps_buffer.foundation_border_types[0, 0]
        ),
    }
    observed_metadata = {
        "trench_axes": np.asarray(timestep.state.world.trench_axes),
        "trench_type": np.asarray(timestep.state.world.trench_type),
        "foundation_border_axes": np.asarray(
            timestep.state.world.foundation_border_axes
        ),
        "foundation_border_type": np.asarray(
            timestep.state.world.foundation_border_type
        ),
    }
    for field, expected in expected_metadata.items():
        expected_batch = np.broadcast_to(
            expected,
            (count,) + expected.shape,
        )
        if not np.array_equal(observed_metadata[field], expected_batch):
            raise RuntimeError(f"F0 exact reset {field} mismatch")

    env_steps = np.asarray(timestep.state.env_steps)
    if env_steps.shape != (count,) or np.any(env_steps != 0):
        raise RuntimeError("F0 resets must start with env_steps == 0")

    provenance_fn = jax.vmap(env.maps_buffer.get_map_provenance)
    slot_indices, family_ids, primary_cell_ids, _ = provenance_fn(
        reset_keys,
        env_params,
    )
    slots = np.asarray(slot_indices, dtype=np.int32)
    families = np.asarray(family_ids, dtype=np.int32)
    primary_cells = np.asarray(primary_cell_ids, dtype=np.int32)
    if np.any(slots != 0):
        raise RuntimeError("F0 reset selected a nonzero manifest slot")
    family_names = {env.maps_buffer.family_names[index] for index in families.tolist()}
    primary_cell_names = {
        env.maps_buffer.primary_cell_names[index] for index in primary_cells.tolist()
    }
    if family_names != {manifest["family"]}:
        raise RuntimeError(f"F0 family provenance mismatch: {family_names}")
    if primary_cell_names != {manifest["primary_cell"]}:
        raise RuntimeError(f"F0 primary-cell provenance mismatch: {primary_cell_names}")

    return {
        "passed": True,
        "episodes": count,
        "reset_seeds": list(RESET_SEEDS),
        "env_steps_min": int(env_steps.min()),
        "env_steps_max": int(env_steps.max()),
        "slot_indices_zero_based": sorted(set(slots.tolist())),
        "family": manifest["family"],
        "primary_cell": manifest["primary_cell"],
        "layer_sha256": {
            field: sha256_file(directory / subdirectory / "img_1.npy")
            for field, subdirectory in source_directories.items()
        },
        "metadata_sha256": sha256_file(directory / "metadata" / "trench_1.json"),
    }


def summarize_rollout(stats: dict, cumulative_rewards: np.ndarray) -> dict:
    count = len(RESET_SEEDS)
    successes = np.asarray(stats["episode_done_once"], dtype=bool)
    terminations = np.asarray(
        stats["episode_terminated_once"],
        dtype=bool,
    )
    lengths = np.asarray(stats["episode_length"], dtype=np.int32)
    completion = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in stats["terminal_completion"].items()
    }
    integrity = stats["integrity"]
    if not bool(integrity.get("supported", False)):
        raise RuntimeError("F0 evaluator lacks state-integrity support")
    integrity_values = {
        key: np.asarray(value) for key, value in integrity.items() if key != "supported"
    }

    for name, values in {
        "successes": successes,
        "terminations": terminations,
        "lengths": lengths,
        **completion,
        **integrity_values,
    }.items():
        if np.asarray(values).shape != (count,):
            raise RuntimeError(
                f"F0 rollout field {name} has shape "
                f"{np.asarray(values).shape}, expected {(count,)}"
            )

    absolute = completion["absolute"]
    completion_one = np.isclose(absolute, 1.0, atol=1e-6)
    if not np.array_equal(successes, completion_one):
        raise RuntimeError("F0 task_done does not match absolute_completion == 1")
    expected_termination = successes | (lengths >= HORIZON)
    if not np.array_equal(terminations, expected_termination):
        raise RuntimeError("F0 evaluator/environment termination disagreement")
    if np.any(
        np.asarray(
            integrity_values["slot_index_zero_based"],
            dtype=np.int32,
        )
        != 0
    ):
        raise RuntimeError("F0 terminal provenance changed manifest slot")

    maximum_mass_residual = np.asarray(
        integrity_values["maximum_mass_residual"],
        dtype=np.int32,
    )
    target_mutation = np.asarray(
        integrity_values["target_mutation"],
        dtype=bool,
    )
    obstacle_mutation = np.asarray(
        integrity_values["obstacle_mutation"],
        dtype=bool,
    )
    nonfinite_state = np.asarray(
        integrity_values["nonfinite_state"],
        dtype=bool,
    )
    integrity_failure = (
        (maximum_mass_residual != 0)
        | target_mutation
        | obstacle_mutation
        | nonfinite_state
    )
    returns = np.asarray(cumulative_rewards[-1], dtype=np.float64)
    if not np.all(np.isfinite(returns)):
        raise RuntimeError("F0 rollout returned non-finite rewards")

    action_sequence = np.asarray(stats["action_sequence"], dtype=np.int32)
    action_had_effect = np.asarray(
        stats["action_had_effect_sequence"],
        dtype=bool,
    )
    if (
        action_sequence.shape != action_had_effect.shape
        or action_sequence.shape[1] != count
    ):
        raise RuntimeError("F0 action trace has an invalid shape")
    if np.any(action_sequence < 0) or np.any(action_sequence >= NUM_TRACKED_ACTIONS):
        raise RuntimeError("F0 action trace contains an invalid action index")

    per_reset = []
    for index, seed in enumerate(RESET_SEEDS):
        timed_out = bool(terminations[index] and lengths[index] >= HORIZON)
        if successes[index] and timed_out:
            reason = "task_done_and_timeout"
        elif successes[index]:
            reason = "task_done"
        elif timed_out:
            reason = "timeout"
        else:
            reason = "horizon_censored"
        per_reset.append(
            {
                "reset_seed": seed,
                "success": bool(successes[index]),
                "terminated": bool(terminations[index]),
                "termination_reason": reason,
                "steps": int(lengths[index]),
                "return": float(returns[index]),
                "terminal_absolute_completion": float(absolute[index]),
                "terminal_dig_completion": float(completion["dig"][index]),
                "terminal_dump_purity": float(completion["dump_purity"][index]),
                "terminal_dump_volume": float(completion["dump_volume"][index]),
                "terminal_unloaded_completion": float(completion["unloaded"][index]),
                "accepted_dump_volume": float(
                    completion["accepted_dump_volume"][index]
                ),
                "illegal_dump_volume": float(completion["illegal_dump_volume"][index]),
                "maximum_mass_residual": int(maximum_mass_residual[index]),
                "no_effect_action_count": int(
                    integrity_values["no_effect_action_count"][index]
                ),
                "target_mutation": bool(target_mutation[index]),
                "obstacle_mutation": bool(obstacle_mutation[index]),
                "nonfinite_state": bool(nonfinite_state[index]),
                "integrity_failure": bool(integrity_failure[index]),
            }
        )

    success_indices = np.flatnonzero(successes)
    successful_trajectory = None
    if success_indices.size:
        index = int(success_indices[0])
        length = int(lengths[index])
        actions = action_sequence[:length, index]
        effects = action_had_effect[:length, index]
        successful_trajectory = {
            "reset_seed": RESET_SEEDS[index],
            "steps": length,
            "actions": actions.tolist(),
            "action_had_effect": effects.tolist(),
            "all_actions_in_discrete_range": bool(
                np.all((actions >= 0) & (actions < NUM_TRACKED_ACTIONS))
            ),
            "final_success": True,
        }

    success_count = int(successes.sum())
    integrity_failure_count = int(integrity_failure.sum())
    return {
        "successes": success_count,
        "episodes": count,
        "success_rate": success_count / count,
        "integrity_failure_count": integrity_failure_count,
        "performance_passed": success_count >= REQUIRED_SUCCESSES,
        "integrity_passed": integrity_failure_count == 0,
        "passed": (
            success_count >= REQUIRED_SUCCESSES and integrity_failure_count == 0
        ),
        "termination_reasons": {
            reason: sum(int(row["termination_reason"] == reason) for row in per_reset)
            for reason in (
                "task_done",
                "timeout",
                "task_done_and_timeout",
                "horizon_censored",
            )
        },
        "per_reset": per_reset,
        "successful_action_trajectory": successful_trajectory,
    }


def consecutive_mastery(records: list[dict]) -> dict:
    passing_pairs = []
    for previous, current in itertools.pairwise(records):
        if (
            current["checkpoint_update"] - previous["checkpoint_update"]
            == CHECKPOINT_CADENCE
            and previous["summary"]["passed"]
            and current["summary"]["passed"]
        ):
            passing_pairs.append(
                [
                    previous["checkpoint_update"],
                    current["checkpoint_update"],
                ]
            )
    trajectory_saved = any(
        record["summary"]["successful_action_trajectory"] is not None
        for record in records
        if record["summary"]["passed"]
    )
    return {
        "required_successes": REQUIRED_SUCCESSES,
        "episodes": len(RESET_SEEDS),
        "required_consecutive_evaluations": 2,
        "checkpoint_cadence": CHECKPOINT_CADENCE,
        "passing_update_pairs": passing_pairs,
        "trajectory_saved_from_passing_checkpoint": trajectory_saved,
        "passed": bool(passing_pairs) and trajectory_saved,
    }


def checkpoint_paths(pattern: str) -> list[Path]:
    paths = sorted({Path(path).resolve() for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {pattern}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-glob",
        required=True,
        help="Glob for update-numbered F0 checkpoints.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--identity",
        choices=tuple(IDENTITIES),
        required=True,
    )
    parser.add_argument(
        "--treatment",
        choices=tuple(F0_TREATMENTS),
        default="corrected_dense_v1",
    )
    parser.add_argument("--expected-checkpoints", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if environment_completion_contract() != "exact_visible_dump_v1":
        raise RuntimeError("F0 requires exact_visible_dump_v1")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    directory = dataset_root / args.identity
    manifest = load_single_manifest(directory)
    identity_spec = IDENTITIES[args.identity]
    if (
        manifest["family"] != identity_spec["family"]
        or manifest["primary_cell"] != identity_spec["primary_cell"]
    ):
        raise RuntimeError("F0 identity manifest does not match its evaluator")

    paths = checkpoint_paths(args.checkpoint_glob)
    if len(paths) != args.expected_checkpoints:
        raise RuntimeError(
            f"expected {args.expected_checkpoints} checkpoints, got {len(paths)}"
        )
    checkpoints = [(path, load_pkl_object(str(path))) for path in paths]
    checkpoints.sort(key=lambda item: int(item[1].get("next_update", 0)))
    updates = [int(checkpoint.get("next_update", 0)) for _, checkpoint in checkpoints]
    expected_updates = list(
        range(
            CHECKPOINT_CADENCE, CHECKPOINT_CADENCE * len(paths) + 1, CHECKPOINT_CADENCE
        )
    )
    if updates != expected_updates:
        raise RuntimeError(
            f"F0 checkpoint cadence mismatch: {updates}, expected {expected_updates}"
        )

    reference_config = checkpoints[0][1]["train_config"]
    checkpoint_gates = {}
    for _, checkpoint in checkpoints:
        if "model" not in checkpoint:
            raise KeyError("F0 checkpoint has no model parameters")
        _validate_checkpoint_architecture(checkpoint, reference_config)
        update = int(checkpoint["next_update"])
        checkpoint_gates[update] = verify_production_checkpoint(
            checkpoint,
            args.identity,
            update,
            args.treatment,
        )

    os.environ["DATASET_PATH"] = str(dataset_root)
    os.environ["DATASET_SIZE"] = "1"
    config = configure_for_identity(
        reference_config,
        args.identity,
        args.treatment,
    )
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    reset_keys = declared_reset_keys()
    reset_verification = verify_single_identity_reset(
        env,
        env_params,
        reset_keys,
        directory,
        manifest,
    )
    model = SimpleNamespace(apply=initialized_state.apply_fn)

    payload = {
        "schema": "terra_f0_identity_eval_v1",
        "completion_contract": "exact_visible_dump_v1",
        "reward_contract": args.treatment,
        "identity": args.identity,
        "map_id": manifest["map_id"],
        "source_id": manifest["source_id"],
        "family": manifest["family"],
        "primary_cell": manifest["primary_cell"],
        "dataset_root": str(dataset_root),
        "dataset_sha256": sha256_file(directory / "dataset.json"),
        "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
        "horizon": HORIZON,
        "deterministic": True,
        "reset_seeds": list(RESET_SEEDS),
        "reset_verification": reset_verification,
        "records": [],
        "mastery_gate": {
            "passed": False,
            "reason": "evaluation_incomplete",
        },
    }
    for checkpoint_path, checkpoint in checkpoints:
        cumulative_rewards, stats, _ = rollout_episode(
            env,
            model,
            checkpoint["model"],
            env_params,
            config,
            max_frames=HORIZON,
            deterministic=True,
            seed=RESET_SEEDS[0],
            use_mcts=False,
            reset_keys=reset_keys,
            record_observations=False,
            record_actions=True,
            preserve_terminal_states=True,
            expected_slot_indices=np.zeros(
                len(RESET_SEEDS),
                dtype=np.int32,
            ),
        )
        summary = summarize_rollout(stats, cumulative_rewards)
        record = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_update": int(checkpoint["next_update"]),
            "checkpoint_gate": checkpoint_gates[int(checkpoint["next_update"])],
            "summary": summary,
        }
        payload["records"].append(record)
        payload["mastery_gate"] = consecutive_mastery(payload["records"])
        with output.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"{checkpoint_path.name}: "
            f"{summary['successes']}/{summary['episodes']} success, "
            f"integrity_failures={summary['integrity_failure_count']}",
            flush=True,
        )

    print(
        "F0_MASTERY_GATE="
        f"{'PASSED' if payload['mastery_gate']['passed'] else 'FAILED'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
