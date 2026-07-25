#!/usr/bin/env python3
"""Evaluate every frozen manifest slot exactly once at a fixed horizon."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

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

LEGACY_COMPLETION_CONTRACT = "legacy_implicit_buffer_v0"


def environment_completion_contract() -> str:
    try:
        from terra.state import CORRECTED_DENSE_CONTRACT
    except ImportError:
        return LEGACY_COMPLETION_CONTRACT
    return str(CORRECTED_DENSE_CONTRACT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_map_indices(keys: jax.Array, count: int) -> np.ndarray:
    def selected(key):
        _, subkey = jax.random.split(key)
        return jax.random.randint(subkey, (), 0, count)

    return np.asarray(jax.vmap(selected)(keys))


def exact_reset_keys(count: int) -> jax.Array:
    found: list[np.ndarray | None] = [None] * count
    start = 0
    while any(key is None for key in found):
        seeds = jnp.arange(start, start + 4096, dtype=jnp.uint32)
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        indices = selected_map_indices(keys, count)
        keys_host = np.asarray(keys)
        for key, index in zip(keys_host, indices):
            if found[int(index)] is None:
                found[int(index)] = key
        start += 4096
        if start > 1_000_000:
            raise RuntimeError(f"could not construct exact reset keys for {count} maps")
    result = jnp.asarray(np.stack(found))
    actual = selected_map_indices(result, count)
    np.testing.assert_array_equal(actual, np.arange(count))
    return result


def load_manifest(directory: Path) -> list[dict]:
    path = directory / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: int(row["slot_index"]))
    expected = list(range(1, len(rows) + 1))
    actual = [int(row["slot_index"]) for row in rows]
    if actual != expected:
        raise RuntimeError(f"{path} does not enumerate contiguous slots: {actual[:8]}")
    return rows


def configure_for_bank(train_config, relative_path: str, count: int):
    config = copy.deepcopy(train_config)
    config.num_devices = 1
    config.num_envs_per_device = count
    config.num_test_rollouts = count
    config.num_minibatches = min(int(getattr(config, "num_minibatches", 32)), count)
    if count % config.num_minibatches != 0:
        raise RuntimeError(
            f"{count} eval maps are not divisible by "
            f"num_minibatches={config.num_minibatches}"
        )
    config.agent_types_override = (0,)
    config.action_types_override = (0,)
    config.curriculum_levels_override = [
        {
            "maps_path": relative_path,
            "max_steps_in_episode": 450,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
    ]
    config.curriculum_increase_level_threshold = 3
    config.curriculum_decrease_level_threshold = 3
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


def verify_exact_reset(
    env,
    env_params,
    reset_keys,
    directory: Path,
    count: int,
) -> None:
    timestep = env.reset(env_params, reset_keys)
    observed = np.asarray(timestep.observation["target_map"])
    if observed.shape[0] != count:
        raise RuntimeError(f"reset produced {observed.shape[0]} maps, expected {count}")
    for index in range(count):
        expected = np.load(directory / "images" / f"img_{index + 1}.npy")
        if not np.array_equal(np.squeeze(observed[index]), expected):
            raise RuntimeError(f"exact reset mismatch at slot {index + 1}")


def grouped_results(
    rows: list[dict],
    successes: np.ndarray,
    terminations: np.ndarray,
    lengths: np.ndarray,
    *,
    horizon: int | None = None,
    completion_metrics: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    completion_metrics = completion_metrics or {}
    per_map = []
    for index, (row, success, terminated, length) in enumerate(
        zip(rows, successes, terminations, lengths)
    ):
        timed_out = bool(
            terminated
            and horizon is not None
            and int(length) >= int(horizon)
        )
        if bool(success) and timed_out:
            termination_reason = "task_done_and_timeout"
        elif bool(success):
            termination_reason = "task_done"
        elif timed_out:
            termination_reason = "timeout"
        elif bool(terminated):
            termination_reason = "other_termination"
        else:
            termination_reason = "horizon_censored"
        metric_values = {
            key: float(np.asarray(values)[index])
            for key, values in completion_metrics.items()
        }
        per_map.append(
            {
                **row,
                "success": bool(success),
                "terminated": bool(terminated),
                "timeout": timed_out,
                "termination_reason": termination_reason,
                "steps": int(length),
                **metric_values,
            }
        )

    def summarize(field: str) -> dict:
        values = sorted({row[field] for row in per_map})
        result = {}
        for value in values:
            selected = [row for row in per_map if row[field] == value]
            successes_here = sum(int(row["success"]) for row in selected)
            result[value] = {
                "successes": successes_here,
                "episodes": len(selected),
                "success_rate": successes_here / len(selected),
                "terminations": sum(int(row["terminated"]) for row in selected),
            }
        return result

    total_successes = int(successes.sum())
    summary = {
        "overall": {
            "successes": total_successes,
            "episodes": len(rows),
            "success_rate": total_successes / len(rows),
            "terminations": int(terminations.sum()),
        },
        "by_family": summarize("family"),
        "by_primary_cell": summarize("primary_cell"),
    }
    family_gate = all(item["successes"] >= 26 for item in summary["by_family"].values())
    subtype_gate = all(
        item["successes"] >= 6 for item in summary["by_primary_cell"].values()
    )
    summary["mastery_gate"] = {
        "family_26_of_32": family_gate,
        "primary_cell_6_of_8": subtype_gate,
        "passed": family_gate and subtype_gate,
    }
    summary["termination_reasons"] = {
        reason: sum(
            int(row["termination_reason"] == reason) for row in per_map
        )
        for reason in (
            "task_done",
            "timeout",
            "task_done_and_timeout",
            "other_termination",
            "horizon_censored",
        )
    }
    return per_map, summary


def checkpoint_paths(args) -> list[Path]:
    paths = [Path(path).resolve() for path in args.checkpoint]
    for pattern in args.checkpoint_glob:
        paths.extend(Path(path).resolve() for path in glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("provide --checkpoint or --checkpoint-glob")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "sealed"),
        default="development",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        choices=("M0", "M1", "M2"),
        default=("M0", "M1", "M2"),
    )
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--expect-completion-contract",
        choices=("exact_visible_dump_v1", LEGACY_COMPLETION_CONTRACT),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon != 450:
        raise ValueError("training-design-v1 fixed evaluation requires horizon 450")
    bank_root = args.bank_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completion_contract = environment_completion_contract()
    if (
        args.expect_completion_contract is not None
        and completion_contract != args.expect_completion_contract
    ):
        raise RuntimeError(
            "completion contract mismatch: expected "
            f"{args.expect_completion_contract}, imported {completion_contract}"
        )

    paths = checkpoint_paths(args)
    checkpoints = [(path, load_pkl_object(str(path))) for path in paths]
    checkpoints.sort(
        key=lambda item: (
            int(item[1].get("next_update", 0)),
            str(item[0]),
        )
    )
    reference_train_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        if "model" not in checkpoint:
            raise KeyError("checkpoint has no model parameters")
        _validate_checkpoint_architecture(checkpoint, reference_train_config)

    records = []
    for stratum in args.strata:
        relative_path = f"{args.split}/{stratum}"
        directory = bank_root / relative_path
        rows = load_manifest(directory)
        count = len(rows)
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(count)
        config = configure_for_bank(reference_train_config, relative_path, count)
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        reset_keys = exact_reset_keys(count)
        verify_exact_reset(env, env_params, reset_keys, directory, count)
        model = SimpleNamespace(apply=initialized_state.apply_fn)

        for checkpoint_path, checkpoint in checkpoints:
            _, stats, _ = rollout_episode(
                env,
                model,
                checkpoint["model"],
                env_params,
                config,
                max_frames=args.horizon,
                deterministic=not args.stochastic,
                seed=args.seed,
                use_mcts=False,
                reset_keys=reset_keys,
                record_observations=False,
            )
            successes = np.asarray(stats["episode_done_once"], dtype=bool)
            terminations = np.asarray(stats["episode_terminated_once"], dtype=bool)
            lengths = np.asarray(stats["episode_length"], dtype=np.int32)
            terminal_completion = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.get("terminal_completion", {}).items()
            }
            if (
                not np.all(np.isfinite(lengths))
                or successes.shape != (count,)
                or terminations.shape != (count,)
            ):
                raise RuntimeError("fixed evaluation returned invalid arrays")
            if completion_contract == "exact_visible_dump_v1":
                absolute = terminal_completion.get("absolute")
                if absolute is None or absolute.shape != (count,):
                    raise RuntimeError(
                        "exact_visible_dump_v1 evaluation did not return "
                        "per-map absolute completion"
                    )
                completion_one = np.isclose(absolute, 1.0, atol=1e-6)
                if not np.array_equal(successes, completion_one):
                    mismatches = np.flatnonzero(successes != completion_one)
                    raise RuntimeError(
                        "task_done <=> absolute_completion == 1 violated at "
                        f"fixed-bank slots {(mismatches + 1).tolist()}"
                    )
            per_map, summary = grouped_results(
                rows,
                successes,
                terminations,
                lengths,
                horizon=args.horizon,
                completion_metrics={
                    f"terminal_{key}": values
                    for key, values in terminal_completion.items()
                },
            )
            record = {
                "schema": "terra_fixed_bank_eval_v2",
                "completion_contract": completion_contract,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_update": int(checkpoint.get("next_update", 0)),
                "bank_root": str(bank_root),
                "split": args.split,
                "stratum": stratum,
                "manifest": str(directory / "manifest.jsonl"),
                "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
                "horizon": args.horizon,
                "deterministic": not args.stochastic,
                "seed": args.seed,
                "exact_manifest_enumeration": True,
                "summary": summary,
                "per_map": per_map,
            }
            records.append(record)
            with output.open("w") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)
                handle.write("\n")
            overall = summary["overall"]
            print(
                f"{checkpoint_path.name} {args.split}/{stratum}: "
                f"{overall['successes']}/{overall['episodes']} success"
            )


if __name__ == "__main__":
    main()
