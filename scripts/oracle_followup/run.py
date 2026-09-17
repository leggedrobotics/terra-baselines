#!/usr/bin/env python3
"""One combined broad continuation: time, actor capacity and foundation release."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
EVALUATION_UPDATES = (10000, 20000, 35000, 50000, 75000, 100000)


def evaluate_checkpoint(checkpoint, update, bank, output):
    """Pause PPO for the fixed panel; an evaluator failure preserves training."""
    if update not in EVALUATION_UPDATES:
        return
    output.mkdir(parents=True, exist_ok=True)
    report = output / f"u{update}.json"
    if report.exists():
        from scripts.foundation_teacher_release.compare import load_report
        try:
            if load_report(report)["checkpoint_update"] != update:
                raise ValueError("Evaluation belongs to another checkpoint update")
        except (ValueError, KeyError, OSError):
            report.rename(report.with_suffix(f".incomplete-{int(time.time())}.json"))
        else:
            return
    env = dict(os.environ, BANK_ROOT=str(bank), CUDA_VISIBLE_DEVICES="0",
               WANDB_MODE="disabled", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    print(f"Fixed evaluation starting at u{update}: {report}", flush=True)
    with (output / f"u{update}.log").open("a") as log:
        result = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", "1800",
             "bash", str(REPO / "scripts/excavation_reliability/eval.sh"),
             str(checkpoint), str(report)],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    status = {"update": update, "checkpoint": str(checkpoint),
              "returncode": result.returncode, "report": str(report)}
    if result.returncode == 0:
        from scripts.foundation_teacher_release.compare import load_report
        try:
            if load_report(report)["checkpoint_update"] != update:
                raise ValueError("Evaluation belongs to another checkpoint update")
        except (ValueError, KeyError, OSError) as error:
            status.update(status="FAILED_VALIDATION", error=str(error))
        else:
            status["status"] = "PASS"
    else:
        status["status"] = "FAILED"
    (output / f"u{update}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Fixed evaluation u{update}: {status['status']}; continuing PPO", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Existing broad campaign inputs: bank and both teachers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-update", type=int, required=True,
                        help="Absolute update ceiling, at most the authorized u100000")
    parser.add_argument("--eval-bank", type=Path)
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="1 GPU x 128 envs, at most 2 new updates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.eval_bank is None) != (args.eval_output is None):
        parser.error("--eval-bank and --eval-output must be provided together")
    args.checkpoint = args.checkpoint.resolve()
    if args.eval_bank is not None:
        args.eval_bank = args.eval_bank.resolve()
        args.eval_output = args.eval_output.resolve()
        if not args.eval_bank.is_dir():
            raise FileNotFoundError(args.eval_bank)
    inputs = args.inputs.resolve()
    for path in (args.checkpoint, inputs / "bank", inputs / "foundation_teacher.pkl",
                 inputs / "trench_teacher.pkl"):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["DATASET_PATH"] = str(inputs / "bank")
    os.environ["DATASET_SIZE"] = "3840"
    os.environ.setdefault("WANDB_MODE", "disabled")

    import jax
    import numpy as np
    from train_mixed import MixedAgentTrainConfig, train_mixed_agents
    from utils.helpers import load_pkl_object, register_checkpoint_config_classes

    register_checkpoint_config_classes()
    parent = load_pkl_object(str(args.checkpoint))
    saved = parent["train_config"]
    values = {field.name: getattr(saved, field.name, field.default)
              for field in dataclasses.fields(MixedAgentTrainConfig) if field.init}
    start = int(parent["next_update"])
    release = parent.get("foundation_teacher_release_state")
    target_update = args.target_update
    if start < 5000 or not start <= target_update <= 100000:
        raise ValueError("Use a broad native parent and an absolute target at or after it, at most u100000")
    if args.smoke and target_update - start > 2:
        raise ValueError("Smoke is limited to two finite updates")
    if int(np.asarray(parent["train_state_step"])) != start * 64:
        raise ValueError("Parent Adam clock does not match 64 steps per update")
    for key in ("model", "optimizer_state"):
        if not all(np.isfinite(np.asarray(leaf)).all()
                   for leaf in jax.tree_util.tree_leaves(parent[key])):
            raise ValueError(f"Nonfinite parent {key}")
    if saved.config_name != "trench_align_v2_generalist_gen":
        raise ValueError("Expected the broad generalist parent")
    if any(getattr(saved, key, 0.) != 0 for key in
           ("lateral_dig_cost", "base_travel_cost", "base_turn_cost",
            "retained_work_setup_cost", "retained_work_travel_cost", "retained_work_turn_cost")):
        raise ValueError("This experiment retains zero added behavior costs")
    if (saved.num_steps, saved.update_epochs, saved.num_minibatches) != (32, 2, 32):
        raise ValueError("Unexpected parent PPO layout")
    if not args.smoke and (saved.num_devices, saved.num_envs_per_device) != (4, 256):
        raise ValueError("Production must preserve the parent's 4x256 batch; smoke checkpoints are diagnostic only")
    if not args.smoke and start % 250:
        raise ValueError("Select a 250-update milestone parent for the fixed evaluation checkpoints")
    devices, envs = (1, 128) if args.smoke else (4, 256)
    output = args.output.resolve()
    overrides = dict(
        name="generalist-oracle-combined",
        num_devices=devices, num_envs_per_device=envs,
        total_timesteps=target_update * devices * envs * saved.num_steps,
        resume_from=str(args.checkpoint.resolve()), warm_start_from=None,
        resume_update=None, load_env_from_checkpoint=True,
        checkpoint_dir=str(output / "checkpoints"), keep_checkpoint_history=True,
        checkpoint_interval=1 if args.smoke else 250,
        log_train_interval=1 if args.smoke else 10, log_eval_interval=0,
        initialization_receipt=str(output / "initialization.json"),
        teacher_checkpoint=str(inputs / "foundation_teacher.pkl"),
        trench_teacher_checkpoint=str(inputs / "trench_teacher.pkl"),
        fail_on_nonfinite=True, finite_check_interval=1 if args.smoke else 10,
        time_observation_mode="remaining",
        migrate_remaining_time=getattr(saved, "time_observation_mode", "none") == "none",
        actor_residual_head=True,
        grow_actor_capacity=not getattr(saved, "actor_residual_head", False),
        cache_teacher_outputs=True,
        foundation_teacher_release_updates=(
            1250 if release is None else 0
        ),
    )
    values.update(overrides)
    config = MixedAgentTrainConfig(**values)
    plan = dict(
        recipe="time_actor_capacity_foundation_release_teacher_cache", parent=str(args.checkpoint.resolve()), start_update=start,
        target_update=target_update, global_envs=devices * envs,
        transitions_per_update=config.env_steps_per_update,
        additional_transitions=(target_update - start) * config.env_steps_per_update,
        smoke=args.smoke, restored_release_state=release,
        configuration_changes={key: {"parent": getattr(saved, key, None), "new": value}
                               for key, value in overrides.items()
                               if getattr(saved, key, None) != value},
        release_transition_duration=1250 * 32768,
        objectives="Corrected physics and reward-v2; every added behavior cost remains zero",
        interpretation="Combined intervention, no causal attribution to individual changes",
        source=dict(baselines=str(REPO), terra=os.environ.get("TERRA_ROOT")),
        evaluation_updates=list(EVALUATION_UPDATES) if args.eval_bank else [],
        evaluation_output=str(args.eval_output) if args.eval_output else None,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "wandb").mkdir()
    os.environ["WANDB_DIR"] = str(output / "wandb")
    os.chdir(output)
    checkpoint_callback = None
    if args.eval_bank is not None:
        def checkpoint_callback(path, update):
            evaluate_checkpoint(path, update, args.eval_bank, args.eval_output)
        # A wall-time limit may have interrupted evaluation immediately after
        # the parent checkpoint was written. Complete that panel on resume.
        checkpoint_callback(args.checkpoint.resolve(), start)
    if start == target_update:
        (output / "result.json").write_text(json.dumps(dict(
            status="ALREADY_AT_TARGET", checkpoint=str(args.checkpoint),
            update=start, adam_step=int(np.asarray(parent["train_state_step"])),
        ), indent=2) + "\n")
        print(f"Already at u{start}; no further training.", flush=True)
        return
    train_mixed_agents(config, checkpoint_callback=checkpoint_callback)
    final = Path(config.checkpoint_dir) / f"{config.name}_FINAL.pkl"
    checkpoint = load_pkl_object(str(final))
    if (checkpoint["next_update"] != target_update
            or int(np.asarray(checkpoint["train_state_step"])) != target_update * 64):
        raise RuntimeError("Final checkpoint does not preserve the native update/Adam clock")
    for key in ("model", "optimizer_state"):
        if not all(np.isfinite(np.asarray(leaf)).all()
                   for leaf in jax.tree_util.tree_leaves(checkpoint[key])):
            raise RuntimeError(f"Nonfinite {key} in final checkpoint")
    result = dict(status="PASS", checkpoint=str(final), update=target_update,
                  adam_step=int(np.asarray(checkpoint["train_state_step"])),
                  foundation_teacher_release_state=checkpoint.get("foundation_teacher_release_state"),
                  remaining_time_migration=checkpoint.get("remaining_time_migration"),
                  actor_capacity_migration=checkpoint.get("actor_capacity_migration"),
                  scope="Native continuation and finite-state check; no policy quality claim")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
