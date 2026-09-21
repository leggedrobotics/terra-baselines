#!/usr/bin/env python3
"""Native broad continuation with optional fading actor demonstrations."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
EVALUATION_UPDATES = (10000, 20000, 35000, 50000, 75000, 100000)


def evaluate_checkpoint(checkpoint, update, bank, output, *,
                        evaluation_updates=EVALUATION_UPDATES, required=False):
    """Pause PPO for the fixed panel; bounded imitation requires a valid result."""
    if update not in evaluation_updates:
        return
    from scripts.foundation_teacher_release.compare import load_report

    checkpoint = Path(checkpoint).resolve()

    def validate_report(path):
        panel = load_report(path)
        if (panel["checkpoint_update"] != update
                or Path(panel["checkpoint"]).resolve() != checkpoint):
            raise ValueError("Evaluation belongs to another checkpoint identity")

    output.mkdir(parents=True, exist_ok=True)
    report = output / f"u{update}.json"
    if report.exists():
        try:
            validate_report(report)
        except (ValueError, KeyError, OSError):
            report.rename(report.with_suffix(f".incomplete-{int(time.time())}.json"))
        else:
            return
    # Restrict the child to one allocated GPU without replacing a Slurm device
    # index or UUID with an unallocated physical device zero on shared nodes.
    evaluation_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env = dict(os.environ, BANK_ROOT=str(bank), CUDA_VISIBLE_DEVICES=evaluation_device,
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
        try:
            validate_report(report)
        except (ValueError, KeyError, OSError) as error:
            status.update(status="FAILED_VALIDATION", error=str(error))
        else:
            status["status"] = "PASS"
    else:
        status["status"] = "FAILED"
    (output / f"u{update}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Fixed evaluation u{update}: {status['status']}", flush=True)
    if required and status["status"] != "PASS":
        raise RuntimeError(f"Required fixed evaluation failed at u{update}: {status}")


def check_imitation_retention(parent_path, current_path, output_path):
    """Engineering stop for this bounded continuation; not a significance test."""
    import math
    from statistics import mean
    from scripts.foundation_teacher_release.compare import load_report

    parent, current = load_report(parent_path), load_report(current_path)
    for key in ("manifest_sha256", "reset_verification", "seed", "horizon"):
        if parent[key] != current[key]:
            raise ValueError(f"Retention panels differ in {key}")
    identity = ("episode_id", "scenario_id", "source_id", "reset_seed",
                "slot_index", "family", "primary_cell")
    before = {row["episode_id"]: row for row in parent["per_map"]}
    after = {row["episode_id"]: row for row in current["per_map"]}
    if before.keys() != after.keys() or any(
            any(before[key][field] != after[key][field] for field in identity) for key in before):
        raise ValueError("Retention panels do not contain the same episode identities")
    pairs = [(before[key], after[key]) for key in before]
    metrics = ("dig_fraction", "terminal_soil_fraction", "off_zone_staged_soil_fraction",
               "loaded_soil_fraction", "longest_material_stall_steps", "longest_task_progress_stall_steps",
               "retained_productive_setups", "retained_work_setups", "mean_workspace_dig_area_m2", "relift_actions",
               "unique_area_per_retained_productive_setup_m2", "retained_work_inter_setup_straight_line_m",
               "fresh_union_edge_adjacency_fraction", "mean_dig_lateral_score", "base_travel_m")

    def summarize(rows):
        summary = dict(episodes=len(rows), parent_successes=sum(a["success"] for a, b in rows),
                       current_successes=sum(b["success"] for a, b in rows),
                       gained=[{key: b[key] for key in identity} for a, b in rows if not a["success"] and b["success"]],
                       lost=[{key: a[key] for key in identity} for a, b in rows if a["success"] and not b["success"]])
        summary["net_lost"] = summary["parent_successes"] - summary["current_successes"]
        for label, subset in (("all", rows), ("common_success", [(a, b) for a, b in rows if a["success"] and b["success"]])):
            summary[label] = {}
            for key in metrics:
                valid = [(a[key], b[key]) for a, b in subset if a.get(key) is not None and b.get(key) is not None
                         and math.isfinite(a[key]) and math.isfinite(b[key])]
                summary[label][key] = dict(paired_n=len(valid), parent=mean(a for a, b in valid) if valid else None,
                                          current=mean(b for a, b in valid) if valid else None)
        return summary

    cohorts = {family: summarize([(a, b) for a, b in pairs if a["family"] == family])
               for family in ("foundation", "trench")}
    cohorts["road_trench"] = summarize([(a, b) for a, b in pairs
                                        if a["family"] == "trench" and "road" in a["primary_cell"]])
    conditions = {name: summarize([(a, b) for a, b in pairs if a["primary_cell"] == name])
                  for name in sorted({a["primary_cell"] for a, b in pairs})}
    thresholds = {"foundation": 4, "trench": 2, "road_trench": 1}
    reasons = [f"{name}: net loss {cohorts[name]['net_lost']} > {limit}"
               for name, limit in thresholds.items() if cohorts[name]["net_lost"] > limit]
    reasons += [f"condition {name}: net loss {stats['net_lost']} > 2"
                for name, stats in conditions.items() if stats["net_lost"] > 2]
    result = dict(status="STOP" if reasons else "PASS", parent_report=str(parent_path),
                  current_report=str(current_path), parent_update=parent["checkpoint_update"],
                  update=current["checkpoint_update"], cohorts=cohorts, conditions=conditions,
                  maximum_net_losses=dict(thresholds, per_condition=2), reasons=reasons,
                  scope="Engineering retention stop, not a statistical test or automatic promotion")
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Continuation retention u{result['update']}: {result['status']} {reasons}", flush=True)
    if reasons:
        raise RuntimeError(f"Bounded continuation stopped by retention check: {reasons}")
    return result


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
    parser.add_argument("--retention-parent-report", type=Path,
                        help="Shared parent panel replayed with this runtime; also gates a PPO-only control")
    parser.add_argument("--extra-evaluation-updates", type=int, nargs="*", default=[],
                        help="Additional absolute saved-checkpoint milestones for a bounded comparison")
    parser.add_argument("--demonstration-npz", type=Path, nargs="+",
                        help="Complete training-only demonstrations, replayed with the current Terra runtime")
    parser.add_argument("--demonstration-coef", type=float,
                        help="Initial actor imitation coefficient; new attachment defaults to 0.01")
    parser.add_argument("--demonstration-fade-transitions", type=int,
                        help="Linear fade in global transitions; new attachment defaults to 24576000")
    parser.add_argument("--demonstration-batch-size", type=int,
                        help="Demonstration samples per device per PPO minibatch; new attachment defaults to 16")
    parser.add_argument("--demonstration-hold-updates", type=int, default=750,
                        help="Auxiliary-free hold after the fade; target must not exceed its end")
    parser.add_argument("--smoke", action="store_true", help="1 GPU x 128 envs, at most 2 new updates")
    parser.add_argument("--qualification", choices=("native", "fade-boundary"),
                        help="At most two diagnostic updates at the exact 4x256 production layout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.eval_bank is None) != (args.eval_output is None):
        parser.error("--eval-bank and --eval-output must be provided together")
    if (args.retention_parent_report or args.extra_evaluation_updates) and args.eval_bank is None:
        parser.error("Retention comparisons and extra milestones require fixed evaluation")
    if args.demonstration_hold_updates < 0:
        parser.error("--demonstration-hold-updates must be nonnegative")
    if args.qualification and (args.smoke or args.eval_bank is not None):
        parser.error("--qualification uses 4x256 without the fixed-panel evaluator or --smoke")
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
    from train_mixed import (MixedAgentTrainConfig, foundation_teacher_release_coef,
                             kickstart_coef_schedule, train_mixed_agents)
    from utils.helpers import load_pkl_object, register_checkpoint_config_classes, save_pkl_object

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
    if args.qualification and not 1 <= target_update - start <= 2:
        raise ValueError("Production-layout qualification requires one or two finite updates")
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
    if not args.smoke and not args.qualification and start % 250:
        raise ValueError("Select a 250-update milestone parent for the fixed evaluation checkpoints")
    devices, envs = (1, 128) if args.smoke else (4, 256)
    diagnostic = args.smoke or bool(args.qualification)
    output = args.output.resolve()
    overrides = dict(
        name="generalist-oracle-combined",
        num_devices=devices, num_envs_per_device=envs,
        total_timesteps=target_update * devices * envs * saved.num_steps,
        resume_from=str(args.checkpoint.resolve()), warm_start_from=None,
        resume_update=None, load_env_from_checkpoint=True,
        checkpoint_dir=str(output / "checkpoints"), keep_checkpoint_history=True,
        checkpoint_interval=1 if diagnostic else 250,
        log_train_interval=1 if diagnostic else 10, log_eval_interval=0,
        initialization_receipt=str(output / "initialization.json"),
        teacher_checkpoint=str(inputs / "foundation_teacher.pkl"),
        trench_teacher_checkpoint=str(inputs / "trench_teacher.pkl"),
        fail_on_nonfinite=True, finite_check_interval=1 if diagnostic else 10,
        time_observation_mode="remaining",
        migrate_remaining_time=getattr(saved, "time_observation_mode", "none") == "none",
        actor_residual_head=True,
        grow_actor_capacity=not getattr(saved, "actor_residual_head", False),
        cache_teacher_outputs=True,
        foundation_teacher_release_updates=(
            1250 if release is None else 0
        ),
    )
    if args.demonstration_npz is not None:
        overrides["demonstration_npz"] = [str(path.resolve()) for path in args.demonstration_npz]
    demonstration_requested = bool(overrides.get("demonstration_npz", values.get("demonstration_npz")))
    for field, requested, default in (
        ("demonstration_coef", args.demonstration_coef, .01),
        ("demonstration_fade_transitions", args.demonstration_fade_transitions, 24576000),
        ("demonstration_batch_size", args.demonstration_batch_size, 16),
    ):
        if requested is not None:
            overrides[field] = requested
        elif demonstration_requested and parent.get("demonstration_state") is None:
            overrides[field] = default
    values.update(overrides)
    config = MixedAgentTrainConfig(**values)
    from utils.demonstrations import (restore_demonstration_schedule,
                                      demonstration_coefficient, demonstration_global_transitions)
    demonstration_state = restore_demonstration_schedule(config, parent, "resume", start)
    comparison_parent = None
    if args.retention_parent_report is not None:
        from scripts.foundation_teacher_release.compare import load_report
        args.retention_parent_report = args.retention_parent_report.resolve()
        comparison_parent = load_report(args.retention_parent_report)
        reference_update = int(comparison_parent["checkpoint_update"])
        if reference_update > start:
            raise ValueError("Retention parent cannot be newer than the continuation")
        if reference_update == start:
            with args.checkpoint.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                checkpoint_digest = digest.hexdigest()
            if comparison_parent.get("checkpoint_sha256") != checkpoint_digest:
                raise ValueError("Shared retention panel belongs to another parent checkpoint")
        if target_update not in args.extra_evaluation_updates:
            raise ValueError("Bounded comparison must evaluate its final target")
    if any(update < start or update > target_update or update % config.checkpoint_interval
           for update in args.extra_evaluation_updates):
        raise ValueError("Extra evaluations must be saved checkpoints within this continuation")
    boundary_parent = None
    if args.qualification == "fade-boundary":
        if (demonstration_state is None or parent.get("demonstration_state") is not None
                or target_update != start + 2):
            raise ValueError("Fade-boundary qualification needs a pre-imitation parent and exactly two updates")
        batch = config.env_steps_per_update
        if demonstration_state["fade_transitions"] % batch:
            raise ValueError("Fade-boundary qualification requires a whole-update fade")
        fade_updates = demonstration_state["fade_transitions"] // batch
        origin = start + 1 - fade_updates
        if origin < 0:
            raise ValueError("Parent clock is too early for the diagnostic fade origin")
        # Only this disposable checkpoint shifts the auxiliary clock. Model,
        # Adam, live update, PPO layout and the 750-update fade remain unchanged.
        demonstration_state = dict(demonstration_state, origin_update=origin,
                                   origin_transitions=origin * batch)
        boundary_parent = dict(parent, demonstration_state=demonstration_state)
        boundary_parent["train_config"] = dataclasses.replace(
            saved, **{key: getattr(config, key) for key in (
                "demonstration_npz", "demonstration_coef",
                "demonstration_fade_transitions", "demonstration_batch_size",
            )},
        )
        config.resume_from = str(output / "diagnostic_boundary_parent.pkl")
        demonstration_state = restore_demonstration_schedule(config, boundary_parent, "resume", start)
    if demonstration_coefficient(demonstration_state, demonstration_global_transitions(
            demonstration_state, start, config.env_steps_per_update)) > 0:
        for path in config.demonstration_npz or ():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    evaluation_updates = set(EVALUATION_UPDATES)
    evaluation_updates.update(args.extra_evaluation_updates)
    demonstration_evaluation = None
    if demonstration_state is not None:
        if not diagnostic and not args.dry_run and args.eval_bank is None:
            raise ValueError("Bounded imitation requires fixed parent/fade/hold evaluation panels")
        teacher_kl = kickstart_coef_schedule(
            start, config.kickstart_kl_coef, config.kickstart_kl_anneal_updates,
            start_update=config.kickstart_start_update,
        )
        teacher_value = kickstart_coef_schedule(
            start, config.kickstart_value_coef, config.kickstart_value_anneal_updates,
            start_update=config.kickstart_start_update,
        )
        foundation_kl = (foundation_teacher_release_coef(release, start)
                         if release is not None else teacher_kl)
        if max(teacher_kl, teacher_value, foundation_kl) > 0:
            raise ValueError("Demonstration continuation requires an already teacher-free parent")
        origin = demonstration_state["origin_update"]
        batch = demonstration_state["transitions_per_update"]
        fade_updates = (demonstration_state["fade_transitions"] + batch - 1) // batch
        fade_end = origin + fade_updates
        hold_end = fade_end + args.demonstration_hold_updates
        if not diagnostic and target_update > hold_end:
            raise ValueError(f"Bounded imitation target exceeds the fade/hold end u{hold_end}")
        # Callbacks run on saved checkpoints. Round up, never evaluate a stage
        # before its final transitions have actually been consumed.
        save_interval = config.checkpoint_interval
        fade_evaluation = ((fade_end + save_interval - 1) // save_interval) * save_interval
        hold_evaluation = ((hold_end + save_interval - 1) // save_interval) * save_interval
        demonstration_evaluation = dict(
            parent_update=origin, fade_end_update=fade_end,
            fade_evaluation_update=fade_evaluation,
            hold_end_update=hold_end, hold_evaluation_update=hold_evaluation,
            hold_updates=args.demonstration_hold_updates,
            automatic_policy_promotion=False,
        )
        evaluation_updates.update((origin, fade_evaluation, hold_evaluation))
    evaluation_updates = sorted(update for update in evaluation_updates if update <= target_update)
    plan = dict(
        recipe=("fixed_geometry_fading_actor_demonstrations" if demonstration_state is not None
                else "time_actor_capacity_foundation_release_teacher_cache"),
        parent=str(args.checkpoint.resolve()), start_update=start,
        target_update=target_update, global_envs=devices * envs,
        transitions_per_update=config.env_steps_per_update,
        additional_transitions=(target_update - start) * config.env_steps_per_update,
        smoke=args.smoke, qualification=args.qualification, restored_release_state=release,
        configuration_changes={key: {"parent": getattr(saved, key, None), "new": value}
                               for key, value in overrides.items()
                               if getattr(saved, key, None) != value},
        release_transition_duration=1250 * 32768,
        demonstration_state=demonstration_state,
        demonstration_evaluation=demonstration_evaluation,
        retention_parent_report=str(args.retention_parent_report) if comparison_parent is not None else None,
        objectives="Corrected physics and reward-v2; every added behavior cost remains zero",
        interpretation="Combined intervention, no causal attribution to individual changes",
        source=dict(baselines=str(REPO), terra=os.environ.get("TERRA_ROOT")),
        evaluation_updates=evaluation_updates if args.eval_bank else [],
        evaluation_output=str(args.eval_output) if args.eval_output else None,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if diagnostic:
        os.environ["WANDB_MODE"] = "disabled"
    if boundary_parent is not None:
        save_pkl_object(boundary_parent, config.resume_from)
    (output / "wandb").mkdir()
    os.environ["WANDB_DIR"] = str(output / "wandb")
    os.chdir(output)
    checkpoint_callback = None
    if args.eval_bank is not None:
        def checkpoint_callback(path, update):
            # A shared panel is computed once, using this immutable runtime.
            # Never substitute the historical training runtime's parent panel.
            if (comparison_parent is not None
                    and update == comparison_parent["checkpoint_update"]):
                if Path(path).resolve() != args.checkpoint:
                    raise ValueError("Shared parent panel checkpoint identity changed")
                return
            evaluate_checkpoint(path, update, args.eval_bank, args.eval_output,
                                evaluation_updates=evaluation_updates,
                                required=demonstration_state is not None or comparison_parent is not None)
            if comparison_parent is not None and update in args.extra_evaluation_updates:
                check_imitation_retention(
                    args.retention_parent_report, args.eval_output / f"u{update}.json",
                    args.eval_output / f"u{update}.retention.json",
                )
            elif demonstration_evaluation is not None and update in (
                    demonstration_evaluation["fade_evaluation_update"],
                    demonstration_evaluation["hold_evaluation_update"]):
                check_imitation_retention(
                    (args.retention_parent_report if comparison_parent is not None
                     else args.eval_output / f"u{demonstration_evaluation['parent_update']}.json"),
                    args.eval_output / f"u{update}.json",
                    args.eval_output / f"u{update}.retention.json",
                )
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
    if demonstration_state is not None:
        final_demo = checkpoint.get("demonstration_state", {})
        for key, value in demonstration_state.items():
            if final_demo.get(key) != value:
                raise RuntimeError(f"Demonstration schedule changed while training: {key}")
    qualification_coefficients = None
    if args.qualification:
        qualification_coefficients = [demonstration_coefficient(
            demonstration_state, demonstration_global_transitions(
                demonstration_state, update, config.env_steps_per_update,
            ),
        ) for update in range(start, target_update)]
        if args.qualification == "fade-boundary":
            if not qualification_coefficients[0] > 0 or qualification_coefficients[1] != 0:
                raise RuntimeError("Diagnostic run did not cross the auxiliary fade boundary")
            if any(float(np.asarray(value)) != 0 for key, value in checkpoint["loss_info"].items()
                   if key.startswith("imitation/")):
                raise RuntimeError("Inactive auxiliary specialization returned nonzero imitation metrics")
    result = dict(status="PASS", checkpoint=str(final), update=target_update,
                  adam_step=int(np.asarray(checkpoint["train_state_step"])),
                  foundation_teacher_release_state=checkpoint.get("foundation_teacher_release_state"),
                  remaining_time_migration=checkpoint.get("remaining_time_migration"),
                  actor_capacity_migration=checkpoint.get("actor_capacity_migration"),
                  demonstration_state=checkpoint.get("demonstration_state"),
                  qualification=args.qualification,
                  qualification_coefficients=qualification_coefficients,
                  scope="Native continuation and finite-state check; no policy quality claim")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
