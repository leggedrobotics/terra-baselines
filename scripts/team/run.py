#!/usr/bin/env python3
"""Jointly controlled team (or one machine), warm-started from a single-agent generalist.

Every agent starts as the parent policy (see utils/team_migration.py); the
recipe (encoder, observations, reward, PPO layout) is the parent's. Teachers
and demonstrations are single-agent tools and stay off. --types sets the
machines (0 excavator, 2 skid steer; all tracked), --scratch trains the same
recipe from a fresh initialization, and --maps-path selects another bank
(its dataset.json then stands for the distance sidecar).

Smoke: python scripts/team/run.py --checkpoint PARENT.pkl --bank BANK_DIR \
           --output OUT --agents 2 --devices 1 --envs 32 --updates 2
Skid steer: ... --types 2 --maps-path train --dataset-size 2048
Makespan fine-tune of a team: ... --resume TEAM.pkl --makespan-cost 2
    --makespan-setup-s 30 --machine-work-observation (a checkpoint without the
    observation is migrated and the reward change is declared once; later
    segments resume ordinarily)
"""
import argparse
import dataclasses
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="single-agent parent")
    parser.add_argument("--bank", type=Path, required=True, help="directory holding the map bank")
    parser.add_argument("--dataset-size", type=int, default=3840)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="team-excavators")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--types", help="comma-separated agent types (default: --agents excavators)")
    parser.add_argument("--maps-path", help="bank subdirectory (default: the parent's)")
    parser.add_argument("--scratch", action="store_true", help="fresh initialization")
    parser.add_argument("--ent-schedule", type=float, nargs=3, metavar=("START", "END", "UPDATES"),
                        help="entropy coefficient cosine schedule (default: the parent's)")
    parser.add_argument("--makespan-cost", type=float, default=0.0,
                        help="team cost on the growth of the busiest machine's executed-plan time")
    parser.add_argument("--makespan-setup-s", type=float, default=0.0,
                        help="executed-plan seconds per new work pose")
    parser.add_argument("--machine-work-observation", action="store_true",
                        help="observe every machine's executed-plan time")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--envs", type=int, default=256, help="environments per device")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", type=Path, help="team checkpoint to continue")
    args = parser.parse_args()

    os.environ["DATASET_PATH"] = str(args.bank.resolve())
    os.environ["DATASET_SIZE"] = str(args.dataset_size)
    os.environ.setdefault("WANDB_MODE", "disabled")

    import numpy as np
    from train_mixed import MixedAgentTrainConfig, train_mixed_agents
    from utils.helpers import load_pkl_object, register_checkpoint_config_classes

    register_checkpoint_config_classes()
    parent = load_pkl_object(str(args.checkpoint))
    saved = parent["train_config"]
    if tuple(saved.agent_types_override or (0,)) != (0,):
        raise ValueError("the parent must be a single tracked excavator")
    values = {field.name: getattr(saved, field.name, field.default)
              for field in dataclasses.fields(MixedAgentTrainConfig) if field.init}
    team = tuple(int(t) for t in args.types.split(",")) if args.types else (0,) * args.agents
    levels = saved.curriculum_levels_override
    sidecar = saved.distance_sidecar_sha256
    if args.maps_path:
        import hashlib
        levels = [dict(level, maps_path=args.maps_path) for level in levels]
        sidecar = hashlib.sha256((args.bank / args.maps_path / "dataset.json").read_bytes()).hexdigest()
    entropy = {}
    if args.ent_schedule:
        entropy = dict(ent_schedule_start=args.ent_schedule[0], ent_schedule_end=args.ent_schedule[1],
                       ent_schedule_steps=int(args.ent_schedule[2]))
    output = args.output.resolve()
    # Behavior costs in effect at the parent checkpoint, held fixed (a ramp is
    # a native-resume schedule of the parent's own run).
    costs = {
        name: float(np.ravel(np.asarray(getattr(parent["env_config"], name)))[0])
        for name in (
            "lateral_dig_cost", "base_travel_cost", "base_turn_cost",
            "retained_work_setup_cost", "retained_work_travel_cost",
            "retained_work_turn_cost",
        )
    }
    costs.update(makespan_cost=args.makespan_cost, makespan_setup_s=args.makespan_setup_s)
    # A resumed checkpoint without the observation is grown once, and a change
    # of the makespan settings is declared as a reward fine-tune; checkpoints
    # of the same treatment resume ordinarily.
    migrate_machine_work = finetune = False
    start_update = 0
    if args.resume:
        resumed_checkpoint = load_pkl_object(str(args.resume))
        resumed = resumed_checkpoint["train_config"]
        start_update = int(resumed_checkpoint["next_update"])
        migrate_machine_work = args.machine_work_observation and not getattr(
            resumed, "machine_work_observation", False)
        finetune = any(float(getattr(resumed, name, 0.0)) != float(costs[name])
                       for name in ("makespan_cost", "makespan_setup_s"))
        del resumed_checkpoint
    diagnostic = args.updates - start_update <= 5
    values.update(
        name=args.name,
        agent_types_override=team, action_types_override=(0,) * len(team),
        curriculum_levels_override=levels, distance_sidecar_sha256=sidecar, **entropy,
        num_devices=args.devices, num_envs_per_device=args.envs,
        total_timesteps=args.updates * args.devices * args.envs * saved.num_steps,
        warm_start_from=None if args.resume or args.scratch else str(args.checkpoint.resolve()),
        resume_from=str(args.resume.resolve()) if args.resume else None,
        resume_update=None, load_env_from_checkpoint=bool(args.resume),
        checkpoint_dir=str(output / "checkpoints"), keep_checkpoint_history=True,
        checkpoint_interval=1 if diagnostic else args.checkpoint_interval,
        log_train_interval=1 if diagnostic else 10, log_eval_interval=0,
        initialization_receipt=str(output / "initialization.json"),
        fail_on_nonfinite=True, finite_check_interval=1 if diagnostic else 10,
        # Parent architecture is already native: no migrations.
        migrate_remaining_time=False, grow_actor_capacity=False,
        migrate_retained_work_context=False,
        # Single-agent teachers and demonstrations do not apply to a team.
        teacher_checkpoint=None, trench_teacher_checkpoint=None,
        teacher_checkpoint_sha256=None, trench_teacher_checkpoint_sha256=None,
        task_teacher_family_ids=None, kickstart_start_update=0,
        kickstart_kl_coef=0.0, kickstart_kl_anneal_updates=0,
        kickstart_value_coef=0.0, kickstart_value_anneal_updates=0,
        cache_teacher_outputs=False, foundation_teacher_release_updates=0,
        demonstration_npz=None, demonstration_coef=0.0,
        demonstration_fade_transitions=0,
        behavior_cost_ramp_updates=0, finetune_foundation_behavior=finetune,
        finetune_task_bank=False, machine_work_observation=args.machine_work_observation,
        migrate_machine_work_observation=migrate_machine_work, **costs,
    )
    output.mkdir(parents=True, exist_ok=True)
    train_mixed_agents(MixedAgentTrainConfig(**values))


if __name__ == "__main__":
    main()
