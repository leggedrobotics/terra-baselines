#!/usr/bin/env python3
"""Continue the broad u5000 teacher experiment without reconstructing its recipe."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Existing broad campaign inputs: bank and both teachers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "foundation_release"), required=True)
    parser.add_argument("--target-update", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="1 GPU x 128 envs, at most 2 new updates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
    if start < 5000 or args.target_update <= start or args.target_update > 7500:
        raise ValueError("This comparison is bounded to native u5000 through u7500")
    if args.smoke and args.target_update - start > 2:
        raise ValueError("The smoke permits at most two new updates")
    if not args.smoke and args.target_update not in (6250, 7500):
        raise ValueError("Production decisions are at u6250 and u7500")
    if args.arm == "control" and release is not None:
        raise ValueError("A released teacher checkpoint cannot become the control")
    if args.arm == "foundation_release" and release is None and start != 5000:
        raise ValueError("The release must start from the shared u5000 checkpoint")
    if int(np.asarray(parent["train_state_step"])) != start * 64:
        raise ValueError("Parent Adam clock does not match 64 steps per update")
    if saved.config_name != "trench_align_v2_generalist_gen":
        raise ValueError("Expected the broad generalist parent")
    if any(getattr(saved, key) != 0 for key in
           ("lateral_dig_cost", "base_travel_cost", "base_turn_cost")):
        raise ValueError("This experiment retains zero added behavior costs")
    if (saved.num_steps, saved.update_epochs, saved.num_minibatches) != (32, 2, 32):
        raise ValueError("Unexpected parent PPO layout")
    if not args.smoke and (saved.num_devices, saved.num_envs_per_device) != (4, 256):
        raise ValueError("Production must preserve the parent's 4x256 batch; smoke checkpoints are diagnostic only")
    devices, envs = (1, 128) if args.smoke else (4, 256)
    output = args.output.resolve()
    overrides = dict(
        name=f"foundation-teacher-{args.arm}",
        num_devices=devices, num_envs_per_device=envs,
        total_timesteps=args.target_update * devices * envs * saved.num_steps,
        resume_from=str(args.checkpoint.resolve()), warm_start_from=None,
        resume_update=None, load_env_from_checkpoint=True,
        checkpoint_dir=str(output / "checkpoints"), keep_checkpoint_history=True,
        checkpoint_interval=1 if args.smoke else 250,
        log_train_interval=1 if args.smoke else 10, log_eval_interval=0,
        initialization_receipt=str(output / "initialization.json"),
        teacher_checkpoint=str(inputs / "foundation_teacher.pkl"),
        trench_teacher_checkpoint=str(inputs / "trench_teacher.pkl"),
        fail_on_nonfinite=True, finite_check_interval=1 if args.smoke else 10,
        foundation_teacher_release_updates=(
            1250 if args.arm == "foundation_release" and release is None else 0
        ),
    )
    values.update(overrides)
    config = MixedAgentTrainConfig(**values)
    plan = dict(
        arm=args.arm, parent=str(args.checkpoint.resolve()), start_update=start,
        target_update=args.target_update, global_envs=devices * envs,
        transitions_per_update=config.env_steps_per_update,
        additional_transitions=(args.target_update - start) * config.env_steps_per_update,
        smoke=args.smoke, restored_release_state=release,
        configuration_changes={key: {"parent": getattr(saved, key, None), "new": value}
                               for key, value in overrides.items()
                               if getattr(saved, key, None) != value},
        source=dict(baselines=str(REPO), terra=os.environ.get("TERRA_ROOT")),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    os.environ["WANDB_DIR"] = str(output / "wandb")
    os.chdir(output)
    train_mixed_agents(config)
    final = Path(config.checkpoint_dir) / f"{config.name}_FINAL.pkl"
    checkpoint = load_pkl_object(str(final))
    if (checkpoint["next_update"] != args.target_update
            or int(np.asarray(checkpoint["train_state_step"])) != args.target_update * 64):
        raise RuntimeError("Final checkpoint does not preserve the native update/Adam clock")
    for key in ("model", "optimizer_state"):
        if not all(np.isfinite(np.asarray(leaf)).all()
                   for leaf in jax.tree_util.tree_leaves(checkpoint[key])):
            raise RuntimeError(f"Nonfinite {key} in final checkpoint")
    result = dict(status="PASS", checkpoint=str(final), update=args.target_update,
                  adam_step=int(np.asarray(checkpoint["train_state_step"])),
                  foundation_teacher_release_state=checkpoint.get("foundation_teacher_release_state"),
                  scope="Native continuation and finite-state check; no policy quality claim")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
