"""Gate one cost increase using the existing fixed-panel reports and native resume.

This is an offline stage launcher, not a training-success or elapsed-update
schedule. Without --execute it only checks evidence and prints the command.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.behavior_cost_ramp import COST_KEYS, SCHEMA, validate_ramp_state

FULL_COSTS = (0.5, 0.01, 0.04)  # Previously called combined 2x.
FRACTIONS = (0.0, 0.25, 0.5, 1.0)
MIN_SUCCESS = 0.90
MAX_SUCCESS_LOSS = 0.03
EVALUATION_INTERVAL = 2500
RAMP_UPDATES = EVALUATION_INTERVAL
STAGE_UPDATES = 2 * EVALUATION_INTERVAL
COUNTS = {"foundation": (64, 64), "trench": (608, 224)}
EVALUATION_PANELS = {
    "foundation": {"manifest_sha256": "68b7807262aa698d4d5e1c638f205939e767d2284b21f77691fead9372e56309",
                   "seed": 20260907, "split": "validation"},
    "trench": {"manifest_sha256": "1216bee3be9fee02531c7e16132ffe61c2cdbbaab40b86e6e7f8e164b0830a3b",
               "seed": 20260724, "split": "development"},
}
IDENTITY_KEYS = ("episode_id", "map_id", "source_id", "scenario_id", "reset_seed", "slot_index", "family")
PANEL_KEYS = ("completion_contract", "manifest_sha256", "seed", "split", "stratum",
              "accepted_bank", "reset_verification")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_report(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, list) and len(value) == 1, "Expected one fixed-panel evaluation")
    return value[0]


def stage(report):
    behavior = report["treatment_fingerprint"]["contract"]["foundation_behavior"]
    require(behavior["executable_dig_observation"] is True, "Keep executable dig observation enabled")
    for index, fraction in enumerate(FRACTIONS):
        if all(math.isclose(behavior[key], cost * fraction, rel_tol=1e-6, abs_tol=1e-9)
               for key, cost in zip(COST_KEYS, FULL_COSTS)):
            return index
    raise ValueError("Report costs are outside the 0/25/50/100 percent combined-cost stages")


def normalized_training(report):
    contract = copy.deepcopy(report["treatment_fingerprint"]["contract"])
    receipt = copy.deepcopy(report["r2_protocol_receipt"])
    del contract["run"]["name"]
    for key in COST_KEYS:
        del contract["foundation_behavior"][key]
        del receipt["foundation_behavior"][key]
    return contract, receipt


def checked_ramp_state(report):
    """A qualifying cost-stage evaluation must have reached its target costs."""
    state = report.get("behavior_cost_ramp_state")
    if state is None:
        return None
    declared = {key: report["r2_protocol_receipt"]["foundation_behavior"][key] for key in COST_KEYS}
    validate_ramp_state(state, report["checkpoint_update"], declared)
    require(report["checkpoint_update"] >= state["start_update"] + state["duration_updates"],
            "Cost-stage evaluation is still mid-ramp; evaluate after the target costs are reached")
    effective = report["treatment_fingerprint"]["contract"]["foundation_behavior"]
    require(all(math.isclose(effective[key], state["target_costs"][key], rel_tol=1e-6, abs_tol=1e-9)
                for key in COST_KEYS), "Evaluation effective costs differ from the completed ramp target")
    return state


def stage_ramp(start_update, start_fraction, target_fraction):
    return {
        "schema": SCHEMA, "start_update": start_update, "duration_updates": RAMP_UPDATES,
        "start_costs": {key: value * start_fraction for key, value in zip(COST_KEYS, FULL_COSTS)},
        "target_costs": {key: value * target_fraction for key, value in zip(COST_KEYS, FULL_COSTS)},
    }


def checked_rows(report, family):
    full_count, family_count = COUNTS[family]
    require(report["deterministic"] is True and report["horizon"] == 450,
            "Use greedy evaluation at the unchanged 450-step horizon")
    require(report["completion_contract"] == "exact_visible_dump_v1", "Require exact completion")
    require(all(report[key] == value for key, value in EVALUATION_PANELS[family].items())
            and report["schema"] == "terra_fixed_bank_eval_v4" and report["stratum"] == "all"
            and report["policy_mode"] == "deterministic" and report["exact_manifest_enumeration"] is True,
            "Use the canonical full development/validation panel and evaluation seed")
    reset = report["reset_verification"]
    require(reset["passed"] is True and reset["env_steps_min"] == reset["env_steps_max"] == 0
            and reset["slots"] == full_count, "Require verified full-start resets")
    integrity = report["summary"]["integrity"]
    require(integrity["passed"] is True and integrity["unavailable"] == 0
            and integrity["failure_count"] == 0, "Evaluation integrity did not pass")
    rows = report["per_map"]
    require(len(rows) == full_count, "Incomplete fixed panel")
    require(len({r["episode_id"] for r in rows}) == full_count, "Duplicate evaluation episodes")
    for row in rows:
        require(type(row["success"]) is bool and not row["integrity_failure"]
                and not row["integrity_unavailable"], "Invalid or unavailable episode result")
    selected = [r for r in rows if r["family"] == family]
    require(len(selected) == family_count, "Wrong task-family cohort")
    return selected


def completion_gate(family, previous, latest, reference, parent_stage=None):
    """Reference is the frozen, accepted zero-cost parent of the first stage."""
    selected = [checked_rows(r, family) for r in (previous, latest, reference)]
    ramps = [checked_ramp_state(r) for r in (previous, latest, reference)]
    current = stage(latest)
    require(stage(previous) == current, "Both evaluations must be from the same cost stage")
    require(stage(reference) == 0, "The completion reference must have zero added costs")
    require(current < len(FRACTIONS) - 1, "Already at the final cost stage")
    require(previous["treatment_fingerprint"] == latest["treatment_fingerprint"]
            and previous["r2_protocol_receipt"] == latest["r2_protocol_receipt"],
            "Evaluations must belong to the same training run and treatment")
    require(latest["checkpoint_update"] - previous["checkpoint_update"] >= EVALUATION_INTERVAL,
            "Use successive retained evaluations at least 2500 updates apart")
    require(previous["checkpoint_sha256"] != latest["checkpoint_sha256"], "Repeated checkpoint")
    require(ramps[2] is None, "The zero-cost reference must precede any behavior-cost ramp")
    if current == 0:
        require(ramps[:2] == [None, None], "Qualify zero-cost learning before starting any behavior-cost ramp")
        require(reference["checkpoint_sha256"] == latest["checkpoint_sha256"],
                "Freeze the latest zero-cost checkpoint as the initial reference")
    else:
        require(parent_stage is not None, "Supply the parent's penalty_stage.json for later increases")
        require(parent_stage["reference_sha256"] == reference["checkpoint_sha256"],
                "Keep the original zero-cost reference across all cost stages")
        require(parent_stage["reference_successes"] == sum(r["success"] for r in selected[2]),
                "Keep the original zero-cost completion result across all cost stages")
        require(parent_stage["family"] == family
                and parent_stage["next_fraction_of_combined_2x"] == FRACTIONS[current]
                and parent_stage["run_name"] == latest["treatment_fingerprint"]["contract"]["run"]["name"],
                "Parent stage record does not belong to this cost stage/run")
        expected_ramp = stage_ramp(parent_stage["parent_update"], FRACTIONS[current - 1], FRACTIONS[current])
        require(parent_stage.get("next_behavior_cost_ramp_state") == expected_ramp
                and ramps[0] == ramps[1] == expected_ramp,
                "Both evaluation ramps must match the parent stage start, duration and cost targets")
        require(previous["checkpoint_update"] >= parent_stage["parent_update"] + EVALUATION_INTERVAL,
                "Both evaluations must follow training under the current costs")
        require(reference["checkpoint_update"] < previous["checkpoint_update"],
                "The zero-cost reference must precede both fine-tune evaluations")
    for report in (previous, reference):
        require(all(report[k] == latest[k] for k in PANEL_KEYS), "Evaluation panel/reset mismatch")
        require(normalized_training(report) == normalized_training(latest),
                "Training configuration changed beyond the staged behavior costs")
        require([tuple(r[k] for k in IDENTITY_KEYS) for r in report["per_map"]]
                == [tuple(r[k] for k in IDENTITY_KEYS) for r in latest["per_map"]],
                "Evaluation episode identities differ")
    counts = [sum(r["success"] for r in rows) for rows in selected]
    n = len(selected[0])
    required = max(math.ceil(MIN_SUCCESS * n), counts[2] - math.floor(MAX_SUCCESS_LOSS * n))
    ready = counts[2] / n >= MIN_SUCCESS and min(counts[:2]) >= required
    fraction = FRACTIONS[current + 1]
    return {
        "ready": ready, "family": family, "episodes": n,
        "previous_successes": counts[0], "latest_successes": counts[1],
        "reference_successes": counts[2], "required_successes": required,
        "previous_update": previous["checkpoint_update"], "parent_update": latest["checkpoint_update"],
        "parent_sha256": latest["checkpoint_sha256"], "reference_sha256": reference["checkpoint_sha256"],
        "current_fraction_of_combined_2x": FRACTIONS[current],
        "next_fraction_of_combined_2x": fraction,
        "next_costs": {key: value * fraction for key, value in zip(COST_KEYS, FULL_COSTS)},
        "next_behavior_cost_ramp_state": stage_ramp(latest["checkpoint_update"], FRACTIONS[current], fraction),
        "target_update": latest["checkpoint_update"] + STAGE_UPDATES,
        "evaluate_at_updates": [latest["checkpoint_update"] + EVALUATION_INTERVAL,
                                latest["checkpoint_update"] + STAGE_UPDATES],
        "latest_by_condition": {
            condition: {"successes": sum(r["success"] for r in selected[1] if r["primary_cell"] == condition),
                        "episodes": sum(r["primary_cell"] == condition for r in selected[1])}
            for condition in sorted({r["primary_cell"] for r in selected[1]})},
    }


def check_training_bank(root, family, expected_sidecar):
    """Read the existing bank identities; dataset paths alone are not identities."""
    root = Path(root).resolve()
    maps_path, count = {"foundation": ("train/all", 256), "trench": ("train_v2_pooled_trench15", 1440)}[family]
    metadata = json.loads((root / maps_path / "dataset.json").read_text())
    if family == "foundation":
        sidecar_path = root / "distance_sidecar/dataset.json"
        sidecar = json.loads(sidecar_path.read_text())
        require(hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == expected_sidecar,
                "Training bank distance sidecar differs from the parent")
        registry_sha = sidecar["source_registry_sha256"]
    else:
        bank = json.loads((root / "dataset.json").read_text())
        require(bank["canonical_distance_sidecar_dataset_sha256"] == expected_sidecar,
                "Training bank distance identity differs from the parent")
        registry_sha = bank["source_registry_sha256"]
    require(metadata["slot_count"] == count
            and metadata["source_registry_sha256"] == registry_sha
            and metadata["distance_protocol_id"] == "obstacle_geodesic_8_physical_global_v1",
            "Training subset does not match the supported bank")
    with (root / "source_registry.jsonl").open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == registry_sha,
                "Training source registry differs from bank metadata")
    return root


def check_native_parent(checkpoint, decision, latest):
    """Use the native trainer's finite/resume checks before entering an allocation stage."""
    import jax
    import numpy as np
    from train_mixed import _assert_finite_tree, _assert_finite_loss_info, _r2_protocol_receipt, _validate_r2_resume_checkpoint
    from utils.helpers import checkpoint_foundation_behavior

    update = decision["parent_update"]
    adam = update * 64
    require("env_config" in checkpoint, "Native continuation requires the saved environment configuration")
    require(checkpoint["next_update"] == update and int(np.asarray(checkpoint["train_state_step"])) == adam,
            "Native checkpoint update/optimizer clock differs from evaluation")
    counts = [int(np.asarray(v)) for v in jax.tree.leaves(checkpoint["optimizer_state"])
              if np.shape(v) == () and np.asarray(v).dtype.kind in "iu"]
    require(counts == [adam], "Actual Adam count differs from the native checkpoint clock")
    require(checkpoint["r2_protocol_receipt"] == latest["r2_protocol_receipt"],
            "Native checkpoint reward protocol differs from evaluation")
    require(checkpoint.get("behavior_cost_ramp_state") == latest.get("behavior_cost_ramp_state"),
            "Native checkpoint behavior-cost ramp differs from evaluation")
    behavior = checkpoint_foundation_behavior(checkpoint)
    expected = latest["treatment_fingerprint"]["contract"]["foundation_behavior"]
    require(all(math.isclose(behavior[k], expected[k], rel_tol=1e-6, abs_tol=1e-9) for k in expected),
            "Native checkpoint behavior differs from evaluation")
    cfg = copy.copy(checkpoint["train_config"])
    run = latest["treatment_fingerprint"]["contract"]["run"]
    require(all(getattr(cfg, key) == run[key] for key in ("name", "seed", "config_name")),
            "Native checkpoint training identity differs from evaluation")
    require(all(getattr(cfg, key) == value for key, value in
                latest["treatment_fingerprint"]["contract"]["ppo"].items()),
            "Native PPO settings differ from evaluation")
    require(checkpoint.get("partial_reset_curriculum") is None
            and checkpoint.get("pooled_sampler_state") is None
            and getattr(cfg, "partial_reset_root", None) is None
            and getattr(cfg, "teacher_checkpoint", None) is None,
            "This cost recipe requires ordinary full-start training without a teacher")
    for field in ("model", "optimizer_state"):
        _assert_finite_tree(checkpoint[field], field)
    _assert_finite_loss_info(checkpoint["loss_info"], update - 1)
    cfg.finetune_foundation_behavior = True
    cfg.finetune_task_bank = False
    cfg.behavior_cost_ramp_updates = RAMP_UPDATES
    for key, value in decision["next_costs"].items():
        setattr(cfg, key, value)
    _validate_r2_resume_checkpoint(checkpoint, _r2_protocol_receipt(cfg), cfg)
    return {"next_update": update, "adam_step": adam, "finite": True}


def load_native_parent(path, decision, latest):
    # Inspection stays on CPU; the subsequent trainer process retains the
    # caller's GPU environment and initializes its own JAX backend.
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo), os.environ.get("TERRA_ROOT", str(repo.parent / "terra"))]
    old_platforms = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from utils.helpers import load_pkl_object, register_checkpoint_config_classes
            register_checkpoint_config_classes()
            return check_native_parent(load_pkl_object(str(path)), decision, latest)
    finally:
        if old_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = old_platforms


def launch_environment(decision, latest, checkpoint, run_dir, dataset_root):
    require(decision["ready"], "Completion gate has not passed; keep added costs at zero/current stage")
    checkpoint = Path(checkpoint).resolve()
    with checkpoint.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == decision["parent_sha256"],
                "Resume checkpoint differs from the latest evaluated checkpoint")
    contract = latest["treatment_fingerprint"]["contract"]
    devices = contract["ppo"]["num_devices"]
    require(type(devices) is int and devices in (1, 2, 4), "Use a supported 1, 2 or 4 GPU parent layout")
    require(contract["ppo"] == {
        "clip_eps": 0.2, "ent_schedule_end": 0.02, "ent_schedule_start": 0.15,
        "ent_schedule_steps": 20000, "flat_minibatch_shuffle": False, "gae_lambda": 0.95,
        "gamma": 0.9984, "lr": 0.0003, "max_grad_norm": 0.5, "num_devices": devices,
        "num_envs_per_device": 512 // devices, "num_minibatches": 32, "num_steps": 32,
        "update_epochs": 2, "use_value_clip": False, "vf_coef": 2.0,
    }, "Parent PPO settings differ from the supported scratch recipe")
    family = decision["family"]
    require(contract["run"]["config_name"] == {
        "foundation": "foundation_reward_sweep", "trench": "trench_align_v2_specialist_spec",
    }[family], "Wrong training-bank preset")
    seed = contract["run"]["seed"]
    fraction = int(100 * decision["next_fraction_of_combined_2x"])
    env = os.environ.copy()
    require(env.get("NUM_DEVICES", str(devices)) == str(devices),
            "A cost-stage fork must keep its parent's GPU layout; migrate and qualify at zero costs first")
    env.update(
        MACHINE=env.get("MACHINE", "cscs"), TERRA_PYTHON=env.get("TERRA_PYTHON", sys.executable),
        INITIALIZATION="resume", BANK_TRANSFER="0", BEHAVIOR_FINETUNE="1",
        BEHAVIOR_COST_RAMP_UPDATES=str(RAMP_UPDATES), NUM_DEVICES=str(devices),
        RESUME_FROM=str(checkpoint), START_UPDATE=str(decision["parent_update"]),
        TARGET_UPDATE=str(decision["target_update"]), SEED=str(seed), TASK_FAMILY=family,
        EXECUTABLE_DIG_OBSERVATION="1", CHECKPOINT_INTERVAL="500",
        RUN_DIR=str(Path(run_dir).resolve()), RUN_NAME=f"{family}-delayed-p{fraction}-s{seed}",
        DATASET_PATH=str(Path(dataset_root).resolve()), WANDB_DIR=str(Path(run_dir).resolve() / "wandb"),
        DATASET_SIZE=str({"foundation": 256, "trench": 1440}[family]),
        DISTANCE_SIDECAR_SHA=contract["reward_action"]["distance_sidecar_sha256"],
        **{key.upper(): str(value) for key, value in decision["next_costs"].items()},
    )
    # A new objective has its own history; ordinary same-objective continuations
    # can subsequently resume that history through train.sh.
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_RESUME", None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(COUNTS), required=True)
    parser.add_argument("--previous-evaluation", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parent-stage", type=Path, help="Parent's penalty_stage.json, required after the first cost stage")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    previous, latest, reference = map(load_report, (
        args.previous_evaluation, args.evaluation, args.reference_evaluation))
    parent_stage = json.loads(args.parent_stage.read_text()) if args.parent_stage else None
    decision = completion_gate(args.family, previous, latest, reference, parent_stage)
    print(json.dumps(decision, indent=2), flush=True)
    if not decision["ready"]:
        return 3
    dataset_root = check_training_bank(args.dataset_root, args.family,
                                     latest["r2_protocol_receipt"]["distance_sidecar_sha256"])
    env = launch_environment(decision, latest, args.checkpoint, args.run_dir, dataset_root)
    decision["parent_validation"] = load_native_parent(args.checkpoint, decision, latest)
    decision["run_name"] = env["RUN_NAME"]
    decision["dataset_root"] = str(dataset_root)
    script = Path(__file__).with_name("train.sh")
    command = subprocess.check_output(["bash", str(script), "--print-args"], env=env, text=True).strip()
    print("Native training arguments: " + command, flush=True)
    require(not args.run_dir.exists(), "Use a fresh output directory for the new cost stage")
    if args.execute:
        require("TRAIN_ENTRY" not in env, "Use the normal trainer for a cost stage")
        args.run_dir.mkdir(parents=True)
        decision["evaluations"] = [str(p.resolve()) for p in
                                   (args.previous_evaluation, args.evaluation, args.reference_evaluation)]
        (args.run_dir / "penalty_stage.json").write_text(json.dumps(decision, indent=2) + "\n")
        return subprocess.call(["bash", str(script)], env=env)
    print("Not executed. Run the same command with --execute inside the checked GPU allocation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
