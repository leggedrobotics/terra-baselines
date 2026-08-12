#!/usr/bin/env python3
"""Validate the 20k nearby-stage compact/10M curriculum result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from scripts import v8_10m_curriculum_gate, v8_10m_stage_b_selection, v8_10m_student
from scripts.euler_v8_deep_xattn_v1 import continuation_contract, stage_gate

SCHEMA = "terra_v8_10m_stage_b_gate_v1"
UPDATES = tuple(range(1000, 20_001, 1000))
REFERENCE_TEACHER = {
    "path": "/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_direct_full_teacher_v1/885b1dbe6d7ac4b4199140a188d147657760eeee/screen/s20260730/G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER/checkpoints/v8_direct_full_885b1dbe6d7a_screen_g_deep_xattn_v8_dense_warm_s20260730-euler-2026-08-04-07-29-25_update_007500.pkl",
    "sha256": "a6bebfffcf4d390df19ade9652d3c96d833eb7d2587ddb1b95035b7ad6a807f6",
    "update": 7500,
    "parameters": 2_856_685,
}
REFERENCE_TEACHER_RUN_CONTRACT = {
    "path": "/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_direct_full_teacher_v1/885b1dbe6d7ac4b4199140a188d147657760eeee/screen/s20260730/G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER/run_contract.env",
    "sha256": "5ebb4de9769d2222839d9845b8eb753d3275fb9b5bab9772d5030470926456fc",
}
REFERENCE_TEACHER_EVAL_ROOT = "/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_10m_curriculum_v1/548253810319831ce6102bea7cd315bff508ba94/reference_teacher_full_v8/s20260730"
REFERENCE_TEACHER_EVIDENCE = {
    "status.env": "dd46643b9ef4493330eb4997ebe7ff32083da46483fa1ea7e0ca9b6bb94fbbfa",
    "eval/main_promotion.json": "7d7129595f0fa33d5fb1bb2f32a0e2a2f0ebcada2e09b933c346e81d1c9e4459",
    "eval/main_development.json": "6c4886822509e8f17f34f4d3904289c878cfc71d08061abc0084d8056f6f7f67",
    "eval/capability_promotion.json": "e8f9cfaf31d085ee4e5378f42e9c838391144456c62692b8ec3b22a68872107e",
    "eval/capability_development.json": "6e0654f70b9d9b7d142e5c0603c3ad92fc4140b0379b934e748a9aec069fa89f",
}


def read_records(
    path: Path,
    split: str,
    condition_ids: tuple[str, ...],
) -> list[dict]:
    records = json.loads(path.read_text())
    if (
        not isinstance(records, list)
        or tuple(record.get("checkpoint_update") for record in records) != UPDATES
    ):
        raise ValueError(f"{path}: expected checkpoint updates {UPDATES}")
    for record in records:
        expected = {
            "schema": stage_gate.EVAL_SCHEMA,
            "completion_contract": stage_gate.COMPLETION_CONTRACT,
            "horizon": 450,
            "deterministic": True,
            "policy_mode": "deterministic",
            "exact_manifest_enumeration": True,
            "split": split,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path}: {key} changed")
        stage_gate.validate_panel_conditions([record], condition_ids)
        if not stage_gate._integrity_passed(record):
            raise ValueError(f"{path}: evaluation integrity failed")
    return records


def validate_checkpoints(
    records: list[dict],
    *,
    arm: str,
    run_dir: Path,
    sampling: dict,
    seed: int,
) -> None:
    expected = v8_10m_curriculum_gate.ARMS[arm]
    for record in records:
        path = Path(str(record["checkpoint"])).resolve()
        if path.parent != run_dir / "checkpoints" or not path.is_file():
            raise ValueError("evaluated checkpoint is outside the Stage-B run")
        if v8_10m_student.sha256_file(path) != record.get("checkpoint_sha256"):
            raise ValueError("evaluated checkpoint hash changed")
        checkpoint = stage_gate._load_checkpoint(path)
        if int(checkpoint.get("next_update", -1)) != int(record["checkpoint_update"]):
            raise ValueError("evaluated checkpoint update changed")
        for key in ("model", "optimizer_state", "pooled_sampler_state", "train_config"):
            if key not in checkpoint:
                raise ValueError(f"evaluated checkpoint lacks {key}")
        continuation_contract._require_finite_tree(checkpoint["model"], "Stage-B model")
        continuation_contract._require_finite_tree(
            checkpoint["optimizer_state"], "Stage-B optimizer"
        )
        count = int(
            sum(
                np.asarray(value).size
                for value in jax.tree_util.tree_leaves(checkpoint["model"])
            )
        )
        if count != expected["parameters"]:
            raise ValueError("Stage-B checkpoint parameter count changed")
        config = checkpoint["train_config"]
        architecture = {
            key: stage_gate._field(config, key)
            for key in v8_10m_student.TEACHER_ARCHITECTURE
        }
        v8_10m_student.validate_architecture(
            architecture, expected["architecture"], "Stage-B checkpoint"
        )
        accepted_bank = stage_gate._field(config, "accepted_bank")
        for key, value in {
            "release_id": stage_gate.RELEASE_ID,
            "terra_revision": stage_gate.TERRA_REVISION,
            "curriculum_stage": "nearby",
            "sampler_profile": "bounded_replay25_v1",
        }.items():
            if stage_gate._field(accepted_bank, key) != value:
                raise ValueError(f"Stage-B accepted-bank {key} changed")
        stage_gate._validate_sampler_state(
            checkpoint["pooled_sampler_state"], sampling, seed, path
        )


def _best_observed(records: list[dict]) -> dict:
    selected = max(
        records,
        key=lambda record: (
            record["summary"]["overall"]["successes"],
            record["summary"]["graded"]["macro_completion"],
            -int(record["checkpoint_update"]),
        ),
    )
    return {
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "checkpoint_update": selected["checkpoint_update"],
        "exact_successes": selected["summary"]["overall"]["successes"],
        "episodes": selected["summary"]["overall"]["episodes"],
        "macro_completion": selected["summary"]["graded"]["macro_completion"],
    }


def build_gate(args: argparse.Namespace) -> dict:
    run_contract_path = args.run_contract.resolve()
    run_dir = run_contract_path.parent
    contract = stage_gate.parse_run_contract(run_contract_path)
    selection_path = args.parent_selection.resolve()
    selection = v8_10m_stage_b_selection.inspect_selection(selection_path)
    parent = selection["parents"][args.arm]
    teacher = dict(REFERENCE_TEACHER)
    expected_contract = {
        "arm": args.arm,
        "phase": "screen",
        "curriculum_stage": "nearby",
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "reward_transition_launched": "false",
        "condition_sampler": "fixed_v8_bounded_replay",
        "sampler_profile": "bounded_replay25_v1",
        "condition_count": "15",
        "maps_per_condition": "96",
        "distinct_training_maps": "1440",
        "horizon": "450",
        "full_resets": "true",
        "updates": "20000",
        "absolute_target_global_transitions": "1310720000",
        "num_devices": "4",
        "num_envs_per_device": "512",
        "num_minibatches": "32",
        "learning_rate": "0.00015",
        "teacher_checkpoint": teacher["path"],
        "teacher_checkpoint_sha256": teacher["sha256"],
        "teacher_checkpoint_update": str(teacher["update"]),
        "teacher_parameter_count": str(teacher["parameters"]),
        "teacher_admission": "fixed_whole_v8_eval_v1",
        "teacher_source_run_contract": REFERENCE_TEACHER_RUN_CONTRACT["path"],
        "teacher_source_run_contract_sha256": REFERENCE_TEACHER_RUN_CONTRACT["sha256"],
        "teacher_fixed_eval_root": REFERENCE_TEACHER_EVAL_ROOT,
        "teacher_fixed_eval_status_sha256": REFERENCE_TEACHER_EVIDENCE["status.env"],
        "teacher_fixed_eval_main_promotion_sha256": REFERENCE_TEACHER_EVIDENCE[
            "eval/main_promotion.json"
        ],
        "teacher_fixed_eval_main_development_sha256": REFERENCE_TEACHER_EVIDENCE[
            "eval/main_development.json"
        ],
        "teacher_fixed_eval_capability_promotion_sha256": REFERENCE_TEACHER_EVIDENCE[
            "eval/capability_promotion.json"
        ],
        "teacher_fixed_eval_capability_development_sha256": REFERENCE_TEACHER_EVIDENCE[
            "eval/capability_development.json"
        ],
        "teacher_fixed_promotion_exact": "552/720",
        "teacher_fixed_promotion_macro": "0.8747645628069424",
        "teacher_fixed_development_exact": "538/720",
        "teacher_fixed_development_macro": "0.8676237401759459",
        "teacher_fixed_capability_promotion_exact": "31/32",
        "teacher_fixed_capability_development_exact": "31/32",
        "kickstart": "current_rollout_kl1_3000_value0p5_1000",
        "kickstart_kl_coef": "1.0",
        "kickstart_kl_anneal_updates": "3000",
        "kickstart_value_coef": "0.5",
        "kickstart_value_anneal_updates": "1000",
        "kickstart_lr_warmup_updates": "200",
        "initialization": "params_only_stage_transition_fresh_optimizer",
        "entropy_schedule": "0.02_to_0.005_over_20000",
        "terra_revision": stage_gate.TERRA_REVISION,
        "training_bank_release_id": stage_gate.RELEASE_ID,
        "training_bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "parent_checkpoint": parent["path"],
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_selection_sha256": selection["receipt_sha256"],
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"run contract {key} changed")
    teacher_path = Path(teacher["path"])
    if v8_10m_student.sha256_file(teacher_path) != teacher["sha256"]:
        raise ValueError("reference teacher checkpoint changed")
    teacher_contract_path = Path(REFERENCE_TEACHER_RUN_CONTRACT["path"])
    if (
        v8_10m_student.sha256_file(teacher_contract_path)
        != REFERENCE_TEACHER_RUN_CONTRACT["sha256"]
    ):
        raise ValueError("reference teacher source contract changed")
    evidence_root = Path(REFERENCE_TEACHER_EVAL_ROOT)
    for relative_path, expected_sha256 in REFERENCE_TEACHER_EVIDENCE.items():
        if v8_10m_student.sha256_file(evidence_root / relative_path) != expected_sha256:
            raise ValueError(f"reference teacher evidence changed: {relative_path}")
    status = stage_gate.parse_run_contract(evidence_root / "status.env")
    if (
        status.get("status") != "PASSED"
        or status.get("teacher_checkpoint_sha256") != teacher["sha256"]
    ):
        raise ValueError("reference teacher evaluation did not pass")
    inspection_path = Path(contract.get("teacher_inspection", "")).resolve()
    if inspection_path.parent != run_dir or not inspection_path.is_file():
        raise ValueError("reference teacher inspection is outside the Stage-B run")
    if v8_10m_student.sha256_file(inspection_path) != contract.get(
        "teacher_inspection_sha256"
    ):
        raise ValueError("reference teacher inspection changed")
    inspection = json.loads(inspection_path.read_text())
    for key, value in {
        "schema": "terra_v8_10m_provisional_teacher_v1",
        "passed": True,
        "same_distribution": True,
        "teacher_checkpoint": teacher["path"],
        "teacher_checkpoint_sha256": teacher["sha256"],
        "teacher_update": teacher["update"],
        "teacher_parameter_count": teacher["parameters"],
    }.items():
        if inspection.get(key) != value:
            raise ValueError(f"reference teacher inspection {key} changed")
    seed = int(contract["seed"])
    bank = stage_gate.load_bank_contract(args.bank_root.resolve())
    sampling = stage_gate.stage_sampling_contract(bank, "nearby", "bounded_replay25_v1")
    main_promotion = read_records(
        args.promotion.resolve(), "promotion", bank["main_ids"]
    )
    main_development = read_records(
        args.development.resolve(), "development", bank["main_ids"]
    )
    capability_promotion = read_records(
        args.capability_promotion.resolve(),
        "promotion",
        stage_gate.CAPABILITY_IDS,
    )
    capability_development = read_records(
        args.capability_development.resolve(),
        "development",
        stage_gate.CAPABILITY_IDS,
    )
    identities = [stage_gate._checkpoint_identity(item) for item in main_promotion]
    for records in (
        main_development,
        capability_promotion,
        capability_development,
    ):
        if [stage_gate._checkpoint_identity(item) for item in records] != identities:
            raise ValueError("Stage-B panels evaluated different checkpoints")
    validate_checkpoints(
        main_promotion,
        arm=args.arm,
        run_dir=run_dir,
        sampling=sampling,
        seed=seed,
    )
    prior = {"retention": selection["retention"]}
    promotion = stage_gate.decide_nearby(
        main_promotion,
        capability_promotion,
        prior,
        bank["core_ids"],
        bank["family_by_condition"],
    )
    development = stage_gate.decide_nearby(
        main_development,
        capability_development,
        prior,
        bank["core_ids"],
        bank["family_by_condition"],
    )
    candidate = main_promotion[-1]
    return {
        "schema": SCHEMA,
        "passed": bool(promotion["passed"] and development["passed"]),
        "arm": args.arm,
        "curriculum_stage": "nearby",
        "next_stage": "full",
        "reward_stage": "dense_skill",
        "reward_transition_launched": False,
        "promotion": promotion,
        "development": development,
        "candidate": {
            "path": candidate["checkpoint"],
            "sha256": candidate["checkpoint_sha256"],
            "update": candidate["checkpoint_update"],
        },
        "best_observed_promotion": _best_observed(main_promotion),
        "best_observed_development": _best_observed(main_development),
        "parent": parent,
        "teacher": teacher,
        "population": sampling,
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": v8_10m_student.sha256_file(path.resolve()),
            }
            for name, path in {
                "run_contract": args.run_contract,
                "parent_selection": args.parent_selection,
                "promotion": args.promotion,
                "development": args.development,
                "capability_promotion": args.capability_promotion,
                "capability_development": args.capability_development,
            }.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=tuple(v8_10m_curriculum_gate.ARMS), required=True
    )
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--capability-promotion", type=Path, required=True)
    parser.add_argument("--capability-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_gate(args)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
