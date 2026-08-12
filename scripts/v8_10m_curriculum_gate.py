#!/usr/bin/env python3
"""Issue the small Stage-A gate for the paired V8 10M curriculum screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from scripts import v8_10m_student
from scripts.euler_v8_deep_xattn_v1 import continuation_contract
from scripts.euler_v8_deep_xattn_v1 import stage_gate

SCHEMA = "terra_v8_10m_curriculum_gate_v1"
UPDATES = (1000, 2000, 3000, 4000)
ARMS = {
    "G-V8-XATTN-REWARM-CONTROL": {
        "parameters": 2_856_685,
        "architecture": v8_10m_student.TEACHER_ARCHITECTURE,
    },
    "G-V8-10M-XATTN-WARM": {
        "parameters": v8_10m_student.TARGET_PARAMETER_COUNT,
        "architecture": v8_10m_student.TARGET_ARCHITECTURE,
    },
}


def read_records(path: Path, split: str) -> list[dict]:
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
            "stratum": "capability",
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path}: {key} changed")
        stage_gate.validate_panel_conditions([record], stage_gate.CAPABILITY_IDS)
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
    expected = ARMS[arm]
    for record in records:
        path = Path(str(record["checkpoint"])).resolve()
        if path.parent != run_dir / "checkpoints" or not path.is_file():
            raise ValueError("evaluated checkpoint is outside the curriculum run")
        if v8_10m_student.sha256_file(path) != record.get("checkpoint_sha256"):
            raise ValueError("evaluated checkpoint hash changed")
        checkpoint = stage_gate._load_checkpoint(path)
        if int(checkpoint.get("next_update", -1)) != int(record["checkpoint_update"]):
            raise ValueError("evaluated checkpoint update changed")
        for key in ("model", "optimizer_state", "pooled_sampler_state", "train_config"):
            if key not in checkpoint:
                raise ValueError(f"evaluated checkpoint lacks {key}")
        continuation_contract._require_finite_tree(
            checkpoint["model"], "curriculum model"
        )
        continuation_contract._require_finite_tree(
            checkpoint["optimizer_state"], "curriculum optimizer"
        )
        count = int(
            sum(
                np.asarray(value).size
                for value in jax.tree_util.tree_leaves(checkpoint["model"])
            )
        )
        if count != expected["parameters"]:
            raise ValueError("curriculum checkpoint parameter count changed")
        config = checkpoint["train_config"]
        architecture = {
            key: stage_gate._field(config, key)
            for key in v8_10m_student.TEACHER_ARCHITECTURE
        }
        v8_10m_student.validate_architecture(
            architecture, expected["architecture"], "curriculum checkpoint"
        )
        accepted_bank = stage_gate._field(config, "accepted_bank")
        for key, value in {
            "release_id": stage_gate.RELEASE_ID,
            "terra_revision": stage_gate.TERRA_REVISION,
            "curriculum_stage": "capability",
        }.items():
            if stage_gate._field(accepted_bank, key) != value:
                raise ValueError(f"curriculum accepted-bank {key} changed")
        stage_gate._validate_sampler_state(
            checkpoint["pooled_sampler_state"], sampling, seed, path
        )


def build_gate(
    *,
    arm: str,
    run_contract_path: Path,
    promotion_path: Path,
    development_path: Path,
    bank_root: Path,
) -> dict:
    run_contract_path = run_contract_path.resolve()
    run_dir = run_contract_path.parent
    contract = stage_gate.parse_run_contract(run_contract_path)
    expected_contract = {
        "arm": arm,
        "phase": "screen",
        "curriculum_stage": "capability",
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "condition_count": "2",
        "horizon": "450",
        "full_resets": "true",
        "updates": "4000",
        "absolute_target_global_transitions": "262144000",
        "num_envs_per_device": "512",
        "num_minibatches": "32",
        "learning_rate": "0.00015",
        "kickstart_kl": "1.0_to_0_over_3000",
        "kickstart_value": "0.5_to_0_over_1000",
        "kickstart_lr_warmup": "0.00005_to_0.00015_over_200",
        "entropy_schedule": "0.02_to_0.005_over_20000",
        "terra_revision": stage_gate.TERRA_REVISION,
        "training_bank_release_id": stage_gate.RELEASE_ID,
        "training_bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"run contract {key} changed")
    seed = int(contract["seed"])
    bank = stage_gate.load_bank_contract(bank_root.resolve())
    sampling = stage_gate.stage_sampling_contract(bank, "capability")
    promotion = read_records(promotion_path.resolve(), "promotion")
    development = read_records(development_path.resolve(), "development")
    if [stage_gate._checkpoint_identity(item) for item in promotion] != [
        stage_gate._checkpoint_identity(item) for item in development
    ]:
        raise ValueError("promotion and development evaluated different checkpoints")
    validate_checkpoints(
        promotion, arm=arm, run_dir=run_dir, sampling=sampling, seed=seed
    )
    promotion_gate = stage_gate.decide_capability(promotion)
    development_gate = stage_gate.decide_capability(development)
    candidate = promotion[-1]
    return {
        "schema": SCHEMA,
        "passed": bool(promotion_gate["passed"] and development_gate["passed"]),
        "arm": arm,
        "curriculum_stage": "capability",
        "next_stage": "nearby",
        "reward_stage": "dense_skill",
        "reward_transition_launched": False,
        "promotion": promotion_gate,
        "development": development_gate,
        "candidate": {
            "path": candidate["checkpoint"],
            "sha256": candidate["checkpoint_sha256"],
            "update": candidate["checkpoint_update"],
        },
        "inputs": {
            "run_contract": {
                "path": str(run_contract_path),
                "sha256": v8_10m_student.sha256_file(run_contract_path),
            },
            "promotion": {
                "path": str(promotion_path.resolve()),
                "sha256": v8_10m_student.sha256_file(promotion_path.resolve()),
            },
            "development": {
                "path": str(development_path.resolve()),
                "sha256": v8_10m_student.sha256_file(development_path.resolve()),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_gate(
        arm=args.arm,
        run_contract_path=args.run_contract,
        promotion_path=args.promotion,
        development_path=args.development,
        bank_root=args.bank_root,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
