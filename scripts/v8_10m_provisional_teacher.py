#!/usr/bin/env python3
"""Validate a same-distribution V8 checkpoint without requiring mastery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import jax
import numpy as np

from scripts import v8_10m_student
from scripts.euler_v8_deep_xattn_v1 import continuation_contract
from scripts.euler_v8_deep_xattn_v1 import stage_gate

SCHEMA = "terra_v8_10m_provisional_teacher_v1"
EXPECTED_TREATMENT = "G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER"
EXPECTED_ARM = "G-DEEP-XATTN-V8-DENSE-WARM"
MINIMUM_UPDATE = 5_000
LEGACY_FULL_SAMPLING_SHA256 = (
    "2a457be780e086c02e0474489b2060d6c577fac0ac429c48ad1a7e1e5e011357"
)


def legacy_full_sampling_contract(bank: dict) -> dict:
    """Reconstruct the exact bank_v4 sampler stored by the source teacher."""
    from utils.accepted_bank import load_accepted_bank

    accepted = load_accepted_bank(
        bank["root"],
        "G-UNIFORM",
        stage_gate.TERRA_REVISION,
        curriculum_stage="full",
        sampler_profile="bank_v4",
    )
    weights = np.asarray(accepted.sampling_probabilities, dtype=np.float64)
    contract = {
        "stage": "full",
        "conditions": [level.condition_id for level in accepted.levels],
        "declared_weights": weights.tolist(),
        "probabilities": (weights / weights.sum()).tolist(),
        "maps_per_condition": accepted.map_count_per_condition,
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != LEGACY_FULL_SAMPLING_SHA256:
        raise ValueError("legacy full-V8 teacher sampler changed")
    return {**contract, "sha256": digest, "sampler_profile": "bank_v4"}


def parse_run_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"{path}:{line_number}: duplicate or empty key")
        values[key] = value
    return values


def checkpoint_update(path: Path) -> int:
    match = re.search(r"_update_([0-9]{6})\.pkl$", path.name)
    if match is None:
        raise ValueError("teacher must be a numbered periodic checkpoint")
    return int(match.group(1))


def inspect_teacher(
    checkpoint_path: Path,
    expected_sha256: str,
    run_contract_path: Path,
    bank_root: Path,
) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    run_contract_path = run_contract_path.resolve()
    bank_root = bank_root.resolve()
    if not checkpoint_path.is_file() or not run_contract_path.is_file():
        raise ValueError("teacher checkpoint or run contract is unavailable")
    stage_gate._require_sha256(expected_sha256, "teacher checkpoint hash")
    if v8_10m_student.sha256_file(checkpoint_path) != expected_sha256:
        raise ValueError("teacher checkpoint hash changed")

    contract = parse_run_contract(run_contract_path)
    expected_contract = {
        "treatment": EXPECTED_TREATMENT,
        "arm": EXPECTED_ARM,
        "curriculum_stage": "full",
        "reward_type": "DENSE",
        "condition_count": "47",
        "terra_revision": stage_gate.TERRA_REVISION,
        "training_bank_release_id": stage_gate.RELEASE_ID,
        "training_bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "horizon": "450",
        "full_resets": "true",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(
                f"teacher run contract {key} must be {value!r}, "
                f"got {contract.get(key)!r}"
            )

    bank = stage_gate.load_bank_contract(bank_root)
    sampling = legacy_full_sampling_contract(bank)
    checkpoint = stage_gate._load_checkpoint(checkpoint_path)
    for key in (
        "model",
        "optimizer_state",
        "train_state_step",
        "pooled_sampler_state",
        "train_config",
    ):
        if key not in checkpoint:
            raise ValueError(f"teacher checkpoint lacks {key}")
    update = checkpoint_update(checkpoint_path)
    if update < MINIMUM_UPDATE or int(checkpoint.get("next_update", -1)) != update:
        raise ValueError("teacher checkpoint update is too early or inconsistent")
    optimizer_step = np.asarray(checkpoint["train_state_step"])
    if optimizer_step.size != 1 or not np.isfinite(optimizer_step).all():
        raise ValueError("teacher optimizer step is not one finite scalar")
    if int(optimizer_step.reshape(())) <= 0:
        raise ValueError("teacher optimizer has not advanced")
    continuation_contract._require_finite_tree(
        checkpoint["model"], "teacher model parameters"
    )
    continuation_contract._require_finite_tree(
        checkpoint["optimizer_state"], "teacher optimizer state"
    )
    stage_gate._validate_sampler_state(
        checkpoint["pooled_sampler_state"],
        sampling,
        int(contract["seed"]),
        checkpoint_path,
    )

    config = checkpoint["train_config"]
    architecture = {
        key: stage_gate._field(config, key)
        for key in v8_10m_student.TEACHER_ARCHITECTURE
    }
    v8_10m_student.validate_architecture(
        architecture, v8_10m_student.TEACHER_ARCHITECTURE, "provisional teacher"
    )
    parameter_count = int(
        sum(
            np.asarray(value).size
            for value in jax.tree_util.tree_leaves(checkpoint["model"])
        )
    )
    if parameter_count != 2_856_685:
        raise ValueError(
            f"provisional teacher must contain 2,856,685 parameters, got {parameter_count:,}"
        )
    accepted_bank = stage_gate._field(config, "accepted_bank")
    for key, value in {
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "curriculum_stage": "full",
    }.items():
        if stage_gate._field(accepted_bank, key) != value:
            raise ValueError(f"teacher accepted-bank {key} changed")

    return {
        "schema": SCHEMA,
        "passed": True,
        "provisional_teacher": True,
        "performance_mastery_gate_waived_by_user": True,
        "same_distribution": True,
        "finite_model_optimizer": True,
        "full_sampler_state_validated": True,
        "teacher_sampler_profile": sampling["sampler_profile"],
        "teacher_sampling_sha256": sampling["sha256"],
        "teacher_arm": EXPECTED_ARM,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": expected_sha256,
        "teacher_update": update,
        "teacher_optimizer_step": int(optimizer_step.reshape(())),
        "teacher_parameter_count": parameter_count,
        "teacher_architecture": v8_10m_student.TEACHER_ARCHITECTURE,
        "run_contract": str(run_contract_path),
        "run_contract_sha256": v8_10m_student.sha256_file(run_contract_path),
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "curriculum_stage": "full",
        "reward_stage": "dense_skill",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = inspect_teacher(
        args.checkpoint,
        args.checkpoint_sha256,
        args.run_contract,
        args.bank_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
