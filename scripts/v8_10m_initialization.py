#!/usr/bin/env python3
"""Compare a qualified V8 teacher with one scale-screen initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eval_fixed_bank import configure_for_bank
from eval_fixed_bank import exact_reset_keys
from eval_fixed_bank import load_manifest
from eval_fixed_bank import verify_exact_reset
from scripts import v8_10m_student
from train_mixed import make_mixed_agent_states
from utils.accepted_bank import load_accepted_bank
from utils.helpers import load_pkl_object
from utils.helpers import register_checkpoint_config_classes
from utils.utils_ppo import obs_to_model_input

ARMS = {
    "G-V8-XATTN-REWARM-CONTROL": {
        "architecture": v8_10m_student.TEACHER_ARCHITECTURE,
        "parameters": 2_856_685,
    },
    "G-V8-10M-XATTN-WARM": {
        "architecture": v8_10m_student.TARGET_ARCHITECTURE,
        "parameters": v8_10m_student.TARGET_PARAMETER_COUNT,
    },
}


def inspect_teacher_source(
    *,
    teacher_receipt: Path | None,
    teacher_inspection: Path | None,
    bank_root: Path,
) -> tuple[dict, Path, str]:
    """Return one validated teacher identity from either admission path."""
    if (teacher_receipt is None) == (teacher_inspection is None):
        raise ValueError("provide exactly one teacher receipt or inspection")
    if teacher_receipt is not None:
        path = teacher_receipt.resolve()
        return (
            v8_10m_student.inspect_teacher(path, bank_root.resolve()),
            path,
            "qualified_receipt",
        )

    path = teacher_inspection.resolve()
    record = json.loads(path.read_text())
    expected = {
        "schema": "terra_v8_10m_provisional_teacher_v1",
        "passed": True,
        "provisional_teacher": True,
        "performance_mastery_gate_waived_by_user": True,
        "same_distribution": True,
        "finite_model_optimizer": True,
        "full_sampler_state_validated": True,
        "teacher_arm": v8_10m_student.TEACHER_ARM,
        "release_id": v8_10m_student.stage_gate.RELEASE_ID,
        "terra_revision": v8_10m_student.stage_gate.TERRA_REVISION,
        "curriculum_stage": "full",
        "reward_stage": "dense_skill",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"provisional teacher inspection {key} changed")
    checkpoint = Path(str(record.get("teacher_checkpoint", ""))).resolve()
    checkpoint_sha = str(record.get("teacher_checkpoint_sha256", ""))
    if (
        not checkpoint.is_file()
        or v8_10m_student.sha256_file(checkpoint) != checkpoint_sha
    ):
        raise ValueError("provisional teacher checkpoint changed after inspection")
    return record, path, "provisional_inspection"


def tree_sha256(tree) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        value = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_outputs(
    teacher_value: np.ndarray,
    teacher_logits: np.ndarray,
    student_value: np.ndarray,
    student_logits: np.ndarray,
) -> dict:
    arrays = {
        "teacher_value": np.asarray(teacher_value, dtype=np.float64),
        "teacher_logits": np.asarray(teacher_logits, dtype=np.float64),
        "student_value": np.asarray(student_value, dtype=np.float64),
        "student_logits": np.asarray(student_logits, dtype=np.float64),
    }
    if arrays["teacher_value"].shape != arrays["student_value"].shape:
        raise ValueError("teacher and student value shapes differ")
    if arrays["teacher_logits"].shape != arrays["student_logits"].shape:
        raise ValueError("teacher and student logit shapes differ")
    if arrays["teacher_logits"].ndim != 2:
        raise ValueError("policy logits must be a batch by action matrix")
    if arrays["teacher_value"].shape[0] != arrays["teacher_logits"].shape[0]:
        raise ValueError("value and policy batch sizes differ")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("teacher/student initialization outputs are non-finite")

    teacher_logits_64 = arrays["teacher_logits"]
    student_logits_64 = arrays["student_logits"]
    teacher_logp = teacher_logits_64 - np.logaddexp.reduce(
        teacher_logits_64, axis=-1, keepdims=True
    )
    student_logp = student_logits_64 - np.logaddexp.reduce(
        student_logits_64, axis=-1, keepdims=True
    )
    teacher_probability = np.exp(teacher_logp)
    per_reset_kl = np.sum(teacher_probability * (teacher_logp - student_logp), axis=-1)
    teacher_action = np.argmax(teacher_logits_64, axis=-1)
    student_action = np.argmax(student_logits_64, axis=-1)
    value_delta = arrays["student_value"] - arrays["teacher_value"]
    return {
        "finite": True,
        "resets": int(teacher_logits_64.shape[0]),
        "actions": int(teacher_logits_64.shape[1]),
        "teacher_to_student_kl_mean": float(per_reset_kl.mean()),
        "teacher_to_student_kl_p95": float(np.percentile(per_reset_kl, 95)),
        "deterministic_action_agreement": float(
            np.mean(teacher_action == student_action)
        ),
        "value_rmse": float(np.sqrt(np.mean(np.square(value_delta)))),
        "value_absolute_error_p95": float(np.percentile(np.abs(value_delta), 95)),
    }


def checkpoint_architecture(checkpoint: dict) -> dict:
    config = checkpoint.get("train_config")
    if config is None:
        raise ValueError("initial checkpoint lacks train_config")
    return {
        key: v8_10m_student.stage_gate._field(config, key)
        for key in v8_10m_student.TEACHER_ARCHITECTURE
    }


def model_outputs(
    *,
    checkpoint: dict,
    bank_root: Path,
    relative_path: str,
    manifest: list[dict],
    reset_keys,
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    count = len(manifest)
    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(count)
    config = configure_for_bank(checkpoint["train_config"], relative_path, count)
    _, env, env_params, initialized_state = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    reset_verification = verify_exact_reset(
        env,
        env_params,
        reset_keys,
        bank_root / relative_path,
        count,
    )
    reset = env.reset(env_params, reset_keys)
    previous_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    model_input = obs_to_model_input(reset.observation, previous_actions, config)
    value, logits = initialized_state.apply_fn(checkpoint["model"], model_input)
    return (
        np.asarray(jax.device_get(value)),
        np.asarray(jax.device_get(logits)),
        reset_verification,
        tree_sha256(reset.observation),
    )


def run_diagnostic(
    *,
    arm: str,
    teacher_receipt: Path | None,
    teacher_inspection: Path | None,
    student_checkpoint_path: Path,
    bank_root: Path,
    terra_revision: str,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unsupported 10M comparison arm: {arm}")
    teacher, teacher_identity_path, teacher_admission = inspect_teacher_source(
        teacher_receipt=teacher_receipt,
        teacher_inspection=teacher_inspection,
        bank_root=bank_root,
    )
    if terra_revision != v8_10m_student.stage_gate.TERRA_REVISION:
        raise ValueError("initialization diagnostic Terra revision changed")
    accepted = load_accepted_bank(
        bank_root.resolve(),
        "G-UNIFORM",
        terra_revision,
        curriculum_stage="full",
    )
    panel = next(
        panel for panel in accepted.evaluation_panels if panel.name == "promotion"
    )
    manifest = load_manifest(bank_root / panel.maps_path)
    # This is a transplant diagnostic, not a behavioral evaluation.  The V8
    # combined main-panel reset_seed values retain episode identities from the
    # component banks and therefore do not select their new combined slots.
    # Enumerate every frozen map slot deterministically so teacher and student
    # see identical observations without pretending those keys are frozen
    # benchmark episodes.
    reset_keys = exact_reset_keys(len(manifest))

    teacher_checkpoint_path = Path(teacher["teacher_checkpoint"]).resolve()
    student_checkpoint_path = student_checkpoint_path.resolve()
    register_checkpoint_config_classes()
    teacher_checkpoint = load_pkl_object(str(teacher_checkpoint_path))
    student_checkpoint = load_pkl_object(str(student_checkpoint_path))
    expected = ARMS[arm]
    v8_10m_student.validate_architecture(
        checkpoint_architecture(student_checkpoint),
        expected["architecture"],
        "initial student",
    )
    parameter_count = int(
        sum(
            value.size
            for value in jax.tree_util.tree_leaves(student_checkpoint["model"])
        )
    )
    if parameter_count != expected["parameters"]:
        raise ValueError(
            f"{arm} must have {expected['parameters']:,} parameters, "
            f"got {parameter_count:,}"
        )

    teacher_value, teacher_logits, teacher_reset, teacher_observation_sha = (
        model_outputs(
            checkpoint=teacher_checkpoint,
            bank_root=bank_root,
            relative_path=panel.maps_path,
            manifest=manifest,
            reset_keys=reset_keys,
        )
    )
    student_value, student_logits, student_reset, student_observation_sha = (
        model_outputs(
            checkpoint=student_checkpoint,
            bank_root=bank_root,
            relative_path=panel.maps_path,
            manifest=manifest,
            reset_keys=reset_keys,
        )
    )
    if teacher_observation_sha != student_observation_sha:
        raise ValueError("teacher and student did not receive identical observations")
    metrics = summarize_outputs(
        teacher_value,
        teacher_logits,
        student_value,
        student_logits,
    )
    return {
        "schema": "terra_v8_10m_initialization_diagnostic_v1",
        "passed": True,
        "diagnostic_only": True,
        "arm": arm,
        "release_id": accepted.release_id,
        "terra_revision": terra_revision,
        "panel": "promotion",
        "exact_frozen_map_slots": len(manifest),
        "reset_key_contract": "deterministic_exact_slot_keys_v1",
        "reset_key_sha256": tree_sha256(reset_keys),
        "teacher_admission": teacher_admission,
        "teacher_identity": str(teacher_identity_path),
        "teacher_identity_sha256": v8_10m_student.sha256_file(teacher_identity_path),
        "teacher_checkpoint": str(teacher_checkpoint_path),
        "teacher_checkpoint_sha256": v8_10m_student.sha256_file(
            teacher_checkpoint_path
        ),
        "student_checkpoint": str(student_checkpoint_path),
        "student_checkpoint_sha256": v8_10m_student.sha256_file(
            student_checkpoint_path
        ),
        "student_parameter_count": parameter_count,
        "observation_sha256": teacher_observation_sha,
        "teacher_reset_verification": teacher_reset,
        "student_reset_verification": student_reset,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    teacher = parser.add_mutually_exclusive_group(required=True)
    teacher.add_argument("--teacher-receipt", type=Path)
    teacher.add_argument("--teacher-inspection", type=Path)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_diagnostic(
        arm=args.arm,
        teacher_receipt=args.teacher_receipt,
        teacher_inspection=args.teacher_inspection,
        student_checkpoint_path=args.student_checkpoint,
        bank_root=args.bank_root.resolve(),
        terra_revision=args.terra_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result["metrics"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
