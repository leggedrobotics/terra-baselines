#!/usr/bin/env python3
"""Inspect a qualified V8 full-stage receipt and its resume checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax
import numpy as np

if __package__:
    from . import stage_gate
else:
    import stage_gate

TARGET_UPDATE = 80_000


def inspect_receipt(path: Path) -> tuple[dict, dict]:
    path = path.resolve()
    raw = stage_gate._read_json(path, dict)
    arm = raw.get("arm")
    if arm not in stage_gate.ARM_ARCHITECTURES:
        raise ValueError(f"{path}: unsupported V8 arm {arm!r}")
    receipt = stage_gate.validate_prior_receipt(path, arm, "full")
    qualification = receipt.get("continuation_qualification")
    if not isinstance(qualification, dict) or (
        qualification.get("qualified_for_120h") is not True
    ):
        raise ValueError(f"{path}: full receipt is not qualified for 120h")

    candidate = receipt["candidate"]
    parts = Path(candidate["path"]).parts
    try:
        seed_token = parts[parts.index("full") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}: cannot derive seed from candidate path") from exc
    if re.fullmatch(r"s[0-9]+", seed_token) is None:
        raise ValueError(f"{path}: candidate path has an invalid seed component")

    run_contract = receipt.get("inputs", {}).get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError(f"{path}: receipt lacks its full-stage run contract")
    run_contract_path = run_contract.get("path")
    run_contract_sha = run_contract.get("sha256")
    stage_gate._require_sha256(run_contract_sha, "full-stage run-contract hash")
    if (
        not isinstance(run_contract_path, str)
        or not Path(run_contract_path).is_absolute()
    ):
        raise ValueError(f"{path}: full-stage run-contract path is not absolute")

    info = {
        "schema": "terra_v8_continuation_inspection_v1",
        "receipt_path": str(path),
        "receipt_sha256": stage_gate.sha256_file(path),
        "arm": arm,
        "seed": int(seed_token[1:]),
        "terra_baselines_revision": receipt["terra_baselines_revision"],
        "candidate_path": candidate["path"],
        "candidate_sha256": candidate["checkpoint_sha256"],
        "candidate_update": candidate["next_update"],
        "parent_run_contract_path": run_contract_path,
        "parent_run_contract_sha256": run_contract_sha,
        "target_update": TARGET_UPDATE,
        "eligibility_scope": "per_arm",
    }
    if not 0 < info["candidate_update"] < TARGET_UPDATE:
        raise ValueError(
            f"{path}: candidate update must be below absolute target {TARGET_UPDATE}"
        )
    return receipt, info


def _require_finite_tree(value: object, label: str) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains non-finite values")


def verify_checkpoint(
    receipt_path: Path,
    checkpoint_path: Path,
    expected_sha256: str,
    expected_arm: str,
    expected_seed: int,
    expected_baselines_revision: str,
) -> dict:
    receipt, info = inspect_receipt(receipt_path)
    expected = {
        "arm": expected_arm,
        "seed": expected_seed,
        "terra_baselines_revision": expected_baselines_revision,
        "candidate_path": str(checkpoint_path),
        "candidate_sha256": expected_sha256,
    }
    for field, value in expected.items():
        if info[field] != value:
            raise ValueError(
                f"continuation {field} must be {value!r}, got {info[field]!r}"
            )

    stage_gate._require_sha256(expected_sha256, "resume checkpoint hash")
    if stage_gate.sha256_file(checkpoint_path) != expected_sha256:
        raise ValueError("resume checkpoint hash differs from qualified receipt")
    candidate = stage_gate.validate_candidate_checkpoint(
        receipt["candidate"], "full", expected_arm
    )
    checkpoint = stage_gate._load_checkpoint(checkpoint_path)
    for field in ("optimizer_state", "train_state_step", "pooled_sampler_state"):
        if field not in checkpoint:
            raise ValueError(f"resume checkpoint lacks required {field}")
    if int(checkpoint["next_update"]) != info["candidate_update"]:
        raise ValueError("resume checkpoint update differs from qualified receipt")
    if int(np.asarray(checkpoint["train_state_step"])) <= 0:
        raise ValueError("resume checkpoint has a fresh optimizer step")
    source_treatment_name = stage_gate._field(checkpoint.get("train_config"), "name")
    if (
        not isinstance(source_treatment_name, str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", source_treatment_name) is None
    ):
        raise ValueError("resume checkpoint has an invalid treatment name")
    _require_finite_tree(checkpoint["model"], "model parameters")
    _require_finite_tree(checkpoint["optimizer_state"], "optimizer state")
    stage_gate._validate_sampler_state(
        checkpoint["pooled_sampler_state"],
        receipt["sampling"],
        expected_seed,
        checkpoint_path,
    )
    return {
        **info,
        "schema": "terra_v8_continuation_checkpoint_v1",
        "passed": True,
        "optimizer_restorable": True,
        "sampler_restorable": True,
        "source_treatment_name": source_treatment_name,
        "candidate": candidate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--receipt", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--checkpoint-sha256", required=True)
    verify.add_argument(
        "--arm", choices=tuple(stage_gate.ARM_ARCHITECTURES), required=True
    )
    verify.add_argument("--seed", type=int, required=True)
    verify.add_argument("--terra-baselines-revision", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        _, result = inspect_receipt(args.receipt)
    else:
        result = verify_checkpoint(
            args.receipt.resolve(),
            args.checkpoint,
            args.checkpoint_sha256,
            args.arm,
            args.seed,
            args.terra_baselines_revision,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
