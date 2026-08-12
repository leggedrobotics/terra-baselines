#!/usr/bin/env python3
"""Freeze the checkpoint prefix and parent Slurm receipt for a V8 tail eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "terra_v8_parent_slurm_job_v1"
RELEASE_ID = "terra_v8_v6_constraints_v7_adjacent_train96_v5"
BANK_ARCHIVE_SHA256 = "dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b"
BANK_DATASET_SHA256 = "715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798"
CHECKPOINT_RE = re.compile(r".+_update_(\d{6})\.pkl")
EXPECTED_UPDATES = tuple(range(500, 8001, 500))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate key {key!r}")
        values[key] = value
    return values


def discover_checkpoint_prefix(checkpoints_dir: Path) -> list[dict]:
    by_update: dict[int, Path] = {}
    for path in sorted(checkpoints_dir.glob("*_update_*.pkl")):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unsupported periodic checkpoint name: {path.name}")
        update = int(match.group(1))
        if update not in EXPECTED_UPDATES:
            raise ValueError(
                f"checkpoint update {update} is outside the full-stage schedule"
            )
        if update in by_update:
            raise ValueError(f"duplicate checkpoint for update {update}")
        by_update[update] = path.resolve()

    observed = sorted(by_update)
    prefix = []
    for update in EXPECTED_UPDATES:
        if update not in by_update:
            if any(later > update for later in observed):
                raise ValueError(f"checkpoint gap at update {update}")
            break
        prefix.append(update)
    if len(prefix) < 2:
        raise ValueError("tail evaluation requires at least checkpoints 500 and 1000")
    return [
        {
            "update": update,
            "path": str(by_update[update]),
            "sha256": sha256_file(by_update[update]),
        }
        for update in prefix
    ]


def build_parent_receipt(
    *,
    run_dir: Path,
    run_contract_path: Path,
    parent_job_id: str,
    parent_state: str,
    parent_exit_code: str,
    parent_partition: str,
    evaluator_job_id: str,
    arm: str,
    seed: int,
    baselines_revision: str,
) -> dict:
    if re.fullmatch(r"[0-9]+", parent_job_id) is None:
        raise ValueError("parent job_id must contain only digits")
    if re.fullmatch(r"[0-9]+", evaluator_job_id) is None:
        raise ValueError("evaluator job_id must contain only digits")
    if parent_state not in {"COMPLETED", "TIMEOUT"}:
        raise ValueError(f"parent job state {parent_state!r} is not tail-evaluable")
    if re.fullmatch(r"[0-9]+:[0-9]+", parent_exit_code) is None:
        raise ValueError("parent exit_code must use Slurm status:signal syntax")
    if parent_state == "COMPLETED" and parent_exit_code != "0:0":
        raise ValueError("a COMPLETED parent must have exit_code 0:0")
    if parent_partition != "gpuhe.24h":
        raise ValueError("full-stage parent must run on gpuhe.24h")
    if re.fullmatch(r"[0-9a-f]{40}", baselines_revision) is None:
        raise ValueError("terra-baselines revision must be a full Git SHA")

    run_dir = run_dir.resolve()
    run_contract_path = run_contract_path.resolve()
    expected_contract_path = run_dir / "run_contract.env"
    if run_contract_path != expected_contract_path:
        raise ValueError("run contract must be the parent run's run_contract.env")
    contract = parse_run_contract(run_contract_path)
    expected = {
        "arm": arm,
        "curriculum_stage": "full",
        "phase": "screen",
        "condition_count": "47",
        "seed": str(seed),
        "updates": "8000",
        "terra_baselines_revision": baselines_revision,
        "training_bank_release_id": RELEASE_ID,
        "training_bank_archive_sha256": BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": BANK_DATASET_SHA256,
        "slurm_job_id": parent_job_id,
        "reward_type": "DENSE",
        "horizon": "450",
        "full_resets": "true",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"run contract {field} must be {value!r}, got {contract.get(field)!r}"
            )

    checkpoints = discover_checkpoint_prefix(run_dir / "checkpoints")
    return {
        "schema": SCHEMA,
        "job_id": parent_job_id,
        "state": parent_state,
        "exit_code": parent_exit_code,
        "partition": parent_partition,
        "run_dir": str(run_dir),
        "checkpoint_updates": [entry["update"] for entry in checkpoints],
        "checkpoints": checkpoints,
        "run_contract": {
            "path": str(run_contract_path),
            "sha256": sha256_file(run_contract_path),
        },
        "terra_baselines_revision": baselines_revision,
        "evaluator_job_id": evaluator_job_id,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--parent-state", required=True)
    parser.add_argument("--parent-exit-code", required=True)
    parser.add_argument("--parent-partition", required=True)
    parser.add_argument("--evaluator-job-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baselines-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_parent_receipt(
        run_dir=args.run_dir,
        run_contract_path=args.run_contract,
        parent_job_id=args.parent_job_id,
        parent_state=args.parent_state,
        parent_exit_code=args.parent_exit_code,
        parent_partition=args.parent_partition,
        evaluator_job_id=args.evaluator_job_id,
        arm=args.arm,
        seed=args.seed,
        baselines_revision=args.baselines_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for checkpoint in receipt["checkpoints"]:
        print(checkpoint["path"])


if __name__ == "__main__":
    main()
