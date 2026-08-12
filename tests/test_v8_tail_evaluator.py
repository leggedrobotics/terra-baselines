import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.euler_v8_deep_xattn_v1 import stage_gate, tail_prefix

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_deep_xattn_v1"
ARM = "G-DEEP-V8-DENSE-WARM"
REVISION = "a" * 40


def make_parent_run(tmp_path: Path, updates=(500, 1000, 1500)) -> Path:
    run_dir = tmp_path / REVISION / "screen" / "full" / "s20260730" / ARM
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for update in updates:
        (checkpoints / f"v8-test-euler-2026-08-03_update_{update:06d}.pkl").write_bytes(
            f"checkpoint-{update}".encode()
        )
    contract = {
        "arm": ARM,
        "curriculum_stage": "full",
        "phase": "screen",
        "condition_count": "47",
        "seed": "20260730",
        "updates": "8000",
        "terra_baselines_revision": REVISION,
        "training_bank_release_id": tail_prefix.RELEASE_ID,
        "training_bank_archive_sha256": tail_prefix.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": tail_prefix.BANK_DATASET_SHA256,
        "slurm_job_id": "12345",
        "reward_type": "DENSE",
        "horizon": "450",
        "full_resets": "true",
    }
    (run_dir / "run_contract.env").write_text(
        "\n".join(f"{key}={value}" for key, value in contract.items()) + "\n"
    )
    return run_dir


def test_tail_receipt_freezes_a_timeout_checkpoint_prefix(tmp_path):
    run_dir = make_parent_run(tmp_path)
    receipt = tail_prefix.build_parent_receipt(
        run_dir=run_dir,
        run_contract_path=run_dir / "run_contract.env",
        parent_job_id="12345",
        parent_state="TIMEOUT",
        parent_exit_code="0:0",
        parent_partition="gpuhe.24h",
        evaluator_job_id="12346",
        arm=ARM,
        seed=20260730,
        baselines_revision=REVISION,
    )

    assert receipt["schema"] == "terra_v8_parent_slurm_job_v1"
    assert receipt["checkpoint_updates"] == [500, 1000, 1500]
    assert [entry["update"] for entry in receipt["checkpoints"]] == [500, 1000, 1500]
    assert all(len(entry["sha256"]) == 64 for entry in receipt["checkpoints"])
    assert receipt["run_contract"]["path"] == str(
        (run_dir / "run_contract.env").resolve()
    )
    receipt_path = tmp_path / "parent_job.json"
    receipt_path.write_text(json.dumps(receipt))
    with patch.object(stage_gate, "REMOTE_RUN_ROOT", tmp_path.resolve()):
        validated = stage_gate.validate_parent_job_receipt(
            receipt_path,
            run_dir / "run_contract.env",
            tail_prefix.parse_run_contract(run_dir / "run_contract.env"),
            [500, 1000, 1500],
        )
    assert validated["job_id"] == "12345"

    receipt["run_contract"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with patch.object(stage_gate, "REMOTE_RUN_ROOT", tmp_path.resolve()):
        with pytest.raises(ValueError, match="run-contract identity changed"):
            stage_gate.validate_parent_job_receipt(
                receipt_path,
                run_dir / "run_contract.env",
                tail_prefix.parse_run_contract(run_dir / "run_contract.env"),
                [500, 1000, 1500],
            )


def test_checkpoint_discovery_rejects_gaps_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="gap at update 1000"):
        tail_prefix.discover_checkpoint_prefix(
            make_parent_run(tmp_path / "gap", updates=(500, 1500)) / "checkpoints"
        )

    duplicate_dir = make_parent_run(tmp_path / "duplicate", updates=(500, 1000))
    (duplicate_dir / "checkpoints" / "other_update_001000.pkl").write_bytes(b"other")
    with pytest.raises(ValueError, match="duplicate checkpoint for update 1000"):
        tail_prefix.discover_checkpoint_prefix(duplicate_dir / "checkpoints")


def test_tail_launcher_is_afterany_and_evaluates_all_four_frozen_panels():
    submit = (LAUNCHER / "submit_tail.sh").read_text()
    sbatch = (LAUNCHER / "evaluate_tail.sbatch").read_text()

    assert "dependency='afterany:$PARENT_JOB_ID'" in submit
    assert "--partition=gpuhe.4h" in submit
    assert "#SBATCH --partition=gpuhe.4h" in sbatch
    assert 'case "$PARENT_STATE" in COMPLETED|TIMEOUT' in sbatch
    assert '--accepted-panel "$PANEL"' in sbatch
    assert '--capability-panel "$PANEL"' in sbatch
    assert '--development "$TAIL_DIR/eval/development.json"' in sbatch
    assert (
        '--capability-development "$TAIL_DIR/eval/capability_development.json"'
        in sbatch
    )
    assert '--parent-job-receipt "$PARENT_RECEIPT"' in sbatch
