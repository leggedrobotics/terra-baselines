from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_gru_recurrence_probe_v1"


def test_submit_zero_prints_pinned_contract_without_ssh():
    environment = {
        **os.environ,
        "SUBMIT": "0",
        "REMOTE_HOST": "must-not-be-contacted.invalid",
    }
    completed = subprocess.run(
        ["bash", str(LAUNCHER / "submit.sh")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SUBMIT=0: contract printed; no SSH" in completed.stdout
    assert "checkpoint_sha256=0985b6338fb02f" in completed.stdout
    assert "runtime_terra_revision=25f855db3d913" in completed.stdout
    assert (
        "fixed_gru_result_sha256="
        "83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2"
        in completed.stdout
    )


def test_sbatch_is_one_gpu_eval_only_and_consumes_fixed_result():
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert "--gpus='rtx_4090:1'" in submit
    assert "--partition='gpuhe.4h'" in submit
    assert "--account='es_hutter'" in submit
    assert 'status --porcelain=v1 --untracked-files=all' in submit
    assert "lquota" in submit and "lquota" in sbatch
    assert "git -C \"$REPO\" archive" in submit
    assert submit.index('if [ "$SUBMIT" = stage ]') < submit.index('JOB_RAW=')
    assert 'remote "mkdir \'$CLAIM_DIR\'"' in submit
    assert "state=submitting" in submit
    assert "cleanup_failed_claim" not in submit
    assert "trap " not in submit
    assert (
        "RUNTIME_TERRA_REVISION=25f855db3d913fd638c4e56b1740437a2b7122ca"
        in submit
    )
    assert (
        "PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
        in submit
    )
    assert (
        "ENVIRONMENT_PROTOCOL_SHA="
        "9917b9238e9e6e844377e6d4a8ca18d1f0defbbacf887642743e579243109367"
        in submit
    )
    assert (
        "BANK_ARCHIVE_SHA="
        "b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725"
        in submit
    )
    assert (
        "BANK_DATASET_SHA="
        "5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851"
        in submit
    )
    assert (
        "PROMOTION_MANIFEST_SHA="
        "dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826"
        in submit
    )
    assert (
        "GRU_U44000_SHA="
        "0985b6338fb02f866b7aadbf065431cd667954a6f9b1a457e3eae9213533569d"
        in submit
    )
    assert "EVAL_SEED=20260807" in submit
    assert "__PENDING" not in submit
    assert (
        "FIXED_GRU_RESULT_SHA="
        "83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2"
        in submit
    )
    assert "TERRA_REMOTE_VENV" not in submit
    assert (
        "RUNTIME_FINGERPRINT_SHA="
        "73c80e3dd483e3202679844228b422f416bbf48b49d6ce35056f3afff91d9b7e"
        in submit
    )
    assert "--fixed-eval \"$FIXED_GRU_RESULT\"" in sbatch
    assert "scripts/gru_recurrence_probe_v1/run_probe.py" in sbatch
    assert "WANDB_MODE=disabled" in sbatch
    assert "EVAL_FORWARD_CHUNK == 120" in sbatch
    assert 'test "$EVAL_SEED" = 20260807' in sbatch
    assert 'assert receipt["horizon"] == 450' in sbatch
    assert 'assert receipt["seed"] == 20260807' in sbatch
    assert 'test -d "$CLAIM_DIR"' in sbatch
    assert 'ln "$OWNER_TMP" "$CLAIM_DIR/job_id"' in sbatch
    assert 'test "$(cat "$CLAIM_DIR/job_id")" = "$SLURM_JOB_ID"' in sbatch
    assert "runtime_fingerprint_sha256=$RUNTIME_FINGERPRINT_SHA" in sbatch
    assert sbatch.index('assert receipt["status"] == "passed"') < sbatch.index(
        '"status=PASSED"'
    )
    assert "train_mixed.py" not in sbatch
    assert "sbatch" not in sbatch
