from pathlib import Path

import pytest

from scripts.euler_v8_deep_xattn_v1 import continuation_contract
from train_mixed import resolve_run_name

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_deep_xattn_v1"
ARM = "G-DEEP-V8-DENSE-WARM"


def test_v8_continuation_is_one_unpaired_true_resume_to_an_absolute_target():
    runner = (ROOT / "scripts" / "run_v8_resume.sh").read_text()
    sbatch = (LAUNCHER / "continue.sbatch").read_text()
    submit = (LAUNCHER / "submit_continuation.sh").read_text()
    contract = (LAUNCHER / "continuation_contract.py").read_text()

    assert "ABSOLUTE_UPDATES=80000" in runner
    assert "--accepted-bank-stage full" in runner
    assert "--exact_run_name" in runner
    assert '--resume_from "$RESUME_CHECKPOINT"' in runner
    assert "--warm_start_from" not in runner
    assert "--teacher_checkpoint" not in runner
    assert "--load_env_from_checkpoint" in runner
    assert '--checkpoint_interval "$CHECKPOINT_INTERVAL"' in runner
    assert '--resnet_blocks_per_stage "2,2,3,3"' in runner
    assert "--num_minibatches 32" in runner
    assert "--no_value_clip" in runner
    assert "--flat_minibatch_shuffle" in runner

    assert "#SBATCH --partition=gpuhe.120h" in sbatch
    assert "#SBATCH --time=119:45:00" in sbatch
    assert "#SBATCH --gpus=rtx_4090:4" in sbatch
    assert 'continuation_contract.py" verify' in sbatch
    assert "WANDB_RESUME=never" in sbatch
    assert "parent_wandb_run_id=$PARENT_WANDB_RUN_ID" in sbatch
    assert "unpaired_single_qualifying_arm" in sbatch
    assert "matched_architecture_pair" in sbatch
    assert '"pairing=$PAIRING"' in sbatch
    assert "initialization=true_resume_optimizer_schedule_sampler" in sbatch
    assert "bit_exact_continuation=false" in sbatch
    assert "restarted_state=environment_rng_action_history" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert 'RUN_NAME="$SOURCE_TREATMENT_NAME"' in sbatch
    assert "source_treatment_name=$SOURCE_TREATMENT_NAME" in sbatch

    assert 'if [ "$#" -ne 1 ]' in submit
    assert "qualified receipt passed" in submit.lower()
    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index('ssh "$REMOTE_HOST"')
    assert "\"cat '$CHECKPOINT'\" | sha256sum" in submit
    assert "sha256sum '$CHECKPOINT'" in submit
    assert "--partition=gpuhe.120h" in submit
    assert "--time=119:45:00" in submit
    assert "--dependency='afterany:$JOB_ID'" in submit
    assert "continuation_tail.sbatch" in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert "TAIL_EVALUATOR_SBATCH" not in submit
    assert 'PAIRING="${PAIRING:-unpaired_single_qualifying_arm}"' in submit
    assert "matched_architecture_pair" in submit

    assert 'qualification.get("qualified_for_120h") is not True' in contract
    assert '"optimizer_state"' in contract
    assert '"pooled_sampler_state"' in contract
    assert "_require_finite_tree" in contract
    assert '"source_treatment_name"' in contract


def test_exact_run_name_preserves_the_source_treatment_identity():
    source = "v8_source-euler-2026-08-03-12-00-00"
    assert resolve_run_name(source, "euler", "2026-08-08-00-00-00", True) == source
    assert resolve_run_name("v8_source", "euler", "timestamp", False) == (
        "v8_source-euler-timestamp"
    )


def test_continuation_inspection_rejects_a_merely_passing_full_receipt(
    tmp_path, monkeypatch
):
    receipt_path = tmp_path / "full.json"
    receipt_path.write_text('{"arm": "G-DEEP-V8-DENSE-WARM"}\n')
    merely_passing = {
        "arm": ARM,
        "continuation_qualification": {"qualified_for_120h": False},
    }
    monkeypatch.setattr(
        continuation_contract.stage_gate,
        "validate_prior_receipt",
        lambda path, arm, stage: merely_passing,
    )

    with pytest.raises(ValueError, match="not qualified for 120h"):
        continuation_contract.inspect_receipt(receipt_path)
