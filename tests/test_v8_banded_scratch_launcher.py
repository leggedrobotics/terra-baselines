from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_banded_scratch_v1"


def test_scratch_stage_a_is_random_dense_and_cannot_skip_its_smoke():
    runner = (ROOT / "scripts" / "run_v8_scratch_capability.sh").read_text()
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert "--accepted-bank-stage capability" in runner
    assert "--accepted-bank-sampler-profile bank_v4" in runner
    assert "--warm_start_from" not in runner
    assert "--teacher_checkpoint" not in runner
    assert "--ent_schedule_start 0.15" in runner
    assert "--ent_schedule_end 0.02" in runner
    assert "sparse" not in runner.lower()

    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index(
        'ssh "$REMOTE_HOST"'
    )
    assert "SMOKE_REVISION" in submit
    assert "smoke_validation.json" in submit
    assert "gpuhe.4h" in submit and "gpuhe.24h" in submit

    assert "UPDATES=6000" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert "initialization=random_no_teacher" in sbatch
    assert "reward_type=DENSE" in sbatch
    assert "--capability-panel" in sbatch
    assert "--expect-completion-contract exact_visible_dump_v1" in sbatch
    assert 'test "${#GPU_NAMES[@]}" -eq 4' in sbatch
    assert 'used < 45.0' in sbatch
