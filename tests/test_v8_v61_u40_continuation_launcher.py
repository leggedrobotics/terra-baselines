from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_v61_u40_continuation"


def test_u40_continuation_keeps_one_exact_resumable_training_path():
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()
    runner = (ROOT / "scripts" / "run_v8_v6_yolo_rv2.sh").read_text()

    assert "dddc691c93ee21488cd7eeb8e01b067bf1f9733c" in submit
    assert "c2d2a94a124759e9f21c2b37930f717e299f0c46" in submit
    assert "17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8" in submit
    assert "PARENT_WANDB_MAX_UPDATE=39991" in submit
    assert "TARGET_UPDATE=70000" in submit
    assert "git -C \"$REPO\" archive --format=tar \"$TRAINING_REVISION\"" in submit

    assert "#SBATCH --partition=gpuhe.24h" in sbatch
    assert "#SBATCH --gpus=rtx_4090:8" in sbatch
    assert "SOURCE_UPDATE=40000" in sbatch
    assert "TARGET_UPDATE=70000" in sbatch
    assert "NUM_ENVS_PER_DEVICE=256" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert "export WANDB_RESUME=must" in sbatch
    assert 'export WANDB_RUN_ID="$PARENT_WANDB_RUN_ID"' in sbatch
    assert 'run.state != "finished"' in sbatch
    assert "observed > 40_000" in sbatch

    assert "--stall_age_observation" in runner
    assert "--accepted-bank-sampler-profile \"$SAMPLER_PROFILE\"" in runner
    assert "--reward_stage reward_v2" in runner
    assert "--checkpoint_interval \"$CHECKPOINT_INTERVAL\"" in runner
    assert '--resume_from "$RESUME_CHECKPOINT"' in runner
    assert "--no-load-env-from-checkpoint" in runner
    assert "--enable_action_mask" not in runner
    assert "--warm_start_from" not in runner
