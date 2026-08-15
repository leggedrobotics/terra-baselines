from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v8_relay_partial_v1.sh"
SBATCH = ROOT / "scripts" / "euler_v8_relay_partial_v1" / "run.sbatch"
SUBMIT = ROOT / "scripts" / "euler_v8_relay_partial_v1" / "submit.sh"


def test_runner_is_one_fresh_relay_recipe():
    text = RUNNER.read_text()
    for token in (
        "--partial-reset-root",
        "--reward-v2-reset-context-observation",
        "--accepted-bank-sampler-profile continuous_banded_v3",
        "--reward_stage reward_v2",
        "--reward_v2_timing_variant 0",
        "--carry_work_observation",
        "--num_devices \"$NUM_DEVICES\"",
        "--num_envs_per_device \"$NUM_ENVS_PER_DEVICE\"",
        "--num_steps \"$NUM_STEPS\"",
        "--num_minibatches \"$NUM_MINIBATCHES\"",
        "--resnet_blocks_per_stage 2,2,3,3",
    ):
        assert token in text
    assert "NUM_DEVICES=8" in text
    assert "NUM_ENVS_PER_DEVICE=256" in text
    assert "NUM_STEPS=32" in text
    assert "NUM_MINIBATCHES=32" in text
    assert "--stall_age_observation" not in text
    assert "--action_logit_masking" not in text
    assert "--warm_start_from" not in text


def test_sbatch_is_two_native_segments_of_one_200k_run():
    text = SBATCH.read_text()
    assert "#SBATCH --partition=gpuhe.120h" in text
    assert "#SBATCH --time=119:45:00" in text
    assert "#SBATCH --gpus=rtx_4090:8" in text
    assert "scratch:100000:none" in text
    assert "resume:200000:" in text
    assert "EXPECTED_PARAMETERS=2306237" in text
    assert "continuous_banded_v3" in text
    assert "partial_reset_bank_sha256" in text
    assert "WANDB_RESUME=never" in text
    assert "WANDB_RESUME=must" in text


def test_submit_stages_one_pair_and_submits_an_afterok_chain():
    text = SUBMIT.read_text()
    assert "SUBMIT must be 0, stage, or 1" in text
    assert "PARTITION=gpuhe.120h" in text
    assert "WALLTIME=119:45:00" in text
    assert "GPU_TYPE=rtx_4090" in text
    assert "GPU_COUNT=8" in text
    assert "TARGET_UPDATE=100000" in text
    assert "TARGET_UPDATE=200000" in text
    assert "--dependency=afterok:$JOB1_ID" in text
    assert "--kill-on-invalid-dep=yes" in text
    assert "RESUME_CHECKPOINT=$REMOTE_PHASE1_CHECKPOINT" in text
