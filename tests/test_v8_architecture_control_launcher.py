from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v8_architecture_control_v1.sh"
LAUNCHER = ROOT / "scripts" / "euler_v8_architecture_control_v1"


def arm_block(source: str, arm: str) -> str:
    return source.split(f"    {arm})", 1)[1].split("        ;;", 1)[0]


def test_architecture_pair_changes_only_the_explicit_model_block():
    runner = RUNNER.read_text()
    compact = arm_block(runner, "compact_xattn")
    atari = arm_block(runner, "atari_base")
    common = runner.split('mkdir -p "$RUN_ROOT/checkpoints"', 1)[1]

    assert "--model_size medium" in compact
    assert "--map_encoder resnet_spatial_8x8_se_xattn" in compact
    assert "--encoder_compute_dtype bfloat16" in compact
    assert "--attention_compute_dtype float32" in compact
    assert '--critic_hidden_dims "512,256"' in compact
    assert '--resnet_stage_channels "24,48,64,96"' in compact
    assert '--resnet_blocks_per_stage "2,2,3,3"' in compact

    assert "--model_size base" in atari
    assert "--map_encoder atari" in atari
    assert "--encoder_compute_dtype float32" in atari
    assert "--attention_compute_dtype encoder" in atari
    assert "--critic_hidden_dims" not in atari
    assert "--resnet_stage_channels" not in atari
    assert "--resnet_blocks_per_stage" not in atari

    for flag in (
        "--config G-V8-CONTINUOUS",
        "--accepted-bank-scope full",
        "--accepted-bank-sampler-profile continuous_banded_v1",
        "--reward_stage dense_skill",
        '--seed "$SEED"',
        '--num_envs_per_device "$NUM_ENVS_PER_DEVICE"',
        '--num_steps "$NUM_STEPS"',
        "--update_epochs 2",
        '--num_minibatches "$NUM_MINIBATCHES"',
        "--lr 3e-4",
        "--ent_schedule_start 0.15",
        "--ent_schedule_end 0.02",
        "--no_value_clip",
        "--flat_minibatch_shuffle",
    ):
        assert flag in common
    assert "--teacher_checkpoint" not in runner
    assert "--warm_start_from" not in runner


def test_submit_uses_paired_smoke_admission_and_requested_queues():
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert "ARMS=(compact_xattn atari_base)" in submit
    assert "SEED=20260807" in submit
    assert "gpuhe.4h" in submit and "GPU_TYPE=rtx_3090" in submit
    assert "gpuhe.120h" in submit and "119:45:00" in submit
    assert "gpuhe.24h" in submit and "23:45:00" in submit
    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index('ssh "$REMOTE_HOST"')
    assert submit.index("architecture_smoke_validation.json") < submit.index(
        'JOB_ID="$(ssh'
    )
    assert "common_training_contract" in submit
    assert "common_training_contract\\\"] == b[\\\"common_training_contract" in submit

    assert "UPDATES=20000" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert "EVAL_INTERVAL=1000" in sbatch
    assert "condition_count=47" in sbatch
    assert "maps_per_condition=96" in sbatch
    assert "sampler_profile=continuous_banded_v1" in sbatch
    assert "reward_stage=dense_skill" in sbatch
    assert "horizon=450" in sbatch
    assert "initialization=random_no_teacher" in sbatch
    assert "model_parameter_count=$EXPECTED_PARAMETERS" in sbatch
    assert '"$RUNTIME_CHECK" --min-devices 4' in sbatch
    assert "P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa" in sbatch
    assert "--accepted-panel" in sbatch and "--capability-panel" in sbatch
    assert "--require-productive-workspace-cycles" in sbatch


def test_smoke_verifier_pins_both_architectures_and_common_ppo_shape():
    verifier = (LAUNCHER / "verify_smoke.py").read_text()
    assert '"compact_xattn"' in verifier
    assert '"parameter_count": 2_856_685' in verifier
    assert '"atari_base"' in verifier
    assert '"parameter_count": 480_137' in verifier
    assert '"map_encoder": "atari"' in verifier
    assert '"critic_hidden_dims": None' in verifier
    assert '"num_envs_per_device": 512' in verifier
    assert '"num_steps": 32' in verifier
    assert '"num_minibatches": 32' in verifier
    assert '"reward_stage": "dense_skill"' in verifier
    assert "require_finite_tree(checkpoint[\"optimizer_state\"]" in verifier
    assert '"schema": "terra_v8_architecture_smoke_v1"' in verifier
