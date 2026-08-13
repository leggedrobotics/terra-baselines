"""Contract for the one supported v6.1 stall-age/v3 continuation."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v8_v6_yolo_rv2.sh"
SBATCH = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "run.sbatch"
SUBMIT = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "submit.sh"
VERIFY = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "verify_resume.py"

SOURCE_SHA = "79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609"
PREPARED_SHA = "68aea1a0f5dc3c05d11319fdf640ade05495125225533bc99ad92592475fcb75"
TERRA_REVISION = "c2d2a94a124759e9f21c2b37930f717e299f0c46"


def test_submit_exposes_only_the_direct_phase2_recipe():
    submit = SUBMIT.read_text()
    assert "usage: submit.sh phase2" in submit
    assert "PHASE=phase2" in submit
    assert "smoke|phase1" not in submit
    assert "ARM_NAME=v6_1_rv2_stall_age_v3" in submit
    assert "SAMPLER_PROFILE=continuous_banded_v3" in submit
    assert f"EXPECTED_RUNTIME_TERRA_REVISION={TERRA_REVISION}" in submit
    assert f"RESUME_SOURCE_SHA={SOURCE_SHA}" in submit

    assert "PARTITION=gpuhe.24h" in submit
    assert "WALLTIME=23:45:00" in submit
    assert "GPU_TYPE=rtx_4090" in submit
    assert "GPU_COUNT=8" in submit
    assert "CPUS=8" in submit
    assert "--account='es_hutter'" in submit
    assert "--job-name='terra-v61-stall-v3'" in submit


def test_prepared_checkpoint_is_a_native_v3_resume():
    submit = SUBMIT.read_text()
    sbatch = SBATCH.read_text()
    runner = RUNNER.read_text()
    verify = VERIFY.read_text()

    prepared_sha = re.search(r"^RESUME_PREPARED_SHA=([0-9a-f]{64})$", submit, re.M)
    assert prepared_sha is not None and prepared_sha.group(1) == PREPARED_SHA
    assert "v8_v61_stall_age_v3_u14000_prepared.pkl" in submit
    assert 'test "${PHASE:?}" = phase2' in sbatch
    assert 'test "${ARM_NAME:?}" = v6_1_rv2_stall_age_v3' in sbatch
    assert 'test "${SAMPLER_PROFILE:?}" = continuous_banded_v3' in sbatch
    assert "SOURCE_UPDATE=14000" in sbatch
    assert "TARGET_UPDATE=40000" in sbatch
    assert "NUM_DEVICES=8" in sbatch
    assert "NUM_ENVS_PER_DEVICE=256" in sbatch
    assert "resume_global_batch=65536_to_65536" in sbatch

    assert '--prepared "$RESUME_CHECKPOINT"' in sbatch
    assert '--prepared-sha256 "$RESUME_CHECKPOINT_SHA"' in sbatch
    assert "restored_sampler_source_rule=continuous_banded_v2" in sbatch
    assert "restored_sampler_rule=continuous_banded_v3" in sbatch
    assert "restored_sampler_migration=materialized_before_resume" in sbatch
    assert "restored_sampler_partial_window_discarded_updates=50" in sbatch
    assert "restored_sampler_partial_window_updates=0" in sbatch
    assert "restored_sampler_open_mass=0.8" in sbatch
    assert "restored_sampler_mastered_replay_mass=0.2" in sbatch

    assert '--resume_from "$RESUME_CHECKPOINT"' in runner
    assert 'test "$RESUME_CHECKPOINT" != none' in runner
    assert "TRAIN_PRESET=G-V8-CONTINUOUS-V3" in runner
    assert "SAMPLER_PROFILE=continuous_banded_v3" in runner
    assert "NUM_DEVICES=8" in runner
    assert "NUM_ENVS_PER_DEVICE=256" in runner
    assert "NUM_STEPS=32" in runner
    assert "NUM_MINIBATCHES=32" in runner
    assert "BLOCKS_PER_STAGE=2,2,3,3" in runner
    assert "AUX_COEF=0" in runner
    assert "VF_COEF=2.0" in runner
    assert "--stall_age_observation" in runner
    assert "--action_logit_masking" not in runner
    assert "${NUM_DEVICES:-" not in runner
    assert "${NUM_ENVS_PER_DEVICE:-" not in runner
    assert "${BLOCKS_PER_STAGE:-" not in runner
    assert "${AUX_COEF:-" not in runner
    assert "${VF_COEF-" not in runner
    assert "${SAMPLER_PROFILE:-" not in runner
    assert '"source_sampler_rule": "continuous_banded_v2"' in verify
    assert '"sampler_rule": "continuous_banded_v3"' in verify
    assert '"sampler_migration": "materialized_before_resume"' in verify


def test_training_contract_changes_only_the_declared_practical_bundle():
    sbatch = SBATCH.read_text()
    assert "STALL_AGE_OBSERVATION=1" in sbatch
    assert "time_remaining_observation=false" in sbatch
    assert 'test "${ACTION_LOGIT_MASKING:?}" = 0' in sbatch
    assert 'action_logit_masking=$([ "$ACTION_LOGIT_MASKING" = 1 ]' in sbatch
    assert (
        "material_stall_age_scalar+continuous_banded_v3_open80_mastered20_cap015"
        in sbatch
    )
    assert "practical_combined_observation_and_curriculum_continuation" in sbatch

    for frozen in (
        "reward_stage=reward_v2",
        "reward_protocol_id=material_potential_v2",
        "reward_v2_step_cost_total=1.0",
        "learning_rate=0.0003",
        "map_encoder=resnet_spatial_8x8_se_sa_xattn",
        "resnet_blocks_per_stage=$BLOCKS_PER_STAGE",
        "num_steps=32",
        "num_minibatches=32",
        "update_epochs=2",
        "model_parameter_count=$EXPECTED_PARAMETERS",
    ):
        assert frozen in sbatch
    assert "verify_smoke.py" not in sbatch
