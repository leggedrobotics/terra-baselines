from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_deep_xattn_v1"


def test_v8_launcher_freezes_the_two_arm_warm_start_contract():
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()
    runner = (ROOT / "scripts" / "run_v8_warm_screen.sh").read_text()

    assert "ARMS=(G-DEEP-V8-DENSE-WARM G-DEEP-XATTN-V8-DENSE-WARM)" in submit
    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index('ssh "$REMOTE_HOST"')
    assert "terra_v8_v6_constraints_v7_adjacent_train96_v5" in submit
    assert "dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b" in submit
    assert "4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc" in submit
    assert "smoke_validation.json" in submit
    assert "gpuhe.4h" in submit and "gpuhe.24h" in submit
    assert "nearby requires both Stage-A gate receipts" in submit
    assert "full requires both Stage-B gate receipts" in submit
    assert "full) SCREEN_UPDATES=8000" in submit
    assert "stage '$STAGE' is not enabled by this launcher revision" in submit
    assert "stage_gate.py" in submit
    assert "check-smoke" in submit
    assert "--parent-sha256 '${PARENT_SHAS[$ARM]}'" in submit
    assert "--prior-gate-sha256 '${GATE_SHAS[$ARM]}'" in submit
    assert "PARENTS[$ARM]" in submit
    assert "GATE_SHAS[$ARM]" in submit
    assert "submit_tail.sh" in submit
    assert "afterany:PARENT_JOB_ID" in submit

    assert "#SBATCH --gpus=rtx_4090:4" in sbatch
    assert "EXPECTED_CONDITIONS=2" in sbatch
    assert "EXPECTED_CONDITIONS=15" in sbatch
    assert "EXPECTED_CONDITIONS=47" in sbatch
    assert "PRIOR_STAGE=nearby" in sbatch
    assert 'test "${#GPU_NAMES[@]}" -eq 4' in sbatch
    assert "scripts/grow_checkpoint.py" in sbatch
    assert "resnet_spatial_8x8_se_xattn" in sbatch
    assert 'INITIAL="$PARENT_CHECKPOINT"' in sbatch
    assert '"teacher_checkpoint_sha256=$TEACHER_SHA"' in sbatch
    assert '"initialization=$INITIALIZATION"' in sbatch
    assert "params_only_stage_transition_fresh_optimizer" in sbatch
    assert "capability_promotion.json" in sbatch
    assert "stage_gate.json" in sbatch
    assert "--prior-receipt" in sbatch
    assert '"reward_type=DENSE"' in sbatch
    assert '"trench_shaping=false"' in sbatch
    assert "--capability-panel" in sbatch
    assert "--expect-completion-contract exact_visible_dump_v1" in sbatch
    assert "status=PASSED" in sbatch
    assert "V8_SCREEN_FULL_${ARM}_TRAINING_FINISHED" in sbatch

    assert "--config G-V8-FIXED" in runner
    assert '--accepted-bank-stage "$STAGE"' in runner
    assert "--kickstart_kl_anneal_updates 1500" in runner
    assert "--kickstart_value_anneal_updates 500" in runner
    assert 'test "$TEACHER_CHECKPOINT" = none' in runner
    assert '--resnet_blocks_per_stage "2,2,3,3"' in runner
    assert "--no_value_clip" in runner
    assert "--flat_minibatch_shuffle" in runner
    assert "sparse" not in runner.lower()
