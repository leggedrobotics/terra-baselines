from pathlib import Path

from configs.training_configs import get_config


ROOT = Path(__file__).resolve().parents[1]


def test_p5b_presets_form_one_factor_star():
    medium = get_config("G-MEDIUM-ADAPTIVE-WARM")
    deep = get_config("G-DEEP-ADAPTIVE-WARM")
    uniform = get_config("G-MEDIUM-UNIFORM-WARM")
    adaptive = get_config("G-ADAPTIVE")
    uniform_reference = get_config("G-UNIFORM")

    for preset in (medium, deep):
        assert preset.accepted_bank_arm == "G-ADAPTIVE"
        assert preset.maps == []
        assert preset.agent_types == (0,)
        assert preset.action_types == (0,)
        assert preset.relocation_progress_mult == 1.5
        assert preset.curriculum == adaptive.curriculum
        assert preset.pooled_sampler == adaptive.pooled_sampler
    assert uniform.accepted_bank_arm == "G-UNIFORM"
    assert uniform.maps == []
    assert uniform.agent_types == medium.agent_types
    assert uniform.action_types == medium.action_types
    assert uniform.relocation_progress_mult == medium.relocation_progress_mult
    assert uniform.curriculum == medium.curriculum
    assert uniform.pooled_sampler == uniform_reference.pooled_sampler

    adaptive_sampler = vars(medium.pooled_sampler).copy()
    uniform_sampler = vars(uniform.pooled_sampler).copy()
    assert adaptive_sampler.pop("rule") == "adaptive"
    assert uniform_sampler.pop("rule") == "uniform"
    assert adaptive_sampler == uniform_sampler


def test_direct_runner_has_one_explicit_architecture_difference():
    script = (ROOT / "scripts/run_p5b_warm_screen.sh").read_text()
    assert 'if [ "$#" -ne 6 ]' in script
    assert "G-MEDIUM-ADAPTIVE-WARM" in script
    assert "G-DEEP-ADAPTIVE-WARM" in script
    assert "G-MEDIUM-UNIFORM-WARM" in script
    assert '--resnet_stage_channels "24,48,64,96"' in script
    assert '--resnet_blocks_per_stage "2,2,3,3"' in script
    assert '--warm_start_from "$INITIAL_CHECKPOINT"' in script
    assert '--teacher_checkpoint "$TEACHER_CHECKPOINT"' in script
    assert "--kickstart_kl_anneal_updates 1500" in script
    assert "--kickstart_value_anneal_updates 500" in script
    assert "--num_minibatches 32" in script
    assert "--no_value_clip" in script
    assert "--flat_minibatch_shuffle" in script
    assert '"$@"' not in script


def test_euler_launcher_is_three_arm_star_and_keeps_screen_contract():
    submit = (ROOT / "scripts/euler_p5b_warm_v1/submit.sh").read_text()
    sbatch = (ROOT / "scripts/euler_p5b_warm_v1/run.sbatch").read_text()

    assert (
        "ARMS=(G-MEDIUM-ADAPTIVE-WARM G-DEEP-ADAPTIVE-WARM "
        "G-MEDIUM-UNIFORM-WARM)"
    ) in submit
    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index(
        'ssh "$REMOTE_HOST"'
    )
    assert "PARENT_SHA=" in submit
    assert "DEEP_SHA=" in submit
    assert (
        "TERRA_REPO=\"/home/lorenzo/moleworks/.worktrees/"
        "terra_simple_mapbank_reward_20260730\""
    ) in submit
    assert "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4" in submit
    assert "git -C \"$REPO\" archive" in submit
    assert "gpuhe.4h" in submit
    assert "gpuhe.24h" in submit
    assert "smoke_validation.json" in submit
    assert 'assert p[\\"passed\\"] is True' in submit
    assert "terra_p5b_warm_euler_campaign" not in submit
    assert "receipt" not in submit.lower()

    assert "#SBATCH --gpus=rtx_4090:4" in sbatch
    assert "home quota launch gate failed" in sbatch
    assert 'test "$MAPS_PER_CONDITION" = 64' in sbatch
    assert "ACCEPTED_BANK_ARM=G-ADAPTIVE" in sbatch
    assert "ACCEPTED_BANK_ARM=G-UNIFORM" in sbatch
    assert "for UPDATE in 500 1000 1500 2000" in sbatch
    assert "--accepted-panel \"$PANEL\"" in sbatch
    assert "--expect-completion-contract exact_visible_dump_v1" in sbatch
    assert "receipt" not in sbatch.lower()
    assert '"primary_family=generalist"' in sbatch
    assert '"training_support=all"' in sbatch
    assert '"condition_sampler=$CONDITION_SAMPLER"' in sbatch
    assert '"initialization=params_only_warm"' in sbatch
    assert "medium:resnet_spatial_8x8_se:mlp:critic-512-256" in sbatch
    assert (
        "medium:resnet_spatial_8x8_se:mlp:critic-512-256:"
        "channels-24x48x64x96:blocks-2x2x3x3"
    ) in sbatch
    assert '"accepted_bank_arm=$ACCEPTED_BANK_ARM"' in sbatch
    assert '"parent_checkpoint_sha256=$PARENT_SHA"' in sbatch
    assert '"initial_checkpoint_sha256=$INITIAL_SHA"' in sbatch
    assert '"initialization=params_only_warm"' in sbatch
    assert '"global_transitions=$GLOBAL_TRANSITIONS"' in sbatch
    assert '"final_checkpoint_update=$UPDATES"' in sbatch
    assert '"$RUN_DIR/run_contract.env"' in sbatch
    assert sbatch.index('> "$RUN_DIR/run_contract.env"') < sbatch.index(
        '\n"${TRAIN_COMMAND[@]}"\n'
    )
    assert '"status=PASSED" >> "$RUN_DIR/run_contract.env"' in sbatch
    assert "/cluster/scratch/lterenzi/codex_terra_edge_runs/p5b_warm_v1" in sbatch
