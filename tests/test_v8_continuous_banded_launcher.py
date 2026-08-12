from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_continuous_banded_v1"


def test_continuous_all47_run_is_scratch_dense_resumable_and_commit_gated():
    runner = (ROOT / "scripts" / "run_v8_continuous_banded_v1.sh").read_text()
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert not (ROOT / "scripts" / "euler_v8_banded_scratch_v1" / "submit.sh").exists()
    assert not (ROOT / "scripts" / "run_v8_scratch_capability.sh").exists()

    assert "--accepted-bank-scope full" in runner
    assert "--config G-V8-CONTINUOUS" in runner
    assert "--accepted-bank-stage" not in runner
    assert "--accepted-bank-sampler-profile continuous_banded_v1" in runner
    assert "--warm_start_from" not in runner
    assert "--teacher_checkpoint" not in runner
    assert '--resume_from "$RESUME_CHECKPOINT"' in runner
    assert "--load_env_from_checkpoint" in runner
    assert "--exact_run_name" in runner
    assert "--ent_schedule_start 0.15" in runner
    assert "--ent_schedule_end 0.02" in runner
    assert "sparse" not in runner.lower()

    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index('ssh "$REMOTE_HOST"')
    assert "SMOKE_REVISION" not in submit
    assert "$BASELINES_REVISION/smoke/full" in submit
    assert "10004034" in submit
    assert "smoke_validation.json" in submit
    assert all(name in submit for name in ("gpuhe.4h", "gpuhe.24h", "gpuhe.120h"))
    assert "RESUME_CHECKPOINT_SHA" in submit

    assert "UPDATES=20000" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert "EVAL_INTERVAL=1000" in sbatch
    assert "support_scope=all47_continuous" in sbatch
    assert "sampler_profile=continuous_banded_v1" in sbatch
    assert "target_probability_support=all_47_positive_from_update_0" in sbatch
    assert "all_family_condition_floor_mass=0.10" in sbatch
    assert "active_depth_band_mass=0.75" in sbatch
    assert "next_depth_band_mass=0.15" in sbatch
    assert "per_condition_mass_cap=none" in sbatch
    assert "foundation_target_probability_mass=0.5" in sbatch
    assert "trench_target_probability_mass=0.5" in sbatch
    assert (
        "actual_exposure_axes=assignment_reset_transition_completed_episode" in sbatch
    )
    assert "family_band_progression=independent" in sbatch
    assert "continuous_family_bands_not_stages" in sbatch
    assert "minimum_completed_episodes=32" in sbatch
    assert "sampler_refresh_interval_updates=150" in sbatch
    assert "curriculum_graph_sha256=$CONTINUOUS_GRAPH_SHA" in sbatch
    assert "verify_continuous_sampler_checkpoint.py" in sbatch
    assert "smoke_sampler_validation.json" in sbatch
    assert "planned_posthoc_eval_spacing_updates" in sbatch
    assert "fixed_evaluation_interval" not in sbatch
    assert "initialization=$INITIALIZATION" in sbatch
    assert "true_resume_optimizer_schedule_sampler" in sbatch
    assert "reward_type=DENSE" in sbatch
    assert "horizon=450" in sbatch
    assert "full_resets=true" in sbatch
    assert "--accepted-panel" in sbatch
    assert "--capability-panel" in sbatch
    assert "--expect-completion-contract exact_visible_dump_v1" in sbatch
    assert 'test "${#GPU_NAMES[@]}" -eq 4' in sbatch
    assert "used < 45.0" in sbatch
