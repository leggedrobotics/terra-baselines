import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_continuous_reward_anneal_v1"


def load_verifier():
    path = LAUNCHER / "verify_pair_smoke.py"
    spec = importlib.util.spec_from_file_location("verify_pair_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pair_launcher_changes_only_reward_schedule_and_uses_long_queue():
    runner = (ROOT / "scripts" / "run_v8_continuous_reward_anneal_v1.sh").read_text()
    submit = (LAUNCHER / "submit.sh").read_text()
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert "dense_skill|annealed_objective" in runner
    assert '--reward_stage "$REWARD_STAGE"' in runner
    assert "--accepted-bank-scope full" in runner
    assert "--accepted-bank-sampler-profile continuous_banded_v1" in runner
    assert "--warm_start_from" not in runner
    assert "--teacher_checkpoint" not in runner
    assert "--exact_run_name" in runner
    assert "--ent_schedule_start 0.15" in runner
    assert "--ent_schedule_end 0.02" in runner

    assert "SEED=20260807" in submit
    assert "ARMS=(constant_dense dense_to_terminal)" in submit
    assert "gpuhe.4h" in submit
    assert "gpuhe.120h" in submit
    assert "119:45:00" in submit
    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index('ssh "$REMOTE_HOST"')
    assert submit.index("verify_pair_smoke.py") < submit.index('    JOB_ID="$(ssh')
    assert "pretrigger_dense_parity" in submit
    assert "training_bank_archive_sha256" in submit
    assert "training_bank_dataset_sha256" in submit

    assert "UPDATES=20000" in sbatch
    assert "CHECKPOINT_INTERVAL=500" in sbatch
    assert "EVAL_INTERVAL=1000" in sbatch
    assert "condition_count=47" in sbatch
    assert "maps_per_condition=96" in sbatch
    assert "sampler_profile=continuous_banded_v1" in sbatch
    assert "reward_anneal_trigger=both_family_active_depth_at_least_2" in sbatch
    assert "reward_anneal_duration_updates=5000" in sbatch
    assert "reward_anneal_shape=one_way_linear_dense_to_terminal" in sbatch
    assert (
        "terminal_success_base_scale_matches_normalized_dense_terminal_component=true"
        in sbatch
    )
    assert "initialization=random_no_teacher" in sbatch
    assert "--require-productive-workspace-cycles" in sbatch
    assert "--accepted-panel" in sbatch and "--capability-panel" in sbatch


def test_pair_verifier_requires_exact_pretrigger_trees_and_sampler_state():
    verifier = load_verifier()
    left = {"x": np.array([1.0, np.nan]), "y": (np.array([2]),)}
    right = {"x": np.array([1.0, np.nan]), "y": (np.array([2]),)}
    verifier.assert_tree_exact(left, right, "tree")
    verifier.assert_nested_exact(
        {"rng": {"state": 7}, "probabilities": np.array([0.25, 0.75])},
        {"rng": {"state": 7}, "probabilities": np.array([0.25, 0.75])},
        "sampler",
    )

    with pytest.raises(ValueError, match="leaf"):
        verifier.assert_tree_exact(
            left,
            {"x": np.array([2.0, np.nan]), "y": (np.array([2]),)},
            "tree",
        )
    with pytest.raises(ValueError, match="values differ"):
        verifier.assert_nested_exact({"schema": "a"}, {"schema": "b"}, "sampler")

    source = (LAUNCHER / "verify_pair_smoke.py").read_text()
    for field in (
        "model",
        "optimizer_state",
        "loss_info",
        "transition_integrity",
        "pooled_sampler_state",
    ):
        assert field in source
    assert '"schema": "terra_reward_anneal_v1"' in source
    assert '"started_update": None' in source
    assert '"duration_updates": 5000' in source
    assert '"last_applied_mix": 0.0' in source
