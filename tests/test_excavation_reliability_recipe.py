"""Protect the experiment's native-resume and compute-budget boundaries."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/excavation_reliability/train.sh"


def arguments(tmp_path, **changes):
    env = dict(os.environ, TERRA_ROOT=str(tmp_path / "terra"),
               DATASET_PATH=str(tmp_path / "bank"), RUN_DIR=str(tmp_path / "run"),
               RUN_NAME="test-generalist", RESUME_FROM=str(tmp_path / "parent.pkl"),
               START_UPDATE="5000", TARGET_UPDATE="100000", FIRST_SEGMENT="1")
    env.update(changes)
    return subprocess.run(["bash", str(SCRIPT), "--print-args"], env=env,
                          text=True, capture_output=True)


def test_recipe_preserves_native_full_bank_and_absolute_budget(tmp_path):
    result = arguments(tmp_path)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert args[args.index("--config") + 1] == "trench_align_v2_generalist_gen"
    assert int(args[args.index("--total_timesteps") + 1]) == 16384 * 100000
    for flag in ("--resume_from", "--load_env_from_checkpoint", "--finetune_foundation_behavior",
                 "--executable_dig_observation", "--keep_checkpoint_history"):
        assert flag in args
    for flag in ("--warm_start_from", "--finetune_task_bank", "--resume_update",
                 "--no-load-env-from-checkpoint", "--flat_minibatch_shuffle", "--enable_action_mask"):
        assert flag not in args
    assert args[args.index("--cache_clear_interval") + 1] == "0"
    assert args[args.index("--checkpoint_interval") + 1] == "500"
    assert not (tmp_path / "run").exists()


def test_ordinary_continuation_removes_behavior_transfer(tmp_path):
    result = arguments(tmp_path, START_UPDATE="17000", FIRST_SEGMENT="0")
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert "--finetune_foundation_behavior" not in args
    assert "--load_env_from_checkpoint" in args


@pytest.mark.parametrize("changes", [
    {"TARGET_UPDATE": "5000"}, {"FIRST_SEGMENT": "2"},
    {"START_UPDATE": "5100", "FIRST_SEGMENT": "1"}, {"CHECKPOINT_INTERVAL": "0"},
])
def test_invalid_or_nonadvancing_recipe_stops_before_execution(tmp_path, changes):
    assert arguments(tmp_path, **changes).returncode != 0
