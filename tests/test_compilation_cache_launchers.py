"""Exercise cache configuration across submit/container boundaries without SSH."""

import os
from pathlib import Path
import re
import subprocess
import tomllib

import pytest


REPO = Path(__file__).resolve().parents[1]


def _script(path, text):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + text)
    path.chmod(0o755)


@pytest.mark.parametrize("cache_enabled", [None, "false", "0"])
def test_cscs_forwards_cache_configuration_to_runtime(tmp_path, cache_enabled):
    commands = tmp_path / "commands"
    commands.mkdir()
    # Emulate SSH's command-string boundary using a local shell. All paths are
    # confined to tmp_path, and sbatch/python/nvidia-smi are harmless stubs.
    _script(commands / "ssh", 'shift 2\nexec bash -c "$*"\n')
    _script(commands / "sbatch", "exit 0\n")
    _script(commands / "nvidia-smi", "echo mock-gpu\n")
    _script(
        commands / "python",
        'test -d "$JAX_COMPILATION_CACHE_DIR"\n'
        'printf "%s|%s\\n" "$JAX_COMPILATION_CACHE_DIR" '
        '"$JAX_ENABLE_COMPILATION_CACHE" >> "$MOCK_PYTHON_LOG"\n',
    )
    remote_root = tmp_path / "remote"
    snapshot = remote_root / "snapshots" / "cache-test"
    snapshot.mkdir(parents=True)
    (snapshot / ".ready").touch()
    images = remote_root / "images"
    images.mkdir()
    (images / "terra-jax+jax24.10-v1.sqsh").touch()
    dataset = remote_root / "dataset"
    dataset.mkdir()
    env = os.environ.copy()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.pop("JAX_ENABLE_COMPILATION_CACHE", None)
    env.update(
        PATH=f"{commands}:{env['PATH']}",
        CSCS_ROOT=str(remote_root),
        CSCS_SSH_TARGET="mock-ssh-only",
        CSCS_IMAGE_NAME="terra-jax",
        CSCS_IMAGE_TAG="jax24.10-v1",
        MOCK_PYTHON_LOG=str(tmp_path / "python_calls"),
    )
    run_root = remote_root / "runs" / "cache-test"
    cache_dir = run_root / "jax-cache"
    if cache_enabled is not None:
        cache_dir = remote_root / "shared-cache"
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        env["JAX_ENABLE_COMPILATION_CACHE"] = cache_enabled

    subprocess.run(
        [
            "bash", str(REPO / "cluster/cscs/submit.sh"),
            "--dataset-path", str(dataset), "--dataset-size", "1",
            "--run-id", "cache-test", "--no-sync", "--test-only",
            "--wandb-mode", "offline",
        ],
        env=env, check=True, capture_output=True, text=True,
    )
    container_env = tomllib.loads((run_root / "terra.edf.toml").read_text())["env"]
    assert container_env["JAX_COMPILATION_CACHE_DIR"] == str(cache_dir)
    assert container_env["JAX_ENABLE_COMPILATION_CACHE"] == (cache_enabled or "true")
    assert not cache_dir.exists()
    for key in (
        "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE",
        "TERRA_RUN_DIR", "DATASET_PATH", "DATASET_SIZE",
    ):
        env[key] = container_env[key]
    subprocess.run(
        ["bash", str(REPO / "cluster/cscs/run_training.sh"), "smoke"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert (tmp_path / "python_calls").read_text().splitlines() == [
        f"{cache_dir}|{cache_enabled or 'true'}",
        f"{cache_dir}|{cache_enabled or 'true'}",
    ]


@pytest.mark.parametrize(
    "launcher", ["cluster/cscs/submit.sh", "scripts/euler_trench_align_v2/submit.sh"]
)
@pytest.mark.parametrize("cache_dir", ["relative/cache", "/tmp/cache with spaces", "/tmp/cache,OTHER=1"])
def test_submit_rejects_unsafe_cache_path_before_ssh(tmp_path, launcher, cache_dir):
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "ssh-called"
    _script(commands / "ssh", 'touch "$MOCK_SSH_MARKER"\nexit 99\n')
    env = os.environ.copy()
    env.update(
        PATH=f"{commands}:{env['PATH']}",
        MOCK_SSH_MARKER=str(marker),
        JAX_COMPILATION_CACHE_DIR=cache_dir,
        JAX_ENABLE_COMPILATION_CACHE="true",
        SUBMIT="0",
    )
    args = ["bash", str(REPO / launcher)]
    if launcher.startswith("cluster/cscs"):
        args += ["--dataset-path", "/remote/dataset", "--dataset-size", "1"]
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "JAX_COMPILATION_CACHE_DIR" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "scenario,expected_parent",
    [
        ("fresh", "unknown"),
        ("retry_without_checkpoint", "unknown"),
        ("legacy_same_run", "base"),
        ("previous_segment", "segment"),
        ("external_checkpoint", "unknown"),
        ("unrelated_run_id", "unknown"),
        ("unrelated_legacy_contract", "unknown"),
        ("missing_contract", "unknown"),
    ],
)
def test_euler_checkpoint_resume_uses_a_new_wandb_segment(tmp_path, scenario, expected_parent):
    launcher = (REPO / "scripts/euler_trench_align_v2/run.sbatch").read_text()
    # Execute the real identity-selection function in isolation from the GPU
    # preflight and Slurm machinery in the rest of the batch script.
    function = re.search(r"(?ms)^configure_wandb_segment\(\) \{\n.*?^\}", launcher)
    assert function is not None
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "fixed-local-name_update_000500.pkl"
    checkpoint.touch()
    revision = "a" * 40
    base_id = f"trench_align_v2_gen_{revision[:10]}_s42"
    previous_segment = f"{base_id}_resume_j100"
    contract = f"terra_baselines_revision={revision}\narm=gen\nseed=42\n"
    if scenario == "previous_segment":
        contract += f"wandb_run_id={previous_segment}\n"
    elif scenario == "unrelated_run_id":
        contract += "wandb_run_id=some-unrelated-run\n"
    elif scenario == "unrelated_legacy_contract":
        contract = contract.replace("arm=gen", "arm=spec")
    if scenario != "missing_contract":
        (run_dir / "run_contract.env").write_text(contract)
    if scenario == "external_checkpoint":
        checkpoint = tmp_path / "external.pkl"
        checkpoint.touch()
    env = os.environ.copy()
    env.update(
        ARM="gen", BASELINES_REVISION=revision, SEED="42",
        SLURM_JOB_ID="200", RUN_DIR=str(run_dir), RUN_NAME="fixed-local-name",
        ATTEMPT="1" if scenario == "retry_without_checkpoint" else "0",
        RESUME_FROM="none" if scenario in {"fresh", "retry_without_checkpoint"} else str(checkpoint),
        WANDB_RUN_ID="untrusted-inherited-id", WANDB_RESUME="allow",
    )
    result = subprocess.run(
        [
            "bash", "-euc", function.group(0) + "\nconfigure_wandb_segment\n"
            'printf "%s\\n" "$WANDB_RUN_ID" "$WANDB_RESUME" '
            '"$WANDB_PARENT_RUN_ID" "$RUN_NAME"\n',
        ],
        env=env, check=True, capture_output=True, text=True,
    )
    run_id, resume, parent_id, local_name = result.stdout.splitlines()
    if scenario in {"fresh", "retry_without_checkpoint"}:
        assert run_id == base_id
        assert resume == ("allow" if scenario == "retry_without_checkpoint" else "never")
    else:
        assert run_id == f"{base_id}_resume_j200"
        assert resume == "never"
    assert parent_id == {"base": base_id, "segment": previous_segment}.get(expected_parent, "unknown")
    assert local_name == "fixed-local-name"
