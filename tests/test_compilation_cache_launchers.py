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
@pytest.mark.parametrize("wandb_mode", ["offline", "disabled"])
def test_cscs_forwards_cache_configuration_to_runtime(tmp_path, cache_enabled, wandb_mode):
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
        'printf "%s|%s|%s\\n" "$JAX_COMPILATION_CACHE_DIR" '
        '"$JAX_ENABLE_COMPILATION_CACHE" "$WANDB_MODE" >> "$MOCK_PYTHON_LOG"\n',
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
            "--wandb-mode", wandb_mode,
        ],
        env=env, check=True, capture_output=True, text=True,
    )
    container_env = tomllib.loads((run_root / "terra.edf.toml").read_text())["env"]
    assert container_env["JAX_COMPILATION_CACHE_DIR"] == str(cache_dir)
    assert container_env["JAX_ENABLE_COMPILATION_CACHE"] == (cache_enabled or "true")
    assert container_env["WANDB_MODE"] == wandb_mode
    assert not cache_dir.exists()
    for key in (
        "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE",
        "TERRA_RUN_DIR", "DATASET_PATH", "DATASET_SIZE", "WANDB_MODE",
    ):
        env[key] = container_env[key]
    subprocess.run(
        ["bash", str(REPO / "cluster/cscs/run_training.sh"), "smoke"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert (tmp_path / "python_calls").read_text().splitlines() == [
        f"{cache_dir}|{cache_enabled or 'true'}|{wandb_mode}",
        f"{cache_dir}|{cache_enabled or 'true'}|{wandb_mode}",
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


@pytest.mark.parametrize("smoke", [False, True])
def test_euler_forwards_segment_overrides_to_sbatch(tmp_path, smoke):
    commands = tmp_path / "commands"
    commands.mkdir()
    _script(
        commands / "git",
        'if [[ "$*" == *"rev-parse HEAD"* ]]; then\n'
        '  if [[ "$2" == "$TERRA_REPO" ]]; then\n'
        '    echo 46b140f8373e098ad832e4968d8136a5ba861bf6\n'
        '  else echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; fi\n'
        'fi\n',
    )
    _script(
        commands / "sha256sum",
        'printf "%s  %s\\n" '
        '1125177d322df6097f8da9f67ec95fe48762e16327f83dc157ec282b24993fb3 "$1"\n',
    )
    _script(
        commands / "ssh",
        'shift 3\n'
        'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
        'if [[ "$*" == "id -un" ]]; then echo "$TERRA_EULER_USER"; fi\n'
        'if [[ "$*" == *"sbatch --parsable"* ]]; then echo 12345; fi\n',
    )
    env = os.environ.copy()
    for key in ("TERRA_TARGET_UPDATE", "TERRA_WALLTIME", "TERRA_PARTITION", "WANDB_MODE"):
        env.pop(key, None)
    env.update(
        PATH=f"{commands}:{env['PATH']}", SUBMIT="1", ARMS="gen",
        TERRA_REPO=str(tmp_path / "terra"), TERRA_EULER_USER="testuser",
        TERRA_EULER_HOME_ROOT="/cluster/home/testuser",
        TERRA_EULER_SCRATCH_ROOT="/cluster/scratch/testuser",
        TERRA_EULER_PROJECT_ROOT="/cluster/project/rsl/testuser",
        REMOTE_HOST="mock-ssh-only", MOCK_SSH_LOG=str(tmp_path / "ssh_calls"),
        JAX_COMPILATION_CACHE_DIR="/cluster/scratch/testuser/shared-cache",
        JAX_ENABLE_COMPILATION_CACHE="false",
    )
    if smoke:
        env.update(TERRA_TARGET_UPDATE="1002", TERRA_WALLTIME="00:45:00",
                   TERRA_PARTITION="gpuhe.4h", WANDB_MODE="disabled")
    subprocess.run(
        ["bash", str(REPO / "scripts/euler_trench_align_v2/submit.sh")],
        env=env, check=True, capture_output=True, text=True,
    )
    sbatch = next(line for line in (tmp_path / "ssh_calls").read_text().splitlines()
                  if "sbatch --parsable" in line)
    assert f"--partition='{env.get('TERRA_PARTITION', 'gpuhe.120h')}'" in sbatch
    assert f"--time='{env.get('TERRA_WALLTIME', '119:45:00')}'" in sbatch
    assert f"TARGET_UPDATE={env.get('TERRA_TARGET_UPDATE', '100000')}" in sbatch
    assert f"WANDB_MODE={env.get('WANDB_MODE', 'online')}" in sbatch
    assert "JAX_COMPILATION_CACHE_DIR=/cluster/scratch/testuser/shared-cache" in sbatch
    assert "JAX_ENABLE_COMPILATION_CACHE=false" in sbatch


@pytest.mark.parametrize("setting,value", [
    ("TERRA_TARGET_UPDATE", "0"),
    ("TERRA_TARGET_UPDATE", "-1"),
    ("TERRA_TARGET_UPDATE", "1.5"),
    ("TERRA_WALLTIME", "00:70:00"),
    ("TERRA_PARTITION", "gpuhe.4h'"),
    ("WANDB_MODE", "disabled,OTHER=1"),
])
def test_euler_rejects_invalid_segment_overrides_before_ssh(tmp_path, setting, value):
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "ssh-called"
    _script(commands / "ssh", 'touch "$MOCK_SSH_MARKER"\nexit 99\n')
    env = os.environ.copy()
    env.update(PATH=f"{commands}:{env['PATH']}", MOCK_SSH_MARKER=str(marker),
               SUBMIT="1", **{setting: value})
    result = subprocess.run(
        ["bash", str(REPO / "scripts/euler_trench_align_v2/submit.sh")],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert setting in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_euler_native_resume_restores_the_saved_environment_config(tmp_path, resume):
    launcher = (REPO / "scripts/euler_trench_align_v2/run.sbatch").read_text()
    start = launcher.index('RESUME_FROM="${RESUME_FROM:-none}"')
    block = launcher[start:launcher.index('\ncase "$ARM" in', start)]
    checkpoint = tmp_path / "source checkpoint.pkl"
    checkpoint.touch()
    env = os.environ.copy()
    env["RESUME_FROM"] = str(checkpoint) if resume else "none"
    result = subprocess.run(
        ["bash", "-euc", block + '\nprintf "%s\\n" "${#RESUME_ARGS[@]}"\n'
         'if (( ${#RESUME_ARGS[@]} )); then printf "%s\\n" "${RESUME_ARGS[@]}"; fi\n'],
        env=env, check=True, capture_output=True, text=True,
    )
    assert result.stdout.splitlines() == (
        ["3", "--resume_from", str(checkpoint), "--load_env_from_checkpoint"]
        if resume else ["0"]
    )


@pytest.mark.parametrize(
    "scenario,expected_parent",
    [
        ("fresh", "unknown"),
        ("obsolete_retry_env_ignored", "unknown"),
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
    env.pop("ATTEMPT", None)
    env.update(
        ARM="gen", BASELINES_REVISION=revision, SEED="42",
        SLURM_JOB_ID="200", RUN_DIR=str(run_dir), RUN_NAME="fixed-local-name",
        RESUME_FROM="none" if scenario in {"fresh", "obsolete_retry_env_ignored"} else str(checkpoint),
        WANDB_RUN_ID="untrusted-inherited-id", WANDB_RESUME="allow",
    )
    if scenario == "obsolete_retry_env_ignored":
        env["ATTEMPT"] = "1"
    result = subprocess.run(
        [
            "bash", "-euc", function.group(0) + "\nconfigure_wandb_segment\n"
            'printf "%s\\n" "$WANDB_RUN_ID" "$WANDB_RESUME" '
            '"$WANDB_PARENT_RUN_ID" "$RUN_NAME"\n',
        ],
        env=env, check=True, capture_output=True, text=True,
    )
    run_id, resume, parent_id, local_name = result.stdout.splitlines()
    if scenario in {"fresh", "obsolete_retry_env_ignored"}:
        assert run_id == base_id
        assert resume == "never"
    else:
        assert run_id == f"{base_id}_resume_j200"
        assert resume == "never"
    assert parent_id == {"base": base_id, "segment": previous_segment}.get(expected_parent, "unknown")
    assert local_name == "fixed-local-name"
