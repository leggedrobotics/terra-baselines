import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_trench_align_v2"
RUNTIME_LOCK = (
    ROOT
    / "cluster"
    / "euler_runtime"
    / "requirements-jax0433-cuda126-cudnn950.txt"
)
RUNTIME_LOCK_SHA = "36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189"


def test_euler_trench_v2_defaults_to_the_coherent_cudnn9_runtime() -> None:
    submit = (LAUNCHER / "submit.sh").read_text(encoding="utf-8")
    sbatch = (LAUNCHER / "run.sbatch").read_text(encoding="utf-8")

    assert "terra_jax0433_cuda126_cudnn950_20260903" in submit
    assert RUNTIME_LOCK_SHA in submit
    assert 'CUDNN_REPAIR="${TERRA_CUDNN_REPAIR:-none}"' in submit
    assert 'MAX_ATTEMPTS="${TERRA_MAX_ATTEMPTS:-0}"' in submit
    assert 'TERRA_SLURM_DEPENDENCY:-none' in submit
    assert "^afterok:[0-9]+$" in submit
    assert 'DEPENDENCY_OPTION="--dependency=$SLURM_DEPENDENCY"' in submit
    assert "RUNTIME_LOCK_SHA=$RUNTIME_LOCK_SHA" in submit

    assert "module load stack/2024-06 eth_proxy" in sbatch
    assert "module load stack/2024-06 cuda/12.1.1 eth_proxy" not in sbatch
    assert "CUDA libraries leaked into LD_LIBRARY_PATH" in sbatch
    assert 'CUDNN_REPAIR_MODE="${CUDNN_REPAIR_MODE:-none}"' in sbatch
    assert 'MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"' in sbatch
    assert '"nvidia-cuda-runtime-cu12": "12.6.77"' in sbatch
    assert '"nvidia-cudnn-cu12": "9.5.0.50"' in sbatch
    assert '"nvidia-nccl-cu12": "2.23.4"' in sbatch
    assert "check_jax_runtime.py\" --min-devices 4" in sbatch
    assert 'sha256sum "$VENV/requirements.lock.txt"' in sbatch


def test_committed_runtime_lock_matches_the_launcher_digest() -> None:
    assert hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest() == RUNTIME_LOCK_SHA


def test_legacy_cudnn8_workarounds_are_explicit_only() -> None:
    sbatch = (LAUNCHER / "run.sbatch").read_text(encoding="utf-8")

    assert "denylist_cache)" in sbatch
    assert "frontend_off)" in sbatch
    assert "none_cudnn9" in sbatch
    assert "CUDNN_REPAIR_MODE must be auto, denylist_cache, frontend_off or none" in sbatch
