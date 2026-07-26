#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-f0-trace
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0_retry1/diagnostics/trajectory_v1/logs/%x_%j.out

set -euo pipefail
umask 002

F0_ROOT="${F0_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0_retry1}"
DIAG_ROOT="${DIAG_ROOT:-$F0_ROOT/diagnostics/trajectory_v1}"
SOURCE="$DIAG_ROOT/source"
OUTPUT="$DIAG_ROOT/trajectory_diagnosis.json"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="$SOURCE/terra:$SOURCE/terra-baselines"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_MODE=disabled
export DATASET_PATH="$F0_ROOT/bank"
export DATASET_SIZE=1

test -f "$DIAG_ROOT/SOURCE_SEALED"
test -f "$F0_ROOT/foundation/EVALUATION_COMPLETE"
test -f "$F0_ROOT/trench/EVALUATION_COMPLETE"
test ! -e "$OUTPUT"
(
    cd "$DIAG_ROOT"
    sha256sum --check manifests/source_files.sha256
)
(
    cd "$F0_ROOT"
    sha256sum --check manifests/bank_files.sha256
)

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

cd "$SOURCE/terra-baselines"
"$VENV/bin/python" -m pytest -q tests/test_diagnose_f0_trajectories.py
"$VENV/bin/python" -u scripts/analysis/diagnose_f0_trajectories.py \
    --run-root "$F0_ROOT" \
    --output "$OUTPUT"

"$VENV/bin/python" - "$OUTPUT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload["schema"] != "terra_f0_trajectory_diagnosis_v1":
    raise SystemExit("unexpected trajectory diagnosis schema")
if payload["gate"] != {
    "bounded_transitions": 57600,
    "gradient_updates": 0,
    "passed": True,
    "replays": 4,
    "selected_traces": 6,
    "validated_evaluator_rows": 128,
}:
    raise SystemExit(f"trajectory diagnosis gate mismatch: {payload['gate']}")
PY

sha256sum "$OUTPUT" | tee "$DIAG_ROOT/trajectory_diagnosis.sha256"
touch "$DIAG_ROOT/DIAGNOSIS_COMPLETE"
echo "F0_TRAJECTORY_DIAGNOSIS_COMPLETE"
