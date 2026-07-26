#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-b0-policy-cross
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_policy_cross_v1/logs/%x_%j.out

set -euo pipefail
umask 002

: "${PANEL:?submit trench_distance or trench_side}"
case "$PANEL" in
    trench_distance)
        PANEL_COUNT=32
        ;;
    trench_side)
        PANEL_COUNT=16
        ;;
    *)
        echo "Unsupported B0 policy-cross panel: $PANEL" >&2
        exit 2
        ;;
esac

TRAINING_ROOT="${TRAINING_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_u1000_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_policy_cross_v1}"
SOURCE="$SOURCE_ROOT/source"
BANK="$TRAINING_ROOT/bank"
CHECKPOINT_DIR="$TRAINING_ROOT/runs/$PANEL/checkpoints"
PANEL_ROOT="$SOURCE_ROOT/runs/$PANEL"
OUTPUT="$PANEL_ROOT/policy_cross.json"
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
export DATASET_PATH="$BANK"
export DATASET_SIZE="$PANEL_COUNT"

mkdir -p "$SOURCE_ROOT/logs" "$PANEL_ROOT"
test -f "$TRAINING_ROOT/runs/$PANEL/EVALUATION_COMPLETE"
test -f "$TRAINING_ROOT/runs/$PANEL/PANEL_DIAGNOSIS_REQUIRED"
test -f "$SOURCE_ROOT/manifests/source_files.sha256"
test -f "$TRAINING_ROOT/manifests/bank_files.sha256"
test ! -e "$OUTPUT"
(
    cd "$SOURCE_ROOT"
    sha256sum --check manifests/source_files.sha256
)
(
    cd "$TRAINING_ROOT"
    sha256sum --check manifests/bank_files.sha256
)

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

cd "$SOURCE/terra-baselines"
"$VENV/bin/python" -m unittest discover \
    -s tests -p 'test_diagnose_b0_policy_cross.py' -q

mapfile -t CHECKPOINTS < <(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*_update_000900.pkl'
)
test "${#CHECKPOINTS[@]}" -eq 1

"$VENV/bin/python" -u scripts/diagnose_b0_policy_cross.py \
    --checkpoint "${CHECKPOINTS[0]}" \
    --bank-root "$BANK" \
    --panel "$PANEL" \
    --planned-updates 1000 \
    --output "$OUTPUT"

"$VENV/bin/python" - "$OUTPUT" "$PANEL" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload["schema"] != "terra_b0_policy_cross_v1":
    raise SystemExit("unexpected B0 policy-cross schema")
if payload["panel"] != sys.argv[2] or payload["checkpoint_update"] != 900:
    raise SystemExit("wrong B0 policy-cross treatment")
if len(payload["records"]) != 10:
    raise SystemExit("incomplete B0 policy cross")
if not all(
    verification["passed"]
    for verification in payload["reset_verifications"].values()
):
    raise SystemExit("B0 policy-cross reset verification failed")
if any(
    record["summary"]["integrity_failure_count"]
    for record in payload["records"]
):
    raise SystemExit("B0 policy-cross integrity failure")
PY

touch "$PANEL_ROOT/POLICY_CROSS_COMPLETE"
echo "B0_${PANEL^^}_POLICY_CROSS_COMPLETE"
