#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-map-v1-eval
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724/logs/%x_%j.out

set -euo pipefail
umask 002

: "${ARM:?submit with --export=ALL,ARM=flat or ARM=staged}"
case "$ARM" in
    flat|staged) ;;
    *)
        echo "Unsupported ARM=$ARM" >&2
        exit 2
        ;;
esac

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$ARM"
OUTPUT="$RUN_ROOT/eval/${ARM}_development_cadence.json"

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

test -f "$RUN_ROOT/${ARM}_TRAINING_COMPLETE"
test ! -e "$OUTPUT"
mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

cd "$SOURCE/terra-baselines"
"$VENV/bin/python" -u eval_fixed_bank.py \
    --checkpoint-glob "$CHECKPOINT_DIR/*_update_*.pkl" \
    --bank-root "$BANK" \
    --split development \
    --strata M0 M1 M2 \
    --horizon 450 \
    --output "$OUTPUT"

touch "$RUN_ROOT/${ARM}_DEVELOPMENT_EVAL_COMPLETE"
echo "${ARM^^}_DEVELOPMENT_EVAL_COMPLETE"
