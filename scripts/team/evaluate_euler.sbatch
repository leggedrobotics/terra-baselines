#!/usr/bin/env bash
# Euler single-GPU evaluation panel (usage: sbatch [resource flags]
# evaluate_euler.sbatch EVAL.env). EVAL.env sets VENV, BANK_ARCHIVE,
# TERRA_ROOT, BASELINES_ROOT, GPU_MODEL and the EVAL_* variables of
# evaluate_panel.sh except EVAL_BANK_ROOT. The panel runs one process at a
# time on the job's GPU.
set -euo pipefail
[[ "$(id -un)" == lterenzi ]] || exit 2
[[ $# == 1 && -r "$1" ]] || { echo "usage: evaluate_euler.sbatch EVAL.env" >&2; exit 2; }
source "$1"
: "${VENV:?}" "${BANK_ARCHIVE:?}" "${GPU_MODEL:?}" "${TMPDIR:?}" "${EVAL_OUTPUT:?}"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv "$GPU_MODEL"; then
    echo "unexpected GPU model; expected $GPU_MODEL" >&2
    exit 3
fi
module purge
module load stack/2024-06 eth_proxy
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export PATH="$VENV/bin:$PATH" EVAL_BANK_ROOT="$TMPDIR/bank" EVAL_SEQUENTIAL=1
export TERRA_ROOT BASELINES_ROOT EVAL_CHECKPOINT EVAL_OUTPUT EVAL_SPECS EVAL_ENVS EVAL_SEED
export EVAL_TYPES EVAL_MAPS_PATH EVAL_DATASET_SIZE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib" WANDB_MODE=disabled
export JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR="$EVAL_OUTPUT/../jax-cache-eval"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
exec bash "$BASELINES_ROOT/scripts/team/evaluate_panel.sh"
