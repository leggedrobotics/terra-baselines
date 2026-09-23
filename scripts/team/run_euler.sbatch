#!/usr/bin/env bash
# One Euler training segment of run.sh on a single GPU (usage: sbatch [resource
# flags] run_euler.sbatch ARM.env). ARM.env sets VENV, BANK_ARCHIVE,
# TERRA_ROOT, BASELINES_ROOT, EXPERIMENT_ROOT, PARENT_CHECKPOINT,
# TARGET_UPDATES, RUN_NAME, GPU_MODEL and EXTRA_ARGS. The bank is unpacked
# into node-local TMPDIR; segments resume the newest checkpoint of the
# experiment, so a chain of 4 h segments continues one run.
set -euo pipefail
[[ "$(id -un)" == lterenzi ]] || exit 2
[[ $# == 1 && -r "$1" ]] || { echo "usage: run_euler.sbatch ARM.env" >&2; exit 2; }
source "$1"
: "${VENV:?}" "${BANK_ARCHIVE:?}" "${GPU_MODEL:?}" "${TMPDIR:?}"
# Hard GPU guard before JAX starts.
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv "$GPU_MODEL"; then
    echo "unexpected GPU model; expected $GPU_MODEL" >&2
    exit 3
fi
module purge
module load stack/2024-06 eth_proxy
if printf '%s' "${LD_LIBRARY_PATH:-}" | grep -Eqi 'cuda|cudnn|cublas|nccl|nvrtc|cupti|cusolver|cusparse|cufft|nvjitlink'; then
    echo "CUDA libraries leaked into LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
    exit 3
fi
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export PATH="$VENV/bin:$PATH" TRAIN_BANK_ROOT="$TMPDIR/bank" DEVICES=1 GPU_KIND="${GPU_MODEL#RTX }"
export TERRA_ROOT BASELINES_ROOT EXPERIMENT_ROOT PARENT_CHECKPOINT TARGET_UPDATES RUN_NAME
export EXTRA_ARGS TEAM_AGENTS="${TEAM_AGENTS:-1}" ENVS_PER_DEVICE="${ENVS_PER_DEVICE:-512}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_MODE=offline WANDB_DIR="$EXPERIMENT_ROOT/wandb"
export JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR="$EXPERIMENT_ROOT/../jax-cache"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
mkdir -p "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
exec bash "$BASELINES_ROOT/scripts/team/run.sh"
