#!/usr/bin/env bash
# usage: smoke_local.sh OUTPUT_DIR
# Local smoke of the machine-rules fine-tune on one RTX 4090: the native
# resume of u100000 with run.sbatch's arguments (train_args.sh), ENVS envs
# (default 64) for UPDATES updates (default 2), DUG_CLEARANCE_M (default
# 0.57), W&B disabled, finite checks and a checkpoint after every update. It
# prints the JAX peak device memory and the peak host RSS at exit, and samples
# the GPU's total memory use every second into gpu_memory.csv.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: smoke_local.sh OUTPUT_DIR" >&2; exit 2; }
OUT="$1"
test ! -e "$OUT"
mkdir -p "$OUT/checkpoints"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_EVAL_PYTHON:-/home/lorenzo/moleworks/.artifacts/terra_gru_rules_ft_20260930/runtime_jax0433_cuda/bin/python}"
TERRA_ROOT="${TERRA_ROOT:-/home/lorenzo/moleworks/.worktrees/terra_machine_rules_20260930/terra}"
BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/train_v3_generalist_512.tar.zst
PRESET=gru_generalist_512_machine_rules
DUG_CLEARANCE_M="${DUG_CLEARANCE_M:-0.57}"
RUN_NAME=gru_rules_ft_smoke
SEED=20260930
NUM_DEVICES=1
ENVS_PER_DEVICE="${ENVS:-64}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * ENVS_PER_DEVICE * 32 * (100000 + ${UPDATES:-2})))
BANK_DISTANCE_SIDECAR_SHA=d9a8c9e61346e319b3ffead150ca44c7e85d016cb5853997cf1ee721286d4506
TEACHER=/home/lorenzo/moleworks/.artifacts/terra_instance_efficiency_20260922/inputs/generalist_u110000.pkl
KL_ANNEAL_UPDATES=10000
CHECKPOINT_DIR="$OUT/checkpoints"
RESUME_FROM=/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/checkpoints/gru_gen512_s20260923_update_100000.pkl
source "$REPO/scripts/euler_gru_machine_rules_ft/train_args.sh"

tar --zstd -xf "$BANK_ARCHIVE" -C "$OUT" bank
export DATASET_PATH="$OUT/bank" DATASET_SIZE=20480
export PYTHONPATH="$TERRA_ROOT:$REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1 WANDB_MODE=disabled
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export WANDB_DIR="$OUT/wandb" XDG_CACHE_HOME="$OUT/xdg-cache" MPLCONFIGDIR="$OUT/matplotlib"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
CUDA_PATHS="$("$PYTHON" -c 'import glob,site; print(":".join(glob.glob(site.getsitepackages()[0]+"/nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$CUDA_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$WANDB_DIR"
{ git -C "$REPO" rev-parse HEAD; git -C "$TERRA_ROOT" rev-parse HEAD; } > "$OUT/revisions.txt"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_memory.csv" &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true' EXIT
"$PYTHON" - "$REPO/train_mixed.py" "${TRAIN_ARGS[@]}" \
    --machine local \
    --finite_check_interval 1 \
    --log_train_interval 1 \
    --checkpoint_interval 1 > "$OUT/train.log" 2>&1 <<'PY'
import runpy
import sys

script = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    peak_rss_kb = next(
        int(line.split()[1]) for line in open("/proc/self/status") if line.startswith("VmHWM:")
    )
    print(
        f"SMOKE_MEMORY jax_peak_bytes_in_use={stats.get('peak_bytes_in_use', -1)} "
        f"jax_bytes_limit={stats.get('bytes_limit', -1)} host_peak_rss_kb={peak_rss_kb}",
        flush=True,
    )
PY
echo "exit=0" >> "$OUT/train.log"
