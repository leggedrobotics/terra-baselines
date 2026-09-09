#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 027
# This is a generated, credential-free file containing pinned paths and hashes.
# shellcheck disable=SC1090
source "${1:?path to staged launch.env}"
# shellcheck disable=SC1091
source "$BASELINES_ROOT/cluster/euler_account.sh"
terra_euler_configure "$TERRA_EULER_USER"
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$TERRA_ROOT/REVISION")" = "$TERRA_REVISION"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$PARENT" | awk '{print $1}')" = "$PARENT_SHA"
test "$(sha256sum "$VENV/requirements.lock.txt" | awk '{print $1}')" = 36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189
module purge
module load stack/2024-06 eth_proxy
task_ld_path="${LD_LIBRARY_PATH:-}"
if [[ "${task_ld_path,,}" =~ cuda|cudnn|cublas|nccl|nvrtc|cupti|cusolver|cusparse|cufft|nvjitlink ]]; then
    echo 'Unexpected inherited CUDA libraries' >&2; exit 3
fi
export TERRA_ROOT BASELINES_ROOT TERRA_PYTHON="$VENV/bin/python" MACHINE=euler SEED=20260909
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
export XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0'
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=true
task_cuda_paths=$("$TERRA_PYTHON" -c 'import glob,site; print(":".join(glob.glob(site.getsitepackages()[0]+"/nvidia/*/lib")))')
export LD_LIBRARY_PATH="$task_cuda_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache" JAX_ENABLE_COMPILATION_CACHE=true
SEGMENT="$RUN_ROOT/segments/$SLURM_JOB_ID"
mkdir -p "$SEGMENT"
cp "$1" "$SEGMENT/launch.env"
lquota > "$SEGMENT/lquota.txt"
task_home_used=$(bash "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT" < "$SEGMENT/lquota.txt")
awk -v used="$task_home_used" 'BEGIN {exit !(used <= 45)}'
scontrol show job "$SLURM_JOB_ID" -o > "$SEGMENT/slurm_job.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv > "$SEGMENT/gpus.csv"
task_allocated_gpu="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
[[ -n "$task_allocated_gpu" && "$task_allocated_gpu" != *,* ]] || exit 3
mapfile -t task_gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#task_gpu_names[@]} > 0 )) || exit 3
for name in "${task_gpu_names[@]}"; do [[ "$name" == *'RTX 4090'* ]] || exit 3; done
"$TERRA_PYTHON" - <<'PY'
import jax
from importlib.metadata import version
devices = jax.devices()
assert len(devices) == 1 and devices[0].platform == 'gpu', devices
assert 'RTX 4090' in devices[0].device_kind, devices
for name, expected in {'jax': '0.4.33', 'jaxlib': '0.4.33', 'jax-cuda12-plugin': '0.4.33',
                       'nvidia-cudnn-cu12': '9.5.0.50', 'nvidia-cuda-runtime-cu12': '12.6.77',
                       'nvidia-nccl-cu12': '2.23.4'}.items():
    assert version(name) == expected, (name, version(name))
print('PASS one RTX4090 and pinned runtime', flush=True)
PY
"$TERRA_PYTHON" -u "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices 1
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export DATASET_PATH="$TMPDIR/bank"
export RUN_NAME="generalist-2x-reliability-s${SEED}"
export RUN_DIR="$SEGMENT/smoke" RESUME_FROM="$PARENT"
export START_UPDATE="$PARENT_UPDATE" TARGET_UPDATE="$((PARENT_UPDATE + 2))"
export FIRST_SEGMENT="$BEHAVIOR_TRANSFER" CHECKPOINT_INTERVAL=1
export WANDB_MODE=disabled FINITE_CHECK_INTERVAL=1
bash "$BASELINES_ROOT/scripts/excavation_reliability/train.sh"
SMOKE_CHECKPOINT="$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
JAX_PLATFORMS=cpu "$TERRA_PYTHON" -u "$BASELINES_ROOT/scripts/excavation_reliability/verify_smoke.py" \
    "$PARENT" "$SMOKE_CHECKPOINT" > "$SEGMENT/smoke_check.log"
# A fresh segment owns its episode receipts and linked W&B history.
export RUN_DIR="$SEGMENT/training" RESUME_FROM="$SMOKE_CHECKPOINT"
export START_UPDATE="$((PARENT_UPDATE + 2))" TARGET_UPDATE=100000 FIRST_SEGMENT=0 CHECKPOINT_INTERVAL=500
export WANDB_MODE=online FINITE_CHECK_INTERVAL=10 WANDB_RESUME=never
export WANDB_RUN_ID="terra-rel2x-${SLURM_JOB_ID}"
export WANDB_NOTES="Native continuation from $PARENT; Terra $TERRA_REVISION; baselines $BASELINES_REVISION"
exec bash "$BASELINES_ROOT/scripts/excavation_reliability/train.sh"
