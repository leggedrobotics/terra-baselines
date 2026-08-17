#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in TARGET_UPDATE RUN_DIR RUN_NAME BASELINES_ROOT BASELINES_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION PROTOCOL_TERRA_REVISION SEED VENV \
    TERRA_EULER_USER TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_DATASET_SHA BANK_RELEASE_ID \
    DISTANCE_SIDECAR_SHA PARTIAL_ARCHIVE PARTIAL_ARCHIVE_SHA PARTIAL_BANK_SHA \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

test "$TARGET_UPDATE" = 100000
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_ARCHIVE_SHA"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
tar --zstd -xf "$PARTIAL_ARCHIVE" -C "$TMPDIR" partial_bank
BANK="$TMPDIR/bank"
PARTIAL_BANK="$TMPDIR/partial_bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1 eth_proxy
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 4
"$PYTHON" - <<'PY'
import jax

assert jax.local_device_count() == 4, jax.devices()
PY
PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" PARTIAL_BANK="$PARTIAL_BANK" "$PYTHON" - <<'PY'
import os
from terra.maps_buffer import partial_reset_bank_sha256

assert partial_reset_bank_sha256(os.environ["PARTIAL_BANK"]) == os.environ["PARTIAL_BANK_SHA"]
PY

EXPECTED_PARAMETERS=2352573
EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import os
from types import SimpleNamespace
import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig
from utils.models import get_model_ready

class Config(dict):
    __getattr__ = dict.__getitem__

config = Config(
    clip_action_maps=True, loaded_max=100,
    local_map_normalization_bounds=(-16, 16),
    maps_net_normalization_bounds=(-10, 10), model_core="mlp",
    actor_core="gru", actor_gru_hidden_dim=64, model_size="medium",
    num_prev_actions=5, critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16", attention_compute_dtype="float32",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(2, 2, 3, 3),
    token_mixer_residual_init_scale=0.1, flatten_reduce_channels=32,
    attn_latent_queries=8, aux_coef=0.0, carry_work_observation=True,
    stall_age_observation=False, reward_v2_reset_context_observation=True,
)
env = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
_, params = get_model_ready(jax.random.PRNGKey(0), config, env)
count = sum(int(np.asarray(x).size) for x in jax.tree.leaves(params))
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
PY

export WANDB_ENTITY WANDB_PROJECT WANDB_MODE=online WANDB_RESUME=never
export WANDB_RUN_ID="v8_relay_gru64r_${BASELINES_REVISION:0:10}_s${SEED}"
export WANDB_DIR="$RUN_DIR/wandb"
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
printf '%s\n' \
    "status=RUNNING" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "actor_core=gru" \
    "actor_gru_hidden_dim=64" \
    "actor_head=concat_skip" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "global_envs=2048" \
    "transitions_per_update=65536" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PYTHON_BIN="$PYTHON" MACHINE=euler
"$BASELINES_ROOT/scripts/run_v8_relay_gru_v2.sh" \
    "$BANK" "$PARTIAL_BANK" "$RUN_NAME" "$TARGET_UPDATE" "$RUN_DIR" \
    "$DISTANCE_SIDECAR_SHA" none
