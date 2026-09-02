#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

# v2 arms with the fresh-trench gate ON under the corrected semantics (parallel
# to the section axis AND on the line; the retired standoff band is off).
# ARM=gen  -> trench_align_v2_generalist_gen  (foundation + trench, 40 cond)
# ARM=spec -> trench_align_v2_specialist_spec (trench only, 15 cond)
# GPU_TYPE selects the node class (rtx_4090 | rtx_3090); the in-job guard
# refuses any other model.

set -euo pipefail
umask 027

for REQUIRED in ARM TARGET_UPDATE RUN_DIR RUN_NAME BASELINES_ROOT \
    BASELINES_REVISION RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION SEED VENV \
    TERRA_EULER_USER TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_MAPS_PATH BANK_DATASET_SIZE \
    BANK_DISTANCE_SIDECAR_SHA GPU_TYPE \
    EXPECTED_PARAMETERS SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$ARM" in
    gen)  CONFIG="trench_align_v2_generalist_gen" ;;
    spec) CONFIG="trench_align_v2_specialist_spec" ;;
    *) echo "ARM must be gen or spec" >&2; exit 2 ;;
esac
case "$GPU_TYPE" in
    rtx_4090) GPU_MODEL="RTX 4090" ;;
    rtx_3090) GPU_MODEL="RTX 3090" ;;
    *) echo "GPU_TYPE must be rtx_4090 or rtx_3090" >&2; exit 2 ;;
esac
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export DATASET_PATH="$TMPDIR/bank"
export DATASET_SIZE="$BANK_DATASET_SIZE"
test -d "$DATASET_PATH/$BANK_MAPS_PATH/metadata"

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
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv "$GPU_MODEL" && exit 3

# RTX 4090 (cc 8.9, cuDNN 8.9.7, driver 580): cuDNN frontend engine 20 returns
# wrong bf16 backward-filter results for these exact PPO minibatch shapes; job
# 12505854 died before update 1 with CUDNN_STATUS_EXECUTION_FAILED. Reuse the
# repair proven in euler_trench_align_generalist_partial_v1: level-4 autotuning,
# the exact engine-20 denylist, and the four-GPU autotune cache that completed
# native PPO updates (ledger revision 58e26fc9). Level 0 reproduced a ~8x slow
# path there, so it is not used. RTX 3090 (cc 8.6) is unaffected and unchanged.
ALGORITHM_DENYLIST="$BASELINES_ROOT/scripts/euler_trench_align_v2/hlo_algorithm_denylist.pbtxt"
AUTOTUNE_CACHE="${TERRA_AUTOTUNE_CACHE:-/cluster/project/rsl/alesweber/terra_runtime/autotune/terra_trench_align_generalist_partial_v1/rtx4090_cc89_cudnn897_jaxlib0426_45e2dbc_u3501.pbtxt}"
AUTOTUNE_CACHE_SHA="698e856cae464e5fea93e0b2121fc8de4d9cb691135571ca4b5d56f3259d16a3"
if [ "$GPU_TYPE" = rtx_4090 ]; then
    test -r "$ALGORITHM_DENYLIST"
    test -r "$AUTOTUNE_CACHE"
    test "$(sha256sum "$AUTOTUNE_CACHE" | awk '{print $1}')" = "$AUTOTUNE_CACHE_SHA"
    export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_algorithm_denylist_path=$ALGORITHM_DENYLIST --xla_gpu_load_autotune_results_from=$AUTOTUNE_CACHE --xla_gpu_dump_autotune_results_to=$RUN_DIR/autotune_results.pbtxt"
    CUDNN_REPAIR="rtx4090_engine20_denylist+autotune_cache_${AUTOTUNE_CACHE_SHA:0:8}"
else
    CUDNN_REPAIR="none"
fi
"$PYTHON" - <<'PY'
import jax

assert jax.local_device_count() == 4, jax.devices()
PY

EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" CONFIG="$CONFIG" "$PYTHON" - <<'PY'
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
    model_size="medium", num_prev_actions=5, critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16", attention_compute_dtype="float32",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(2, 2, 3, 3),
    token_mixer_residual_init_scale=0.1, flatten_reduce_channels=32,
    attn_latent_queries=8, aux_coef=0.0, carry_work_observation=True,
    stall_age_observation=False, reward_v2_reset_context_observation=False,
    trench_alignment_observation=True,
)
env = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
_, params = get_model_ready(jax.random.PRNGKey(0), config, env)
count = sum(int(np.asarray(x).size) for x in jax.tree.leaves(params))
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
PY

export WANDB_ENTITY WANDB_PROJECT WANDB_MODE=online WANDB_RESUME=never
export WANDB_RUN_ID="trench_align_v2_${ARM}_${BASELINES_REVISION:0:10}_s${SEED}"
export WANDB_DIR="$RUN_DIR/wandb"
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

NUM_DEVICES=4
NUM_ENVS_PER_DEVICE=512
NUM_STEPS=32
NUM_MINIBATCHES=32
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * TARGET_UPDATE))

printf '%s\n' \
    "status=RUNNING" \
    "experiment=trench_align_v2_generalist" \
    "arm=$ARM" \
    "gpu_type=$GPU_TYPE" \
    "cudnn_repair=$CUDNN_REPAIR" \
    "config=$CONFIG" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_maps_path=$BANK_MAPS_PATH" \
    "bank_dataset_size=$BANK_DATASET_SIZE" \
    "gate_semantics=v2_yaw_parallel_and_on_the_line" \
    "gate_max_offset_m=$("$PYTHON" -c 'from terra.config import EnvConfig; print(EnvConfig().trench_dig_max_offset_m)')" \
    "reward_stage=reward_v2" \
    "reward_v2_timing_variant=0" \
    "distance_protocol_id=obstacle_geodesic_8_physical_global_v1" \
    "distance_sidecar_sha256=$BANK_DISTANCE_SIDECAR_SHA" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "seed=$SEED" \
    "num_devices=$NUM_DEVICES" \
    "num_envs_per_device=$NUM_ENVS_PER_DEVICE" \
    "num_steps=$NUM_STEPS" \
    "num_minibatches=$NUM_MINIBATCHES" \
    "transitions_per_update=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS))" \
    "target_update=$TARGET_UPDATE" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

exec "$PYTHON" -u "$BASELINES_ROOT/train_mixed.py" \
    --config "$CONFIG" \
    --machine euler \
    --name "$RUN_NAME" \
    --exact_run_name \
    --seed "$SEED" \
    --num_devices "$NUM_DEVICES" \
    --num_envs_per_device "$NUM_ENVS_PER_DEVICE" \
    --num_steps "$NUM_STEPS" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --update_epochs 2 \
    --num_minibatches "$NUM_MINIBATCHES" \
    --lr 3e-4 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se_sa_xattn \
    --encoder_compute_dtype bfloat16 \
    --attention_compute_dtype float32 \
    --critic_hidden_dims 512,256 \
    --resnet_stage_channels 24,48,64,96 \
    --resnet_blocks_per_stage 2,2,3,3 \
    --token_mixer_residual_init_scale 0.1 \
    --flatten_reduce_channels 32 \
    --attn_latent_queries 8 \
    --aux_coef 0 \
    --vf_coef 2.0 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps 20000 \
    --no_value_clip \
    --carry_work_observation \
    --reward_stage reward_v2 \
    --reward_v2_timing_variant 0 \
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1 \
    --distance_sidecar_sha256 "$BANK_DISTANCE_SIDECAR_SHA" \
    --fail_on_nonfinite \
    --finite_check_interval 10 \
    --log_train_interval 10 \
    --log_eval_interval 100 \
    --eval_episodes 100 \
    --checkpoint_interval 500 \
    --cache_clear_interval 1000 \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_DIR/checkpoints"
