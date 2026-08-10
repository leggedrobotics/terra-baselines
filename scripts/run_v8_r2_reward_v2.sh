#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "usage: run_v8_r2_reward_v2.sh control|reward_v2 BANK_ROOT PREPARED_FORK RUN_NAME ABSOLUTE_UPDATES RUN_ROOT SIDECAR_SHA256" >&2
    exit 2
fi

ARM="$1"
BANK_ROOT="$2"
PREPARED_FORK="$3"
RUN_NAME="$4"
ABSOLUTE_UPDATES="$5"
RUN_ROOT="$6"
SIDECAR_SHA256="$7"
case "$ARM" in
    control)
        REWARD_STAGE=dense_skill
        DISTANCE_PROTOCOL_ID=legacy_dataset_distance
        ;;
    reward_v2)
        REWARD_STAGE=reward_v2
        DISTANCE_PROTOCOL_ID=obstacle_geodesic_8_physical_global_v1
        ;;
    *) echo "unsupported R2 arm '$ARM'" >&2; exit 2 ;;
esac
[[ "$ABSOLUTE_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABSOLUTE_UPDATES must be positive" >&2
    exit 2
}
[[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "SIDECAR_SHA256 must be lowercase SHA-256" >&2
    exit 2
}
test -f "$PREPARED_FORK"

: "${TERRA_ROOT:?set TERRA_ROOT to the committed R2 Terra source}"
: "${PROTOCOL_TERRA_REVISION:?set the immutable V8 bank protocol revision}"
: "${SEED:?set the matched R2 seed}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-512}"
NUM_STEPS="${NUM_STEPS:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * ABSOLUTE_UPDATES))

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config G-V8-CONTINUOUS-V2 \
    --machine "${MACHINE:-euler}" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-scope full \
    --accepted-bank-sampler-profile continuous_banded_v2 \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
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
    --map_encoder resnet_spatial_8x8_se_xattn \
    --encoder_compute_dtype bfloat16 \
    --attention_compute_dtype float32 \
    --critic_hidden_dims 512,256 \
    --resnet_stage_channels 24,48,64,96 \
    --resnet_blocks_per_stage 2,2,3,3 \
    --ent_schedule_start 0.02 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps 1 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --prepared_fork_from "$PREPARED_FORK" \
    --load_env_from_checkpoint \
    --carry_work_observation \
    --distance_protocol_id "$DISTANCE_PROTOCOL_ID" \
    --distance_sidecar_sha256 "$SIDECAR_SHA256" \
    --reward_stage "$REWARD_STAGE" \
    --kickstart_lr_warmup_updates 100 \
    --fail_on_nonfinite \
    --finite_check_interval "$FINITE_CHECK_INTERVAL" \
    --log_train_interval "$LOG_TRAIN_INTERVAL" \
    --log_eval_interval 0 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL" \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints"
