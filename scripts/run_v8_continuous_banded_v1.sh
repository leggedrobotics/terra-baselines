#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "usage: run_v8_continuous_banded_v1.sh BANK_ROOT RUN_NAME ABSOLUTE_UPDATES RUN_ROOT [RESUME_CHECKPOINT]" >&2
    exit 2
fi

BANK_ROOT="$1"
RUN_NAME="$2"
ABSOLUTE_UPDATES="$3"
RUN_ROOT="$4"
RESUME_CHECKPOINT="${5:-}"

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable Terra source}"
: "${TERRA_REVISION:?set TERRA_REVISION to the frozen V8 protocol revision}"
: "${SEED:?set the training seed}"
[[ "$ABSOLUTE_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABSOLUTE_UPDATES must be a positive integer" >&2
    exit 2
}

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

RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
    test -f "$RESUME_CHECKPOINT"
    RESUME_ARGS=(--resume_from "$RESUME_CHECKPOINT" --load_env_from_checkpoint)
fi

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config G-V8-CONTINUOUS \
    --machine "${MACHINE:-euler}" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-scope full \
    --accepted-bank-sampler-profile continuous_banded_v1 \
    --terra-revision "$TERRA_REVISION" \
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
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps "$ABSOLUTE_UPDATES" \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval "$FINITE_CHECK_INTERVAL" \
    --log_train_interval "$LOG_TRAIN_INTERVAL" \
    --log_eval_interval 0 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL" \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints" \
    "${RESUME_ARGS[@]}"
