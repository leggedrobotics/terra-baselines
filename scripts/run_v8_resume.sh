#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: run_v8_resume.sh ARM BANK_ROOT RESUME_CHECKPOINT RUN_NAME" >&2
    exit 2
fi

ARM="$1"
BANK_ROOT="$2"
RESUME_CHECKPOINT="$3"
RUN_NAME="$4"
ABSOLUTE_UPDATES=80000

case "$ARM" in
    G-DEEP-V8-DENSE-WARM)
        MAP_ENCODER=resnet_spatial_8x8_se
        ATTENTION_DTYPE=encoder
        ;;
    G-DEEP-XATTN-V8-DENSE-WARM)
        MAP_ENCODER=resnet_spatial_8x8_se_xattn
        ATTENTION_DTYPE=float32
        ;;
    *) echo "unknown V8 arm '$ARM'" >&2; exit 2 ;;
esac

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable Terra source}"
: "${TERRA_REVISION:?set TERRA_REVISION to the bank Terra revision}"
: "${RUN_ROOT:?set RUN_ROOT to the scratch run directory}"
: "${SEED:?set the original full-stage training seed}"
test -f "$RESUME_CHECKPOINT"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-1024}"
NUM_STEPS="${NUM_STEPS:-32}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
MACHINE="${MACHINE:-euler}"

TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * ABSOLUTE_UPDATES))
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config G-V8-FIXED \
    --machine "$MACHINE" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-stage full \
    --terra-revision "$TERRA_REVISION" \
    --name "$RUN_NAME" \
    --exact_run_name \
    --seed "$SEED" \
    --num_devices "$NUM_DEVICES" \
    --num_envs_per_device "$NUM_ENVS_PER_DEVICE" \
    --num_steps "$NUM_STEPS" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --update_epochs 2 \
    --num_minibatches 32 \
    --lr 3e-4 \
    --model_size medium \
    --model_core mlp \
    --map_encoder "$MAP_ENCODER" \
    --encoder_compute_dtype bfloat16 \
    --attention_compute_dtype "$ATTENTION_DTYPE" \
    --critic_hidden_dims "512,256" \
    --resnet_stage_channels "24,48,64,96" \
    --resnet_blocks_per_stage "2,2,3,3" \
    --resume_from "$RESUME_CHECKPOINT" \
    --load_env_from_checkpoint \
    --ent_schedule_start 0.02 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 10000 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval "$FINITE_CHECK_INTERVAL" \
    --log_train_interval "$LOG_TRAIN_INTERVAL" \
    --log_eval_interval 0 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL" \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints"
