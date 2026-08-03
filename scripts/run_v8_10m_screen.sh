#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "usage: run_v8_10m_screen.sh ARM BANK_ROOT INITIAL TEACHER RUN_NAME NUM_UPDATES RUN_ROOT" >&2
    exit 2
fi

ARM="$1"
BANK_ROOT="$2"
INITIAL_CHECKPOINT="$3"
TEACHER_CHECKPOINT="$4"
RUN_NAME="$5"
NUM_UPDATES="$6"
RUN_ROOT="$7"

case "$ARM" in
    G-V8-XATTN-REWARM-CONTROL)
        STAGE_CHANNELS=24,48,64,96
        ;;
    G-V8-10M-XATTN-WARM)
        STAGE_CHANNELS=64,128,192,256
        ;;
    *) echo "unknown V8 scale arm '$ARM'" >&2; exit 2 ;;
esac

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable Terra source}"
: "${TERRA_REVISION:?set TERRA_REVISION to the frozen V8 Terra revision}"
: "${SEED:?set the paired training seed}"
test -f "$INITIAL_CHECKPOINT"
test -f "$TEACHER_CHECKPOINT"
[[ "$NUM_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_UPDATES must be a positive integer" >&2
    exit 2
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-1024}"
NUM_STEPS="${NUM_STEPS:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * NUM_UPDATES))

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config G-V8-FIXED \
    --machine "${MACHINE:-euler}" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-stage full \
    --terra-revision "$TERRA_REVISION" \
    --name "$RUN_NAME" \
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
    --resnet_stage_channels "$STAGE_CHANNELS" \
    --resnet_blocks_per_stage 2,2,3,3 \
    --warm_start_from "$INITIAL_CHECKPOINT" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --kickstart_kl_coef 1.0 \
    --kickstart_kl_anneal_updates 1500 \
    --kickstart_value_coef 0.5 \
    --kickstart_value_anneal_updates 500 \
    --kickstart_lr_warmup_updates 100 \
    --ent_schedule_start 0.02 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 10000 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --no-load-env-from-checkpoint \
    --fail_on_nonfinite \
    --finite_check_interval "$FINITE_CHECK_INTERVAL" \
    --log_train_interval "$LOG_TRAIN_INTERVAL" \
    --log_eval_interval 0 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL" \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints"
