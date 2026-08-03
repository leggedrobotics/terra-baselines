#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "usage: run_v8_warm_screen.sh ARM BANK_ROOT STAGE INITIAL TEACHER RUN_NAME NUM_UPDATES" >&2
    exit 2
fi

ARM="$1"
BANK_ROOT="$2"
STAGE="$3"
INITIAL_CHECKPOINT="$4"
TEACHER_CHECKPOINT="$5"
RUN_NAME="$6"
NUM_UPDATES="$7"

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
case "$STAGE" in capability|nearby|full) ;; *) echo "invalid V8 stage '$STAGE'" >&2; exit 2 ;; esac

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable Terra source}"
: "${TERRA_REVISION:?set TERRA_REVISION to the bank Terra revision}"
: "${RUN_ROOT:?set RUN_ROOT to the scratch run directory}"
: "${SEED:?set the paired training seed}"
test -f "$INITIAL_CHECKPOINT"
if [ "$STAGE" = capability ]; then
    test "$TEACHER_CHECKPOINT" != none
    test -f "$TEACHER_CHECKPOINT"
else
    test "$TEACHER_CHECKPOINT" = none
fi
[[ "$NUM_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_UPDATES must be a positive integer" >&2
    exit 2
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-1024}"
NUM_STEPS="${NUM_STEPS:-32}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
ENT_SCHEDULE_START="${ENT_SCHEDULE_START:-0.02}"
ENT_SCHEDULE_END="${ENT_SCHEDULE_END:-0.005}"
ENT_SCHEDULE_STEPS="${ENT_SCHEDULE_STEPS:-10000}"
MACHINE="${MACHINE:-euler}"

TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * NUM_UPDATES))
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

TRAIN_ARGS=(
    --config G-V8-FIXED
    --machine "$MACHINE"
    --accepted-bank-root "$BANK_ROOT"
    --accepted-bank-stage "$STAGE"
    --terra-revision "$TERRA_REVISION"
    --name "$RUN_NAME"
    --seed "$SEED"
    --num_devices "$NUM_DEVICES"
    --num_envs_per_device "$NUM_ENVS_PER_DEVICE"
    --num_steps "$NUM_STEPS"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --update_epochs 2
    --num_minibatches 32
    --lr 3e-4
    --model_size medium
    --model_core mlp
    --map_encoder "$MAP_ENCODER"
    --encoder_compute_dtype bfloat16
    --attention_compute_dtype "$ATTENTION_DTYPE"
    --critic_hidden_dims "512,256"
    --resnet_stage_channels "24,48,64,96"
    --resnet_blocks_per_stage "2,2,3,3"
    --warm_start_from "$INITIAL_CHECKPOINT"
    --ent_schedule_start "$ENT_SCHEDULE_START"
    --ent_schedule_end "$ENT_SCHEDULE_END"
    --ent_schedule_steps "$ENT_SCHEDULE_STEPS"
    --no_value_clip
    --flat_minibatch_shuffle
    --no-load-env-from-checkpoint
    --fail_on_nonfinite
    --finite_check_interval "$FINITE_CHECK_INTERVAL"
    --log_train_interval "$LOG_TRAIN_INTERVAL"
    --log_eval_interval 0
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL"
    --keep_checkpoint_history
    --checkpoint_dir "$RUN_ROOT/checkpoints"
)
if [ "$TEACHER_CHECKPOINT" != none ]; then
    TRAIN_ARGS+=(
        --teacher_checkpoint "$TEACHER_CHECKPOINT"
        --kickstart_kl_coef 1.0
        --kickstart_kl_anneal_updates 1500
        --kickstart_value_coef 0.5
        --kickstart_value_anneal_updates 500
        --kickstart_lr_warmup_updates 100
    )
fi

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" "${TRAIN_ARGS[@]}"
