#!/usr/bin/env bash
# Run one P5b parameters-only warm-start treatment.
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: run_p5b_warm_screen.sh ARM BANK_ROOT INITIAL_CHECKPOINT TEACHER_CHECKPOINT RUN_NAME NUM_UPDATES" >&2
    exit 2
fi
ARM="$1"
BANK_ROOT="$2"
INITIAL_CHECKPOINT="$3"
TEACHER_CHECKPOINT="$4"
RUN_NAME="$5"
NUM_UPDATES="$6"

case "$ARM" in
    G-MEDIUM-ADAPTIVE-WARM|G-MEDIUM-UNIFORM-WARM|F-MEDIUM-UNIFORM-WARM|T-MEDIUM-UNIFORM-WARM)
        ARCHITECTURE_ARGS=()
        ;;
    G-DEEP-ADAPTIVE-WARM|G-DEEP-UNIFORM-WARM)
        ARCHITECTURE_ARGS=(
            --resnet_stage_channels "24,48,64,96"
            --resnet_blocks_per_stage "2,2,3,3"
        )
        ;;
    *) echo "unknown P5b arm '$ARM'" >&2; exit 2 ;;
esac

: "${TERRA_ROOT:?set TERRA_ROOT to the archived Terra source}"
: "${TERRA_REVISION:?set TERRA_REVISION to the archived Terra revision}"
: "${RUN_ROOT:?set RUN_ROOT to the scratch run directory}"
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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
MACHINE="${MACHINE:-euler}"
ENT_SCHEDULE_START="${ENT_SCHEDULE_START:-0.15}"
ENT_SCHEDULE_END="${ENT_SCHEDULE_END:-0.005}"
ENT_SCHEDULE_STEPS="${ENT_SCHEDULE_STEPS:-7600}"

for value in \
    "$NUM_DEVICES" "$NUM_ENVS_PER_DEVICE" "$NUM_STEPS" \
    "$CHECKPOINT_INTERVAL" "$FINITE_CHECK_INTERVAL" \
    "$LOG_TRAIN_INTERVAL" "$CACHE_CLEAR_INTERVAL"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "operational counts must be nonnegative integers" >&2
        exit 2
    }
done
[[ "$ENT_SCHEDULE_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ENT_SCHEDULE_STEPS must be a positive integer" >&2
    exit 2
}
test "$NUM_DEVICES" -gt 0
test "$NUM_ENVS_PER_DEVICE" -gt 0
test "$NUM_STEPS" -gt 0
test "$CHECKPOINT_INTERVAL" -gt 0
case "$MACHINE" in local|euler) ;; *) echo "unsupported MACHINE=$MACHINE" >&2; exit 2 ;; esac

TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * NUM_UPDATES))
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config "$ARM" \
    --machine "$MACHINE" \
    --accepted-bank-root "$BANK_ROOT" \
    --terra-revision "$TERRA_REVISION" \
    --name "$RUN_NAME" \
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
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    "${ARCHITECTURE_ARGS[@]}" \
    --warm_start_from "$INITIAL_CHECKPOINT" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --kickstart_kl_coef 1.0 \
    --kickstart_kl_anneal_updates 1500 \
    --kickstart_value_coef 0.5 \
    --kickstart_value_anneal_updates 500 \
    --kickstart_lr_warmup_updates 100 \
    --ent_schedule_start "$ENT_SCHEDULE_START" \
    --ent_schedule_end "$ENT_SCHEDULE_END" \
    --ent_schedule_steps "$ENT_SCHEDULE_STEPS" \
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
