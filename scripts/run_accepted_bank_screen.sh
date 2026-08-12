#!/usr/bin/env bash
# Run one accepted-bank screen locally or inside an existing allocation.
set -euo pipefail

ARM="${1:?usage: run_accepted_bank_screen.sh ARM BANK_ROOT RUN_NAME NUM_UPDATES}"
BANK_ROOT="${2:?missing accepted bank root}"
RUN_NAME="${3:?missing run name}"
NUM_UPDATES="${4:?missing PPO update count}"
if [ "$#" -ne 4 ]; then
    echo "usage: run_accepted_bank_screen.sh ARM BANK_ROOT RUN_NAME NUM_UPDATES" >&2
    echo "training arguments are frozen; use the documented operational environment variables" >&2
    exit 2
fi

case "$ARM" in
    F-ANCHOR|F-SPECIALIST|T-ANCHOR|T-SPECIALIST|G-UNIFORM|G-ADAPTIVE) ;;
    *) echo "unknown arm '$ARM'" >&2; exit 2 ;;
esac

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable Terra source archive}"
: "${TERRA_REVISION:?set TERRA_REVISION to the immutable Terra revision in the source manifest}"
: "${RUN_ROOT:?set RUN_ROOT to the run-artifact directory}"
: "${SEED:?set SEED explicitly; paired G arms must use the same seed}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-1}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-1024}"
NUM_STEPS="${NUM_STEPS:-32}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
MACHINE="${MACHINE:-local}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * NUM_UPDATES))

case "$MACHINE" in
    local|euler) ;;
    *) echo "MACHINE must be local or euler, got '$MACHINE'" >&2; exit 2 ;;
esac
for value in \
    "$NUM_DEVICES" "$NUM_ENVS_PER_DEVICE" "$NUM_STEPS" \
    "$NUM_UPDATES" "$FINITE_CHECK_INTERVAL" "$CHECKPOINT_INTERVAL" \
    "$LOG_TRAIN_INTERVAL" "$CACHE_CLEAR_INTERVAL"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "operational integer arguments must be nonnegative integers" >&2
        exit 2
    }
done
if [ "$NUM_DEVICES" -eq 0 ] || [ "$NUM_ENVS_PER_DEVICE" -eq 0 ] \
    || [ "$NUM_STEPS" -eq 0 ] || [ "$NUM_UPDATES" -eq 0 ] \
    || [ "$CHECKPOINT_INTERVAL" -eq 0 ]; then
    echo "device, environment, step, update, and checkpoint counts must be positive" >&2
    exit 2
fi

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
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 7600 \
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
