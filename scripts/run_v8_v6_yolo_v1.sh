#!/usr/bin/env bash
# spatial_v6_3m: the whole V6 readout redesign at once, from scratch.
#
# One arm, one GPU. Batch accounting versus the 4x4090 architecture control:
# that run used 4 devices x 512 envs x 32 steps / 32 minibatches = 512 samples
# per optimizer step and 2 epochs x 32 minibatches = 64 optimizer steps per
# update. This run keeps BOTH of those (512-sample minibatches, 64 optimizer
# steps per update) on a single device, so only the global batch shrinks to a
# quarter (524,288 -> 131,072 transitions per update). Updates are therefore
# comparable in count but not in transitions; report both.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: run_v8_v6_yolo_v1.sh BANK_ROOT RUN_NAME UPDATES RUN_ROOT" >&2
    exit 2
fi

BANK_ROOT="$1"
RUN_NAME="$2"
UPDATES="$3"
RUN_ROOT="$4"

: "${TERRA_ROOT:?set TERRA_ROOT to the immutable runtime Terra source}"
: "${PROTOCOL_TERRA_REVISION:?set the Terra revision frozen into the V8 bank}"
: "${SEED:?set the paired training seed}"
[[ "$UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "UPDATES must be a positive integer" >&2
    exit 2
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DEVICES="${NUM_DEVICES:-1}"
NUM_ENVS_PER_DEVICE="${NUM_ENVS_PER_DEVICE:-512}"
NUM_STEPS="${NUM_STEPS:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"
LOG_TRAIN_INTERVAL="${LOG_TRAIN_INTERVAL:-10}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * UPDATES))

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
    --reward_stage dense_skill \
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
    --critic_hidden_dims "512,256" \
    --resnet_stage_channels "24,48,64,96" \
    --resnet_blocks_per_stage "3,3,2,2" \
    --token_mixer_residual_init_scale 0.1 \
    --flatten_reduce_channels 32 \
    --attn_latent_queries 8 \
    --aux_coef 0.25 \
    --vf_coef 0.5 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps "$UPDATES" \
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
