#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "usage: run_v8_relay_gru_v2.sh BANK_ROOT PARTIAL_RESET_ROOT RUN_NAME TARGET_UPDATES RUN_ROOT DISTANCE_SHA RESUME_CHECKPOINT_OR_NONE" >&2
    exit 2
fi

BANK_ROOT="$1"
PARTIAL_RESET_ROOT="$2"
RUN_NAME="$3"
TARGET_UPDATES="$4"
RUN_ROOT="$5"
DISTANCE_SHA="$6"
RESUME_CHECKPOINT="$7"

: "${TERRA_ROOT:?set TERRA_ROOT to the committed Terra runtime}"
: "${PROTOCOL_TERRA_REVISION:?set the immutable accepted-bank Terra revision}"
: "${SEED:?set the training seed}"
[[ "$TARGET_UPDATES" =~ ^[1-9][0-9]*$ ]]
[[ "$DISTANCE_SHA" =~ ^[0-9a-f]{64}$ ]]
test -d "$BANK_ROOT"
test -d "$PARTIAL_RESET_ROOT"

NUM_DEVICES=4
NUM_ENVS_PER_DEVICE=512
NUM_STEPS=32
NUM_MINIBATCHES=32
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * TARGET_UPDATES))

RESUME_ARGS=()
if [ "$RESUME_CHECKPOINT" != none ]; then
    test -f "$RESUME_CHECKPOINT"
    RESUME_ARGS=(--resume_from "$RESUME_CHECKPOINT" --no-load-env-from-checkpoint)
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config G-V8-CONTINUOUS-V3 \
    --machine "${MACHINE:-euler}" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-scope full \
    --accepted-bank-sampler-profile continuous_banded_v3 \
    --partial-reset-root "$PARTIAL_RESET_ROOT" \
    --reward-v2-reset-context-observation \
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
    --actor-core gru \
    --actor-gru-hidden-dim 64 \
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
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1 \
    --distance_sidecar_sha256 "$DISTANCE_SHA" \
    --reward_stage reward_v2 \
    --reward_v2_timing_variant 0 \
    --fail_on_nonfinite \
    --finite_check_interval 10 \
    --log_train_interval 10 \
    --log_eval_interval 0 \
    --checkpoint_interval 500 \
    --cache_clear_interval 1000 \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints" \
    "${RESUME_ARGS[@]}"
