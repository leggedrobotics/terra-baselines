#!/usr/bin/env bash
# Continue the v6.1 reward-v2 policy with one material-stall-age scalar and the
# final family-free continuous_banded_v3 curriculum. The launcher supplies the
# selected v6.1 architecture, reshapes 4x512 to 8x256 without changing the
# 2,048 environments or 65,536 transitions/update, and gives this runner a
# checkpoint already materialized onto the final observation and sampler.
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: run_v8_v6_yolo_rv2.sh BANK_ROOT RUN_NAME UPDATES RUN_ROOT SIDECAR_SHA256 RESUME_CHECKPOINT" >&2
    exit 2
fi

BANK_ROOT="$1"
RUN_NAME="$2"
UPDATES="$3"
RUN_ROOT="$4"
SIDECAR_SHA256="$5"
RESUME_CHECKPOINT="$6"
# A warm resume: params, optimizer clock and sampler history carry over; the
# environments reset fresh. The env axis is not restored either way (the
# checkpoint stores env_config, not env state), and phase2 changes the env
# count, so the checkpoint's env_config is deliberately NOT reapplied — the
# fixed flags below rebuild it. The checkpoint materializer has already
# performed the sampler migration.
test "$RESUME_CHECKPOINT" != none || {
    echo "the continuation requires a prepared resume checkpoint" >&2
    exit 2
}
test -f "$RESUME_CHECKPOINT" || {
    echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
}
[[ "$UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "UPDATES must be positive" >&2
    exit 2
}
[[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "SIDECAR_SHA256 must be lowercase SHA-256" >&2
    exit 2
}

: "${TERRA_ROOT:?set TERRA_ROOT to the committed R2 Terra source}"
: "${PROTOCOL_TERRA_REVISION:?set the immutable V8 bank protocol revision}"
: "${SEED:?set the R2 system seed}"

# One recipe only. The checkpoint materializer has already added the stall-age
# weights and migrated the source sampler, so the runner never selects a mode.
NUM_DEVICES=8
NUM_ENVS_PER_DEVICE=256
NUM_STEPS=32
NUM_MINIBATCHES=32
BLOCKS_PER_STAGE=2,2,3,3
AUX_COEF=0
VF_COEF=2.0
SAMPLER_PROFILE=continuous_banded_v3
TRAIN_PRESET=G-V8-CONTINUOUS-V3
CHECKPOINT_INTERVAL=500
FINITE_CHECK_INTERVAL=10
LOG_TRAIN_INTERVAL=10
CACHE_CLEAR_INTERVAL=1000
ENTROPY_SCHEDULE_STEPS=20000

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TOTAL_TIMESTEPS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * UPDATES))

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export PYTHONPATH="$TERRA_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

exec "$PYTHON_BIN" -u "$REPO/train_mixed.py" \
    --config "$TRAIN_PRESET" \
    --machine "${MACHINE:-euler}" \
    --accepted-bank-root "$BANK_ROOT" \
    --accepted-bank-scope full \
    --accepted-bank-sampler-profile "$SAMPLER_PROFILE" \
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
    --map_encoder resnet_spatial_8x8_se_sa_xattn \
    --encoder_compute_dtype bfloat16 \
    --attention_compute_dtype float32 \
    --critic_hidden_dims 512,256 \
    --resnet_stage_channels 24,48,64,96 \
    --resnet_blocks_per_stage "$BLOCKS_PER_STAGE" \
    --token_mixer_residual_init_scale 0.1 \
    --flatten_reduce_channels 32 \
    --attn_latent_queries 8 \
    --aux_coef "$AUX_COEF" \
    --vf_coef "$VF_COEF" \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps "$ENTROPY_SCHEDULE_STEPS" \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --carry_work_observation \
    --stall_age_observation \
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1 \
    --distance_sidecar_sha256 "$SIDECAR_SHA256" \
    --reward_stage reward_v2 \
    --fail_on_nonfinite \
    --finite_check_interval "$FINITE_CHECK_INTERVAL" \
    --log_train_interval "$LOG_TRAIN_INTERVAL" \
    --log_eval_interval 0 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --cache_clear_interval "$CACHE_CLEAR_INTERVAL" \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_ROOT/checkpoints" \
    --resume_from "$RESUME_CHECKPOINT" \
    --no-load-env-from-checkpoint
