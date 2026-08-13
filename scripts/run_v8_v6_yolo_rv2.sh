#!/usr/bin/env bash
# v6_3m_yolo_rv2 / v6_1_rv2: the V6 readout redesign carried on top of reward-v2.
#
# This is scripts/run_v8_r2_reward_v2.sh with eight flags changed and nothing
# else: map_encoder se_sa_xattn, blocks (3,3,2,2), token mixer residual 0.1,
# 1x1 flatten shrink to 32, 8 latent queries, aux decoder 0.25, vf_coef 0.5,
# and --action_logit_masking (D3; requires the terra runtime that exposes
# obs['action_mask'], branch experiment/v8-v6-yolo-rv2-20260810 @ 04c67bba).
# Every reward-v2 contract flag (preset, sampler profile, carry-work channel,
# distance protocol + sidecar SHA, reward stage, value-clip, minibatch shuffle,
# dtypes, critic, LR, entropy schedule) is byte-identical to the baseline, and
# so is the 4 x 512 x 32 / 32 batch shape, so updates AND transitions match.
#
# Three of the eight are env-tunable because the v6.1 arm reverts them after
# day-2 evidence: BLOCKS_PER_STAGE (v6.1 keeps the baseline's 2,2,3,3),
# AUX_COEF (v6.1 uses 0.1) and VF_COEF (empty drops the flag entirely, so
# train_mixed's default 2.0 applies, as in the baseline). The defaults below
# are the original v6_3m_yolo_rv2 values, so an unset environment reproduces
# that arm byte-for-byte.
#
# phase2 adds one material-stall-age scalar to v6.1 at u14000. It reshapes
# 4x512 to 8x256, preserving the 2,048 environments and 65,536 transitions per
# update, and restores continuous_banded_v2 without a curriculum migration.
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: run_v8_v6_yolo_rv2.sh BANK_ROOT RUN_NAME UPDATES RUN_ROOT SIDECAR_SHA256 RESUME_CHECKPOINT_OR_NONE" >&2
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
# same launcher flags rebuild it. The stall-age continuation stays on v2 and
# therefore preserves its partial sampler window without migration.
RESUME_ARGS=()
if [ "$RESUME_CHECKPOINT" != none ]; then
    test -f "$RESUME_CHECKPOINT" || {
        echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
        exit 2
    }
    RESUME_ARGS=(
        --resume_from "$RESUME_CHECKPOINT"
        --no-load-env-from-checkpoint
    )
fi
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

# D3 masking variant: 1 (default) passes --action_logit_masking and requires
# the terra runtime that exposes obs['action_mask'] (04c67bba); 0 launches the
# no-mask ablation arm on the baseline's terra (3051054b) — flag-off on the
# mask terra would still pay the mask's per-step simulation cost for nothing.
ACTION_LOGIT_MASKING="${ACTION_LOGIT_MASKING:-1}"
MASK_ARGS=()
if [ "$ACTION_LOGIT_MASKING" = 1 ]; then
    MASK_ARGS=(--action_logit_masking)
fi

STALL_AGE_OBSERVATION="${STALL_AGE_OBSERVATION:-0}"
case "$STALL_AGE_OBSERVATION" in 0|1) ;; *) echo "STALL_AGE_OBSERVATION must be 0 or 1" >&2; exit 2 ;; esac
STALL_AGE_ARGS=()
if [ "$STALL_AGE_OBSERVATION" = 1 ]; then
    STALL_AGE_ARGS=(--stall_age_observation)
fi

# The three v6.1-reverted knobs. VF_COEF="" drops --vf_coef so the trainer
# default (2.0, the baseline's) applies; any other value is passed through.
BLOCKS_PER_STAGE="${BLOCKS_PER_STAGE:-3,3,2,2}"
AUX_COEF="${AUX_COEF:-0.25}"
VF_COEF="${VF_COEF-0.5}"
VF_COEF_ARGS=()
if [ -n "$VF_COEF" ]; then
    VF_COEF_ARGS=(--vf_coef "$VF_COEF")
fi

# Sampler rule. The default and this phase2 continuation are both v2. Other
# profiles remain available only for their existing scratch launch modes.
SAMPLER_PROFILE="${SAMPLER_PROFILE:-continuous_banded_v2}"
case "$SAMPLER_PROFILE" in
    continuous_banded_v2) TRAIN_PRESET=G-V8-CONTINUOUS-V2 ;;
    continuous_banded_v3) TRAIN_PRESET=G-V8-CONTINUOUS-V3 ;;
    continuous_banded_v4) TRAIN_PRESET=G-V8-CONTINUOUS-V4 ;;
    *) echo "unsupported SAMPLER_PROFILE '$SAMPLER_PROFILE'" >&2; exit 2 ;;
esac

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
ENTROPY_SCHEDULE_STEPS="${ENTROPY_SCHEDULE_STEPS:-20000}"
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
    "${VF_COEF_ARGS[@]}" \
    "${MASK_ARGS[@]}" \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps "$ENTROPY_SCHEDULE_STEPS" \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --carry_work_observation \
    "${STALL_AGE_ARGS[@]}" \
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
    "${RESUME_ARGS[@]}"
