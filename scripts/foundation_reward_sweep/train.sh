#!/usr/bin/env bash
# One GPU per process. Source this file to reuse FOUNDATION_TRAIN_ARGS without
# executing Python, or run `bash train.sh --print-args` for a local CLI review.
set -euo pipefail

SWEEP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_PYTHON:-python}"
MACHINE="${MACHINE:-euler}"
: "${DATASET_PATH:?Set the new bank root}"
: "${DISTANCE_SIDECAR_SHA:?Set the recorded bank distance protocol SHA}"
: "${RUN_DIR:?Set a separate directory for this arm and segment}"
: "${RUN_NAME:?Set a unique arm name}"
: "${TARGET_UPDATE:?Set the absolute target update}"
: "${EXECUTABLE_DIG_OBSERVATION:?Set 0 for legacy or 1 for executable affordance}"
SEED="${SEED:-20260907}"
INITIALIZATION="${INITIALIZATION:-resume}"
case "$INITIALIZATION" in
    scratch)
        START_UPDATE="${START_UPDATE:-0}"
        BANK_TRANSFER="${BANK_TRANSFER:-0}"
        [[ "$START_UPDATE" == 0 && -z "${RESUME_FROM:-}" && "$BANK_TRANSFER" == 0 ]] || {
            echo "scratch requires START_UPDATE=0, no RESUME_FROM and BANK_TRANSFER=0" >&2; exit 2;
        }
        ;;
    resume)
        : "${RESUME_FROM:?Set the native parent or continuation checkpoint}"
        : "${START_UPDATE:?Set the checkpoint next_update}"
        BANK_TRANSFER="${BANK_TRANSFER:-1}"
        ;;
    *) echo "INITIALIZATION must be scratch or resume" >&2; exit 2 ;;
esac
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
LATERAL_DIG_COST="${LATERAL_DIG_COST:-0}"
BASE_TRAVEL_COST="${BASE_TRAVEL_COST:-0}"
BASE_TURN_COST="${BASE_TURN_COST:-0}"
TASK_FAMILY="${TASK_FAMILY:-foundation}"
case "$TASK_FAMILY" in
    foundation)
        TRAIN_CONFIG=foundation_reward_sweep
        MAPS_PATH=train/all
        export DATASET_SIZE="${DATASET_SIZE:-256}"
        ;;
    trench)
        TRAIN_CONFIG=trench_align_v2_specialist_spec
        MAPS_PATH=train_v2_pooled_trench15
        export DATASET_SIZE="${DATASET_SIZE:-1440}"
        ;;
    *) echo "TASK_FAMILY must be foundation or trench" >&2; exit 2 ;;
esac

for value in "$START_UPDATE" "$TARGET_UPDATE" "$SEED"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "updates and seed must be nonnegative integers" >&2; exit 2; }
done
(( TARGET_UPDATE > START_UPDATE )) || { echo "TARGET_UPDATE must exceed START_UPDATE" >&2; exit 2; }
[[ "$CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "CHECKPOINT_INTERVAL must be a positive integer" >&2; exit 2; }
[[ "$DISTANCE_SIDECAR_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid distance sidecar SHA" >&2; exit 2; }
[[ "$EXECUTABLE_DIG_OBSERVATION" =~ ^[01]$ && "$BANK_TRANSFER" =~ ^[01]$ ]] || {
    echo "EXECUTABLE_DIG_OBSERVATION and BANK_TRANSFER must be 0 or 1" >&2; exit 2;
}
for value in "$LATERAL_DIG_COST" "$BASE_TRAVEL_COST" "$BASE_TURN_COST"; do
    [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || {
        echo "behavior costs must be nonnegative numbers" >&2; exit 2;
    }
done

FOUNDATION_TRAIN_ARGS=(
    --config "$TRAIN_CONFIG" --machine "$MACHINE"
    --name "$RUN_NAME" --exact_run_name --seed "$SEED"
    --num_devices 1 --num_envs_per_device 512 --num_steps 32
    --total_timesteps "$((TARGET_UPDATE * 16384))"
    --update_epochs 2 --num_minibatches 32
    --lr 3e-4 --model_size medium --model_core mlp
    --map_encoder resnet_spatial_8x8_se_sa_xattn
    --encoder_compute_dtype bfloat16 --attention_compute_dtype float32
    --critic_hidden_dims '512,256' --resnet_stage_channels '24,48,64,96'
    --resnet_blocks_per_stage '2,2,3,3' --token_mixer_residual_init_scale 0.1
    --flatten_reduce_channels 32 --attn_latent_queries 8 --aux_coef 0
    --vf_coef 2.0 --ent_schedule_start 0.15 --ent_schedule_end 0.02
    --ent_schedule_steps 20000 --no_value_clip
    --carry_work_observation --relocation_distance_observation
    --admissible_dig_observation --reward_stage reward_v2
    --reward_v2_timing_variant 0
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1
    --distance_sidecar_sha256 "$DISTANCE_SIDECAR_SHA"
    --lateral_dig_cost "$LATERAL_DIG_COST"
    --base_travel_cost "$BASE_TRAVEL_COST" --base_turn_cost "$BASE_TURN_COST"
    --fail_on_nonfinite --finite_check_interval 10
    --log_train_interval 10 --log_eval_interval 0
    --checkpoint_interval "$CHECKPOINT_INTERVAL" --cache_clear_interval 0
    --keep_checkpoint_history --checkpoint_dir "$RUN_DIR/checkpoints"
)
if [[ "$EXECUTABLE_DIG_OBSERVATION" == 1 ]]; then
    FOUNDATION_TRAIN_ARGS+=(--executable_dig_observation)
fi
if [[ "$INITIALIZATION" == resume ]]; then
    FOUNDATION_TRAIN_ARGS+=(--resume_from "$RESUME_FROM")
    if [[ "$BANK_TRANSFER" == 1 ]]; then
        FOUNDATION_TRAIN_ARGS+=(--finetune_task_bank --finetune_foundation_behavior --no-load-env-from-checkpoint)
    else
        FOUNDATION_TRAIN_ARGS+=(--load_env_from_checkpoint)
    fi
fi

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi
if [[ "${1:-}" == --print-args ]]; then
    printf '%q ' "${FOUNDATION_TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
[[ $# == 0 ]] || { echo "Usage: bash train.sh [--print-args]" >&2; exit 2; }
if [[ "$INITIALIZATION" == resume ]]; then test -r "$RESUME_FROM"; fi
test -d "$DATASET_PATH/$MAPS_PATH"
export PYTHONPATH="${TERRA_ROOT:-$(dirname "$SWEEP_REPO")/terra}:$SWEEP_REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUN_DIR/jax-cache}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR/checkpoints" "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
printf '%s\n' "task_family=$TASK_FAMILY" "initialization=$INITIALIZATION" "parent_checkpoint=${RESUME_FROM:-none}" "start_update=$START_UPDATE" \
    "target_update=$TARGET_UPDATE" "transitions_per_update=16384" \
    "additional_transitions=$(((TARGET_UPDATE - START_UPDATE) * 16384))" \
    "adam_steps_per_update=64" > "$RUN_DIR/training_budget.env"
cd "$RUN_DIR"
exec "$PYTHON" -u "${TRAIN_ENTRY:-$SWEEP_REPO/train_mixed.py}" "${FOUNDATION_TRAIN_ARGS[@]}"
