#!/usr/bin/env bash
# Same-bank native continuation of the mixed generalist with the 2x costs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${TERRA_ROOT:?}" "${DATASET_PATH:?}" "${RUN_DIR:?}" "${RESUME_FROM:?}"
: "${START_UPDATE:?}" "${TARGET_UPDATE:?}" "${RUN_NAME:?}"
FIRST_SEGMENT="${FIRST_SEGMENT:-0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
for value in "$START_UPDATE" "$TARGET_UPDATE" "$CHECKPOINT_INTERVAL"; do
    [[ "$value" =~ ^[0-9]+$ ]] || exit 2
done
(( TARGET_UPDATE > START_UPDATE && CHECKPOINT_INTERVAL > 0 )) || exit 2
[[ "$FIRST_SEGMENT" =~ ^[01]$ ]] || exit 2
if [[ "$FIRST_SEGMENT" == 1 ]]; then (( START_UPDATE == 5000 )) || exit 2; fi
export DATASET_SIZE=3840
ARGS=(
    --config trench_align_v2_generalist_gen --machine "${MACHINE:-local}"
    --name "$RUN_NAME" --exact_run_name --seed "${SEED:-20260909}"
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
    --admissible_dig_observation --executable_dig_observation
    --reward_stage reward_v2 --reward_v2_timing_variant 0
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1
    --distance_sidecar_sha256 f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
    --lateral_dig_cost 0.5 --base_travel_cost 0.01 --base_turn_cost 0.04
    --resume_from "$RESUME_FROM" --load_env_from_checkpoint
    --fail_on_nonfinite --finite_check_interval "${FINITE_CHECK_INTERVAL:-10}"
    --log_train_interval 10 --log_eval_interval 100 --eval_episodes 100
    --checkpoint_interval "$CHECKPOINT_INTERVAL" --cache_clear_interval 0
    --keep_checkpoint_history --checkpoint_dir "$RUN_DIR/checkpoints"
)
if [[ "$FIRST_SEGMENT" == 1 ]]; then ARGS+=(--finetune_foundation_behavior); fi
if [[ "${1:-}" == --print-args ]]; then printf '%q ' "${ARGS[@]}"; printf '\n'; exit 0; fi
[[ $# == 0 ]] || exit 2
test -r "$RESUME_FROM"
test -d "$DATASET_PATH/train_v2_pooled_generalist"
export PYTHONPATH="$TERRA_ROOT:$REPO"
export WANDB_DIR="$RUN_DIR/wandb" WANDB_MODE="${WANDB_MODE:-disabled}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUN_DIR/jax-cache}"
export JAX_ENABLE_COMPILATION_CACHE=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p "$RUN_DIR/checkpoints" "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
printf '%s\n' "parent_checkpoint=$RESUME_FROM" "start_update=$START_UPDATE" \
    "target_update=$TARGET_UPDATE" "transitions_per_update=16384" \
    "additional_transitions=$(((TARGET_UPDATE - START_UPDATE) * 16384))" \
    "adam_steps_per_update=64" "same_bank_native_resume=true" \
    "first_behavior_segment=$FIRST_SEGMENT" > "$RUN_DIR/training_budget.env"
cd "$RUN_DIR"
exec "${TERRA_PYTHON:-python}" -u "$REPO/train_mixed.py" "${ARGS[@]}"
