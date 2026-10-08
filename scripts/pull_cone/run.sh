#!/usr/bin/env bash
# Scratch student under the +-30 degree pull cone, turn-keeping moves and the
# native dump observation (preset gru_generalist_512_pull_cone). GRU110000
# guides only compatible bulk foundation states (no precision lanes, no
# trenches) through its own legacy observation view, fading out over 3000
# updates, as in the scratch-teacher campaign.
set -euo pipefail
PHASE="${1:?smoke or production}"
[[ "$PHASE" == smoke || "$PHASE" == production ]]
: "${TERRA_RUN_DIR:?}" "${TERRA_EXPERIMENT_INPUTS:?}" "${DATASET_PATH:?}"
BASELINES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAIR_ROOT="$(dirname "$BASELINES_ROOT")"
export PYTHONPATH="$PAIR_ROOT/terra:$BASELINES_ROOT:$TERRA_EXPERIMENT_INPUTS/geometry_dependencies"
export DATASET_SIZE=20480 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
export WANDB_DIR="$TERRA_RUN_DIR/wandb"
export JAX_COMPILATION_CACHE_DIR="$TERRA_RUN_DIR/jax-cache"
export JAX_ENABLE_COMPILATION_CACHE=true
mkdir -p "$TERRA_RUN_DIR/checkpoints" "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$TERRA_RUN_DIR"
NUM_DEVICES="${NUM_DEVICES:-4}"
python -u "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices "$NUM_DEVICES"
PARENT="$TERRA_EXPERIMENT_INPUTS/gru_rules_ft_c057_s20260930_update_110000.pkl"
ARGS=(
    --config gru_generalist_512_pull_cone
    --name "gru_pull_cone_teacher_s20261006_${PHASE}" --exact_run_name --seed 20261006
    --machine daint --num_devices "$NUM_DEVICES" --num_envs_per_device 512
    --num_steps 32 --update_epochs 2 --num_minibatches 32 --total_timesteps 50000000000
    --lr 3e-4 --model_size medium --model_core mlp --actor-core gru --actor-gru-hidden-dim 64
    --map_encoder resnet_spatial_8x8_se_sa_xattn
    --encoder_compute_dtype bfloat16 --attention_compute_dtype float32
    --critic_hidden_dims 512,256 --resnet_stage_channels 24,48,64,96
    --resnet_blocks_per_stage 2,2,3,3 --token_mixer_residual_init_scale 0.1
    --flatten_reduce_channels 32 --attn_latent_queries 8
    --aux_coef 0 --vf_coef 2 --ent_schedule_start 0.02 --ent_schedule_end 0.02
    --ent_schedule_steps 20000 --no_value_clip --global_minibatch_advantage_norm
    --carry_work_observation --relocation_distance_observation
    --admissible_dig_observation --executable_dig_observation
    --time_observation_mode remaining --reward_stage reward_v2 --reward_v2_timing_variant 0
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1
    --distance_sidecar_sha256 d9a8c9e61346e319b3ffead150ca44c7e85d016cb5853997cf1ee721286d4506
    --lateral_dig_cost 0 --base_travel_cost 0 --base_turn_cost 0
    --dug_clearance_m 0.57 --pull_direction_alignment --dig_pull_min_length_m 2.5
    --precision_required_band_observation --precision_episode_fraction 0.5
    --pull_direction_training_slots "${TERRA_TRAINING_SLOTS:-$TERRA_EXPERIMENT_INPUTS/training_slots.json}"
    --teacher_checkpoint "$PARENT" --recurrent_teacher --teacher_bulk_compatibility
    --kickstart_start_update 0 --kickstart_kl_coef 1 --kickstart_kl_anneal_updates 3000
    --kickstart_value_coef 0 --kickstart_value_anneal_updates 0 --kickstart_lr_warmup_updates 0
    --no-load-env-from-checkpoint
    --fail_on_nonfinite --finite_check_interval 100
    --log_train_interval 1 --log_eval_interval 0 --cache_clear_interval 0
    --checkpoint_interval 100 --keep_checkpoint_history
    --checkpoint_dir "$TERRA_RUN_DIR/checkpoints"
)
if [[ "$PHASE" == smoke ]]; then
    export WANDB_MODE=disabled
    ARGS+=(--num_envs_per_device 8 --num_steps 2 --update_epochs 1 --num_minibatches 2
           --total_timesteps "$((NUM_DEVICES * 16))" --finite_check_interval 1 --checkpoint_interval 1)
fi
printf '%s\n' "${ARGS[@]}" > "$TERRA_RUN_DIR/arguments.txt"
python -u "$BASELINES_ROOT/train_mixed.py" "${ARGS[@]}"
if [[ "$PHASE" == smoke ]]; then
    JAX_PLATFORMS=cpu python "$BASELINES_ROOT/scripts/pull_cone/verify_smoke.py" \
        "$TERRA_RUN_DIR/checkpoints/gru_pull_cone_teacher_s20261006_smoke_FINAL.pkl" \
        "$TERRA_RUN_DIR/verification.json"
fi
