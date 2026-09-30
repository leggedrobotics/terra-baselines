# train_mixed arguments of the machine-rules fine-tune, shared by run.sbatch
# (Euler) and smoke_local.sh (local smoke). Source it with PRESET,
# DUG_CLEARANCE_M, RUN_NAME, SEED, NUM_DEVICES, ENVS_PER_DEVICE,
# TOTAL_TIMESTEPS, BANK_DISTANCE_SIDECAR_SHA, TEACHER, KL_ANNEAL_UPDATES,
# CHECKPOINT_DIR and RESUME_FROM set; callers add the logging, finite-check
# and checkpoint intervals. The PPO recipe and architecture are the GRU
# release's (scripts/euler_gru_generalist_512/run.sbatch); the rules come from
# the preset and --dug_clearance_m; the teacher arguments repeat the parent's
# schedule, which a native resume must keep (its KL weight is 0 after u10000).
TRAIN_ARGS=(
    --config "$PRESET"
    --dug_clearance_m "$DUG_CLEARANCE_M"
    --name "$RUN_NAME"
    --exact_run_name
    --seed "$SEED"
    --num_devices "$NUM_DEVICES"
    --num_envs_per_device "$ENVS_PER_DEVICE"
    --num_steps 32
    --total_timesteps "$TOTAL_TIMESTEPS"
    --update_epochs 2
    --num_minibatches 32
    --lr 3e-4
    --model_size medium
    --model_core mlp
    --actor-core gru
    --actor-gru-hidden-dim 64
    --map_encoder resnet_spatial_8x8_se_sa_xattn
    --encoder_compute_dtype bfloat16
    --attention_compute_dtype float32
    --critic_hidden_dims 512,256
    --resnet_stage_channels 24,48,64,96
    --resnet_blocks_per_stage 2,2,3,3
    --token_mixer_residual_init_scale 0.1
    --flatten_reduce_channels 32
    --attn_latent_queries 8
    --aux_coef 0
    --vf_coef 2
    --ent_schedule_start 0.02
    --ent_schedule_end 0.02
    --ent_schedule_steps 20000
    --no_value_clip
    --global_minibatch_advantage_norm
    --carry_work_observation
    --relocation_distance_observation
    --admissible_dig_observation
    --executable_dig_observation
    --time_observation_mode remaining
    --reward_stage reward_v2
    --reward_v2_timing_variant 0
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1
    --distance_sidecar_sha256 "$BANK_DISTANCE_SIDECAR_SHA"
    --lateral_dig_cost 0
    --base_travel_cost 0
    --base_turn_cost 0
    --teacher_checkpoint "$TEACHER"
    --trench_teacher_checkpoint "$TEACHER"
    --kickstart_start_update 0
    --kickstart_kl_coef 1
    --kickstart_kl_anneal_updates "$KL_ANNEAL_UPDATES"
    --kickstart_value_coef 0
    --kickstart_value_anneal_updates 0
    --kickstart_lr_warmup_updates 0
    --fail_on_nonfinite
    --log_eval_interval 0
    --cache_clear_interval 0
    --keep_checkpoint_history
    --checkpoint_dir "$CHECKPOINT_DIR"
    --resume_from "$RESUME_FROM"
    --load_env_from_checkpoint
)
