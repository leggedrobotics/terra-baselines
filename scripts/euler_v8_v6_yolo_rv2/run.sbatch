#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

# One phase and one arm. The checkpoint was prepared offline with zero stall-age
# embeddings and a v2 -> final-v3 sampler migration, including clearing its
# 50-update partial window. Runtime restore is therefore a native v3 resume.
test "${PHASE:?}" = phase2
test "${ARM_NAME:?}" = v6_1_rv2_stall_age_v3
test "${ACTION_LOGIT_MASKING:?}" = 0
test "${SAMPLER_PROFILE:?}" = continuous_banded_v3
EXPECTED_RUNTIME_TERRA_REVISION=c2d2a94a124759e9f21c2b37930f717e299f0c46
RUN_PREFIX=v8_v61_stall_age_v3
BLOCKS_PER_STAGE=2,2,3,3
AUX_COEF=0
EXPECTED_AUX_LEAVES=0
VF_COEF=
CONTRACT_VF_COEF=2.0
EXPECTED_PARAMETERS=2304829
EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK=2304813
STALL_AGE_OBSERVATION=1
BUNDLED_CHANGES=material_stall_age_scalar+continuous_banded_v3_open80_mastered20_cap015
EXPERIMENT_NAME=v8_v61_stall_age_v3_continuation
SCREEN_SEMANTICS=practical_combined_observation_and_curriculum_continuation
INHERITED_CONTRACT=reward_v2+global_distance_ledger+carry_input+v6_1_architecture
EXPECTED_HORIZON=450
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4

for REQUIRED in TERRA_EULER_USER TERRA_EULER_HOME_ROOT VENV RUN_BASE \
    WANDB_ENTITY WANDB_PROJECT \
    PHASE BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT \
    RUNTIME_TERRA_REVISION R2_HORIZON SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_TREE_SHA BANK_RELEASE_ID DISTANCE_ARTIFACT_SHA \
    MATERIALIZATION_RECEIPT MATERIALIZATION_RECEIPT_SHA \
    STATIC_RECEIPT_MANIFEST STATIC_RECEIPT_MANIFEST_SHA D4A_RECEIPT \
    D4A_RECEIPT_SHA D4A_MANIFEST D4A_MANIFEST_SHA \
    ARM_NAME ACTION_LOGIT_MASKING RESUME_SOURCE_CHECKPOINT RESUME_SOURCE_SHA \
    RESUME_CHECKPOINT RESUME_CHECKPOINT_SHA \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
TERRA_EULER_USER="${TERRA_EULER_USER:?}"
TERRA_EULER_HOME_ROOT="${TERRA_EULER_HOME_ROOT:?}"
VENV="${VENV:?}"
RUN_BASE="${RUN_BASE:?}"
WANDB_ENTITY="${WANDB_ENTITY:?}"
WANDB_PROJECT="${WANDB_PROJECT:?}"
test "$SEED" = 20260807
# The stall-age runtime is based on 46b5a1dd and keeps timing variant 0. The R2
# receipt below catches any accidental reward-timing change.
test "$RUNTIME_TERRA_REVISION" = "$EXPECTED_RUNTIME_TERRA_REVISION"
test "$R2_HORIZON" -eq "$EXPECTED_HORIZON"
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
for REMOTE_PATH in "$TERRA_EULER_HOME_ROOT" "$VENV" "$RUN_BASE"; do
    case "$REMOTE_PATH" in /cluster/*) ;; *) exit 2 ;; esac
    case "$REMOTE_PATH" in *[!a-zA-Z0-9_./-]*) exit 2 ;; esac
done
for WANDB_NAME in "$WANDB_ENTITY" "$WANDB_PROJECT"; do
    case "$WANDB_NAME" in ''|*[!a-zA-Z0-9_.-]*) exit 2 ;; esac
done

# Phase2 reshapes 4x512 to 8x256, preserving 2,048 total envs,
# 65,536 transitions/update, 2,048 transitions/minibatch, and every absolute
# optimizer/schedule clock.
SOURCE_UPDATE=14000
TARGET_UPDATE=40000
test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h
START_UPDATE="$SOURCE_UPDATE"
UPDATES="$TARGET_UPDATE"
NUM_DEVICES=8
NUM_ENVS_PER_DEVICE=256
EXPECTED_CPUS=8
CHECKPOINT_INTERVAL=500
EVAL_INTERVAL=1000
LOG_TRAIN_INTERVAL=10
FINITE_CHECK_INTERVAL=10
CACHE_CLEAR_INTERVAL=1000
[[ "$RESUME_CHECKPOINT" = /* ]]
[[ "$RESUME_CHECKPOINT_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$RESUME_SOURCE_CHECKPOINT" = /* ]]
[[ "$RESUME_SOURCE_SHA" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')" = "$RESUME_CHECKPOINT_SHA"
test "$(sha256sum "$RESUME_SOURCE_CHECKPOINT" | awk '{print $1}')" = "$RESUME_SOURCE_SHA"

RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/s$SEED/$ARM_NAME"
test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
for SPEC in \
    "$BANK_ARCHIVE:$BANK_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA" \
    "$STATIC_RECEIPT_MANIFEST:$STATIC_RECEIPT_MANIFEST_SHA" \
    "$D4A_RECEIPT:$D4A_RECEIPT_SHA" \
    "$D4A_MANIFEST:$D4A_MANIFEST_SHA"; do
    FILE="${SPEC%:*}" EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$FILE" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "${SLURM_CPUS_PER_TASK:-}" = "$EXPECTED_CPUS"
test -d "$RUN_DIR" -a ! -e "$RUN_DIR/run_contract.env"

HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
module load stack/2024-06 cuda/12.1.1
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
module load eth_proxy
"$PYTHON" -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for LIB_GLOB in \
    "$SITE_PACKAGES"/nvidia/cudnn/lib/libcudnn.so* \
    "$SITE_PACKAGES"/nvidia/cuda_cupti/lib/libcupti.so* \
    "$SITE_PACKAGES"/nvidia/cublas/lib/libcublas.so* \
    "$SITE_PACKAGES"/nvidia/cuda_nvrtc/lib/libnvrtc.so* \
    "$SITE_PACKAGES"/nvidia/nccl/lib/libnccl.so*; do
    compgen -G "$LIB_GLOB" >/dev/null
done

# One device per GPU. The failed 8x3090/512 smoke motivated the treatment's
# batch-preserving 8x4090/256 shape. The full first PPO update below remains the
# decisive cuDNN execution smoke; this cheap preflight catches missing CUDA or
# NCCL before W&B/training initialization.
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq "$NUM_DEVICES"
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done
# cuda first keeps every default on GPU; cpu must be reachable for the
# host-side model.init (see utils/models.py get_model_ready).
export JAX_PLATFORMS=cuda,cpu
NUM_DEVICES="$NUM_DEVICES" "$PYTHON" - <<'PY'
import os

import jax
import jax.numpy as jnp
from jax import lax

devices = int(os.environ["NUM_DEVICES"])
assert len(jax.devices()) == devices
def loss(x, kernel):
    y = lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return lax.psum(jnp.mean(y * y), "devices")
step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((devices, 1, 16, 16, 4), dtype=jnp.float32)
k = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, k)
assert gradient.shape == x.shape and jnp.all(jnp.isfinite(gradient))
print(jax.devices(), float(jnp.linalg.norm(gradient)))
PY
"$PYTHON" - <<'PY'
from terra.config import (
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    REWARD_V2_SHAPING_WEIGHT,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SUCCESS_BONUS,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.state import CORRECTED_DENSE_CONTRACT

assert int(RewardStage.REWARD_V2) == 3
assert REWARD_V2_PROTOCOL_ID == "material_potential_v2"
assert REWARD_V2_DISTANCE_PROTOCOL_ID == "obstacle_geodesic_8_physical_global_v1"
assert CORRECTED_DENSE_CONTRACT == "exact_visible_dump_v1"
assert (
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_SUCCESS_BONUS,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SHAPING_WEIGHT,
) == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)
PY
# Assert the one supported model shape, including its zero-initialized
# stall-age embeddings and carry-work input.
BLOCKS_PER_STAGE="$BLOCKS_PER_STAGE" AUX_COEF="$AUX_COEF" \
    STALL_AGE_OBSERVATION="$STALL_AGE_OBSERVATION" \
    EXPECTED_AUX_LEAVES="$EXPECTED_AUX_LEAVES" \
    EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" \
    EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK="$EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK" \
    "$PYTHON" - <<'PY'
import os
from types import SimpleNamespace

import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import get_model_ready

BLOCKS = tuple(int(x) for x in os.environ["BLOCKS_PER_STAGE"].split(","))
AUX_COEF = float(os.environ["AUX_COEF"])
EXPECTED_AUX_LEAVES = int(os.environ["EXPECTED_AUX_LEAVES"])
EXPECTED = int(os.environ["EXPECTED_PARAMETERS"])
EXPECTED_WITHOUT_CARRY_WORK = int(os.environ["EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK"])
STALL_AGE_OBSERVATION = bool(int(os.environ["STALL_AGE_OBSERVATION"]))


class Config(dict):
    __getattr__ = dict.__getitem__


def config(**extra):
    return Config(
        clip_action_maps=True,
        loaded_max=100,
        local_map_normalization_bounds=(-16, 16),
        maps_net_normalization_bounds=(-10, 10),
        model_core="mlp",
        model_size="medium",
        num_prev_actions=5,
        critic_hidden_dims=(512, 256),
        encoder_compute_dtype="bfloat16",
        attention_compute_dtype="float32",
        map_encoder="resnet_spatial_8x8_se_sa_xattn",
        resnet_stage_channels=(24, 48, 64, 96),
        resnet_blocks_per_stage=BLOCKS,
        token_mixer_residual_init_scale=0.1,
        flatten_reduce_channels=32,
        attn_latent_queries=8,
        aux_coef=AUX_COEF,
        **extra,
    )


environment = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
assert environment.batch_cfg.agent.num_state_obs == 9
counts = {}
for carry in (False, True):
    _, params = get_model_ready(
        jax.random.PRNGKey(0),
        config(
            carry_work_observation=carry,
            stall_age_observation=STALL_AGE_OBSERVATION,
        ),
        environment,
    )
    counts[carry] = sum(
        int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)
    )
    aux = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]
        if "aux_decoder" in jax.tree_util.keystr(path)
    ]
    assert len(aux) == EXPECTED_AUX_LEAVES, aux
    stall = [
        (jax.tree_util.keystr(path), np.asarray(value))
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
        if "stall_age_" in jax.tree_util.keystr(path)
    ]
    if STALL_AGE_OBSERVATION:
        assert len(stall) == 2, stall
        assert {value.shape for _, value in stall} == {(704,)}, stall
        assert all(np.count_nonzero(value) == 0 for _, value in stall), stall
    else:
        assert not stall, stall
assert counts[False] == EXPECTED_WITHOUT_CARRY_WORK, counts
assert counts[True] == EXPECTED, counts
print("v6 readout parameters", BLOCKS, counts)
PY

OBSERVED_MAPS="$("$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS" = 96
GRAPH_SHA="$("$PYTHON" -c 'import collections,sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full", sampler_profile=sys.argv[3]); assert len(b.levels)==47 and b.map_count_per_condition==96 and b.sampling_probabilities==(); assert collections.Counter(b.curriculum_depths)=={0:2,1:13,2:32}; print(b.curriculum_graph_sha256)' "$BANK" "$PROTOCOL_TERRA_REVISION" "$SAMPLER_PROFILE")"
[[ "$GRAPH_SHA" =~ ^[0-9a-f]{64}$ ]]

# Verify both immutable inputs and the already-materialized v2 -> final-v3
# sampler transition before initializing W&B or compiling PPO.
JAX_PLATFORMS=cpu "$PYTHON" \
    "$BASELINES_ROOT/scripts/euler_v8_v6_yolo_rv2/verify_resume.py" \
    --source "$RESUME_SOURCE_CHECKPOINT" \
    --source-sha256 "$RESUME_SOURCE_SHA" \
    --prepared "$RESUME_CHECKPOINT" \
    --prepared-sha256 "$RESUME_CHECKPOINT_SHA" \
    --distance-artifact-sha256 "$DISTANCE_ARTIFACT_SHA" \
    --output "$RUN_DIR/resume_validation.json"
JAX_PLATFORMS=cpu "$PYTHON" \
    "$BASELINES_ROOT/scripts/prepare_v61_stall_age_continuation.py" \
    --source "$RESUME_SOURCE_CHECKPOINT" \
    --verify "$RESUME_CHECKPOINT" \
    > "$RUN_DIR/stall_age_prepared_validation.txt"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY WANDB_PROJECT
export WANDB_RUN_ID="${RUN_PREFIX}_${BASELINES_REVISION:0:10}_${PHASE}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
RUN_NAME="${RUN_PREFIX}_${BASELINES_REVISION:0:12}_s${SEED}"
# Transitions this job runs, not the lineage total: phase2 starts at 14,000.
TRANSITIONS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * 32 * (UPDATES - START_UPDATE)))
SAMPLER_CONTRACT=(
    "restored_sampler_source_rule=continuous_banded_v2"
    "restored_sampler_rule=continuous_banded_v3"
    "restored_sampler_migration=materialized_before_resume"
    "restored_sampler_partial_window_discarded_updates=50"
    "restored_sampler_partial_window_updates=0"
    "restored_sampler_last_refresh_update=13950"
    "restored_sampler_open_mass=0.8"
    "restored_sampler_mastered_replay_mass=0.2"
)
RUNTIME_TERRA_BASE=46b5a1dd_reward_v2_baseline_timing_plus_stall_age_c2d2a94a
RESUME_CONTRACT=(
    "resume_from=$RESUME_CHECKPOINT"
    "resume_prepared_sha256=$RESUME_CHECKPOINT_SHA"
    "resume_source_checkpoint=$RESUME_SOURCE_CHECKPOINT"
    "resume_source_sha256=$RESUME_SOURCE_SHA"
    "resume_preparation=zero_stall_age_embeddings+v2_to_v3_sampler_migration+clear_partial_window"
    "resume_restores=model_params+optimizer_state+optimizer_clock+sampler_mastery+competence+closed_window+refresh_grid+sampler_rng"
    "resume_env_state=fresh_resets_4x512_to_8x256_same_global_env_count_2048"
    "resume_global_batch=65536_to_65536_transitions_per_update"
)
COMPARISON_ROLE=practical_v6_1_continuation
SELECTION_SPAN="$((START_UPDATE + 1000))_to_${UPDATES}_promotion_only"
INITIALIZATION="resume_v6_1_rv2_u${START_UPDATE}"
OPTIMIZER_INITIAL_STEP=$((START_UPDATE * 2 * 32))
RESTART_POLICY=resume_from_verified_native_v3_stall_age_checkpoint
printf '%s\n' \
    "status=RUNNING" \
    "experiment=$EXPERIMENT_NAME" \
    "arm=$ARM_NAME" \
    "comparison_role=$COMPARISON_ROLE" \
    "screen_semantics=$SCREEN_SEMANTICS" \
    "full_system_selection_updates=$SELECTION_SPAN" \
    "promotion_selection_rule=combined_exact_then_47_condition_macro_then_worst_then_earliest" \
    "bundled_changes=$BUNDLED_CHANGES" \
    "inherited_from_baseline=$INHERITED_CONTRACT" \
    "runtime_terra_base=$RUNTIME_TERRA_BASE" \
    "action_logit_masking=$([ "$ACTION_LOGIT_MASKING" = 1 ] && echo true || echo false)" \
    "stall_age_observation=$([ "$STALL_AGE_OBSERVATION" = 1 ] && echo true || echo false)" \
    "stall_age_cap_steps=$([ "$STALL_AGE_OBSERVATION" = 1 ] && echo 32 || echo none)" \
    "time_remaining_observation=false" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "absolute_start_update=$START_UPDATE" \
    "absolute_target_update=$UPDATES" \
    "updates=$((UPDATES - START_UPDATE))" \
    "transitions=$TRANSITIONS" \
    "support_scope=all47_continuous_from_update0" \
    "condition_count=47" \
    "maps_per_condition=96" \
    "sampler_profile=$SAMPLER_PROFILE" \
    "${SAMPLER_CONTRACT[@]}" \
    "curriculum_graph_sha256=$GRAPH_SHA" \
    "reward_stage=reward_v2" \
    "reward_protocol_id=material_potential_v2" \
    "distance_protocol_id=obstacle_geodesic_8_physical_global_v1" \
    "distance_artifact_sha256=$DISTANCE_ARTIFACT_SHA" \
    "carry_channel=agent_states_8_normalized_carry_work" \
    "horizon=$R2_HORIZON" \
    "full_resets=true" \
    "initialization=$INITIALIZATION" \
    "warm_start_from=none" \
    "${RESUME_CONTRACT[@]}" \
    "optimizer_initial_local_step=$OPTIMIZER_INITIAL_STEP" \
    "entropy_schedule=0.15_to_0.02_over_20000" \
    "ppo_gamma=0.9984" \
    "reward_v2_distance_ref_m=16.0" \
    "reward_v2_distance_bound=2.5" \
    "reward_v2_success_bonus=6.0" \
    "reward_v2_horizon_failure_penalty=1.0" \
    "reward_v2_alpha=1.0" \
    "reward_v2_beta=1.5" \
    "reward_v2_step_cost_total=1.0" \
    "reward_v2_shaping_weight=1.0" \
    "resume_supported=phase_boundaries_only" \
    "restart_policy=$RESTART_POLICY" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "euler_user=$TERRA_EULER_USER" \
    "euler_home_root=$TERRA_EULER_HOME_ROOT" \
    "runtime_venv=$VENV" \
    "wandb_entity=$WANDB_ENTITY" \
    "wandb_project=$WANDB_PROJECT" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "training_bank_tree_sha256=$BANK_TREE_SHA" \
    "materialization_receipt_sha256=$MATERIALIZATION_RECEIPT_SHA" \
    "static_admission_manifest_sha256=$STATIC_RECEIPT_MANIFEST_SHA" \
    "d4a_receipt_sha256=$D4A_RECEIPT_SHA" \
    "d4a_manifest_sha256=$D4A_MANIFEST_SHA" \
    "map_encoder=resnet_spatial_8x8_se_sa_xattn" \
    "resnet_stage_channels=24,48,64,96" \
    "resnet_blocks_per_stage=$BLOCKS_PER_STAGE" \
    "token_mixer_residual_init_scale=0.1" \
    "flatten_reduce_channels=32" \
    "attn_latent_queries=8" \
    "aux_coef=$AUX_COEF" \
    "vf_coef=$CONTRACT_VF_COEF" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "model_parameter_count_without_carry_work=$EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK" \
    "num_devices=$NUM_DEVICES" \
    "num_envs_per_device=$NUM_ENVS_PER_DEVICE" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.0003" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "planned_fixed_panel_spacing_updates=${EVAL_INTERVAL:-none}" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PROTOCOL_TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export ACTION_LOGIT_MASKING BLOCKS_PER_STAGE AUX_COEF VF_COEF
export STALL_AGE_OBSERVATION
export NUM_DEVICES NUM_ENVS_PER_DEVICE SAMPLER_PROFILE
export NUM_STEPS=32 NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL FINITE_CHECK_INTERVAL CACHE_CLEAR_INTERVAL
export ENTROPY_SCHEDULE_STEPS=20000
# UPDATES is the ABSOLUTE num_updates, so a resume runs [START_UPDATE, UPDATES).
"$BASELINES_ROOT/scripts/run_v8_v6_yolo_rv2.sh" \
    "$BANK" "$RUN_NAME" "$UPDATES" "$RUN_DIR" "$DISTANCE_ARTIFACT_SHA" \
    "$RESUME_CHECKPOINT"

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1
    printf '%s\n' "${matches[0]}"
}

find_one "${RUN_NAME}_update_$(printf '%06d' "$UPDATES").pkl" >/dev/null
printf '%s\n' status=TRAINING_TARGET_REACHED >> "$RUN_DIR/run_contract.env"
# Keep the scarce allocation for training; evaluate checkpoints in a separate
# batched observer job.
echo "V8_V61_STALL_AGE_V3_TARGET_REACHED"
