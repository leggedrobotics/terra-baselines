#!/usr/bin/env bash
# GRU generalist from random initialization on the 512-maps-per-condition bank.
# Early policy KL to the frozen u110000 feed-forward generalist (task-teacher
# path, one checkpoint for both families) fades 1 -> 0; plain PPO afterwards.
set -euo pipefail
umask 027

for REQUIRED in TARGET_UPDATE RUN_DIR RUN_NAME BASELINES_ROOT BASELINES_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION SEED VENV RUNTIME_LOCK_SHA \
    TERRA_EULER_USER TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT WANDB_MODE \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_MAPS_PATH BANK_DATASET_SIZE \
    BANK_DISTANCE_SIDECAR_SHA TEACHER TEACHER_SHA NUM_DEVICES ENVS_PER_DEVICE \
    KL_ANNEAL_UPDATES GPU_TYPE EXPECTED_PARAMETERS SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

# One run may have several segments (a 24 h chain and a 120 h job): each
# resumes from the newest checkpoint and only one trains at a time. A segment
# with TAKEOVER=1 stops the running segment instead of exiting.
LOCK="$RUN_DIR/active_segment"
mkdir -p "$RUN_DIR/checkpoints"
claim_lock() {
    if mkdir "$LOCK" 2>/dev/null; then
        echo "$SLURM_JOB_ID" > "$LOCK/job"
        return 0
    fi
    sleep 15
    local holder
    holder="$(cat "$LOCK/job" 2>/dev/null || true)"
    if [ -n "$holder" ] && [ "$holder" != "$SLURM_JOB_ID" ] && \
        squeue -h -j "$holder" -o %T 2>/dev/null | grep -q RUNNING; then
        if [ "${TAKEOVER:-0}" != 1 ]; then
            echo "segment $holder is training this run; nothing to do"
            return 1
        fi
        echo "taking over from segment $holder"
        scancel "$holder"
        for _ in $(seq 60); do
            squeue -h -j "$holder" 2>/dev/null | grep -q . || break
            sleep 10
        done
    fi
    mkdir -p "$LOCK"
    echo "$SLURM_JOB_ID" > "$LOCK/job"
}
release_lock() {
    if [ "$(cat "$LOCK/job" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then
        rm -f "$LOCK/job"
        rmdir "$LOCK" 2>/dev/null || true
    fi
}
claim_lock || exit 0
trap release_lock EXIT
trap 'release_lock; exit 143' TERM INT

RESUME_FROM="${RESUME_FROM:-none}"
if [ "$RESUME_FROM" = latest ]; then
    if compgen -G "$RUN_DIR/checkpoints/*_FINAL.pkl" > /dev/null; then
        echo "run already reached its target update"
        exit 0
    fi
    # Checkpoint names carry zero-padded updates and are written atomically.
    RESUME_FROM="$( (ls "$RUN_DIR"/checkpoints/*_update_*.pkl 2>/dev/null || true) | sort | tail -n 1)"
    RESUME_FROM="${RESUME_FROM:-none}"
    echo "resume_from=$RESUME_FROM"
fi
if [ "$RESUME_FROM" != none ]; then
    test -r "$RESUME_FROM"
    RESUME_ARGS=(--resume_from "$RESUME_FROM" --load_env_from_checkpoint)
else
    RESUME_ARGS=(--no-load-env-from-checkpoint)
fi
case "$GPU_TYPE" in
    rtx_4090) GPU_MODEL="RTX 4090" ;;
    rtx_3090) GPU_MODEL="RTX 3090" ;;
    *) echo "GPU_TYPE must be rtx_4090 or rtx_3090" >&2; exit 2 ;;
esac

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$TEACHER" | awk '{print $1}')" = "$TEACHER_SHA"
test "$(sha256sum "$VENV/requirements.lock.txt" | awk '{print $1}')" = "$RUNTIME_LOCK_SHA"

# Hard GPU guard before JAX or W&B start.
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq "$NUM_DEVICES"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv "$GPU_MODEL"; then
    echo "unexpected GPU model; expected $GPU_MODEL" >&2
    exit 3
fi

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUN_DIR/jax-cache}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
mkdir -p "$RUN_DIR/wandb" "$JAX_COMPILATION_CACHE_DIR"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export DATASET_PATH="$TMPDIR/bank"
export DATASET_SIZE="$BANK_DATASET_SIZE"
test -d "$DATASET_PATH/$BANK_MAPS_PATH/metadata"

module purge
module load stack/2024-06 eth_proxy
PYTHON="$VENV/bin/python"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
if printf '%s' "${LD_LIBRARY_PATH:-}" | grep -Eqi 'cuda|cudnn|cublas|nccl|nvrtc|cupti|cusolver|cusparse|cufft|nvjitlink'; then
    echo "CUDA libraries leaked into LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
    exit 3
fi
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"

"$PYTHON" - "$VENV/requirements.lock.txt" <<'PY'
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement
from packaging.version import Version

mismatches = []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line.strip())
    (pin,) = list(requirement.specifier)
    observed = metadata.version(requirement.name)
    if pin.operator != "==" or Version(observed) != Version(pin.version):
        mismatches.append(f"{requirement.name}: expected {pin}, observed {observed}")
if mismatches:
    raise SystemExit("runtime lock mismatch:\n" + "\n".join(mismatches))
print("PASS runtime lock")
PY
"$PYTHON" "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices "$NUM_DEVICES"

EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import os
from types import SimpleNamespace
import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig
from utils.models import get_model_ready

class Config(dict):
    __getattr__ = dict.__getitem__

config = Config(
    clip_action_maps=True, loaded_max=100,
    local_map_normalization_bounds=(-16, 16),
    maps_net_normalization_bounds=(-10, 10), model_core="mlp",
    actor_core="gru", actor_gru_hidden_dim=64,
    model_size="medium", num_prev_actions=5, critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16", attention_compute_dtype="float32",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96), resnet_blocks_per_stage=(2, 2, 3, 3),
    token_mixer_residual_init_scale=0.1, flatten_reduce_channels=32,
    attn_latent_queries=8, aux_coef=0.0, carry_work_observation=True,
    stall_age_observation=False, reward_v2_reset_context_observation=False,
    trench_alignment_observation=True, relocation_distance_observation=True,
    admissible_dig_observation=True, time_observation_mode="remaining",
)
env = SimpleNamespace(batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)))
_, params = get_model_ready(jax.random.PRNGKey(0), config, env)
count = sum(int(np.asarray(x).size) for x in jax.tree.leaves(params))
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
print(f"PASS model parameters {count}")
PY

# One W&B run per Slurm segment; continuations carry the job id.
BASE_RUN_ID="$RUN_NAME"
if [ "$RESUME_FROM" = none ]; then
    export WANDB_RUN_ID="$BASE_RUN_ID" WANDB_RESUME=never
else
    export WANDB_RUN_ID="${BASE_RUN_ID}_resume_j${SLURM_JOB_ID}" WANDB_RESUME=never
fi
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_DIR="$RUN_DIR/wandb"

scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.$SLURM_JOB_ID.txt"
TRANSITIONS_PER_UPDATE=$((NUM_DEVICES * ENVS_PER_DEVICE * 32))
TOTAL_TIMESTEPS=$((TRANSITIONS_PER_UPDATE * TARGET_UPDATE))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=gru_generalist_512" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "resume_from=$RESUME_FROM" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "wandb_mode=$WANDB_MODE" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "runtime_venv=$VENV" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_maps_path=$BANK_MAPS_PATH" \
    "bank_dataset_size=$BANK_DATASET_SIZE" \
    "distance_sidecar_sha256=$BANK_DISTANCE_SIDECAR_SHA" \
    "teacher_sha256=$TEACHER_SHA" \
    "kickstart_kl=1_to_0_cosine_over_${KL_ANNEAL_UPDATES}_updates" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "seed=$SEED" \
    "gpu=${NUM_DEVICES}x$GPU_TYPE" \
    "envs_per_device=$ENVS_PER_DEVICE" \
    "transitions_per_update=$TRANSITIONS_PER_UPDATE" \
    "target_update=$TARGET_UPDATE" > "$RUN_DIR/run_contract.env"
cp "$RUN_DIR/run_contract.env" "$RUN_DIR/run_contract.job${SLURM_JOB_ID}.env"

TRAIN_RC=0
"$PYTHON" -u "$BASELINES_ROOT/train_mixed.py" \
    --config gru_generalist_512 \
    --machine euler \
    --name "$RUN_NAME" \
    --exact_run_name \
    --seed "$SEED" \
    --num_devices "$NUM_DEVICES" \
    --num_envs_per_device "$ENVS_PER_DEVICE" \
    --num_steps 32 \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --update_epochs 2 \
    --num_minibatches 32 \
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
    --vf_coef 2 \
    --ent_schedule_start 0.02 \
    --ent_schedule_end 0.02 \
    --ent_schedule_steps 20000 \
    --no_value_clip \
    --global_minibatch_advantage_norm \
    --carry_work_observation \
    --relocation_distance_observation \
    --admissible_dig_observation \
    --executable_dig_observation \
    --time_observation_mode remaining \
    --reward_stage reward_v2 \
    --reward_v2_timing_variant 0 \
    --distance_protocol_id obstacle_geodesic_8_physical_global_v1 \
    --distance_sidecar_sha256 "$BANK_DISTANCE_SIDECAR_SHA" \
    --lateral_dig_cost 0 \
    --base_travel_cost 0 \
    --base_turn_cost 0 \
    --teacher_checkpoint "$TEACHER" \
    --trench_teacher_checkpoint "$TEACHER" \
    --kickstart_start_update 0 \
    --kickstart_kl_coef 1 \
    --kickstart_kl_anneal_updates "$KL_ANNEAL_UPDATES" \
    --kickstart_value_coef 0 \
    --kickstart_value_anneal_updates 0 \
    --kickstart_lr_warmup_updates 0 \
    --fail_on_nonfinite \
    --finite_check_interval 10 \
    --log_train_interval 10 \
    --log_eval_interval 0 \
    --checkpoint_interval 500 \
    --cache_clear_interval 0 \
    --keep_checkpoint_history \
    --checkpoint_dir "$RUN_DIR/checkpoints" \
    "${RESUME_ARGS[@]}" || TRAIN_RC=$?

if [ "$TRAIN_RC" -eq 0 ]; then
    echo "status=COMPLETED" >> "$RUN_DIR/run_contract.env"
    exit 0
fi
echo "status=FAILED train_rc=$TRAIN_RC" >> "$RUN_DIR/run_contract.env"
exit "$TRAIN_RC"
