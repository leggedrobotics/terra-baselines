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

SOURCE_UPDATE=40000
TARGET_UPDATE=70000
NUM_DEVICES=8
NUM_ENVS_PER_DEVICE=256
NUM_STEPS=32
NUM_MINIBATCHES=32
CHECKPOINT_INTERVAL=500
EXPECTED_CPUS=8
EXPECTED_BASELINES_REVISION=dddc691c93ee21488cd7eeb8e01b067bf1f9733c
EXPECTED_TERRA_REVISION=c2d2a94a124759e9f21c2b37930f717e299f0c46
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4

for REQUIRED in TERRA_EULER_USER TERRA_EULER_HOME_ROOT VENV RUN_DIR \
    WANDB_ENTITY WANDB_PROJECT PARENT_WANDB_RUN_ID PARENT_WANDB_MAX_UPDATE \
    PHASE ARM_NAME BASELINES_ROOT BASELINES_REVISION LAUNCHER_REVISION \
    LAUNCHER_SBATCH_SHA RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION \
    R2_HORIZON SEED BANK_ARCHIVE BANK_SHA BANK_DATASET_SHA BANK_RELEASE_ID \
    DISTANCE_ARTIFACT_SHA RESUME_CHECKPOINT RESUME_CHECKPOINT_SHA \
    RESUME_VALIDATOR RESUME_VALIDATOR_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

test "$PHASE" = phase3
test "$ARM_NAME" = v6_1_rv2_stall_age_v3
test "$BASELINES_REVISION" = "$EXPECTED_BASELINES_REVISION"
test "$RUNTIME_TERRA_REVISION" = "$EXPECTED_TERRA_REVISION"
test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$R2_HORIZON" = 450
test "$SEED" = 20260807
test "$PARENT_WANDB_MAX_UPDATE" = 39991
test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h
test "${SLURM_CPUS_PER_TASK:-}" = "$EXPECTED_CPUS"
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"

for PATH_VALUE in "$TERRA_EULER_HOME_ROOT" "$VENV" "$RUN_DIR" \
    "$BASELINES_ROOT" "$RUNTIME_TERRA_ROOT" "$BANK_ARCHIVE" \
    "$RESUME_CHECKPOINT" "$RESUME_VALIDATOR"; do
    case "$PATH_VALUE" in /cluster/*) ;; *) echo "non-cluster path: $PATH_VALUE" >&2; exit 2 ;; esac
    case "$PATH_VALUE" in *[!a-zA-Z0-9_./-]*) echo "unsafe path: $PATH_VALUE" >&2; exit 2 ;; esac
done
for SHA in "$BANK_SHA" "$BANK_DATASET_SHA" "$DISTANCE_ARTIFACT_SHA" \
    "$RESUME_CHECKPOINT_SHA" "$RESUME_VALIDATOR_SHA" "$LAUNCHER_SBATCH_SHA"; do
    [[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: $SHA" >&2; exit 2; }
done
for WANDB_NAME in "$WANDB_ENTITY" "$WANDB_PROJECT" "$PARENT_WANDB_RUN_ID"; do
    [[ "$WANDB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid W&B value: $WANDB_NAME" >&2; exit 2; }
done

test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')" = "$RESUME_CHECKPOINT_SHA"
test "$(sha256sum "$RESUME_VALIDATOR" | awk '{print $1}')" = "$RESUME_VALIDATOR_SHA"
test -d "$RUN_DIR" -a ! -e "$RUN_DIR/run_contract.env"

HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
module load eth_proxy
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
"$PYTHON" -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq "$NUM_DEVICES"
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done

# Exercise both the cuDNN conv-backward and NCCL all-reduce paths before W&B
# changes state or the full PPO graph compiles.
export JAX_PLATFORMS=cuda,cpu
NUM_DEVICES="$NUM_DEVICES" "$PYTHON" - <<'PY'
import os

import jax
import jax.numpy as jnp
from jax import lax

n = int(os.environ["NUM_DEVICES"])
assert len(jax.devices()) == n

def loss(x, kernel):
    y = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return lax.psum(jnp.mean(y * y), "devices")

step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((n, 1, 16, 16, 4), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, kernel)
assert gradient.shape == x.shape and jnp.all(jnp.isfinite(gradient))
print(jax.devices(), float(jnp.linalg.norm(gradient)))
PY

OBSERVED_MAPS="$("$PYTHON" \
    "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" \
    "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS" = 96

JAX_PLATFORMS=cpu "$PYTHON" "$RESUME_VALIDATOR" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --sha256 "$RESUME_CHECKPOINT_SHA" \
    --output "$RUN_DIR/resume_validation.json"

# The source checkpoint is at or ahead of the last parent history row, so this
# segment may safely continue the same absolute train/update axis.
WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" \
PARENT_WANDB_RUN_ID="$PARENT_WANDB_RUN_ID" \
PARENT_WANDB_MAX_UPDATE="$PARENT_WANDB_MAX_UPDATE" "$PYTHON" - <<'PY'
import os

import wandb

path = f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/{os.environ["PARENT_WANDB_RUN_ID"]}'
run = wandb.Api(timeout=30).run(path)
if run.state != "finished":
    raise RuntimeError(f"parent W&B run state is {run.state!r}, expected 'finished'")
updates = [
    row["train/update"]
    for row in run.scan_history(keys=["train/update"], page_size=1000)
    if row.get("train/update") is not None
]
observed = int(max(updates))
expected = int(os.environ["PARENT_WANDB_MAX_UPDATE"])
if observed != expected or observed > 40_000:
    raise RuntimeError(
        f"parent W&B max train/update is {observed}, expected {expected} and <= 40000"
    )
print(f"parent_wandb_state={run.state} parent_wandb_max_train_update={observed}")
PY

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
export WANDB_ENTITY WANDB_PROJECT
export WANDB_RUN_ID="$PARENT_WANDB_RUN_ID"
export WANDB_RESUME=must
export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

RUN_NAME=v8_v61_stall_age_v3_dddc691c93ee_s20260807
TRANSITIONS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * (TARGET_UPDATE - SOURCE_UPDATE)))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=v8_v61_stall_age_v3_continuation" \
    "phase=$PHASE" \
    "arm=$ARM_NAME" \
    "seed=$SEED" \
    "absolute_start_update=$SOURCE_UPDATE" \
    "absolute_target_update=$TARGET_UPDATE" \
    "updates=$((TARGET_UPDATE - SOURCE_UPDATE))" \
    "transitions=$TRANSITIONS" \
    "expected_segment_exit=CONTINUABLE_AT_WALLTIME" \
    "resume_from=$RESUME_CHECKPOINT" \
    "resume_checkpoint_sha256=$RESUME_CHECKPOINT_SHA" \
    "resume_next_update=$SOURCE_UPDATE" \
    "optimizer_initial_step=$((SOURCE_UPDATE * 64))" \
    "fresh_environment_resets=true" \
    "sampler_profile=continuous_banded_v3" \
    "sampler_mastered_at_resume=46" \
    "sampler_window_updates_at_resume=100" \
    "reward_stage=reward_v2" \
    "distance_protocol_id=obstacle_geodesic_8_physical_global_v1" \
    "distance_artifact_sha256=$DISTANCE_ARTIFACT_SHA" \
    "stall_age_observation=true" \
    "time_remaining_observation=false" \
    "action_logit_masking=false" \
    "num_devices=$NUM_DEVICES" \
    "num_envs_per_device=$NUM_ENVS_PER_DEVICE" \
    "num_steps=$NUM_STEPS" \
    "num_minibatches=$NUM_MINIBATCHES" \
    "update_epochs=2" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "training_bank_release_id=$BANK_RELEASE_ID" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "training_terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "launcher_revision=$LAUNCHER_REVISION" \
    "launcher_sbatch_sha256=$LAUNCHER_SBATCH_SHA" \
    "resume_validator_sha256=$RESUME_VALIDATOR_SHA" \
    "wandb_entity=$WANDB_ENTITY" \
    "wandb_project=$WANDB_PROJECT" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "wandb_resume=must" \
    "parent_wandb_max_train_update=$PARENT_WANDB_MAX_UPDATE" \
    "euler_user=$TERRA_EULER_USER" \
    "runtime_venv=$VENV" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PROTOCOL_TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
"$BASELINES_ROOT/scripts/run_v8_v6_yolo_rv2.sh" \
    "$BANK" "$RUN_NAME" "$TARGET_UPDATE" "$RUN_DIR" "$DISTANCE_ARTIFACT_SHA" \
    "$RESUME_CHECKPOINT"

test -f "$RUN_DIR/checkpoints/${RUN_NAME}_update_070000.pkl"
printf '%s\n' status=TRAINING_TARGET_REACHED >> "$RUN_DIR/run_contract.env"
echo V8_V61_STALL_AGE_V3_U70000_TARGET_REACHED
