#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=71:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in ARM RUN_DIR RUN_NAME BASELINES_ROOT BASELINES_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION PROTOCOL_TERRA_REVISION \
    SEED VENV TERRA_EULER_USER TERRA_EULER_HOME_ROOT WANDB_ENTITY \
    WANDB_PROJECT WANDB_RUN_ID BANK_ARCHIVE BANK_ARCHIVE_SHA \
    BANK_DATASET_SHA BANK_RELEASE_ID DISTANCE_SIDECAR_SHA PARTIAL_ARCHIVE \
    PARTIAL_ARCHIVE_SHA PARTIAL_BANK_SHA TARGET_UPDATE SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$ARM" in
    control) EXPECTED_PARAMETERS=2352573 ;;
    feedback) EXPECTED_PARAMETERS=2361021 ;;
    *) echo "ARM must be control or feedback" >&2; exit 2 ;;
esac
test "$TARGET_UPDATE" = 50000
test "${SLURM_JOB_PARTITION:-}" = gpuhe.120h
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_ARCHIVE_SHA"
test "$BANK_RELEASE_ID" = terra_v8_v6_constraints_v7_adjacent_train96_v5
test "$PROTOCOL_TERRA_REVISION" = a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4

JOB_DESCRIPTION="$(scontrol show job "$SLURM_JOB_ID" -o)"
grep -q 'ReqTRES=.*gres/gpu=4' <<<"$JOB_DESCRIPTION"
grep -q 'ReqTRES=.*gres/gpu:nvidia_geforce_rtx_4090=4' <<<"$JOB_DESCRIPTION"
grep -q 'AllocTRES=.*gres/gpu=4' <<<"$JOB_DESCRIPTION"
grep -q 'AllocTRES=.*gres/gpu:nvidia_geforce_rtx_4090=4' <<<"$JOB_DESCRIPTION"
HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45) }'

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
CONTRACT="$RUN_DIR/run_contract.env"
test ! -e "$CONTRACT"
printf '%s\n' \
    "status=RUNNING_PREFLIGHT" \
    "arm=$ARM" \
    "seed=$SEED" \
    "absolute_target_update=$TARGET_UPDATE" \
    "initialization=fresh_scratch" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "training_bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "partial_reset_archive_sha256=$PARTIAL_ARCHIVE_SHA" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "movement_feasibility_observation=$([ "$ARM" = feedback ] && echo true || echo false)" \
    "previous_outcome_observation=$([ "$ARM" = feedback ] && echo true || echo false)" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "actor_core=gru" \
    "actor_gru_hidden_dim=64" \
    "actor_head=concat_skip" \
    "action_logit_masking=false" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "transitions_per_update=65536" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "wandb_run_id=$WANDB_RUN_ID" > "$CONTRACT"

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
tar --zstd -xf "$PARTIAL_ARCHIVE" -C "$TMPDIR" partial_bank
BANK="$TMPDIR/bank"
PARTIAL_BANK="$TMPDIR/partial_bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1 eth_proxy
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done
"$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_movement_feedback_v1/check_jax_runtime.py" --min-devices 4

OBSERVED_MAPS="$("$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$BANK_RELEASE_ID")"
test "$OBSERVED_MAPS" = 96
PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" PARTIAL_BANK="$PARTIAL_BANK" "$PYTHON" - <<'PY'
import os
from terra.maps_buffer import partial_reset_bank_sha256

observed = partial_reset_bank_sha256(os.environ["PARTIAL_BANK"])
assert observed == os.environ["PARTIAL_BANK_SHA"], observed
PY

"$PYTHON" -m pytest -q "$BASELINES_ROOT/tests/test_v61_stall_age.py" \
    -k movement_feedback_adds_only_zero_embeddings_and_preserves_gru_u0

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PYTHON_BIN="$PYTHON" MACHINE=euler
SMOKE_RUN="$TMPDIR/update1-$ARM"
SMOKE_NAME="${RUN_NAME}_update1_smoke"
PRODUCTION_WANDB_RUN_ID="$WANDB_RUN_ID"
export WANDB_MODE=disabled WANDB_RESUME=never WANDB_RUN_ID="$SMOKE_NAME"
SMOKE_STARTED_SECONDS=$SECONDS
"$BASELINES_ROOT/scripts/run_v8_movement_feedback_v1.sh" \
    "$ARM" "$BANK" "$PARTIAL_BANK" "$SMOKE_NAME" 1 "$SMOKE_RUN" \
    "$DISTANCE_SIDECAR_SHA"
SMOKE_FINAL="$SMOKE_RUN/checkpoints/${SMOKE_NAME}_FINAL.pkl"
test -f "$SMOKE_FINAL"
ARM="$ARM" EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" SMOKE_FINAL="$SMOKE_FINAL" \
    "$PYTHON" - <<'PY'
import os
import jax
import numpy as np
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(os.environ["SMOKE_FINAL"])
assert checkpoint["next_update"] == 1
assert int(np.asarray(checkpoint["train_state_step"])) == 64
config = checkpoint["train_config"]
enabled = os.environ["ARM"] == "feedback"
assert bool(config.movement_feasibility_observation) is enabled
assert bool(config.previous_outcome_observation) is enabled
assert bool(config.action_logit_masking) is False
model_leaves = jax.tree.leaves(checkpoint["model"])
assert sum(int(np.asarray(leaf).size) for leaf in model_leaves) == int(
    os.environ["EXPECTED_PARAMETERS"]
)
assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in model_leaves)
assert all(
    np.all(np.isfinite(np.asarray(leaf)))
    for leaf in jax.tree.leaves(checkpoint["optimizer_state"])
)
assert all(
    np.all(np.isfinite(np.asarray(leaf)))
    for leaf in jax.tree.leaves(checkpoint["loss_info"])
)
for name, value in checkpoint["loss_info"].items():
    if name.endswith("_finite_fraction") or name.endswith("_all_finite"):
        assert float(np.asarray(value)) == 1.0, (name, value)
assert checkpoint["transition_integrity"]["maximum_mass_residual"] == 0
assert checkpoint["transition_integrity"]["target_mutation_count"] == 0
assert checkpoint["transition_integrity"]["obstacle_mutation_count"] == 0
print("PASS finite update-1 checkpoint")
PY
printf '%s\n' \
    "update1_smoke_seconds=$((SECONDS - SMOKE_STARTED_SECONDS))" \
    "status=UPDATE1_SMOKE_PASSED" >> "$CONTRACT"

export WANDB_ENTITY WANDB_PROJECT WANDB_MODE=online WANDB_RESUME=never
export WANDB_RUN_ID="$PRODUCTION_WANDB_RUN_ID"
export WANDB_DIR="$RUN_DIR/wandb"
"$PYTHON" -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
printf '%s\n' "status=PRODUCTION_RUNNING" >> "$CONTRACT"

"$BASELINES_ROOT/scripts/run_v8_movement_feedback_v1.sh" \
    "$ARM" "$BANK" "$PARTIAL_BANK" "$RUN_NAME" "$TARGET_UPDATE" \
    "$RUN_DIR" "$DISTANCE_SIDECAR_SHA"

FINAL="$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
test -f "$FINAL"
printf '%s\n' "status=TARGET_REACHED" >> "$CONTRACT"
echo "V8_MOVEMENT_FEEDBACK_${ARM^^}_TARGET_REACHED"
