#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-v8-cont-unpaired

set -euo pipefail
umask 002

for REQUIRED in ARM PAIRING BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_RELEASE_ID QUALIFIED_RECEIPT QUALIFIED_RECEIPT_SHA \
    RESUME_CHECKPOINT RESUME_CHECKPOINT_SHA RESUME_UPDATE PARENT_WANDB_RUN_ID \
    PARENT_RUN_CONTRACT_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "${SLURM_JOB_PARTITION:-}" = gpuhe.120h
test "$RESUME_UPDATE" -lt 80000

case "$ARM" in
    G-DEEP-V8-DENSE-WARM) ARCHITECTURE_LABEL=deep-se ;;
    G-DEEP-XATTN-V8-DENSE-WARM) ARCHITECTURE_LABEL=deep-se-xattn ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac
case "$PAIRING" in
    unpaired_single_qualifying_arm) PAIRING_SUFFIX=unpaired ;;
    matched_architecture_pair) PAIRING_SUFFIX=matched ;;
    *) echo "unsupported pairing '$PAIRING'" >&2; exit 2 ;;
esac

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/continuation/full/s$SEED/$ARM-$PAIRING_SUFFIX"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$QUALIFIED_RECEIPT" | awk '{print $1}')" = "$QUALIFIED_RECEIPT_SHA"
test "$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')" = "$RESUME_CHECKPOINT_SHA"
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"
[[ "$PARENT_WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]

HOME_USED_GB="$(lquota | awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }'

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
module load eth_proxy
test -r "$HOME/.netrc"
grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' "$HOME/.netrc"
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

JAX_PLATFORMS=cpu "$PYTHON" \
    "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/continuation_contract.py" verify \
    --receipt "$QUALIFIED_RECEIPT" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --checkpoint-sha256 "$RESUME_CHECKPOINT_SHA" \
    --arm "$ARM" --seed "$SEED" \
    --terra-baselines-revision "$BASELINES_REVISION" \
    > "$RUN_DIR/continuation_checkpoint_validation.json"
SOURCE_TREATMENT_NAME="$(
    "$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["source_treatment_name"])' \
        "$RUN_DIR/continuation_checkpoint_validation.json"
)"
[[ "$SOURCE_TREATMENT_NAME" =~ ^[A-Za-z0-9_-]+$ ]]

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported GPU: $GPU_NAME" >&2
        exit 3
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$MAPS_PER_CONDITION" = 96
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full"); assert len(b.levels)==47; assert b.map_count_per_condition==96; assert abs(sum(b.sampling_probabilities)-1.0)<1e-12' "$BANK" "$TERRA_REVISION"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_${BASELINES_REVISION:0:10}_cont_full_${ARM,,}_${SLURM_JOB_ID}"
export WANDB_RUN_ID="${WANDB_RUN_ID//-/_}"
export WANDB_RESUME=never
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_NOTES="true-resume continuation of W&B run $PARENT_WANDB_RUN_ID from checkpoint $RESUME_CHECKPOINT_SHA; pairing=$PAIRING"
export WANDB_JOB_TYPE=v8-full-continuation
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

# Preserve the source training name so pre/post-resume fixed evaluations have
# the same path-independent treatment fingerprint. W&B linkage uses a new run
# ID and does not require changing the policy's treatment name.
RUN_NAME="$SOURCE_TREATMENT_NAME"
ABSOLUTE_UPDATES=80000
ABSOLUTE_TRANSITIONS=$((4 * 1024 * 32 * ABSOLUTE_UPDATES))
REMAINING_TRANSITIONS=$((4 * 1024 * 32 * (ABSOLUTE_UPDATES - RESUME_UPDATE)))
printf '%s\n' \
    "status=RUNNING_OR_CONTINUABLE" \
    "arm=$ARM" \
    "architecture=$ARCHITECTURE_LABEL" \
    "pairing=$PAIRING" \
    "curriculum_stage=full" \
    "reward_stage=dense_skill" \
    "reward_type=DENSE" \
    "condition_sampler=fixed_v8_stage_weights" \
    "condition_count=47" \
    "seed=$SEED" \
    "phase=continuation" \
    "resume_update=$RESUME_UPDATE" \
    "absolute_target_update=$ABSOLUTE_UPDATES" \
    "absolute_target_transitions=$ABSOLUTE_TRANSITIONS" \
    "remaining_transitions_at_launch=$REMAINING_TRANSITIONS" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "resume_checkpoint_path=$RESUME_CHECKPOINT" \
    "resume_checkpoint_sha256=$RESUME_CHECKPOINT_SHA" \
    "qualified_receipt=$QUALIFIED_RECEIPT" \
    "qualified_receipt_sha256=$QUALIFIED_RECEIPT_SHA" \
    "parent_run_contract_sha256=$PARENT_RUN_CONTRACT_SHA" \
    "parent_wandb_run_id=$PARENT_WANDB_RUN_ID" \
    "source_treatment_name=$SOURCE_TREATMENT_NAME" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "wandb_resume=never" \
    "initialization=true_resume_optimizer_schedule_sampler" \
    "statistical_continuation=true" \
    "bit_exact_continuation=false" \
    "restarted_state=environment_rng_action_history" \
    "trench_shaping=false" \
    "horizon=450" \
    "full_resets=true" \
    "checkpoint_interval=500" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$TERRA"
export TERRA_REVISION
export RUN_ROOT="$RUN_DIR"
export PYTHON_BIN="$PYTHON"
export MACHINE=euler
export NUM_DEVICES=4
export NUM_ENVS_PER_DEVICE=1024
export NUM_STEPS=32
export CHECKPOINT_INTERVAL=500
export FINITE_CHECK_INTERVAL=10
export LOG_TRAIN_INTERVAL=10
export CACHE_CLEAR_INTERVAL=1000
TRAIN_COMMAND=("$BASELINES_ROOT/scripts/run_v8_resume.sh" "$ARM" "$BANK" "$RESUME_CHECKPOINT" "$RUN_NAME")
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.sh"
printf '\n' >> "$RUN_DIR/train_command.sh"
"${TRAIN_COMMAND[@]}"

printf '%s\n' "target_update=$ABSOLUTE_UPDATES" > "$RUN_DIR/training_target_reached.env"
