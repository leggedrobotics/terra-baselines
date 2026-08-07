#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

for REQUIRED in PHASE BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE \
    BANK_SHA BANK_DATASET_SHA BANK_RELEASE_ID SMOKE_JOB_ID SMOKE_RUN \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        FINITE_CHECK_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        UPDATES=6000
        CHECKPOINT_INTERVAL=500
        LOG_TRAIN_INTERVAL=10
        FINITE_CHECK_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/work/rsl/lterenzi/terra_v8_banded_scratch_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/capability/s$SEED/G-V8-XATTN-SCRATCH"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"
if [ "$PHASE" = screen ]; then
    SMOKE_STATE="$(sacct -n -X -P -j "$SMOKE_JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$SMOKE_JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$SMOKE_STATE" = COMPLETED
    test -f "$SMOKE_RUN/smoke_validation.json"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$SMOKE_RUN/smoke_validation.json"
fi

HOME_USED_GB="$(lquota | awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" = screen ]; then
    module load eth_proxy
    test -r "$HOME/.netrc"
fi
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    case "$GPU_NAME" in
        "NVIDIA GeForce RTX 3090"|"NVIDIA GeForce RTX 4090") ;;
        *) echo "unsupported GPU: $GPU_NAME" >&2; exit 3 ;;
    esac
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
OBSERVED_MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS_PER_CONDITION" = 96
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="capability", sampler_profile="bank_v4"); assert len(b.levels)==2; assert b.sampling_probabilities==(0.5,0.5)' "$BANK" "$TERRA_REVISION"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_banded_scratch_${BASELINES_REVISION:0:10}_${PHASE}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

RUN_NAME="v8_banded_scratch_${BASELINES_REVISION:0:12}_${PHASE}_capability_s${SEED}"
GLOBAL_TRANSITIONS=$((4 * 512 * 32 * UPDATES))
printf '%s\n' \
    "arm=G-V8-XATTN-SCRATCH" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "updates=$UPDATES" \
    "absolute_target_global_transitions=$GLOBAL_TRANSITIONS" \
    "curriculum_stage=capability" \
    "sampler_profile=bank_v4" \
    "population_capability=1.0" \
    "population_foundation=0.5" \
    "population_trench=0.5" \
    "reward_stage=dense_skill" \
    "reward_type=DENSE" \
    "horizon=450" \
    "full_resets=true" \
    "initialization=random_no_teacher" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "model_parameter_count=2856685" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "learning_rate=0.0003" \
    "entropy_schedule=0.15_to_0.02_over_${UPDATES}" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "smoke_job_id=$SMOKE_JOB_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$TERRA" TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL FINITE_CHECK_INTERVAL CACHE_CLEAR_INTERVAL
"$BASELINES_ROOT/scripts/run_v8_scratch_capability.sh" \
    "$BANK" "$RUN_NAME" "$UPDATES" "$RUN_DIR"

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1
    printf '%s\n' "${matches[0]}"
}
if [ "$PHASE" = smoke ]; then
    PERIODIC="$(find_one "${RUN_NAME}-euler-*_update_000001.pkl")"
    FINAL="$(find_one "${RUN_NAME}-euler-*_FINAL.pkl")"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/select_promotion.py" verify-smoke \
        --arm G-UNIFORM --periodic "$PERIODIC" --final "$FINAL" \
        --output "$RUN_DIR/smoke_validation.json"
    printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
    echo "V8_BANDED_SCRATCH_SMOKE_PASSED"
    exit 0
fi

CHECKPOINT_ARGS=()
for UPDATE in $(seq 500 500 6000); do
    TOKEN="$(printf '%06d' "$UPDATE")"
    CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}-euler-*_update_${TOKEN}.pkl")")
done
for PANEL in promotion development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$RUN_DIR/eval/capability_${PANEL}.json"
done
printf '%s\n' status=EVALUATED >> "$RUN_DIR/run_contract.env"
