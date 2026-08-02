#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-p5b-warm
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/p5b_warm_v1/slurm-%x-%j.out

set -euo pipefail
umask 002

: "${PHASE:?missing PHASE=smoke|screen}"
: "${ARM:?missing ARM}"
: "${BASELINES_ROOT:?missing BASELINES_ROOT}"
: "${BASELINES_REVISION:?missing BASELINES_REVISION}"
: "${DEEP_CHECKPOINT:?missing DEEP_CHECKPOINT}"
: "${DEEP_SHA:?missing DEEP_SHA}"
: "${SEED:?missing SEED}"
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${TMPDIR:?Euler did not provide TMPDIR}"

P5_SHA="f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa"
P5_ARCHIVE="/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst"
TERRA_REVISION="a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
PARENT="/cluster/scratch/lterenzi/codex_terra_edge_runs/accepted_bank_v1/$P5_SHA/screen/s20260730/G-ADAPTIVE/checkpoints/accepted_f8aac348d64c_screen_g_adaptive_s20260730-euler-2026-08-01-19-45-09_update_002000.pkl"
PARENT_SHA="76b5189955735741b0cd4b3444fbda8ffdb8be4b29582509dafad85fa7cfb45a"
VENV="/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426"
RUN_BASE="/cluster/scratch/lterenzi/codex_terra_edge_runs/p5b_warm_v1"

case "$ARM" in
    G-MEDIUM-ADAPTIVE-WARM|G-DEEP-ADAPTIVE-WARM)
        ACCEPTED_BANK_ARM=G-ADAPTIVE
        CONDITION_SAMPLER=adaptive
        ;;
    G-MEDIUM-UNIFORM-WARM)
        ACCEPTED_BANK_ARM=G-UNIFORM
        CONDITION_SAMPLER=uniform
        ;;
    *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac
case "$PHASE" in
    smoke)
        EXPECTED_PARTITION=gpuhe.4h
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        EXPECTED_PARTITION=gpuhe.24h
        UPDATES=2000
        CHECKPOINT_INTERVAL=500
        LOG_TRAIN_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported PHASE=$PHASE" >&2; exit 2 ;;
esac

test "${SLURM_JOB_PARTITION:-}" = "$EXPECTED_PARTITION"
test "${SLURM_CPUS_PER_TASK:-}" = 4
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$PARENT" | awk '{print $1}')" = "$PARENT_SHA"
test "$(sha256sum "$DEEP_CHECKPOINT" | awk '{print $1}')" = "$DEEP_SHA"
HOME_USED_GB="$(lquota | awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/bank campaign/runtime/check_jax_runtime.py
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/campaign/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" = screen ]; then
    module load eth_proxy
    test -r "$HOME/.netrc"
    grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' "$HOME/.netrc"
fi
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

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || { echo "unsupported GPU: $GPU_NAME" >&2; exit 3; }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK")"
test "$MAPS_PER_CONDITION" = 64
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], sys.argv[2], sys.argv[3]); assert len(b.levels) == 32 and b.map_count_per_condition == 64' "$BANK" "$ACCEPTED_BANK_ARM" "$TERRA_REVISION"

case "$ARM" in
    G-MEDIUM-ADAPTIVE-WARM|G-MEDIUM-UNIFORM-WARM)
        INITIAL="$PARENT"
        ARCHITECTURE_LABEL=medium:resnet_spatial_8x8_se:mlp:critic-512-256
        ;;
    G-DEEP-ADAPTIVE-WARM)
        INITIAL="$DEEP_CHECKPOINT"
        ARCHITECTURE_LABEL=medium:resnet_spatial_8x8_se:mlp:critic-512-256:channels-24x48x64x96:blocks-2x2x3x3
        ;;
esac
INITIAL_SHA="$(sha256sum "$INITIAL" | awk '{print $1}')"
GLOBAL_TRANSITIONS=$((4 * 1024 * 32 * UPDATES))
RUN_NAME="p5b_${BASELINES_REVISION:0:12}_${PHASE}_${ARM,,}_s${SEED}"
RUN_NAME="${RUN_NAME//-/_}"
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/s$SEED/$ARM"
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
if [ "$PHASE" = screen ]; then
    SMOKE_DIR="$RUN_BASE/$BASELINES_REVISION/smoke/s$SEED/$ARM"
    test -f "$SMOKE_DIR/smoke_validation.json"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$SMOKE_DIR/smoke_validation.json"
    grep -qx 'status=PASSED' "$SMOKE_DIR/run_contract.env"
fi

export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="p5b_${BASELINES_REVISION:0:10}_${PHASE}_${ARM,,}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled
    export WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online
    export WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

export TERRA_ROOT="$TERRA"
export TERRA_REVISION
export RUN_ROOT="$RUN_DIR"
export PYTHON_BIN="$PYTHON"
export MACHINE=euler
export NUM_DEVICES=4
export NUM_ENVS_PER_DEVICE=1024
export NUM_STEPS=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL CACHE_CLEAR_INTERVAL
export FINITE_CHECK_INTERVAL=10
if [ "$PHASE" = smoke ]; then export FINITE_CHECK_INTERVAL=1; fi
TRAIN_COMMAND=("$BASELINES_ROOT/scripts/run_p5b_warm_screen.sh" "$ARM" "$BANK" "$INITIAL" "$PARENT" "$RUN_NAME" "$UPDATES")
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.sh"
printf '\n' >> "$RUN_DIR/train_command.sh"
printf '%s\n' \
    "arm=$ARM" \
    "accepted_bank_arm=$ACCEPTED_BANK_ARM" \
    "primary_family=generalist" \
    "training_support=all" \
    "condition_sampler=$CONDITION_SAMPLER" \
    "architecture=$ARCHITECTURE_LABEL" \
    "seed=$SEED" \
    "phase=$PHASE" \
    "updates=$UPDATES" \
    "global_transitions=$GLOBAL_TRANSITIONS" \
    "final_checkpoint_update=$UPDATES" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "parent_checkpoint_sha256=$PARENT_SHA" \
    "initial_checkpoint_sha256=$INITIAL_SHA" \
    "teacher_checkpoint_sha256=$PARENT_SHA" \
    "initialization=params_only_warm" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "wandb_run_id=$WANDB_RUN_ID" > "$RUN_DIR/run_contract.env"
"${TRAIN_COMMAND[@]}"

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1 || { echo "expected one checkpoint matching $pattern, got ${#matches[@]}" >&2; return 1; }
    printf '%s\n' "${matches[0]}"
}
if [ "$PHASE" = smoke ]; then
    PERIODIC="$(find_one "${RUN_NAME}-euler-*_update_000001.pkl")"
    FINAL="$(find_one "${RUN_NAME}-euler-*_FINAL.pkl")"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/select_promotion.py" verify-smoke \
        --arm "$ACCEPTED_BANK_ARM" --periodic "$PERIODIC" --final "$FINAL" \
        --output "$RUN_DIR/smoke_validation.json"
else
    CHECKPOINT_ARGS=()
    for UPDATE in 500 1000 1500 2000; do
        TOKEN="$(printf '%06d' "$UPDATE")"
        CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}-euler-*_update_${TOKEN}.pkl")")
    done
    for PANEL in promotion development; do
        JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
            "${CHECKPOINT_ARGS[@]}" \
            --bank-root "$BANK" --accepted-panel "$PANEL" \
            --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$RUN_DIR/eval/${PANEL}.json"
    done
fi

printf '%s\n' "status=PASSED" >> "$RUN_DIR/run_contract.env"
echo "P5B_${PHASE^^}_${ARM}_PASSED"
