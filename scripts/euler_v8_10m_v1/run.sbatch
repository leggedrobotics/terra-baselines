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

: "${PHASE:?missing PHASE}"
: "${ARM:?missing ARM}"
: "${BASELINES_ROOT:?missing BASELINES_ROOT}"
: "${BASELINES_REVISION:?missing BASELINES_REVISION}"
: "${SEED:?missing SEED}"
: "${BANK_ARCHIVE:?missing BANK_ARCHIVE}"
: "${BANK_SHA:?missing BANK_SHA}"
: "${BANK_DATASET_SHA:?missing BANK_DATASET_SHA}"
: "${BANK_RELEASE_ID:?missing BANK_RELEASE_ID}"
: "${TEACHER_RECEIPT:?missing TEACHER_RECEIPT}"
: "${TEACHER_RECEIPT_SHA:?missing TEACHER_RECEIPT_SHA}"
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${TMPDIR:?Euler did not provide TMPDIR}"

case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h
        # Deliberately unreachable for a normal 24-hour run. Wall time, not a
        # deceptively short update target, bounds the screen; the paired tail
        # evaluator consumes the longest common complete checkpoint prefix.
        UPDATES=20000
        CHECKPOINT_INTERVAL=500
        LOG_TRAIN_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac
case "$ARM" in
    G-V8-XATTN-REWARM-CONTROL)
        EXPECTED_PARAMETERS=2856685
        EXPECTED_STAGE_CHANNELS=24,48,64,96
        ;;
    G-V8-10M-XATTN-WARM)
        EXPECTED_PARAMETERS=10257209
        EXPECTED_STAGE_CHANNELS=64,128,192,256
        ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_10m_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/s$SEED/$ARM"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$TEACHER_RECEIPT" | awk '{print $1}')" = "$TEACHER_RECEIPT_SHA"
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" = screen ]; then module load eth_proxy; fi
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
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported GPU: $GPU_NAME" >&2
        exit 3
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$MAPS_PER_CONDITION" = 96
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full"); assert len(b.levels)==47; assert b.map_count_per_condition==96' "$BANK" "$TERRA_REVISION"

JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_student.py" \
    inspect-teacher --receipt "$TEACHER_RECEIPT" --bank-root "$BANK" \
    > "$RUN_DIR/teacher_inspection.json"
read -r TEACHER_CHECKPOINT TEACHER_SHA TEACHER_UPDATE <<< "$(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["teacher_checkpoint"], d["teacher_checkpoint_sha256"], d["teacher_update"])' "$RUN_DIR/teacher_inspection.json"
)"
test "$(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}')" = "$TEACHER_SHA"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
INITIAL_CHECKPOINT="$TEACHER_CHECKPOINT"
if [ "$ARM" = G-V8-10M-XATTN-WARM ]; then
    INITIAL_CHECKPOINT="$RUN_DIR/initial_checkpoint.pkl"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/grow_checkpoint.py" \
        --src "$TEACHER_CHECKPOINT" \
        --out "$INITIAL_CHECKPOINT" \
        --map_encoder resnet_spatial_8x8_se_xattn \
        --model_size medium \
        --critic_hidden_dims 512,256 \
        --encoder_compute_dtype bfloat16 \
        --attention_compute_dtype float32 \
        --resnet_stage_channels 64,128,192,256 \
        --resnet_blocks_per_stage 2,2,3,3
fi
OBSERVED_PARAMETERS="$(JAX_PLATFORMS=cpu "$PYTHON" -c 'import sys,jax; from utils.helpers import load_pkl_object; c=load_pkl_object(sys.argv[1]); print(sum(x.size for x in jax.tree_util.tree_leaves(c["model"])))' "$INITIAL_CHECKPOINT")"
test "$OBSERVED_PARAMETERS" = "$EXPECTED_PARAMETERS"
INITIAL_SHA="$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')"
JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_initialization.py" \
    --arm "$ARM" --teacher-receipt "$TEACHER_RECEIPT" \
    --student-checkpoint "$INITIAL_CHECKPOINT" --bank-root "$BANK" \
    --terra-revision "$TERRA_REVISION" \
    --output "$RUN_DIR/initialization_diagnostic.json"
INITIALIZATION_DIAGNOSTIC_SHA="$(sha256sum "$RUN_DIR/initialization_diagnostic.json" | awk '{print $1}')"

export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_10m_${BASELINES_REVISION:0:10}_${PHASE}_${ARM,,}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled
    export WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online
    export WANDB_DIR="$RUN_DIR/wandb"
fi

RUN_NAME="v8_10m_${BASELINES_REVISION:0:12}_${PHASE}_full_${ARM,,}_s${SEED}"
RUN_NAME="${RUN_NAME//-/_}"
GLOBAL_TRANSITIONS=$((4 * 1024 * 32 * UPDATES))
printf '%s\n' \
    "arm=$ARM" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "absolute_target_update=$UPDATES" \
    "absolute_target_global_transitions=$GLOBAL_TRANSITIONS" \
    "curriculum_stage=full" \
    "reward_type=DENSE" \
    "horizon=450" \
    "full_resets=true" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "teacher_receipt=$TEACHER_RECEIPT" \
    "teacher_receipt_sha256=$TEACHER_RECEIPT_SHA" \
    "teacher_checkpoint=$TEACHER_CHECKPOINT" \
    "teacher_checkpoint_sha256=$TEACHER_SHA" \
    "teacher_update=$TEACHER_UPDATE" \
    "initial_checkpoint_sha256=$INITIAL_SHA" \
    "initialization_diagnostic_sha256=$INITIALIZATION_DIAGNOSTIC_SHA" \
    "model_parameter_count=$OBSERVED_PARAMETERS" \
    "resnet_stage_channels=$EXPECTED_STAGE_CHANNELS" \
    "resnet_blocks_per_stage=2,2,3,3" \
    "num_devices=4" \
    "num_envs_per_device=1024" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.0003" \
    "kickstart_kl=1.0_to_0_over_1500" \
    "kickstart_value=0.5_to_0_over_500" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$TERRA"
export TERRA_REVISION
export PYTHON_BIN="$PYTHON"
export MACHINE=euler
export NUM_DEVICES=4
export NUM_ENVS_PER_DEVICE=1024
export NUM_STEPS=32
export NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL CACHE_CLEAR_INTERVAL
export FINITE_CHECK_INTERVAL=10
if [ "$PHASE" = smoke ]; then export FINITE_CHECK_INTERVAL=1; fi
"$BASELINES_ROOT/scripts/run_v8_10m_screen.sh" \
    "$ARM" "$BANK" "$INITIAL_CHECKPOINT" "$TEACHER_CHECKPOINT" \
    "$RUN_NAME" "$UPDATES" "$RUN_DIR"

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
fi
printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
echo "V8_10M_${PHASE^^}_${ARM}_PASSED"
