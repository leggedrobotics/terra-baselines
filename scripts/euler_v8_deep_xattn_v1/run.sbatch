#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-v8-deep-xattn

set -euo pipefail
umask 002

: "${PHASE:?missing PHASE}"
: "${STAGE:?missing STAGE}"
: "${ARM:?missing ARM}"
: "${BASELINES_ROOT:?missing BASELINES_ROOT}"
: "${BASELINES_REVISION:?missing BASELINES_REVISION}"
: "${SEED:?missing SEED}"
: "${SCREEN_UPDATES:?missing SCREEN_UPDATES}"
: "${BANK_ARCHIVE:?missing BANK_ARCHIVE}"
: "${BANK_SHA:?missing BANK_SHA}"
: "${BANK_DATASET_SHA:?missing BANK_DATASET_SHA}"
: "${BANK_RELEASE_ID:?missing BANK_RELEASE_ID}"
: "${PARENT_CHECKPOINT:?missing PARENT_CHECKPOINT}"
: "${PARENT_SHA:?missing PARENT_SHA}"
: "${PRIOR_GATE_RECEIPT:?missing PRIOR_GATE_RECEIPT}"
: "${PRIOR_GATE_SHA:?missing PRIOR_GATE_SHA}"
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${TMPDIR:?Euler did not provide TMPDIR}"

case "$STAGE" in
    capability)
        EXPECTED_CONDITIONS=2
        V8_SAMPLER_PROFILE=bank_v4
        CONDITION_SAMPLER=fixed_v8_stage_weights
        test "$PRIOR_GATE_RECEIPT" = none
        test "$PRIOR_GATE_SHA" = none
        ;;
    nearby)
        EXPECTED_CONDITIONS=15
        V8_SAMPLER_PROFILE=bounded_replay25_v1
        CONDITION_SAMPLER=fixed_v8_bounded_replay
        PRIOR_STAGE=capability
        test "$PRIOR_GATE_RECEIPT" != none
        test "$PRIOR_GATE_SHA" != none
        ;;
    full)
        EXPECTED_CONDITIONS=47
        V8_SAMPLER_PROFILE=bounded_replay25_v1
        CONDITION_SAMPLER=fixed_v8_bounded_replay
        PRIOR_STAGE=nearby
        test "$PRIOR_GATE_RECEIPT" != none
        test "$PRIOR_GATE_SHA" != none
        ;;
    *) echo "stage '$STAGE' is not enabled by this launcher revision" >&2; exit 2 ;;
esac
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
        UPDATES="$SCREEN_UPDATES"
        CHECKPOINT_INTERVAL=500
        LOG_TRAIN_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        test $((UPDATES % 500)) -eq 0
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac
case "$ARM" in
    G-DEEP-V8-DENSE-WARM)
        ARCHITECTURE_LABEL=deep-se
        ;;
    G-DEEP-XATTN-V8-DENSE-WARM)
        ARCHITECTURE_LABEL=deep-se-xattn
        ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/$STAGE/s$SEED/$ARM"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')" = "$PARENT_SHA"
if [ "$STAGE" != capability ]; then
    test "$(sha256sum "$PRIOR_GATE_RECEIPT" | awk '{print $1}')" = "$PRIOR_GATE_SHA"
fi
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"

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
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported GPU: $GPU_NAME" >&2
        exit 3
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$MAPS_PER_CONDITION" = 96
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage=sys.argv[3], sampler_profile=sys.argv[5]); assert len(b.levels)==int(sys.argv[4]); assert b.map_count_per_condition==96; assert abs(sum(b.sampling_probabilities)-1.0)<1e-12; assert b.sampler_profile==sys.argv[5]' "$BANK" "$TERRA_REVISION" "$STAGE" "$EXPECTED_CONDITIONS" "$V8_SAMPLER_PROFILE"
if [ "$STAGE" != capability ]; then
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/stage_gate.py" inspect \
        --receipt "$PRIOR_GATE_RECEIPT" --stage "$PRIOR_STAGE" --arm "$ARM" \
        --expect-candidate "$PARENT_CHECKPOINT" \
        --expect-candidate-sha256 "$PARENT_SHA" --verify-checkpoint >/dev/null
fi

INITIAL="$PARENT_CHECKPOINT"
if [ "$STAGE" = capability ] && [ "$ARM" = G-DEEP-XATTN-V8-DENSE-WARM ]; then
    INITIAL="$TMPDIR/deep_xattn_initial.pkl"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/grow_checkpoint.py" \
        --src "$PARENT_CHECKPOINT" \
        --out "$INITIAL" \
        --map_encoder resnet_spatial_8x8_se_xattn \
        --model_size medium \
        --critic_hidden_dims 512,256 \
        --encoder_compute_dtype bfloat16 \
        --attention_compute_dtype float32 \
        --resnet_stage_channels 24,48,64,96 \
        --resnet_blocks_per_stage 2,2,3,3
fi
INITIAL_SHA="$(sha256sum "$INITIAL" | awk '{print $1}')"
if [ "$STAGE" = capability ]; then
    TEACHER_CHECKPOINT="$PARENT_CHECKPOINT"
    TEACHER_SHA="$PARENT_SHA"
    INITIALIZATION=params_only_warm_fresh_optimizer
else
    TEACHER_CHECKPOINT=none
    TEACHER_SHA=none
    INITIALIZATION=params_only_stage_transition_fresh_optimizer
fi

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_${BASELINES_REVISION:0:10}_${PHASE}_${STAGE}_${ARM,,}_${SLURM_JOB_ID}"
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

RUN_NAME="v8_${BASELINES_REVISION:0:12}_${PHASE}_${STAGE}_${ARM,,}_s${SEED}"
RUN_NAME="${RUN_NAME//-/_}"
GLOBAL_TRANSITIONS=$((4 * 1024 * 32 * UPDATES))
printf '%s\n' \
    "arm=$ARM" \
    "architecture=$ARCHITECTURE_LABEL" \
    "curriculum_stage=$STAGE" \
    "reward_stage=dense_skill" \
    "reward_type=DENSE" \
    "condition_sampler=$CONDITION_SAMPLER" \
    "sampler_profile=$V8_SAMPLER_PROFILE" \
    "condition_count=$EXPECTED_CONDITIONS" \
    "seed=$SEED" \
    "phase=$PHASE" \
    "updates=$UPDATES" \
    "global_transitions=$GLOBAL_TRANSITIONS" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "parent_checkpoint_path=$PARENT_CHECKPOINT" \
    "parent_checkpoint_sha256=$PARENT_SHA" \
    "initial_checkpoint_sha256=$INITIAL_SHA" \
    "teacher_checkpoint_sha256=$TEACHER_SHA" \
    "prior_gate_receipt=$PRIOR_GATE_RECEIPT" \
    "prior_gate_receipt_sha256=$PRIOR_GATE_SHA" \
    "initialization=$INITIALIZATION" \
    "trench_shaping=false" \
    "horizon=450" \
    "full_resets=true" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "wandb_run_id=$WANDB_RUN_ID" > "$RUN_DIR/run_contract.env"

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
export ENT_SCHEDULE_START=0.02
export ENT_SCHEDULE_END=0.005
export ENT_SCHEDULE_STEPS=10000
if [ "$PHASE" = smoke ]; then export FINITE_CHECK_INTERVAL=1; fi
TRAIN_COMMAND=("$BASELINES_ROOT/scripts/run_v8_warm_screen.sh" "$ARM" "$BANK" "$STAGE" "$INITIAL" "$TEACHER_CHECKPOINT" "$RUN_NAME" "$UPDATES")
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.sh"
printf '\n' >> "$RUN_DIR/train_command.sh"
"${TRAIN_COMMAND[@]}"

# Full-stage and 120-hour allocations are evaluated in a separate `afterany`
# job.  Keeping evaluation out of the training tail preserves every complete
# checkpoint even when Slurm stops training at the wall-time boundary.
if [ "$PHASE" = screen ] && [ "$STAGE" = full ]; then
    printf '%s\n' status=TRAINING_FINISHED >> "$RUN_DIR/run_contract.env"
    echo "V8_SCREEN_FULL_${ARM}_TRAINING_FINISHED"
    exit 0
fi

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1 || {
        echo "expected one checkpoint matching $pattern, got ${#matches[@]}" >&2
        return 1
    }
    printf '%s\n' "${matches[0]}"
}

if [ "$PHASE" = smoke ]; then
    PERIODIC="$(find_one "${RUN_NAME}-euler-*_update_000001.pkl")"
    FINAL="$(find_one "${RUN_NAME}-euler-*_FINAL.pkl")"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/select_promotion.py" verify-smoke \
        --arm G-UNIFORM --periodic "$PERIODIC" --final "$FINAL" \
        --output "$RUN_DIR/smoke_validation.json"
else
    CHECKPOINT_ARGS=()
    for UPDATE in $(seq 500 500 "$UPDATES"); do
        TOKEN="$(printf '%06d' "$UPDATE")"
        CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}-euler-*_update_${TOKEN}.pkl")")
    done
    for PANEL in promotion development; do
        PANEL_OPTION=--accepted-panel
        if [ "$STAGE" = capability ]; then PANEL_OPTION=--capability-panel; fi
        JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
            "${CHECKPOINT_ARGS[@]}" \
            --bank-root "$BANK" "$PANEL_OPTION" "$PANEL" \
            --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$RUN_DIR/eval/${PANEL}.json"
    done
    if [ "$STAGE" != capability ]; then
        for PANEL in promotion development; do
            JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
                "${CHECKPOINT_ARGS[@]}" \
                --bank-root "$BANK" --capability-panel "$PANEL" \
                --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
                --expect-completion-contract exact_visible_dump_v1 \
                --output "$RUN_DIR/eval/capability_${PANEL}.json"
        done
    fi
    GATE_ARGS=(
        promote --stage "$STAGE" --arm "$ARM" --bank-root "$BANK"
        --promotion "$RUN_DIR/eval/promotion.json"
        --run-contract "$RUN_DIR/run_contract.env"
        --output "$RUN_DIR/stage_gate.json"
    )
    if [ "$STAGE" != capability ]; then
        GATE_ARGS+=(
            --capability "$RUN_DIR/eval/capability_promotion.json"
            --prior-receipt "$PRIOR_GATE_RECEIPT"
        )
    fi
    JAX_PLATFORMS=cpu "$PYTHON" \
        "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/stage_gate.py" \
        "${GATE_ARGS[@]}"
fi

printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
echo "V8_${PHASE^^}_${STAGE^^}_${ARM}_PASSED"
