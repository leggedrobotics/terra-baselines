#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

for REQUIRED in PHASE BASELINES_ROOT BASELINES_REVISION SEED \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_RELEASE_ID SMOKE_JOB_ID SMOKE_RUN SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "$SEED" = 20260807

ARM=v6_3m
EXPECTED_PARAMETERS=2134755

case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        test "$SMOKE_JOB_ID" = none
        test "$SMOKE_RUN" = none
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        FINITE_CHECK_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.120h
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        [[ "$SMOKE_RUN" = /* ]]
        UPDATES=20000
        CHECKPOINT_INTERVAL=500
        EVAL_INTERVAL=1000
        LOG_TRAIN_INTERVAL=10
        FINITE_CHECK_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac

# The bank protocol stays frozen at a6e6e5bc even though the runtime Terra has
# moved on (docs/EXPERIMENTS_RUNNING.md); the runtime revision is submitted.
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/work/rsl/lterenzi/terra_v8_v6_yolo_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/all47/s$SEED/$ARM"
PAIR_ID="v8_v6_yolo_v1_${BASELINES_REVISION:0:12}_${RUNTIME_TERRA_REVISION:0:12}_s${SEED}"

[[ "$RUNTIME_TERRA_REVISION" =~ ^[0-9a-f]{40}$ ]]
test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"

if [ "$PHASE" = screen ]; then
    SMOKE_STATE="$(sacct -n -X -P -j "$SMOKE_JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$SMOKE_JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$SMOKE_STATE" = COMPLETED
    for RECEIPT in smoke_validation.json smoke_sampler_validation.json architecture_smoke_validation.json run_contract.env; do
        test -f "$SMOKE_RUN/$RECEIPT"
    done
    python3 -c 'import json,sys; assert all(json.load(open(p))["passed"] is True for p in sys.argv[1:])' \
        "$SMOKE_RUN/smoke_validation.json" \
        "$SMOKE_RUN/smoke_sampler_validation.json" \
        "$SMOKE_RUN/architecture_smoke_validation.json"
    test "$(awk -F= '$1=="pair_id" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$PAIR_ID"
    test "$(awk -F= '$1=="arm" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$ARM"
    test "$(awk -F= '$1=="reward_stage" {print $2}' "$SMOKE_RUN/run_contract.env")" = dense_skill
    test "$(awk -F= '$1=="sampler_profile" {print $2}' "$SMOKE_RUN/run_contract.env")" = continuous_banded_v2
    test "$(awk -F= '$1=="runtime_terra_revision" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$RUNTIME_TERRA_REVISION"
    test "$(awk -F= '$1=="terra_baselines_revision" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$BASELINES_REVISION"
    test "$(awk -F= '$1=="training_bank_archive_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$BANK_SHA"
    test "$(awk -F= '$1=="training_bank_dataset_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$BANK_DATASET_SHA"
fi

HOME_USED_GB="$(lquota | awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$RUNTIME_TERRA_ROOT"
BANK="$TMPDIR/bank"
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
test "${#GPU_NAMES[@]}" -eq 1
case "${GPU_NAMES[0]}" in
    "NVIDIA GeForce RTX 3090"|"NVIDIA GeForce RTX 4090") ;;
    *) echo "unsupported GPU: ${GPU_NAMES[0]}" >&2; exit 3 ;;
esac
if [ "$PHASE" = screen ]; then
    test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
fi
export JAX_PLATFORMS=cuda
"$PYTHON" -c 'import jax; assert len(jax.devices()) >= 1; print(jax.devices())'
"$PYTHON" -c 'from terra.config import EnvConfig,RewardStage; assert EnvConfig().reward_stage == RewardStage.DENSE_SKILL; assert float(EnvConfig().terminal_reward_mix) == 0.0'
OBSERVED_MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS_PER_CONDITION" = 96
CONTINUOUS_GRAPH_SHA="$("$PYTHON" -c 'import collections,sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full", sampler_profile="continuous_banded_v2"); assert len(b.levels)==47; assert b.map_count_per_condition==96; assert b.sampling_probabilities==(); assert collections.Counter(b.curriculum_depths)=={0:2,1:13,2:32}; print(b.curriculum_graph_sha256)' "$BANK" "$PROTOCOL_TERRA_REVISION")"
[[ "$CONTINUOUS_GRAPH_SHA" =~ ^[0-9a-f]{64}$ ]]

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_v6_yolo_${BASELINES_REVISION:0:10}_${PHASE}_${ARM}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

RUN_NAME="v8_v6_yolo_${BASELINES_REVISION:0:12}_${ARM}_s${SEED}"
if [ "$PHASE" = smoke ]; then
    RUN_NAME="${RUN_NAME}_smoke"
fi
GLOBAL_TRANSITIONS=$((1 * 512 * 32 * UPDATES))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=v8_v6_readout_redesign_yolo_v1" \
    "pair_id=$PAIR_ID" \
    "arm=$ARM" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "updates=$UPDATES" \
    "absolute_target_global_transitions=$GLOBAL_TRANSITIONS" \
    "support_scope=all47_continuous" \
    "condition_count=47" \
    "maps_per_condition=96" \
    "sampler_profile=continuous_banded_v2" \
    "curriculum_graph_sha256=$CONTINUOUS_GRAPH_SHA" \
    "reward_stage=dense_skill" \
    "horizon=450" \
    "full_resets=true" \
    "initialization=random_no_teacher" \
    "teacher_checkpoint=none" \
    "warm_start_from=none" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "model_size=medium" \
    "map_encoder=resnet_spatial_8x8_se_sa_xattn" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "flatten_reduce_channels=32" \
    "attn_latent_queries=8" \
    "token_mixer_residual_init_scale=0.1" \
    "aux_coef=0.25" \
    "vf_coef=0.5" \
    "num_devices=1" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "optimizer_steps_per_update=64" \
    "samples_per_optimizer_step=512" \
    "global_batch_vs_4gpu_reference=one_quarter" \
    "learning_rate=0.0003" \
    "entropy_schedule=0.15_to_0.02_over_${UPDATES}" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "planned_fixed_panel_spacing_updates=${EVAL_INTERVAL:-none}" \
    "primary_comparison=fixed_panel_exact_and_macro_completion" \
    "smoke_job_id=$SMOKE_JOB_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$TERRA" PROTOCOL_TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export NUM_DEVICES=1 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL FINITE_CHECK_INTERVAL CACHE_CLEAR_INTERVAL
"$BASELINES_ROOT/scripts/run_v8_v6_yolo_v1.sh" "$BANK" "$RUN_NAME" "$UPDATES" "$RUN_DIR"

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1
    printf '%s\n' "${matches[0]}"
}

if [ "$PHASE" = smoke ]; then
    PERIODIC="$(find_one "${RUN_NAME}_update_000001.pkl")"
    FINAL="$(find_one "${RUN_NAME}_FINAL.pkl")"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/select_promotion.py" verify-smoke \
        --arm G-UNIFORM --periodic "$PERIODIC" --final "$FINAL" \
        --output "$RUN_DIR/smoke_validation.json"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/verify_continuous_sampler_checkpoint.py" \
        "$PERIODIC" "$FINAL" --output "$RUN_DIR/smoke_sampler_validation.json"
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_v6_yolo_v1/verify_smoke.py" \
        --run "$RUN_DIR" --output "$RUN_DIR/architecture_smoke_validation.json"
    printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
    echo "V8_V6_YOLO_SMOKE_PASSED arm=$ARM"
    exit 0
fi

find_one "${RUN_NAME}_update_020000.pkl" >/dev/null
printf '%s\n' status=TRAINING_TARGET_REACHED >> "$RUN_DIR/run_contract.env"
CHECKPOINT_ARGS=()
for UPDATE in $(seq "$EVAL_INTERVAL" "$EVAL_INTERVAL" "$UPDATES"); do
    TOKEN="$(printf '%06d' "$UPDATE")"
    CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}_update_${TOKEN}.pkl")")
done
for PANEL in promotion development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/${PANEL}.json"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/capability_${PANEL}.json"
done
printf '%s\n' status=EVALUATED >> "$RUN_DIR/run_contract.env"
echo "V8_V6_YOLO_EVALUATED arm=$ARM"
