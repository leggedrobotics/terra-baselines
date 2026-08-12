#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

for REQUIRED in PHASE ARM BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE \
    BANK_SHA BANK_DATASET_SHA BANK_RELEASE_ID PARENT_CHECKPOINT PARENT_SHA \
    PARENT_SELECTION PARENT_SELECTION_SHA TEACHER_CHECKPOINT TEACHER_SHA \
    TEACHER_RUN_CONTRACT TEACHER_RUN_CONTRACT_SHA TEACHER_FIXED_EVAL_ROOT \
    TEACHER_FIXED_EVAL_STATUS_SHA TEACHER_FIXED_EVAL_MAIN_PROMOTION_SHA \
    TEACHER_FIXED_EVAL_MAIN_DEVELOPMENT_SHA \
    TEACHER_FIXED_EVAL_CAPABILITY_PROMOTION_SHA \
    TEACHER_FIXED_EVAL_CAPABILITY_DEVELOPMENT_SHA \
    SMOKE_JOB_ID SMOKE_RUN \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

STAGE=nearby
EXPECTED_CONDITIONS=15
MAPS_PER_CONDITION=96
DISTINCT_TRAINING_MAPS=1440
V8_SAMPLER_PROFILE=bounded_replay25_v1
case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        test "$SMOKE_JOB_ID" = none
        test "$SMOKE_RUN" = none
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.120h
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        [[ "$SMOKE_RUN" = /* ]]
        UPDATES=20000
        CHECKPOINT_INTERVAL=1000
        LOG_TRAIN_INTERVAL=10
        CACHE_CLEAR_INTERVAL=2000
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
RUN_BASE=/cluster/work/rsl/lterenzi/terra_v8_10m_nearby_long_v1
RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/$STAGE/s$SEED/$ARM"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')" = "$PARENT_SHA"
test "$(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}')" = "$TEACHER_SHA"
test "$(sha256sum "$TEACHER_RUN_CONTRACT" | awk '{print $1}')" = "$TEACHER_RUN_CONTRACT_SHA"
test "$(sha256sum "$TEACHER_FIXED_EVAL_ROOT/status.env" | awk '{print $1}')" = "$TEACHER_FIXED_EVAL_STATUS_SHA"
test "$(awk -F= '$1=="status" {print $2}' "$TEACHER_FIXED_EVAL_ROOT/status.env")" = PASSED
test "$(sha256sum "$TEACHER_FIXED_EVAL_ROOT/eval/main_promotion.json" | awk '{print $1}')" = "$TEACHER_FIXED_EVAL_MAIN_PROMOTION_SHA"
test "$(sha256sum "$TEACHER_FIXED_EVAL_ROOT/eval/main_development.json" | awk '{print $1}')" = "$TEACHER_FIXED_EVAL_MAIN_DEVELOPMENT_SHA"
test "$(sha256sum "$TEACHER_FIXED_EVAL_ROOT/eval/capability_promotion.json" | awk '{print $1}')" = "$TEACHER_FIXED_EVAL_CAPABILITY_PROMOTION_SHA"
test "$(sha256sum "$TEACHER_FIXED_EVAL_ROOT/eval/capability_development.json" | awk '{print $1}')" = "$TEACHER_FIXED_EVAL_CAPABILITY_DEVELOPMENT_SHA"
test "$(sha256sum "$PARENT_SELECTION" | awk '{print $1}')" = "$PARENT_SELECTION_SHA"
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/run_contract.env"
if [ "$PHASE" = screen ]; then
    SMOKE_STATE="$(sacct -n -X -P -j "$SMOKE_JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$SMOKE_JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$SMOKE_STATE" = COMPLETED
    test -f "$SMOKE_RUN/smoke_validation.json"
    test -f "$SMOKE_RUN/run_contract.env"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$SMOKE_RUN/smoke_validation.json"
    test "$(awk -F= '$1=="status" {print $2}' "$SMOKE_RUN/run_contract.env")" = PASSED
    test "$(awk -F= '$1=="parent_checkpoint_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$PARENT_SHA"
    test "$(awk -F= '$1=="parent_selection_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$PARENT_SELECTION_SHA"
    test "$(awk -F= '$1=="teacher_checkpoint_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_SHA"
    test "$(awk -F= '$1=="teacher_fixed_eval_status_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_FIXED_EVAL_STATUS_SHA"
    test "$(awk -F= '$1=="teacher_fixed_eval_main_promotion_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_FIXED_EVAL_MAIN_PROMOTION_SHA"
    test "$(awk -F= '$1=="teacher_fixed_eval_main_development_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_FIXED_EVAL_MAIN_DEVELOPMENT_SHA"
    test "$(awk -F= '$1=="teacher_fixed_eval_capability_promotion_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_FIXED_EVAL_CAPABILITY_PROMOTION_SHA"
    test "$(awk -F= '$1=="teacher_fixed_eval_capability_development_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$TEACHER_FIXED_EVAL_CAPABILITY_DEVELOPMENT_SHA"
fi

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
OBSERVED_MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition "$MAPS_PER_CONDITION" --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS_PER_CONDITION" = "$MAPS_PER_CONDITION"
"$PYTHON" -c 'import sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="nearby", sampler_profile="bounded_replay25_v1"); assert len(b.levels)==15; assert b.map_count_per_condition==96; assert abs(sum(b.sampling_probabilities)-1.0)<1e-12' "$BANK" "$TERRA_REVISION"
SELECTION_INFO="$(JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_stage_b_selection.py" inspect --receipt "$PARENT_SELECTION" --arm "$ARM")"
read -r SELECTED_PATH SELECTED_SHA SELECTED_UPDATE RECEIPT_SHA <<< "$("$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["path"],d["sha256"],d["update"],d["receipt_sha256"])' <<< "$SELECTION_INFO")"
test "$SELECTED_PATH" = "$PARENT_CHECKPOINT"
test "$SELECTED_SHA" = "$PARENT_SHA"
test "$RECEIPT_SHA" = "$PARENT_SELECTION_SHA"
JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_provisional_teacher.py" \
    --checkpoint "$TEACHER_CHECKPOINT" --checkpoint-sha256 "$TEACHER_SHA" \
    --run-contract "$TEACHER_RUN_CONTRACT" --bank-root "$BANK" \
    --output "$RUN_DIR/teacher_inspection.json"
TEACHER_INSPECTION_SHA="$(sha256sum "$RUN_DIR/teacher_inspection.json" | awk '{print $1}')"
read -r TEACHER_UPDATE TEACHER_PARAMETERS <<< "$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] is True; print(d["teacher_update"],d["teacher_parameter_count"])' "$RUN_DIR/teacher_inspection.json")"
test "$TEACHER_UPDATE" = 7500
test "$TEACHER_PARAMETERS" = 2856685

OBSERVED_PARAMETERS="$(JAX_PLATFORMS=cpu "$PYTHON" -c 'import sys,jax; from utils.helpers import load_pkl_object,register_checkpoint_config_classes; register_checkpoint_config_classes(); c=load_pkl_object(sys.argv[1]); print(sum(x.size for x in jax.tree_util.tree_leaves(c["model"])))' "$PARENT_CHECKPOINT" | tail -1)"
test "$OBSERVED_PARAMETERS" = "$EXPECTED_PARAMETERS"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_10m_b_${BASELINES_REVISION:0:10}_${PHASE}_${ARM,,}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

RUN_NAME="v8_10m_b_${BASELINES_REVISION:0:12}_${PHASE}_${STAGE}_${ARM,,}_s${SEED}"
RUN_NAME="${RUN_NAME//-/_}"
GLOBAL_TRANSITIONS=$((4 * 512 * 32 * UPDATES))
printf '%s\n' \
    "arm=$ARM" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "updates=$UPDATES" \
    "absolute_target_global_transitions=$GLOBAL_TRANSITIONS" \
    "curriculum_stage=$STAGE" \
    "condition_sampler=fixed_v8_bounded_replay" \
    "sampler_profile=$V8_SAMPLER_PROFILE" \
    "condition_count=$EXPECTED_CONDITIONS" \
    "maps_per_condition=$MAPS_PER_CONDITION" \
    "distinct_training_maps=$DISTINCT_TRAINING_MAPS" \
    "population_capability_replay=0.25" \
    "population_nearby_core=0.75" \
    "population_foundation=0.50" \
    "population_trench=0.50" \
    "reward_stage=dense_skill" \
    "reward_type=DENSE" \
    "reward_transition_launched=false" \
    "trench_shaping=false" \
    "horizon=450" \
    "full_resets=true" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "parent_checkpoint=$PARENT_CHECKPOINT" \
    "parent_checkpoint_sha256=$PARENT_SHA" \
    "parent_checkpoint_update=$SELECTED_UPDATE" \
    "parent_selection=$PARENT_SELECTION" \
    "parent_selection_sha256=$PARENT_SELECTION_SHA" \
    "initial_checkpoint_sha256=$PARENT_SHA" \
    "teacher_checkpoint=$TEACHER_CHECKPOINT" \
    "teacher_checkpoint_sha256=$TEACHER_SHA" \
    "teacher_checkpoint_update=$TEACHER_UPDATE" \
    "teacher_parameter_count=$TEACHER_PARAMETERS" \
    "teacher_admission=fixed_whole_v8_eval_v1" \
    "teacher_source_run_contract=$TEACHER_RUN_CONTRACT" \
    "teacher_source_run_contract_sha256=$TEACHER_RUN_CONTRACT_SHA" \
    "teacher_inspection=$RUN_DIR/teacher_inspection.json" \
    "teacher_inspection_sha256=$TEACHER_INSPECTION_SHA" \
    "teacher_fixed_eval_root=$TEACHER_FIXED_EVAL_ROOT" \
    "teacher_fixed_eval_status_sha256=$TEACHER_FIXED_EVAL_STATUS_SHA" \
    "teacher_fixed_eval_main_promotion_sha256=$TEACHER_FIXED_EVAL_MAIN_PROMOTION_SHA" \
    "teacher_fixed_eval_main_development_sha256=$TEACHER_FIXED_EVAL_MAIN_DEVELOPMENT_SHA" \
    "teacher_fixed_eval_capability_promotion_sha256=$TEACHER_FIXED_EVAL_CAPABILITY_PROMOTION_SHA" \
    "teacher_fixed_eval_capability_development_sha256=$TEACHER_FIXED_EVAL_CAPABILITY_DEVELOPMENT_SHA" \
    "teacher_fixed_promotion_exact=552/720" \
    "teacher_fixed_promotion_macro=0.8747645628069424" \
    "teacher_fixed_development_exact=538/720" \
    "teacher_fixed_development_macro=0.8676237401759459" \
    "teacher_fixed_capability_promotion_exact=31/32" \
    "teacher_fixed_capability_development_exact=31/32" \
    "initialization=params_only_stage_transition_fresh_optimizer" \
    "model_parameter_count=$OBSERVED_PARAMETERS" \
    "resnet_stage_channels=$EXPECTED_STAGE_CHANNELS" \
    "resnet_blocks_per_stage=2,2,3,3" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.00015" \
    "kickstart=current_rollout_kl1_3000_value0p5_1000" \
    "kickstart_kl_coef=1.0" \
    "kickstart_kl_anneal_updates=3000" \
    "kickstart_value_coef=0.5" \
    "kickstart_value_anneal_updates=1000" \
    "kickstart_lr_warmup_updates=200" \
    "entropy_schedule=0.02_to_0.005_over_20000" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "smoke_job_id=$SMOKE_JOB_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$TERRA" TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
export V8_SAMPLER_PROFILE CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL CACHE_CLEAR_INTERVAL
export FINITE_CHECK_INTERVAL=10
if [ "$PHASE" = smoke ]; then export FINITE_CHECK_INTERVAL=1; fi
"$BASELINES_ROOT/scripts/run_v8_10m_screen.sh" \
    "$ARM" "$BANK" "$STAGE" "$PARENT_CHECKPOINT" "$TEACHER_CHECKPOINT" \
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
    printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
    echo "V8_10M_STAGE_B_SMOKE_PASSED arm=$ARM"
    exit 0
fi

CHECKPOINT_ARGS=()
for UPDATE in $(seq 1000 1000 20000); do
    TOKEN="$(printf '%06d' "$UPDATE")"
    CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}-euler-*_update_${TOKEN}.pkl")")
done
for PANEL in promotion development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$RUN_DIR/eval/${PANEL}.json"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$RUN_DIR/eval/capability_${PANEL}.json"
done
JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_stage_b_gate.py" \
    --arm "$ARM" --run-contract "$RUN_DIR/run_contract.env" \
    --parent-selection "$PARENT_SELECTION" --bank-root "$BANK" \
    --promotion "$RUN_DIR/eval/promotion.json" \
    --development "$RUN_DIR/eval/development.json" \
    --capability-promotion "$RUN_DIR/eval/capability_promotion.json" \
    --capability-development "$RUN_DIR/eval/capability_development.json" \
    --output "$RUN_DIR/stage_gate.json"
GATE_PASSED="$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$RUN_DIR/stage_gate.json")"
printf '%s\n' "status=EVALUATED" "stage_gate_passed=$GATE_PASSED" >> "$RUN_DIR/run_contract.env"
echo "V8_10M_STAGE_B_EVALUATED arm=$ARM gate=$GATE_PASSED"
