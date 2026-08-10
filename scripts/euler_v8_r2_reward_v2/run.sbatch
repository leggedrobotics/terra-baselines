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

EXPECTED_RUNTIME_TERRA_REVISION=3051054bc4c713d95905d3f954e6eabf55d6a85a
EXPECTED_HORIZON=450
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_r2_reward_v2/runs

for REQUIRED in PHASE BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT \
    RUNTIME_TERRA_REVISION R2_HORIZON SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_TREE_SHA BANK_RELEASE_ID DISTANCE_ARTIFACT_SHA \
    MATERIALIZATION_RECEIPT MATERIALIZATION_RECEIPT_SHA \
    STATIC_RECEIPT_MANIFEST STATIC_RECEIPT_MANIFEST_SHA D4A_RECEIPT \
    D4A_RECEIPT_SHA D4A_MANIFEST D4A_MANIFEST_SHA SMOKE_JOB_ID SMOKE_RUN \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "$SEED" = 20260807
test "$RUNTIME_TERRA_REVISION" = "$EXPECTED_RUNTIME_TERRA_REVISION"
test "$R2_HORIZON" -eq "$EXPECTED_HORIZON"

case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        test "$SMOKE_JOB_ID" = none -a "$SMOKE_RUN" = none
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
        UPDATES=40000
        CHECKPOINT_INTERVAL=500
        EVAL_INTERVAL=1000
        LOG_TRAIN_INTERVAL=10
        FINITE_CHECK_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac

RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/s$SEED/reward_v2_scratch"
test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
for SPEC in \
    "$BANK_ARCHIVE:$BANK_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA" \
    "$STATIC_RECEIPT_MANIFEST:$STATIC_RECEIPT_MANIFEST_SHA" \
    "$D4A_RECEIPT:$D4A_RECEIPT_SHA" \
    "$D4A_MANIFEST:$D4A_MANIFEST_SHA"; do
    FILE="${SPEC%:*}" EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$FILE" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR" -a ! -e "$RUN_DIR/run_contract.env"

if [ "$PHASE" = screen ]; then
    SMOKE_STATE="$(sacct -n -X -P -j "$SMOKE_JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$SMOKE_JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$SMOKE_STATE" = COMPLETED
    test -f "$SMOKE_RUN/smoke_validation.json" -a -f "$SMOKE_RUN/run_contract.env"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$SMOKE_RUN/smoke_validation.json"
    test "$(awk -F= '$1=="runtime_terra_revision" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$RUNTIME_TERRA_REVISION"
    test "$(awk -F= '$1=="terra_baselines_revision" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$BASELINES_REVISION"
    test "$(awk -F= '$1=="distance_artifact_sha256" {print $2}' "$SMOKE_RUN/run_contract.env")" = "$DISTANCE_ARTIFACT_SHA"
fi

HOME_USED_GB="$(lquota | awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" = screen ]; then
    module load eth_proxy
    test -r "$HOME/.netrc"
fi
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    if [ "$PHASE" = smoke ]; then
        test "$GPU_NAME" = "NVIDIA GeForce RTX 3090"
    else
        test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
    fi
done
export JAX_PLATFORMS=cuda
"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
from jax import lax

assert len(jax.devices()) == 4
def loss(x, kernel):
    y = lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return lax.psum(jnp.mean(y * y), "devices")
step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((4, 1, 16, 16, 4), dtype=jnp.float32)
k = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, k)
assert gradient.shape == x.shape and jnp.all(jnp.isfinite(gradient))
print(jax.devices(), float(jnp.linalg.norm(gradient)))
PY
"$PYTHON" - <<'PY'
from terra.config import (
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    REWARD_V2_SHAPING_WEIGHT,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SUCCESS_BONUS,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.state import CORRECTED_DENSE_CONTRACT

assert int(RewardStage.REWARD_V2) == 3
assert REWARD_V2_PROTOCOL_ID == "material_potential_v2"
assert REWARD_V2_DISTANCE_PROTOCOL_ID == "obstacle_geodesic_8_physical_global_v1"
assert CORRECTED_DENSE_CONTRACT == "exact_visible_dump_v1"
assert (
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_SUCCESS_BONUS,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SHAPING_WEIGHT,
) == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)
PY

OBSERVED_MAPS="$("$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS" = 96
GRAPH_SHA="$("$PYTHON" -c 'import collections,sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full", sampler_profile="continuous_banded_v2"); assert len(b.levels)==47 and b.map_count_per_condition==96 and b.sampling_probabilities==(); assert collections.Counter(b.curriculum_depths)=={0:2,1:13,2:32}; print(b.curriculum_graph_sha256)' "$BANK" "$PROTOCOL_TERRA_REVISION")"
[[ "$GRAPH_SHA" =~ ^[0-9a-f]{64}$ ]]

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="v8_r2_scratch_${BASELINES_REVISION:0:10}_${PHASE}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
RUN_NAME="v8_r2_scratch_${BASELINES_REVISION:0:12}_s${SEED}"
if [ "$PHASE" = smoke ]; then RUN_NAME="${RUN_NAME}_smoke"; fi
TRANSITIONS=$((4 * 512 * 32 * UPDATES))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=v8_r2_scratch_system_improvement" \
    "comparison_role=historical_descriptive_reference" \
    "historical_baseline_job_id=10128518" \
    "historical_baseline_checkpoint_sha256=0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834" \
    "full_system_selection_updates=1000_to_40000_promotion_only" \
    "promotion_selection_rule=combined_exact_then_47_condition_macro_then_worst_then_earliest" \
    "bundled_changes=reward_v2+global_distance_ledger+carry_input+continuous_banded_v2+runtime_3051054+budget_40000" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "absolute_start_update=0" \
    "absolute_target_update=$UPDATES" \
    "updates=$UPDATES" \
    "transitions=$TRANSITIONS" \
    "support_scope=all47_continuous_from_update0" \
    "condition_count=47" \
    "maps_per_condition=96" \
    "sampler_profile=continuous_banded_v2" \
    "initial_sampler_family_mass_foundation=0.5" \
    "initial_sampler_family_mass_trench=0.5" \
    "initial_sampler_depth_mass_d0=0.11346390374331551" \
    "initial_sampler_depth_mass_d1=0.3836076203208556" \
    "initial_sampler_depth_mass_d2=0.5029284759358292" \
    "curriculum_graph_sha256=$GRAPH_SHA" \
    "reward_stage=reward_v2" \
    "reward_protocol_id=material_potential_v2" \
    "distance_protocol_id=obstacle_geodesic_8_physical_global_v1" \
    "distance_artifact_sha256=$DISTANCE_ARTIFACT_SHA" \
    "carry_channel=agent_states_8_normalized_carry_work" \
    "horizon=$R2_HORIZON" \
    "full_resets=true" \
    "initialization=random_no_teacher" \
    "warm_start_from=none" \
    "prepared_fork_from=none" \
    "optimizer_initial_local_step=0" \
    "entropy_schedule=0.15_to_0.02_over_20000" \
    "ppo_gamma=0.9984" \
    "reward_v2_distance_ref_m=16.0" \
    "reward_v2_distance_bound=2.5" \
    "reward_v2_success_bonus=6.0" \
    "reward_v2_horizon_failure_penalty=1.0" \
    "reward_v2_alpha=1.0" \
    "reward_v2_beta=1.5" \
    "reward_v2_step_cost_total=1.0" \
    "reward_v2_shaping_weight=1.0" \
    "resume_supported=false" \
    "restart_policy=discard_run_restart_from_update0" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "training_bank_tree_sha256=$BANK_TREE_SHA" \
    "materialization_receipt_sha256=$MATERIALIZATION_RECEIPT_SHA" \
    "static_admission_manifest_sha256=$STATIC_RECEIPT_MANIFEST_SHA" \
    "d4a_receipt_sha256=$D4A_RECEIPT_SHA" \
    "d4a_manifest_sha256=$D4A_MANIFEST_SHA" \
    "model_parameter_count=2856701" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.0003" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "planned_fixed_panel_spacing_updates=${EVAL_INTERVAL:-none}" \
    "smoke_job_id=$SMOKE_JOB_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PROTOCOL_TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL FINITE_CHECK_INTERVAL CACHE_CLEAR_INTERVAL
export ENTROPY_SCHEDULE_STEPS=20000
"$BASELINES_ROOT/scripts/run_v8_r2_reward_v2.sh" \
    "$BANK" "$RUN_NAME" "$UPDATES" "$RUN_DIR" "$DISTANCE_ARTIFACT_SHA"

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1
    printf '%s\n' "${matches[0]}"
}

if [ "$PHASE" = smoke ]; then
    find_one "${RUN_NAME}_update_000001.pkl" >/dev/null
    find_one "${RUN_NAME}_FINAL.pkl" >/dev/null
    JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_r2_reward_v2/verify_smoke.py" \
        --run "$RUN_DIR" --distance-artifact-sha256 "$DISTANCE_ARTIFACT_SHA" \
        --output "$RUN_DIR/smoke_validation.json"
    printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
    echo "V8_R2_SCRATCH_SMOKE_PASSED"
    exit 0
fi

find_one "${RUN_NAME}_update_040000.pkl" >/dev/null
printf '%s\n' status=TRAINING_TARGET_REACHED >> "$RUN_DIR/run_contract.env"
CHECKPOINT_ARGS=()
for UPDATE in $(seq "$EVAL_INTERVAL" "$EVAL_INTERVAL" "$UPDATES"); do
    TOKEN="$(printf '%06d' "$UPDATE")"
    CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}_update_${TOKEN}.pkl")")
done
"$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
    "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel promotion \
    --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
    --expect-completion-contract exact_visible_dump_v1 \
    --expect-reward-protocol-id material_potential_v2 \
    --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
    --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
    --require-productive-workspace-cycles \
    --output "$RUN_DIR/eval/promotion.json"
"$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
    "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel promotion \
    --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
    --expect-completion-contract exact_visible_dump_v1 \
    --expect-reward-protocol-id material_potential_v2 \
    --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
    --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
    --require-productive-workspace-cycles \
    --output "$RUN_DIR/eval/capability_promotion.json"
"$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_r2_reward_v2/select_promotion.py" \
    --promotion "$RUN_DIR/eval/promotion.json" \
    --capability-promotion "$RUN_DIR/eval/capability_promotion.json" \
    --output "$RUN_DIR/eval/promotion_selection.json"
SELECTED_CHECKPOINT="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$RUN_DIR/eval/promotion_selection.json")"
SELECTED_UPDATE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_update"])' "$RUN_DIR/eval/promotion_selection.json")"
SELECTED_SHA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_sha256"])' "$RUN_DIR/eval/promotion_selection.json")"
case "$SELECTED_CHECKPOINT" in "$RUN_DIR/checkpoints/"*) ;; *) exit 3 ;; esac
test "$(sha256sum "$SELECTED_CHECKPOINT" | awk '{print $1}')" = "$SELECTED_SHA"

for PANEL in development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$SELECTED_CHECKPOINT" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --expect-reward-protocol-id material_potential_v2 \
        --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
        --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/${PANEL}.json"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$SELECTED_CHECKPOINT" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --expect-reward-protocol-id material_potential_v2 \
        --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
        --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/capability_${PANEL}.json"
done
SELECTION_SHA="$(sha256sum "$RUN_DIR/eval/promotion_selection.json" | awk '{print $1}')"
printf '%s\n' \
    status=EVALUATED \
    "selected_checkpoint_update=$SELECTED_UPDATE" \
    "selected_checkpoint_sha256=$SELECTED_SHA" \
    "promotion_selection_receipt_sha256=$SELECTION_SHA" >> "$RUN_DIR/run_contract.env"
echo "V8_R2_SCRATCH_EVALUATED"
