#!/usr/bin/env bash
set -euo pipefail

EXPECTED_RUNTIME_TERRA_REVISION=3051054bc4c713d95905d3f954e6eabf55d6a85a
R2_HORIZON=450
if [ "$#" -ne 1 ]; then
    echo "usage: submit.sh smoke|screen" >&2
    exit 2
fi
PHASE="$1"
RUNTIME_TERRA_REVISION="$EXPECTED_RUNTIME_TERRA_REVISION"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_v8_r2_reward_v2_20260810}"
ARTIFACT_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_training_inputs_20260810
ADMISSION_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_admission_20260810
BASE_BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
TREATMENT_BANK_ARCHIVE="$ARTIFACT_ROOT/treatment_bank.tar.zst"
PREPARED_FORK="$ARTIFACT_ROOT/prepared_fork.pkl"
PREPARED_RECEIPT="$ARTIFACT_ROOT/prepared_fork_receipt.json"
MATERIALIZATION_RECEIPT="$ARTIFACT_ROOT/treatment_bank_receipt.json"
STATIC_RECEIPT_MANIFEST="$ADMISSION_ROOT/static_v2/receipt_manifest.json"
D4A_RECEIPT="$ADMISSION_ROOT/d4a/d4a_receipt.json"
D4A_MANIFEST="$ADMISSION_ROOT/d4a/receipt_manifest.json"

SEED=20260807
CAMPAIGN=terra_v8_r2_reward_v2
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BASE_BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BASE_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
BASE_TREE_SHA=08d32fd198803dc6a7539b43c818534ad09e8a0bbe21daf7770ede9cd13b7528
TREATMENT_BANK_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
TREATMENT_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
TREATMENT_TREE_SHA=225e13aacd9047e7f241facd3397fd66794e3094a883cc6dc26304decc24d388
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
STATIC_RECEIPT_MANIFEST_SHA=9b16c391dbe0c108f4b79833f1940c5fc0ba31903a1e7edbfec1797aa53740d9
PREPARED_SHA=8e01ebd3dfd99b36cea90a251dfe4a4e305228abeb2f5ecba633a9fc6805b1d0
PREPARED_RECEIPT_SHA=d119f443613d4959d5f63918971c50c5ad204e4b6c1d65ec985c3fc31b005185
MATERIALIZATION_RECEIPT_SHA=631fac8c3b78ff2c5a9e94ea4032244c9ef05dc6c984b603e4318121a263d3f1
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN/runs
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN/inputs
ARMS=(control reward_v2)

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean" >&2; exit 3;
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "runtime Terra must be committed and clean" >&2; exit 3;
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$EXPECTED_RUNTIME_TERRA_REVISION"
LOCAL_TERRA_PYTHON=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
test -x "$LOCAL_TERRA_PYTHON"
PYTHONPATH="$TERRA_REPO" JAX_PLATFORMS=cpu PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$LOCAL_TERRA_PYTHON" - <<'PY'
from terra.config import (
    DENSE_REWARD_PROTOCOL_ID,
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
from terra.maps_buffer import LEGACY_DISTANCE_PROTOCOL_ID
from terra.state import CORRECTED_DENSE_CONTRACT

assert int(RewardStage.REWARD_V2) == 3
assert DENSE_REWARD_PROTOCOL_ID == "dense_skill_legacy_relocation_v1"
assert REWARD_V2_PROTOCOL_ID == "material_potential_v2"
assert LEGACY_DISTANCE_PROTOCOL_ID == "legacy_dataset_distance"
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

for SPEC in \
    "$BASE_BANK_ARCHIVE:$BASE_BANK_SHA" \
    "$TREATMENT_BANK_ARCHIVE:$TREATMENT_BANK_SHA" \
    "$PREPARED_FORK:$PREPARED_SHA" \
    "$PREPARED_RECEIPT:$PREPARED_RECEIPT_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA" \
    "$STATIC_RECEIPT_MANIFEST:$STATIC_RECEIPT_MANIFEST_SHA"; do
    PATH_LOCAL="${SPEC%:*}"
    EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$PATH_LOCAL" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "$(tar --zstd -xOf "$BASE_BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BASE_DATASET_SHA"
test "$(tar --zstd -xOf "$TREATMENT_BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$TREATMENT_DATASET_SHA"
python3 - "$MATERIALIZATION_RECEIPT" "$PREPARED_RECEIPT" <<'PY'
import json, sys
material, prepared = (json.load(open(path)) for path in sys.argv[1:])
assert material["schema"] == "terra_v8_r2_materialized_distance_bank_v1"
assert material["status"] == "passed"
assert material["base_bank"]["unchanged"] is True
assert material["pair_equivalence"]["scenarios"] == 7520
assert material["pair_equivalence"]["physical_arrays_preserved"] is True
assert material["pair_equivalence"]["metadata_and_pose_sidecars_preserved"] is True
assert prepared["schema"] == "terra_v8_r2_prepared_fork_v1"
assert prepared["passed"] is True and prepared["output_preserving"] is True
assert prepared["source_checkpoint_sha256"] == "0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834"
assert prepared["target_sampler_rule"] == "continuous_banded_v2"
assert prepared["target_config_name"] == "G-V8-CONTINUOUS-V2"
assert prepared["target_bank_sampler_profile"] == "continuous_banded_v2"
PY

test -f "$D4A_RECEIPT" -a -f "$D4A_MANIFEST" || {
    echo "D4a durable replay receipt is still missing; R2 launch remains blocked" >&2
    exit 4
}
D4A_RECEIPT_SHA="$(sha256sum "$D4A_RECEIPT" | awk '{print $1}')"
D4A_MANIFEST_SHA="$(sha256sum "$D4A_MANIFEST" | awk '{print $1}')"
python3 - "$D4A_RECEIPT" "$D4A_MANIFEST" "$D4A_RECEIPT_SHA" <<'PY'
import hashlib, json, pathlib, sys
receipt = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
assert receipt["schema"] == "terra_v8_r2_d4a_replay_v1"
assert receipt["status"] == "passed"
assert receipt["all_targeted_traces_match_frozen_rows"] is True
assert receipt["targeted_trace_count"] == 9
targeted = receipt["frozen_parity"]
assert targeted["episodes"] == 720 and targeted["all_nine_equal"] is True
assert all(targeted["targeted_checks"].values())
assert targeted["cross_hardware_policy_drift_is_non_gating"] is True
assert receipt["reset_verification"]["passed"] is True
assert receipt["action_tensor"]["shape"] == [450, 720]
assert receipt["action_tensor"]["dtype"] == "int32"
command = receipt["command_contract"]
assert command["seed"] == 20260807 and command["horizon"] == 450
assert command["deterministic"] is True and command["wandb"] == "disabled"
assert command["baselines_revision"] == "dcc4f955347182e57e6f16e9df81a3f170564d97"
assert command["terra_revision"] == "eb3835c1d17af81e970b973ed5abf687ca6f3a26"
assert command["bank_declared_terra_revision"] == "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
assert receipt["analysis_script_sha256"] == "268f73abfbfea05eff0ee3a41b7995fd6900b5a3268ba5ae91e7153a5d8dc7a4"
assert receipt["analysis_support_sha256"] == "8c182bbcb906d581222b6637ad7d45ca45b196fe748b5a867cee44c46510e3a7"
assert receipt["source_file_sha256"]["scripts/analysis/d4a_ledger.py"] == receipt["analysis_support_sha256"]
ledger = receipt["ledger"]
lift_tolerance = ledger["tolerance"]["lift"]
assert lift_tolerance["rule"] == "max(absolute_floor, ulp_multiplier * max_float32_spacing)"
assert lift_tolerance["absolute_floor"] == 1e-6
assert lift_tolerance["ulp_multiplier"] == 4.0
assert ledger["lift_gate_passed"] is True
assert ledger["failed_lift_event_count"] == 0
assert ledger["max_inert_transition_error"] <= ledger["tolerance"]["inert"]
assert ledger["max_dump_progress_telescope_error"] <= ledger["tolerance"]["telescope"]
diagnostics_path = pathlib.Path(sys.argv[1]).parent / receipt["lift_diagnostics"]
diagnostics = json.load(open(diagnostics_path))
assert diagnostics["schema"] == "terra_v8_r2_d4a_lift_diagnostics_v1"
assert diagnostics["status"] == "passed"
assert diagnostics["failed_lift_event_count"] == 0
assert diagnostics["written_before_gate_raise"] is True
diagnostics_sha = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
assert diagnostics_sha == receipt["lift_diagnostics_sha256"]
assert manifest["files"][diagnostics_path.name]["sha256"] == diagnostics_sha
assert manifest["files"]["d4a_receipt.json"]["sha256"] == sys.argv[3]
PY

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
echo "phase=$PHASE seed=$SEED absolute_start=20000 additional_updates=$([ "$PHASE" = smoke ] && echo 1 || echo 6000) arms=${ARMS[*]}"
echo "terra_baselines_revision=$BASELINES_REVISION runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "d4a_receipt_sha256=$D4A_RECEIPT_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
    exit 0
fi

if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_TERRA/REVISION'"; then
    PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION' && test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"

upload() {
    local source="$1" destination="$2" expected_sha="$3"
    if ! ssh "$REMOTE_HOST" "test -f '$destination'"; then
        local partial="$destination.partial.$$"
        scp -q "$source" "$REMOTE_HOST:$partial"
        ssh "$REMOTE_HOST" "test \"\$(sha256sum '$partial' | awk '{print \$1}')\" = '$expected_sha' && mv '$partial' '$destination'"
    fi
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected_sha'"
}

REMOTE_BASE_BANK="$REMOTE_INPUTS/base-bank-$BASE_BANK_SHA.tar.zst"
REMOTE_TREATMENT_BANK="$REMOTE_INPUTS/treatment-bank-$TREATMENT_BANK_SHA.tar.zst"
REMOTE_PREPARED="$REMOTE_INPUTS/prepared-$PREPARED_SHA.pkl"
REMOTE_PREPARED_RECEIPT="$REMOTE_INPUTS/prepared-receipt-$PREPARED_RECEIPT_SHA.json"
REMOTE_MATERIALIZATION_RECEIPT="$REMOTE_INPUTS/materialization-$MATERIALIZATION_RECEIPT_SHA.json"
REMOTE_STATIC_MANIFEST="$REMOTE_INPUTS/static-admission-$STATIC_RECEIPT_MANIFEST_SHA.json"
REMOTE_D4A_RECEIPT="$REMOTE_INPUTS/d4a-$D4A_RECEIPT_SHA.json"
REMOTE_D4A_MANIFEST="$REMOTE_INPUTS/d4a-manifest-$D4A_MANIFEST_SHA.json"
upload "$BASE_BANK_ARCHIVE" "$REMOTE_BASE_BANK" "$BASE_BANK_SHA"
upload "$TREATMENT_BANK_ARCHIVE" "$REMOTE_TREATMENT_BANK" "$TREATMENT_BANK_SHA"
upload "$PREPARED_FORK" "$REMOTE_PREPARED" "$PREPARED_SHA"
upload "$PREPARED_RECEIPT" "$REMOTE_PREPARED_RECEIPT" "$PREPARED_RECEIPT_SHA"
upload "$MATERIALIZATION_RECEIPT" "$REMOTE_MATERIALIZATION_RECEIPT" "$MATERIALIZATION_RECEIPT_SHA"
upload "$STATIC_RECEIPT_MANIFEST" "$REMOTE_STATIC_MANIFEST" "$STATIC_RECEIPT_MANIFEST_SHA"
upload "$D4A_RECEIPT" "$REMOTE_D4A_RECEIPT" "$D4A_RECEIPT_SHA"
upload "$D4A_MANIFEST" "$REMOTE_D4A_MANIFEST" "$D4A_MANIFEST_SHA"

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=23:45:00; GPU_TYPE=rtx_4090 ;;
esac
declare -A SMOKE_JOB_IDS=()
declare -A SMOKE_RUNS=()
if [ "$PHASE" = screen ]; then
    for ARM in "${ARMS[@]}"; do
        SMOKE_RUN="$REMOTE_RUNS/$BASELINES_REVISION/smoke/s$SEED/$ARM"
        SMOKE_RUNS[$ARM]="$SMOKE_RUN"
        ssh "$REMOTE_HOST" "test -f '$SMOKE_RUN/smoke_validation.json' -a -f '$SMOKE_RUN/run_contract.env'"
        ssh "$REMOTE_HOST" "python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))[\"passed\"] is True' '$SMOKE_RUN/smoke_validation.json'"
        SMOKE_JOB_ID="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        SMOKE_JOB_IDS[$ARM]="$SMOKE_JOB_ID"
        SMOKE_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
        test "$SMOKE_STATE" = COMPLETED
        for EXPECTED in \
            "arm=$ARM" \
            "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
            "terra_baselines_revision=$BASELINES_REVISION" \
            "prepared_fork_sha256=$PREPARED_SHA" \
            "d4a_receipt_sha256=$D4A_RECEIPT_SHA"; do
            KEY="${EXPECTED%%=*}" VALUE="${EXPECTED#*=}"
            ssh "$REMOTE_HOST" "test \"\$(awk -F= -v key='$KEY' '\$1==key {print \$2}' '$SMOKE_RUN/run_contract.env')\" = '$VALUE'"
        done
    done
fi

RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/s$SEED"
ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT'"
declare -a NEW_JOB_IDS=()
declare -a NEW_RUN_DIRS=()
declare -A PAIR_JOB_IDS=()

cleanup_new_pair() {
    local status="$1"
    trap - ERR INT TERM
    set +e
    if [ "${#NEW_JOB_IDS[@]}" -gt 0 ]; then
        ssh "$REMOTE_HOST" "scancel -- ${NEW_JOB_IDS[*]}"
    fi
    for RUN_DIR in "${NEW_RUN_DIRS[@]}"; do
        # rmdir is intentionally the only cleanup: non-empty evidence survives.
        ssh "$REMOTE_HOST" "rmdir -- '$RUN_DIR'" || true
    done
    exit "$status"
}
trap 'cleanup_new_pair $?' ERR
trap 'cleanup_new_pair 130' INT TERM

for ARM in "${ARMS[@]}"; do
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "test ! -e '$RUN_DIR' && mkdir '$RUN_DIR'"
    NEW_RUN_DIRS+=("$RUN_DIR")
done
for ARM in "${ARMS[@]}"; do
    case "$ARM" in
        control)
            REWARD_STAGE=dense_skill
            DISTANCE_PROTOCOL_ID=legacy_dataset_distance
            DISTANCE_ARTIFACT_SHA=$BASE_DATASET_SHA
            BANK_ARCHIVE=$REMOTE_BASE_BANK
            BANK_SHA=$BASE_BANK_SHA
            BANK_DATASET_SHA=$BASE_DATASET_SHA
            BANK_TREE_SHA=$BASE_TREE_SHA
            ;;
        reward_v2)
            REWARD_STAGE=reward_v2
            DISTANCE_PROTOCOL_ID=obstacle_geodesic_8_physical_global_v1
            DISTANCE_ARTIFACT_SHA=$DISTANCE_SIDECAR_SHA
            BANK_ARCHIVE=$REMOTE_TREATMENT_BANK
            BANK_SHA=$TREATMENT_BANK_SHA
            BANK_DATASET_SHA=$TREATMENT_DATASET_SHA
            BANK_TREE_SHA=$TREATMENT_TREE_SHA
            ;;
    esac
    RUN_DIR="$RUN_PARENT/$ARM"
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,REWARD_STAGE=$REWARD_STAGE,DISTANCE_PROTOCOL_ID=$DISTANCE_PROTOCOL_ID,DISTANCE_ARTIFACT_SHA=$DISTANCE_ARTIFACT_SHA,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,R2_HORIZON=$R2_HORIZON,SEED=$SEED,BANK_ARCHIVE=$BANK_ARCHIVE,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_TREE_SHA=$BANK_TREE_SHA,BANK_RELEASE_ID=$RELEASE_ID,PREPARED_FORK=$REMOTE_PREPARED,PREPARED_SHA=$PREPARED_SHA,PREPARED_RECEIPT=$REMOTE_PREPARED_RECEIPT,PREPARED_RECEIPT_SHA=$PREPARED_RECEIPT_SHA,MATERIALIZATION_RECEIPT=$REMOTE_MATERIALIZATION_RECEIPT,MATERIALIZATION_RECEIPT_SHA=$MATERIALIZATION_RECEIPT_SHA,STATIC_RECEIPT_MANIFEST=$REMOTE_STATIC_MANIFEST,STATIC_RECEIPT_MANIFEST_SHA=$STATIC_RECEIPT_MANIFEST_SHA,D4A_RECEIPT=$REMOTE_D4A_RECEIPT,D4A_RECEIPT_SHA=$D4A_RECEIPT_SHA,D4A_MANIFEST=$REMOTE_D4A_MANIFEST,D4A_MANIFEST_SHA=$D4A_MANIFEST_SHA,SMOKE_JOB_ID=${SMOKE_JOB_IDS[$ARM]:-none},SMOKE_RUN=${SMOKE_RUNS[$ARM]:-none}"
    JOB_ID_RAW="$(ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_r2_reward_v2/run.sbatch' | sbatch --hold --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' --exclude='eu-g6-064' --job-name='terra-r2-$ARM' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
    JOB_ID="${JOB_ID_RAW%%;*}"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]]
    NEW_JOB_IDS+=("$JOB_ID")
    PAIR_JOB_IDS[$ARM]="$JOB_ID"
    ssh "$REMOTE_HOST" "scontrol show job '$JOB_ID' -o | grep -q 'JobState=PENDING' && scontrol show job '$JOB_ID' -o | grep -q 'Reason=JobHeldUser'"
done
test "${#NEW_JOB_IDS[@]}" -eq "${#ARMS[@]}"
ssh "$REMOTE_HOST" "scontrol release ${NEW_JOB_IDS[*]}"
trap - ERR INT TERM
for ARM in "${ARMS[@]}"; do
    echo "$PHASE $ARM ${PAIR_JOB_IDS[$ARM]}"
done
