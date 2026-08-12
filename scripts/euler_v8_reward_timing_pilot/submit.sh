#!/usr/bin/env bash
# Submit the rv2p1_scratch smoke (1 update) or main run (14000 updates).
#   SUBMIT=0     local contracts only (default)
#   SUBMIT=stage stage code and inputs, no Slurm mutation
#   SUBMIT=1     submit
set -euo pipefail

R2_HORIZON=450
if [ "$#" -ne 1 ]; then
    echo "usage: submit.sh smoke|phase1" >&2
    exit 2
fi
PHASE="$1"
case "$PHASE" in smoke|phase1) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|1) ;;
    *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;;
esac

ARM_NAME=rv2p1_scratch
SAMPLER_PROFILE=continuous_banded_v3
REWARD_V2_TIMING_VARIANT=1
EXPECTED_RUNTIME_TERRA_REVISION=46b5a1ddcd3b0e3a0d9e637af2e4ea94af51b4c8
TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_reward_timing_20260812

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=cluster/euler_account.sh
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-lterenzi}"
TERRA_REPO="${TERRA_REPO:-$TERRA_REPO_DEFAULT}"
ARTIFACT_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_training_inputs_20260810
BANK_ARCHIVE="$ARTIFACT_ROOT/treatment_bank.tar.zst"
MATERIALIZATION_RECEIPT="$ARTIFACT_ROOT/treatment_bank_receipt.json"

SEED=20260807
CAMPAIGN=terra_v8_rv2p1
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
BANK_TREE_SHA=225e13aacd9047e7f241facd3397fd66794e3094a883cc6dc26304decc24d388
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
MATERIALIZATION_RECEIPT_SHA=631fac8c3b78ff2c5a9e94ea4032244c9ef05dc6c984b603e4318121a263d3f1
REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"

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
    EnvConfig,
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
    REWARD_V2_TIMING_BASELINE,
    REWARD_V2_TIMING_V21,
    REWARD_V2_TIMING_V21_ID,
    REWARD_V2_V21_SHAPING_GAMMA,
    REWARD_V2_V21_STEP_COST_TOTAL,
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
assert (REWARD_V2_TIMING_BASELINE, REWARD_V2_TIMING_V21) == (0, 1)
assert REWARD_V2_TIMING_V21_ID == "gamma1_stepcost_3.6"
assert (REWARD_V2_V21_SHAPING_GAMMA, REWARD_V2_V21_STEP_COST_TOTAL) == (1.0, 3.6)
assert EnvConfig().reward_v2_timing_variant == REWARD_V2_TIMING_BASELINE
PY
PYTHONPATH="$REPO" python3 -c "
from utils.pooled_sampler import CONTINUOUS_MAX_MASS, CONTINUOUS_RULES
assert '$SAMPLER_PROFILE' in CONTINUOUS_RULES
assert CONTINUOUS_MAX_MASS == 0.15
"

for SPEC in \
    "$BANK_ARCHIVE:$BANK_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA"; do
    PATH_LOCAL="${SPEC%:*}"
    EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$PATH_LOCAL" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "$(tar --zstd -xOf "$BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$EXPECTED_RUNTIME_TERRA_REVISION"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
echo "phase=$PHASE arm=$ARM_NAME seed=$SEED updates=$([ "$PHASE" = smoke ] && echo 1 || echo 14000)"
echo "reward_v2_timing=gamma1_stepcost_3.6 variant=$REWARD_V2_TIMING_VARIANT sampler=$SAMPLER_PROFILE"
echo "terra_baselines_revision=$BASELINES_REVISION runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "euler_user=$TERRA_EULER_USER remote_host=$REMOTE_HOST remote_venv=$REMOTE_VENV"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
    exit 0
fi

REMOTE_ID="$(ssh -o BatchMode=yes "$REMOTE_HOST" 'id -un')"
test "$REMOTE_ID" = "$TERRA_EULER_USER" || {
    echo "remote account mismatch: expected $TERRA_EULER_USER, got $REMOTE_ID" >&2
    exit 3
}
ssh "$REMOTE_HOST" \
    "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"

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

REMOTE_BANK="$REMOTE_INPUTS/treatment-bank-$BANK_SHA.tar.zst"
REMOTE_MATERIALIZATION_RECEIPT="$REMOTE_INPUTS/materialization-$MATERIALIZATION_RECEIPT_SHA.json"
upload "$BANK_ARCHIVE" "$REMOTE_BANK" "$BANK_SHA"
upload "$MATERIALIZATION_RECEIPT" "$REMOTE_MATERIALIZATION_RECEIPT" "$MATERIALIZATION_RECEIPT_SHA"

if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: code and pinned inputs staged; no W&B or Slurm mutation"
    exit 0
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    phase1) PARTITION=gpuhe.24h; WALLTIME=23:45:00; GPU_TYPE=rtx_4090 ;;
esac
SMOKE_JOB_ID=none
SMOKE_RUN=none
if [ "$PHASE" = phase1 ]; then
    SMOKE_RUN="$REMOTE_RUNS/$BASELINES_REVISION/smoke/s$SEED/$ARM_NAME"
    ssh "$REMOTE_HOST" "test -f '$SMOKE_RUN/smoke_validation.json' -a -f '$SMOKE_RUN/run_contract.env'"
    ssh "$REMOTE_HOST" "python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))[\"passed\"] is True' '$SMOKE_RUN/smoke_validation.json'"
    SMOKE_JOB_ID="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
    [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
    SMOKE_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
    test "$SMOKE_STATE" = COMPLETED
    for EXPECTED in \
        "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
        "terra_baselines_revision=$BASELINES_REVISION" \
        "distance_artifact_sha256=$DISTANCE_SIDECAR_SHA" \
        "reward_v2_timing_variant=$REWARD_V2_TIMING_VARIANT" \
        "sampler_profile=$SAMPLER_PROFILE"; do
        KEY="${EXPECTED%%=*}" VALUE="${EXPECTED#*=}"
        ssh "$REMOTE_HOST" "test \"\$(awk -F= -v key='$KEY' '\$1==key {print \$2}' '$SMOKE_RUN/run_contract.env')\" = '$VALUE'"
    done
fi

RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/s$SEED/$ARM_NAME"
ssh "$REMOTE_HOST" "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"
JOB_ID=""
cleanup_new_job() {
    local status="$1"
    trap - ERR INT TERM
    set +e
    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        ssh "$REMOTE_HOST" "scancel -- '$JOB_ID'"
    fi
    # rmdir is intentionally the only cleanup: non-empty evidence survives.
    ssh "$REMOTE_HOST" "rmdir -- '$RUN_DIR'" || true
    exit "$status"
}
trap 'cleanup_new_job $?' ERR
trap 'cleanup_new_job 130' INT TERM

EXPORTS="ALL,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,VENV=$REMOTE_VENV,RUN_BASE=$REMOTE_RUNS,PHASE=$PHASE,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,R2_HORIZON=$R2_HORIZON,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_TREE_SHA=$BANK_TREE_SHA,BANK_RELEASE_ID=$RELEASE_ID,DISTANCE_ARTIFACT_SHA=$DISTANCE_SIDECAR_SHA,MATERIALIZATION_RECEIPT=$REMOTE_MATERIALIZATION_RECEIPT,MATERIALIZATION_RECEIPT_SHA=$MATERIALIZATION_RECEIPT_SHA,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN"
JOB_ID_RAW="$(ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_reward_timing_pilot/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' --exclude='eu-g6-064' --job-name='terra-rv2p1' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_ID_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
trap - ERR INT TERM
echo "$PHASE $ARM_NAME $JOB_ID"
