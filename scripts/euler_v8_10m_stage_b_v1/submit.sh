#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: submit.sh smoke|screen SEED STAGE_B_SELECTION.json" >&2
    exit 2
fi
PHASE="$1"
SEED="$2"
SELECTION="$3"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
test -f "$SELECTION"
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_PYTHON="${LOCAL_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID=terra_v8_10m_nearby_long_v1
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
REMOTE_INPUTS=$REMOTE_RUNS/inputs
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
ARMS=(G-V8-XATTN-REWARM-CONTROL G-V8-10M-XATTN-WARM)

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
SELECTION="$(realpath "$SELECTION")"
SELECTION_SHA="$(sha256sum "$SELECTION" | awk '{print $1}')"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
REMOTE_SELECTION="$REMOTE_INPUTS/selections/$SELECTION_SHA.json"
declare -A PARENTS PARENT_SHAS PARENT_UPDATES
for ARM in "${ARMS[@]}"; do
    INFO="$(PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" JAX_PLATFORMS=cpu "$LOCAL_PYTHON" "$REPO/scripts/v8_10m_stage_b_selection.py" inspect --receipt "$SELECTION" --arm "$ARM")"
    read -r PARENT_VALUE PARENT_SHA_VALUE PARENT_UPDATE_VALUE RECEIPT_SHA <<< "$("$LOCAL_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["path"],d["sha256"],d["update"],d["receipt_sha256"])' <<< "$INFO")"
    PARENTS[$ARM]="$PARENT_VALUE"
    PARENT_SHAS[$ARM]="$PARENT_SHA_VALUE"
    PARENT_UPDATES[$ARM]="$PARENT_UPDATE_VALUE"
    test "$RECEIPT_SHA" = "$SELECTION_SHA"
done
TEACHER_ARM=G-V8-XATTN-REWARM-CONTROL
TEACHER_CHECKPOINT="${PARENTS[$TEACHER_ARM]}"
TEACHER_SHA="${PARENT_SHAS[$TEACHER_ARM]}"

echo "phase=$PHASE stage=nearby seed=$SEED updates=$([ "$PHASE" = smoke ] && echo 1 || echo 20000)"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "selection_sha256=$SELECTION_SHA"
echo "teacher_arm=$TEACHER_ARM teacher_sha256=$TEACHER_SHA"
for ARM in "${ARMS[@]}"; do
    echo "$ARM parent_update=${PARENT_UPDATES[$ARM]} parent_sha256=${PARENT_SHAS[$ARM]}"
done
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation"
    exit 0
fi

for ARM in "${ARMS[@]}"; do
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '${PARENTS[$ARM]}' | awk '{print \$1}')\" = '${PARENT_SHAS[$ARM]}'"
done
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS/selections' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SELECTION'"; then
    PARTIAL="$REMOTE_SELECTION.partial.$$"
    scp -q "$SELECTION" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$SELECTION_SHA' && mv '$PARTIAL' '$REMOTE_SELECTION'"
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    screen) PARTITION=gpuhe.120h; WALLTIME=119:45:00; GPU_TYPE=rtx_4090 ;;
esac
SMOKE_REVISION="${SMOKE_REVISION:-$BASELINES_REVISION}"
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/nearby/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    SMOKE_JOB_ID=none
    SMOKE_RUN=none
    if [ "$PHASE" = screen ]; then
        SMOKE_RUN="$REMOTE_RUNS/$SMOKE_REVISION/smoke/nearby/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "test -f '$SMOKE_RUN/run_contract.env' && test -f '$SMOKE_RUN/smoke_validation.json'"
        SMOKE_JOB_ID="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        SMOKE_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
        test "$SMOKE_STATE" = COMPLETED
    fi
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,PARENT_CHECKPOINT=${PARENTS[$ARM]},PARENT_SHA=${PARENT_SHAS[$ARM]},PARENT_SELECTION=$REMOTE_SELECTION,PARENT_SELECTION_SHA=$SELECTION_SHA,TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT,TEACHER_SHA=$TEACHER_SHA,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_stage_b_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' --exclude='eu-g6-064' --job-name='terra-v8-10m-b-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
    )"
    echo "$PHASE nearby $ARM $JOB_ID"
done
