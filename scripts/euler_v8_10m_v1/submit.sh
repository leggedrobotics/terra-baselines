#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: submit.sh smoke|screen SEED REMOTE_TEACHER_RECEIPT.json" >&2
    exit 2
fi
PHASE="$1"
SEED="$2"
TEACHER_RECEIPT="$3"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
[[ "$TEACHER_RECEIPT" = /* ]] || {
    echo "teacher receipt must be an absolute Euler path" >&2
    exit 2
}
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_ID=terra_v8_10m_v1
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
REMOTE_INPUTS=$REMOTE_RUNS/inputs
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
ARMS=(G-V8-XATTN-REWARM-CONTROL G-V8-10M-XATTN-WARM)

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"

echo "phase=$PHASE seed=$SEED absolute_target_update=$([ "$PHASE" = smoke ] && echo 1 || echo 20000)"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "release_id=$RELEASE_ID"
echo "teacher_receipt=$TEACHER_RECEIPT"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation"
    for ARM in "${ARMS[@]}"; do
        echo "future sbatch: phase=$PHASE arm=$ARM seed=$SEED"
    done
    if [ "$PHASE" = screen ]; then
        echo "future paired evaluator: afterany:<control-job>:<10m-job>"
    fi
    exit 0
fi

ssh "$REMOTE_HOST" "test -f '$TEACHER_RECEIPT'"
TEACHER_RECEIPT_SHA="$(ssh "$REMOTE_HOST" "sha256sum '$TEACHER_RECEIPT' | awk '{print \$1}'")"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"

if [ "$PHASE" = screen ]; then
    for ARM in "${ARMS[@]}"; do
        SMOKE="$REMOTE_RUNS/$BASELINES_REVISION/smoke/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "test -f '$SMOKE/smoke_validation.json' && python3 -c 'import json; assert json.load(open(\"$SMOKE/smoke_validation.json\"))[\"passed\"] is True' && test -f '$SMOKE/initialization_diagnostic.json' && python3 -c 'import json; d=json.load(open(\"$SMOKE/initialization_diagnostic.json\")); assert d[\"schema\"] == \"terra_v8_10m_initialization_diagnostic_v1\"; assert d[\"passed\"] is True; assert d[\"exact_frozen_resets\"] == 720' && test \"\$(sha256sum '$SMOKE/initialization_diagnostic.json' | awk '{print \$1}')\" = \"\$(awk -F= '\$1==\"initialization_diagnostic_sha256\" {print \$2}' '$SMOKE/run_contract.env')\" && test \"\$(awk -F= '\$1==\"teacher_receipt_sha256\" {print \$2}' '$SMOKE/run_contract.env')\" = '$TEACHER_RECEIPT_SHA'"
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=23:45:00 ;;
esac
JOB_IDS=()
RUN_DIRS=()
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,TEACHER_RECEIPT=$TEACHER_RECEIPT,TEACHER_RECEIPT_SHA=$TEACHER_RECEIPT_SHA"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --exclude='eu-g6-064' --job-name='terra-v8-10m-${PHASE}-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
    )"
    JOB_IDS+=("$JOB_ID")
    RUN_DIRS+=("$RUN_DIR")
    echo "$PHASE $ARM $JOB_ID"
done

if [ "$PHASE" = screen ]; then
    test "${#JOB_IDS[@]}" -eq 2
    test "${#RUN_DIRS[@]}" -eq 2
    CONTROL_JOB_ID="${JOB_IDS[0]}"
    TREATMENT_JOB_ID="${JOB_IDS[1]}"
    CONTROL_RUN="${RUN_DIRS[0]}"
    TREATMENT_RUN="${RUN_DIRS[1]}"
    COMPARE_EXPORTS="ALL,CONTROL_JOB_ID=$CONTROL_JOB_ID,TREATMENT_JOB_ID=$TREATMENT_JOB_ID,CONTROL_RUN=$CONTROL_RUN,TREATMENT_RUN=$TREATMENT_RUN,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID"
    COMPARE_JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/compare.sbatch' | sbatch --parsable --dependency='afterany:$CONTROL_JOB_ID:$TREATMENT_JOB_ID' --kill-on-invalid-dep=yes --partition='gpuhe.24h' --time='23:45:00' --exclude='eu-g6-064' --job-name='terra-v8-10m-compare' --output='$RUN_PARENT/paired_benchmark_%j.out' --export='$COMPARE_EXPORTS'"
    )"
    echo "paired-evaluator $COMPARE_JOB_ID parents=$CONTROL_JOB_ID,$TREATMENT_JOB_ID"
fi
