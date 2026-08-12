#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: submit_continuation_tail.sh QUALIFIED_FULL_RECEIPT.json CONTINUATION_JOB_ID CONTINUATION_RUN_DIR" >&2
    exit 2
fi
RECEIPT="$1"
CONTINUATION_JOB_ID="$2"
CONTINUATION_RUN_DIR="$3"
[[ "$CONTINUATION_JOB_ID" =~ ^[0-9]+$ ]] || { echo "invalid continuation job id" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_PYTHON="${LOCAL_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID=terra_v8_deep_xattn_v1
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798

test -f "$RECEIPT"
INFO="$(JAX_PLATFORMS=cpu "$LOCAL_PYTHON" "$REPO/scripts/euler_v8_deep_xattn_v1/continuation_contract.py" inspect --receipt "$RECEIPT")"
read -r ARM SEED REVISION RECEIPT_SHA <<< "$(
    "$LOCAL_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["arm"], d["seed"], d["terra_baselines_revision"], d["receipt_sha256"])' <<< "$INFO"
)"
test "$REVISION" = "$(git -C "$REPO" rev-parse HEAD)"
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
EXPECTED_PARENT="$RUN_ROOT/$REVISION/continuation/full/s$SEED"
case "$CONTINUATION_RUN_DIR" in
    "$EXPECTED_PARENT/$ARM-unpaired"|"$EXPECTED_PARENT/$ARM-matched") ;;
    *) echo "continuation run directory changed campaign identity" >&2; exit 3 ;;
esac

REMOTE_SOURCE="$REMOTE_WORK/$REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
REMOTE_RECEIPT="$REMOTE_INPUTS/gates/$RECEIPT_SHA.json"
echo "continuation_job_id=$CONTINUATION_JOB_ID"
echo "continuation_run_dir=$CONTINUATION_RUN_DIR"
echo "arm=$ARM seed=$SEED revision=$REVISION"
echo "qualified_receipt_sha256=$RECEIPT_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, or Slurm mutation"
    echo "future dependency=afterany:$CONTINUATION_JOB_ID partition=gpuhe.24h"
    exit 0
fi

ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$REVISION'"
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_RECEIPT' | awk '{print \$1}')\" = '$RECEIPT_SHA'"
ssh "$REMOTE_HOST" "test -d '$CONTINUATION_RUN_DIR' && test ! -e '$CONTINUATION_RUN_DIR/continuation_tail'"
EXPORTS="ALL,CONTINUATION_JOB_ID=$CONTINUATION_JOB_ID,CONTINUATION_RUN_DIR=$CONTINUATION_RUN_DIR,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,QUALIFIED_RECEIPT=$REMOTE_RECEIPT,QUALIFIED_RECEIPT_SHA=$RECEIPT_SHA"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/continuation_tail.sbatch' | sbatch --parsable --dependency='afterany:$CONTINUATION_JOB_ID' --kill-on-invalid-dep=yes --partition=gpuhe.24h --time=23:45:00 --exclude='eu-g6-064' --job-name='terra-v8-cont-tail-$ARM' --output='$CONTINUATION_RUN_DIR/continuation_tail_%j.out' --export='$EXPORTS'"
)"
echo "continuation-tail $ARM $JOB_ID parent=$CONTINUATION_JOB_ID"
