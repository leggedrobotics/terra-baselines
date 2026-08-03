#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: submit_continuation.sh QUALIFIED_FULL_RECEIPT.json" >&2
    exit 2
fi

RECEIPT="$1"
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac
PAIRING="${PAIRING:-unpaired_single_qualifying_arm}"
case "$PAIRING" in
    unpaired_single_qualifying_arm) PAIRING_SUFFIX=unpaired ;;
    matched_architecture_pair) PAIRING_SUFFIX=matched ;;
    *) echo "unsupported PAIRING '$PAIRING'" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_p5c_authority_20260802}"
TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
CAMPAIGN_ID=terra_v8_deep_xattn_v1
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
LOCAL_PYTHON="${LOCAL_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}"
CONTRACT_TOOL="$REPO/scripts/euler_v8_deep_xattn_v1/continuation_contract.py"
TAIL_EVALUATOR_REL=scripts/euler_v8_deep_xattn_v1/continuation_tail.sbatch

test -x "$LOCAL_PYTHON"
test -f "$RECEIPT"
INFO="$(JAX_PLATFORMS=cpu "$LOCAL_PYTHON" "$CONTRACT_TOOL" inspect --receipt "$RECEIPT")"
read -r ARM SEED BASELINES_REVISION CHECKPOINT CHECKPOINT_SHA RESUME_UPDATE \
    RECEIPT_SHA PARENT_CONTRACT PARENT_CONTRACT_SHA <<< "$(
        "$LOCAL_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["arm"], d["seed"], d["terra_baselines_revision"], d["candidate_path"], d["candidate_sha256"], d["candidate_update"], d["receipt_sha256"], d["parent_run_contract_path"], d["parent_run_contract_sha256"])' <<< "$INFO"
    )"
test "$BASELINES_REVISION" = "$(git -C "$REPO" rev-parse HEAD)"
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before continuation" >&2
    exit 3
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "paired Terra authority must be clean" >&2
    exit 3
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$TERRA_REVISION"
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
test -f "$REPO/$TAIL_EVALUATOR_REL"

echo "pairing=$PAIRING"
echo "arm=$ARM seed=$SEED resume_update=$RESUME_UPDATE target_update=80000"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "qualified_receipt_sha256=$RECEIPT_SHA"
echo "resume_checkpoint_sha256=$CHECKPOINT_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local qualified receipt passed; no SSH, W&B, scratch, or Slurm mutation"
    echo "future sbatch: partition=gpuhe.120h time=119:45:00 arm=$ARM"
    exit 0
fi

REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
REMOTE_RECEIPT="$REMOTE_INPUTS/gates/$RECEIPT_SHA.json"
test "$(ssh "$REMOTE_HOST" "cat '$CHECKPOINT'" | sha256sum | awk '{print $1}')" = "$CHECKPOINT_SHA"
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$CHECKPOINT' | awk '{print \$1}')\" = '$CHECKPOINT_SHA'"
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARENT_CONTRACT' | awk '{print \$1}')\" = '$PARENT_CONTRACT_SHA'"
PARENT_WANDB_RUN_ID="$(
    ssh "$REMOTE_HOST" "awk -F= '\$1 == \"wandb_run_id\" {print substr(\$0, index(\$0, \"=\") + 1)}' '$PARENT_CONTRACT'"
)"
[[ "$PARENT_WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]

if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS/gates' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_RECEIPT'"; then
    PARTIAL="$REMOTE_RECEIPT.partial.$$"
    scp -q "$RECEIPT" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$RECEIPT_SHA' && mv '$PARTIAL' '$REMOTE_RECEIPT'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_RECEIPT' | awk '{print \$1}')\" = '$RECEIPT_SHA'"

RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/continuation/full/s$SEED"
RUN_DIR="$RUN_PARENT/$ARM-$PAIRING_SUFFIX"
ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
EXPORTS="ALL,ARM=$ARM,PAIRING=$PAIRING,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,QUALIFIED_RECEIPT=$REMOTE_RECEIPT,QUALIFIED_RECEIPT_SHA=$RECEIPT_SHA,RESUME_CHECKPOINT=$CHECKPOINT,RESUME_CHECKPOINT_SHA=$CHECKPOINT_SHA,RESUME_UPDATE=$RESUME_UPDATE,PARENT_WANDB_RUN_ID=$PARENT_WANDB_RUN_ID,PARENT_RUN_CONTRACT_SHA=$PARENT_CONTRACT_SHA"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/continue.sbatch' | sbatch --parsable --partition=gpuhe.120h --time=119:45:00 --exclude='eu-g6-064' --job-name='terra-v8-cont-$PAIRING_SUFFIX-$ARM' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
)"
echo "continuation $ARM $JOB_ID parent_wandb_run_id=$PARENT_WANDB_RUN_ID"
EVAL_JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/$TAIL_EVALUATOR_REL' | sbatch --parsable --dependency='afterany:$JOB_ID' --kill-on-invalid-dep=yes --partition=gpuhe.24h --time=23:45:00 --exclude='eu-g6-064' --job-name='terra-v8-cont-tail-$ARM' --output='$RUN_DIR/continuation_tail_%j.out' --export='$EXPORTS,CONTINUATION_JOB_ID=$JOB_ID,CONTINUATION_RUN_DIR=$RUN_DIR'"
)"
echo "continuation-tail $ARM $EVAL_JOB_ID dependency=afterany:$JOB_ID"
