#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: submit_tail.sh PARENT_JOB_ID ARM SEED STAGE_B_GATE.json" >&2
    exit 2
fi
PARENT_JOB_ID="$1"
ARM="$2"
SEED="$3"
GATE_PATH="$4"
[[ "$PARENT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "invalid parent job id" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed" >&2; exit 2; }
case "$ARM" in
    G-DEEP-V8-DENSE-WARM|G-DEEP-XATTN-V8-DENSE-WARM) ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

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
GATE_TOOL="$REPO/scripts/euler_v8_deep_xattn_v1/stage_gate.py"

test -f "$GATE_PATH"
GATE_INFO="$("$LOCAL_PYTHON" "$GATE_TOOL" inspect \
    --receipt "$GATE_PATH" --stage nearby --arm "$ARM")"
read -r GATE_CANDIDATE GATE_CANDIDATE_SHA GATE_SHA NEXT_STAGE <<< "$(
    "$LOCAL_PYTHON" -c \
        'import json,sys; d=json.load(sys.stdin); print(d["candidate_path"], d["candidate_sha256"], d["receipt_sha256"], d["next_stage"])' \
        <<< "$GATE_INFO"
)"
test "$NEXT_STAGE" = full
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "paired Terra authority must be clean" >&2
    exit 3
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$TERRA_REVISION"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
REMOTE_GATE="$REMOTE_INPUTS/gates/$GATE_SHA.json"
RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/screen/full/s$SEED/$ARM"

echo "parent_job_id=$PARENT_JOB_ID arm=$ARM seed=$SEED"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "stage_b_gate_sha256=$GATE_SHA"
echo "stage_b_candidate=$GATE_CANDIDATE"
echo "stage_b_candidate_sha256=$GATE_CANDIDATE_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, or Slurm mutation"
    echo "future dependency=afterany:$PARENT_JOB_ID partition=gpuhe.4h"
    exit 0
fi

if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" |
        ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "test -d '$RUN_DIR' && test ! -e '$RUN_DIR/tail_eval'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS/gates'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_GATE'"; then
    PARTIAL="$REMOTE_GATE.partial.$$"
    scp -q "$GATE_PATH" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$GATE_SHA' && mv '$PARTIAL' '$REMOTE_GATE'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_GATE' | awk '{print \$1}')\" = '$GATE_SHA'"
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"
ssh "$REMOTE_HOST" "mkdir '$RUN_DIR/tail_eval'"

EXPORTS="ALL,PARENT_JOB_ID=$PARENT_JOB_ID,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,RUN_DIR=$RUN_DIR,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,PRIOR_GATE_RECEIPT=$REMOTE_GATE,PRIOR_GATE_SHA=$GATE_SHA"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/evaluate_tail.sbatch' | sbatch --parsable --dependency='afterany:$PARENT_JOB_ID' --kill-on-invalid-dep=yes --partition=gpuhe.4h --time=03:45:00 --exclude='eu-g6-064' --job-name='terra-v8-tail-$ARM' --output='$RUN_DIR/tail_eval/slurm_%j.out' --export='$EXPORTS'"
)"
echo "tail full $ARM $JOB_ID parent=$PARENT_JOB_ID"
