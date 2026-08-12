#!/usr/bin/env bash
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID=terra_v8_10m_curriculum_v1
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
REMOTE_INPUTS=$REMOTE_RUNS/inputs
SEED=20260730
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
TEACHER_CHECKPOINT=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_direct_full_teacher_v1/885b1dbe6d7ac4b4199140a188d147657760eeee/screen/s20260730/G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER/checkpoints/v8_direct_full_885b1dbe6d7a_screen_g_deep_xattn_v8_dense_warm_s20260730-euler-2026-08-04-07-29-25_update_007500.pkl
TEACHER_SHA=a6bebfffcf4d390df19ade9652d3c96d833eb7d2587ddb1b95035b7ad6a807f6

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
OUTPUT_ROOT="$REMOTE_RUNS/$BASELINES_REVISION/reference_teacher_full_v8/s$SEED"
OUTPUT_PARENT="$(dirname "$OUTPUT_ROOT")"

echo "terra_baselines_revision=$BASELINES_REVISION"
echo "teacher_checkpoint_sha256=$TEACHER_SHA"
echo "output_root=$OUTPUT_ROOT"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, or Slurm mutation"
    exit 0
fi

ssh "$REMOTE_HOST" "test \"\$(sha256sum '$TEACHER_CHECKPOINT' | awk '{print \$1}')\" = '$TEACHER_SHA'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$OUTPUT_PARENT' && test ! -e '$OUTPUT_ROOT'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
EXPORTS="ALL,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT,TEACHER_SHA=$TEACHER_SHA,OUTPUT_ROOT=$OUTPUT_ROOT"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/teacher_full_v8_eval.sbatch' | sbatch --parsable --partition=gpuhe.4h --time=03:45:00 --gpus=rtx_4090:1 --exclude='eu-g6-064' --job-name='terra-v8-reference-teacher-eval' --output='$OUTPUT_PARENT/teacher_eval_%j.out' --export='$EXPORTS'"
)"
echo "reference-teacher-eval $JOB_ID"
