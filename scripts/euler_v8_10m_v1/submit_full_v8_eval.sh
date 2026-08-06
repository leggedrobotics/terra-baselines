#!/usr/bin/env bash
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID=terra_v8_10m_curriculum_v1
RUN_REVISION=2a195b6c7112e56684d6088f1c9a073f3a3ff047
SEED=20260730
CONTROL_JOB_ID=9685873
TREATMENT_JOB_ID=9685874
RUN_PARENT=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/$RUN_REVISION/screen/capability/s$SEED
CONTROL_RUN=$RUN_PARENT/G-V8-XATTN-REWARM-CONTROL
TREATMENT_RUN=$RUN_PARENT/G-V8-10M-XATTN-WARM
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before evaluation" >&2
    exit 3
}
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"

echo "control_job=$CONTROL_JOB_ID treatment_job=$TREATMENT_JOB_ID"
echo "run_revision=$RUN_REVISION evaluator_revision=$BASELINES_REVISION"
echo "primary_metric=fixed/development/exact_success_rate"
echo "reward_type=DENSE reward_transition_launched=false"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, or Slurm mutation"
    exit 0
fi

ssh "$REMOTE_HOST" "test -f '$CONTROL_RUN/run_contract.env' && test -f '$TREATMENT_RUN/run_contract.env'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"

OUTPUT_DIR="$RUN_PARENT/whole_v8_fixed_$BASELINES_REVISION"
test -z "$(ssh "$REMOTE_HOST" "test -e '$OUTPUT_DIR' && echo exists || true")"
EXPORTS="ALL,CONTROL_RUN=$CONTROL_RUN,TREATMENT_RUN=$TREATMENT_RUN,CONTROL_JOB_ID=$CONTROL_JOB_ID,TREATMENT_JOB_ID=$TREATMENT_JOB_ID,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/full_v8_eval.sbatch' | sbatch --parsable --partition=gpuhe.24h --time=23:45:00 --exclude='eu-g6-064' --job-name='terra-v8-stagea-whole-eval' --output='$RUN_PARENT/whole_v8_eval_%j.out' --export='$EXPORTS'"
)"
echo "whole-V8 fixed evaluation $JOB_ID"
