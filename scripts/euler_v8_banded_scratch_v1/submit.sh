#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: submit.sh smoke|screen SEED" >&2
    exit 2
fi
PHASE="$1"
SEED="$2"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID=terra_v8_banded_scratch_v1
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/work/rsl/lterenzi/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
SMOKE_REVISION="${SMOKE_REVISION:-$BASELINES_REVISION}"

echo "phase=$PHASE stage=capability seed=$SEED updates=$([ "$PHASE" = smoke ] && echo 1 || echo 6000)"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "initialization=random_no_teacher"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, storage, W&B, or Slurm mutation"
    exit 0
fi

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

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=23:45:00; GPU_TYPE=rtx_4090 ;;
esac
RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/capability/s$SEED"
RUN_DIR="$RUN_PARENT/G-V8-XATTN-SCRATCH"
ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
SMOKE_JOB_ID=none
SMOKE_RUN=none
if [ "$PHASE" = screen ]; then
    SMOKE_RUN="$REMOTE_RUNS/$SMOKE_REVISION/smoke/capability/s$SEED/G-V8-XATTN-SCRATCH"
    ssh "$REMOTE_HOST" "test -f '$SMOKE_RUN/run_contract.env' && test -f '$SMOKE_RUN/smoke_validation.json'"
    SMOKE_JOB_ID="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
    [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
    SMOKE_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
    test "$SMOKE_STATE" = COMPLETED
fi
EXPORTS="ALL,PHASE=$PHASE,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN"
JOB_ID="$(ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_banded_scratch_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' --exclude='eu-g6-064' --job-name='terra-v8-scratch-a' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
echo "$PHASE capability G-V8-XATTN-SCRATCH $JOB_ID"
