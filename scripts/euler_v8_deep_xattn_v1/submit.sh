#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: submit.sh smoke|screen [capability] [SEED]" >&2
    exit 2
fi
PHASE="$1"
STAGE="${2:-capability}"
SEED="${3:-20260730}"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
if [ "$STAGE" != capability ]; then
    echo "only capability is launchable until a prior-stage gate receipt and promoted parent are supplied" >&2
    exit 2
fi
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
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
PARENT=/cluster/scratch/lterenzi/codex_terra_edge_runs/p5c_low_entropy_v1/3478af87950d3d35059344b078209d00785c8481/screen/s20260730/G-DEEP-UNIFORM-WARM/checkpoints/p5c_3478af87950d_screen_g_deep_uniform_warm_s20260730-euler-2026-08-03-00-39-08_update_004000.pkl
PARENT_SHA=4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
ARMS=(G-DEEP-V8-DENSE-WARM G-DEEP-XATTN-V8-DENSE-WARM)

SCREEN_UPDATES=2000

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
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
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"

echo "phase=$PHASE stage=$STAGE seed=$SEED updates=$SCREEN_UPDATES"
echo "terra_revision=$TERRA_REVISION"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "release_id=$RELEASE_ID"
echo "bank_archive_sha256=$BANK_SHA"
echo "bank_dataset_sha256=$BANK_DATASET_SHA"
echo "parent_checkpoint_sha256=$PARENT_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation"
    for ARM in "${ARMS[@]}"; do
        echo "future sbatch: phase=$PHASE stage=$STAGE arm=$ARM seed=$SEED"
    done
    exit 0
fi

ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARENT' | awk '{print \$1}')\" = '$PARENT_SHA'"
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
        SMOKE="$REMOTE_RUNS/$BASELINES_REVISION/smoke/$STAGE/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "test -f '$SMOKE/smoke_validation.json' && python3 -c 'import json; assert json.load(open(\"$SMOKE/smoke_validation.json\"))[\"passed\"] is True' && grep -qx status=PASSED '$SMOKE/run_contract.env'"
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=23:45:00 ;;
esac
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/$STAGE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,PHASE=$PHASE,STAGE=$STAGE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,SCREEN_UPDATES=$SCREEN_UPDATES,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,PARENT_CHECKPOINT=$PARENT,PARENT_SHA=$PARENT_SHA"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --exclude='eu-g6-064' --job-name='terra-v8-${PHASE}-${STAGE}-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
    )"
    echo "$PHASE $STAGE $ARM $JOB_ID"
done
