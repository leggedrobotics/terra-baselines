#!/usr/bin/env bash
# Sync one clean terra-baselines revision and optionally submit the P5b star.
# shellcheck disable=SC2029  # Validated paths intentionally expand client-side.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: submit.sh smoke|screen [SEED]" >&2
    exit 2
fi
PHASE="$1"
case "$PHASE" in smoke|screen) ;; *) echo "PHASE must be smoke or screen" >&2; exit 2 ;; esac
SEED="${2:-20260730}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_p5c_authority_20260802}"
TERRA_REVISION="a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
DEEP_CHECKPOINT="/home/lorenzo/moleworks/.artifacts/terra_p5b_parent_20260802/g_adaptive_u2000_deep_se_grown.pkl"
DEEP_SHA="6bf014c7b9074564df9e1b36fd4e4106bfeb61f1dfa17b7fbd728314c958ba9b"
PARENT_REMOTE="/cluster/scratch/lterenzi/codex_terra_edge_runs/accepted_bank_v1/f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa/screen/s20260730/G-ADAPTIVE/checkpoints/accepted_f8aac348d64c_screen_g_adaptive_s20260730-euler-2026-08-01-19-45-09_update_002000.pkl"
PARENT_SHA="76b5189955735741b0cd4b3444fbda8ffdb8be4b29582509dafad85fa7cfb45a"
REMOTE_HOST="${REMOTE_HOST:-euler}"
CAMPAIGN_ID="${CAMPAIGN_ID:-p5b_warm_v1}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-p5b}"
ENT_SCHEDULE_START="${ENT_SCHEDULE_START:-0.15}"
ENT_SCHEDULE_END="${ENT_SCHEDULE_END:-0.005}"
ENT_SCHEDULE_STEPS="${ENT_SCHEDULE_STEPS:-7600}"
SCREEN_UPDATES="${SCREEN_UPDATES:-2000}"
ARMS_STRING="${ARMS_STRING:-G-MEDIUM-ADAPTIVE-WARM G-DEEP-ADAPTIVE-WARM G-MEDIUM-UNIFORM-WARM}"
read -r -a ARMS <<< "$ARMS_STRING"
DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL="${DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL:-}"
DIAGNOSTIC_CONTROL_SHA="${DIAGNOSTIC_CONTROL_SHA:-}"
TRAIN_BANK_ARCHIVE_LOCAL="${TRAIN_BANK_ARCHIVE_LOCAL:-}"
TRAIN_BANK_SHA="${TRAIN_BANK_SHA:-}"
TRAIN_BANK_DATASET_SHA="${TRAIN_BANK_DATASET_SHA:-}"
TRAIN_BANK_RELEASE_ID="${TRAIN_BANK_RELEASE_ID:-}"
TRAIN_MAPS_PER_CONDITION="${TRAIN_MAPS_PER_CONDITION:-64}"
REMOTE_WORK="/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID"
REMOTE_INPUTS="/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs"
REMOTE_RUNS="/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID"

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "paired Terra authority must be clean: $TERRA_REPO" >&2
    exit 3
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$TERRA_REVISION" || {
    echo "paired Terra authority has the wrong environment revision" >&2
    exit 3
}
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
test -f "$DEEP_CHECKPOINT"
test "$(sha256sum "$DEEP_CHECKPOINT" | awk '{print $1}')" = "$DEEP_SHA" || {
    echo "depth-grown checkpoint SHA mismatch" >&2
    exit 3
}
if [ -n "$DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL" ] || [ -n "$DIAGNOSTIC_CONTROL_SHA" ]; then
    if [ -z "$DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL" ] || [ -z "$DIAGNOSTIC_CONTROL_SHA" ]; then
        echo "diagnostic control archive and SHA must be supplied together" >&2
        exit 3
    fi
    test -f "$DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL"
    test "$(sha256sum "$DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL" | awk '{print $1}')" = "$DIAGNOSTIC_CONTROL_SHA" || {
        echo "diagnostic control archive SHA mismatch" >&2
        exit 3
    }
fi
if [ -n "$TRAIN_BANK_ARCHIVE_LOCAL" ] || [ -n "$TRAIN_BANK_SHA" ] || [ -n "$TRAIN_BANK_DATASET_SHA" ]; then
    if [ -z "$TRAIN_BANK_ARCHIVE_LOCAL" ] || [ -z "$TRAIN_BANK_SHA" ] || [ -z "$TRAIN_BANK_DATASET_SHA" ]; then
        echo "training bank archive, archive SHA, and dataset SHA must be supplied together" >&2
        exit 3
    fi
    test -f "$TRAIN_BANK_ARCHIVE_LOCAL"
    test "$(sha256sum "$TRAIN_BANK_ARCHIVE_LOCAL" | awk '{print $1}')" = "$TRAIN_BANK_SHA" || {
        echo "training bank archive SHA mismatch" >&2
        exit 3
    }
    test "$(tar --zstd -xOf "$TRAIN_BANK_ARCHIVE_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$TRAIN_BANK_DATASET_SHA" || {
        echo "training bank dataset SHA mismatch" >&2
        exit 3
    }
fi
[[ "$TRAIN_MAPS_PER_CONDITION" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAIN_MAPS_PER_CONDITION must be a positive integer" >&2
    exit 3
}
if [ -n "$TRAIN_BANK_RELEASE_ID" ] && [ "$SUBMIT" = 1 ] && [ -z "$TRAIN_BANK_ARCHIVE_LOCAL" ]; then
    echo "SUBMIT=1 for a named training-bank release requires the immutable local archive" >&2
    exit 3
fi

REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_DEEP="$REMOTE_INPUTS/g_adaptive_u2000_deep_se_grown-$DEEP_SHA.pkl"
REMOTE_CONTROL=""
if [ -n "$DIAGNOSTIC_CONTROL_SHA" ]; then
    REMOTE_CONTROL="$REMOTE_INPUTS/diagnostic-control-$DIAGNOSTIC_CONTROL_SHA.tar.zst"
fi
REMOTE_TRAIN_BANK=""
if [ -n "$TRAIN_BANK_SHA" ]; then
    REMOTE_TRAIN_BANK="$REMOTE_INPUTS/training-bank-$TRAIN_BANK_SHA.tar.zst"
fi
echo "phase=$PHASE seed=$SEED"
echo "terra_revision=$TERRA_REVISION terra_repo=$TERRA_REPO"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "remote_source=$REMOTE_SOURCE"
echo "parent_checkpoint=$PARENT_REMOTE"
echo "deep_checkpoint=$REMOTE_DEEP"
echo "campaign=$CAMPAIGN_ID prefix=$EXPERIMENT_PREFIX"
echo "entropy=$ENT_SCHEDULE_START:$ENT_SCHEDULE_END:$ENT_SCHEDULE_STEPS screen_updates=$SCREEN_UPDATES"
echo "diagnostic_control=${REMOTE_CONTROL:-none}"
echo "training_bank_release=${TRAIN_BANK_RELEASE_ID:-legacy-p5}"
echo "training_bank_maps_per_condition=$TRAIN_MAPS_PER_CONDITION"
echo "training_bank_archive=${REMOTE_TRAIN_BANK:-${TRAIN_BANK_ARCHIVE_LOCAL:-PENDING}}"
echo "training_bank_archive_sha256=${TRAIN_BANK_SHA:-PENDING}"
echo "training_bank_dataset_sha256=${TRAIN_BANK_DATASET_SHA:-PENDING}"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no sync, scratch, W&B, or Slurm mutation"
    for ARM in "${ARMS[@]}"; do
        echo "future sbatch: phase=$PHASE arm=$ARM seed=$SEED"
    done
    exit 0
fi

ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARENT_REMOTE' | awk '{print \$1}')\" = '$PARENT_SHA'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    REMOTE_PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$REMOTE_PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$REMOTE_PARTIAL/terra-baselines/REVISION' && mv '$REMOTE_PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_DEEP'"; then
    REMOTE_DEEP_PARTIAL="$REMOTE_DEEP.partial.$$"
    scp -q "$DEEP_CHECKPOINT" "$REMOTE_HOST:$REMOTE_DEEP_PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_DEEP_PARTIAL' | awk '{print \$1}')\" = '$DEEP_SHA' && mv '$REMOTE_DEEP_PARTIAL' '$REMOTE_DEEP'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_DEEP' | awk '{print \$1}')\" = '$DEEP_SHA'"
if [ -n "$REMOTE_CONTROL" ]; then
    if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_CONTROL'"; then
        REMOTE_CONTROL_PARTIAL="$REMOTE_CONTROL.partial.$$"
        scp -q "$DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL" "$REMOTE_HOST:$REMOTE_CONTROL_PARTIAL"
        ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_CONTROL_PARTIAL' | awk '{print \$1}')\" = '$DIAGNOSTIC_CONTROL_SHA' && mv '$REMOTE_CONTROL_PARTIAL' '$REMOTE_CONTROL'"
    fi
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_CONTROL' | awk '{print \$1}')\" = '$DIAGNOSTIC_CONTROL_SHA'"
fi
if [ -n "$REMOTE_TRAIN_BANK" ]; then
    if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_TRAIN_BANK'"; then
        REMOTE_TRAIN_BANK_PARTIAL="$REMOTE_TRAIN_BANK.partial.$$"
        scp -q "$TRAIN_BANK_ARCHIVE_LOCAL" "$REMOTE_HOST:$REMOTE_TRAIN_BANK_PARTIAL"
        ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_TRAIN_BANK_PARTIAL' | awk '{print \$1}')\" = '$TRAIN_BANK_SHA' && mv '$REMOTE_TRAIN_BANK_PARTIAL' '$REMOTE_TRAIN_BANK'"
    fi
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_TRAIN_BANK' | awk '{print \$1}')\" = '$TRAIN_BANK_SHA'"
fi

if [ "$PHASE" = screen ]; then
    for ARM in "${ARMS[@]}"; do
        SMOKE="$REMOTE_RUNS/$BASELINES_REVISION/smoke/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "
            set -e
            test -f '$SMOKE/smoke_validation.json'
            python3 -c 'import json; p=json.load(open(\"$SMOKE/smoke_validation.json\")); assert p[\"passed\"] is True'
            grep -qx 'status=PASSED' '$SMOKE/run_contract.env'
            grep -qx 'arm=$ARM' '$SMOKE/run_contract.env'
            grep -qx 'seed=$SEED' '$SMOKE/run_contract.env'
            grep -qx 'entropy_schedule=$ENT_SCHEDULE_START:$ENT_SCHEDULE_END:$ENT_SCHEDULE_STEPS' '$SMOKE/run_contract.env'
            grep -qx 'diagnostic_control_archive_sha256=${DIAGNOSTIC_CONTROL_SHA:-none}' '$SMOKE/run_contract.env'
            grep -qx 'training_bank_release_id=${TRAIN_BANK_RELEASE_ID:-legacy-p5}' '$SMOKE/run_contract.env'
            grep -qx 'training_bank_archive_sha256=${TRAIN_BANK_SHA:-legacy-p5-campaign}' '$SMOKE/run_contract.env'
            grep -qx 'training_bank_dataset_sha256=${TRAIN_BANK_DATASET_SHA:-legacy-p5-campaign}' '$SMOKE/run_contract.env'
            grep -qx 'train_maps_per_condition=$TRAIN_MAPS_PER_CONDITION' '$SMOKE/run_contract.env'
        "
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=20:00:00 ;;
esac
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,DEEP_CHECKPOINT=$REMOTE_DEEP,DEEP_SHA=$DEEP_SHA,SEED=$SEED,CAMPAIGN_ID=$CAMPAIGN_ID,EXPERIMENT_PREFIX=$EXPERIMENT_PREFIX,ENT_SCHEDULE_START=$ENT_SCHEDULE_START,ENT_SCHEDULE_END=$ENT_SCHEDULE_END,ENT_SCHEDULE_STEPS=$ENT_SCHEDULE_STEPS,SCREEN_UPDATES=$SCREEN_UPDATES,CONTROL_ARCHIVE=$REMOTE_CONTROL,CONTROL_SHA=$DIAGNOSTIC_CONTROL_SHA,TRAIN_BANK_ARCHIVE=$REMOTE_TRAIN_BANK,TRAIN_BANK_SHA=$TRAIN_BANK_SHA,TRAIN_BANK_DATASET_SHA=$TRAIN_BANK_DATASET_SHA,TRAIN_BANK_RELEASE_ID=$TRAIN_BANK_RELEASE_ID,TRAIN_MAPS_PER_CONDITION=$TRAIN_MAPS_PER_CONDITION"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "
            cat '$REMOTE_SOURCE/scripts/euler_p5b_warm_v1/run.sbatch' |
            sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' \\
                --exclude='eu-g6-064' --job-name='terra-${EXPERIMENT_PREFIX}-${PHASE}-${ARM}' \\
                --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'
        "
    )"
    echo "$PHASE $ARM $JOB_ID"
done
