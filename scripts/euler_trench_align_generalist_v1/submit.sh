#!/usr/bin/env bash
# Stage, smoke, or submit one gate-on 37-condition Continuous Banded v3 run.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|stage|smoke|1) ;; *) echo "SUBMIT must be 0, stage, smoke, or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818}"
BANK_ARCHIVE=/media/lorenzo/T7/codex/terra_trench_alignment_u30000_20260822/terra_v8_trench_aligned_generalist_37cond_20260822.tar.zst
BANK_ARCHIVE_SHA=e84c75f27ebc2fa4c48c0a127a21386b4cf4b0ee3a87daa442759bb7c11be680
BANK_DATASET_SIZE=96
BANK_DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
LOADER_RECEIPT=/media/lorenzo/T7/codex/terra_trench_alignment_u30000_20260822/receipts/generalist_37cond_loader_validation.json
TERRA_REVISION_PIN=b3599b30a67dd4d1de77c3f45871e4d6d6651c7f
EXPECTED_PARAMETERS=2309053
SEED=20260822
TARGET_UPDATE=100000

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_trench_align_generalist_v1
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(git -C "$TERRA_REPO" status --porcelain)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$RUNTIME_TERRA_REVISION" = "$TERRA_REVISION_PIN"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
LOADER_RECEIPT="$LOADER_RECEIPT" python3 - <<'PY'
import json, os
receipt = json.load(open(os.environ["LOADER_RECEIPT"]))
assert receipt["accepted_datasets"] == 37, receipt
assert receipt["rejected_datasets"] == 0, receipt
assert receipt["accepted_slots"] == 3552, receipt
PY

echo "terra_baselines_revision=$BASELINES_REVISION"
echo "runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "scope=37_supported_conditions sampler=continuous_banded_v3 gate=on seed=$SEED"
if test "$SUBMIT" = 0; then
    echo "SUBMIT=0: local contract passed; no external mutation"
    exit 0
fi

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"

REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | remote "tar -xf - -C '$PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
remote "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"

REMOTE_BANK="$REMOTE_INPUTS/generalist-37cond-$BANK_ARCHIVE_SHA.tar.zst"
if ! remote "test -f '$REMOTE_BANK'"; then
    scp -q -o BatchMode=yes "$BANK_ARCHIVE" "$REMOTE_HOST:$REMOTE_BANK.partial.$$"
    remote "test \"\$(sha256sum '$REMOTE_BANK.partial.$$' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA' && mv -T '$REMOTE_BANK.partial.$$' '$REMOTE_BANK'"
fi
remote "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA'"

if test "$SUBMIT" = stage; then
    echo "SUBMIT=stage: immutable source and bank staged; no Slurm mutation"
    exit 0
fi

if test "$SUBMIT" = smoke; then
    RUN_ROLE=smoke
    THIS_TARGET=1
    PARTITION=gpuhe.4h
    WALLTIME=03:45:00
else
    RUN_ROLE=production
    THIS_TARGET=$TARGET_UPDATE
    PARTITION=gpuhe.120h
    WALLTIME=119:45:00
    SMOKE_DIR="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED/smoke"
    remote "test -f '$SMOKE_DIR/completion.env' && grep -qx 'status=COMPLETE' '$SMOKE_DIR/completion.env' && grep -qx 'target_update=1' '$SMOKE_DIR/completion.env'"
fi

remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED/$RUN_ROLE"
RUN_NAME="trench_align_generalist_${BASELINES_REVISION:0:12}_s${SEED}_${RUN_ROLE}"
remote "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"
EXPORTS="ALL,RUN_ROLE=$RUN_ROLE,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_DATASET_SIZE=$BANK_DATASET_SIZE,BANK_DISTANCE_SIDECAR_SHA=$BANK_DISTANCE_SIDECAR_SHA,EXPECTED_PARAMETERS=$EXPECTED_PARAMETERS,TARGET_UPDATE=$THIS_TARGET"
JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_trench_align_generalist_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='$PARTITION' --time='$WALLTIME' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='eu-g6-064' --job-name='terra-trench-generalist-$RUN_ROLE' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
echo "role=$RUN_ROLE job_id=$JOB_ID run_dir=$RUN_DIR target_update=$THIS_TARGET"
