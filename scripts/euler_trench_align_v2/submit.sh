#!/usr/bin/env bash
# Fresh-trench dig-alignment pilot: submits BOTH matched-seed arms (C0 gate-off,
# T1 gate-on) as two 4x RTX 4090 jobs. SUBMIT=0 local contract check only,
# SUBMIT=stage stages code+inputs without Slurm mutation, SUBMIT=1 submits.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|1) ;;
    *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818}"

# ---- pinned pilot inputs (filled from the enrichment manifest) --------------
BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819/terra_v2_generalist_pooled_bank_20260901.tar.zst
BANK_ARCHIVE_SHA=1125177d322df6097f8da9f67ec95fe48762e16327f83dc157ec282b24993fb3
BANK_MAPS_PATH=train_v2_pooled_generalist
BANK_DATASET_SIZE=3840
# Frozen R2 receipt: canonical_distance_sidecar_dataset_sha256 of the enriched
# bank (.artifacts/terra_v8_trench_finite_enriched_20260819/dataset.json).
BANK_DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
EXPECTED_PARAMETERS=2307645
TRENCH_TERRA_REVISION_PIN=fd7195751f238bc3c0afd0ad60385741021de35b
SEED=20260901
TARGET_UPDATE=100000
# -----------------------------------------------------------------------------

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_trench_align_v2_generalist
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

# The baselines worktree carries a concurrent in-flight isaac_sim change from
# another session; git archive ships HEAD, so those paths are excluded from the
# cleanliness contract instead of being committed here.
test -z "$(git -C "$REPO" status --porcelain -- . \
    ':(exclude)isaac_sim' ':(exclude)tests/test_trench_pose_alignment.py')"
test -z "$(git -C "$TERRA_REPO" status --porcelain)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$RUNTIME_TERRA_REVISION" = "$TRENCH_TERRA_REVISION_PIN"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
grep -q "path: &v2gen_bank $BANK_MAPS_PATH\$" \
    "$REPO/configs/training_configs.yaml"

echo "terra_baselines_revision=$BASELINES_REVISION"
echo "runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "arms=c0,t1 seed=$SEED devices=4 envs_per_device=512 target=$TARGET_UPDATE"
if [ "$SUBMIT" = 0 ]; then
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

REMOTE_BANK="$REMOTE_INPUTS/trench-bank-$BANK_ARCHIVE_SHA.tar.zst"
if ! remote "test -f '$REMOTE_BANK'"; then
    scp -q -o BatchMode=yes "$BANK_ARCHIVE" "$REMOTE_HOST:$REMOTE_BANK.partial.$$"
    remote "test \"\$(sha256sum '$REMOTE_BANK.partial.$$' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA' && mv -T '$REMOTE_BANK.partial.$$' '$REMOTE_BANK'"
fi
remote "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA'"

PARTITION=gpuhe.120h
WALLTIME=119:45:00
remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: exact source and inputs staged; no Slurm mutation"
    exit 0
fi

for ARM in gen; do
    RUN_NAME="trench_align_v2gen_${ARM}_${BASELINES_REVISION:0:12}_s${SEED}"
    RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED/$ARM"
    remote "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,ARM=$ARM,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_MAPS_PATH=$BANK_MAPS_PATH,BANK_DATASET_SIZE=$BANK_DATASET_SIZE,BANK_DISTANCE_SIDECAR_SHA=$BANK_DISTANCE_SIDECAR_SHA,EXPECTED_PARAMETERS=$EXPECTED_PARAMETERS,TARGET_UPDATE=$TARGET_UPDATE"
    JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_trench_align_v2/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='$PARTITION' --time='$WALLTIME' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='eu-g6-064' --job-name='terra-trench-v2gen' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
    JOB_ID="${JOB_RAW%%;*}"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]]
    printf '%s\n' "arm=$ARM job_id=$JOB_ID run_dir=$RUN_DIR"
done
printf '%s\n' "target_update=$TARGET_UPDATE seed=$SEED"
