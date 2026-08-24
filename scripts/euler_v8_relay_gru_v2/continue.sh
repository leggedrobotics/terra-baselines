#!/usr/bin/env bash
# Queue a wallclock continuation of the running GRU v2 scratch job.
#
# The scratch job (default 10991006) will hit its 119:45 limit around u91k at
# the measured ~4.7 s/update wall rate. This submits one dependent job
# (afterany) that resumes from the newest checkpoint in the SAME run dir and
# trains to the original 100k target; if the scratch job somehow reaches 100k,
# the continuation exits 0 without training.
set -euo pipefail

DEPEND_JOB="${DEPEND_JOB:-10991006}"
SCRATCH_REVISION=33d26213327d66921b66753a5a6018a37d6f2e81
WANDB_RUN_ID_PIN="v8_relay_gru64r_${SCRATCH_REVISION:0:10}_s20260817"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_relay_main_integration_20260815}"
FULL_BANK_ARCHIVE_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
FULL_BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
PARTIAL_BANK_ARCHIVE_SHA=eb200b151f6b47d9f2ea5f53f6b13cdb45b595a54029fd5d866ec732fea1c8b8
PARTIAL_BANK_SHA=fb73b1d12dfad98c9aa79680d4d3ac178bf84b537e1be1e822535c65473a23f5
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
SEED=20260817

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_v8_relay_gru_v2
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(git -C "$TERRA_REPO" status --porcelain)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$RUNTIME_TERRA_REVISION" = 25f855db3d913fd638c4e56b1740437a2b7122ca

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "squeue -j '$DEPEND_JOB' -h -o %T | grep -Eq 'RUNNING|PENDING'"

# Stage this revision's source (training code identical to scratch; scripts/docs differ).
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
remote "test -e '$REMOTE_TERRA'"

REMOTE_BANK="$REMOTE_INPUTS/full-bank-$FULL_BANK_ARCHIVE_SHA.tar.zst"
REMOTE_PARTIAL="$REMOTE_INPUTS/partial-bank-$PARTIAL_BANK_ARCHIVE_SHA.tar.zst"
remote "test -f '$REMOTE_BANK' && test -f '$REMOTE_PARTIAL'"

RUN_NAME="v8_relay_gru64r_${SCRATCH_REVISION:0:12}_s${SEED}"
RUN_DIR="$REMOTE_RUNS/$SCRATCH_REVISION/s$SEED"
remote "test -d '$RUN_DIR/checkpoints'"

EXPORTS="ALL,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_TERRA_REVISION=$PROTOCOL_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,WANDB_RUN_ID=$WANDB_RUN_ID_PIN,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$FULL_BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$FULL_BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,DISTANCE_SIDECAR_SHA=$DISTANCE_SIDECAR_SHA,PARTIAL_ARCHIVE=$REMOTE_PARTIAL,PARTIAL_ARCHIVE_SHA=$PARTIAL_BANK_ARCHIVE_SHA,PARTIAL_BANK_SHA=$PARTIAL_BANK_SHA,TARGET_UPDATE=100000,PHASE=resume,RESUME_CHECKPOINT=latest"

JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_relay_gru_v2/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.120h' --time='119:45:00' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='eu-g6-064' --dependency='afterany:$DEPEND_JOB' --job-name='terra-v8-gru64r-c' --output='$RUN_DIR/slurm_resume_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
printf '%s\n' "continuation_job_id=$JOB_ID" "dependency=afterany:$DEPEND_JOB" "run_dir=$RUN_DIR" "target_update=100000"
