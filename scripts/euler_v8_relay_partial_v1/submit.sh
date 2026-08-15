#!/usr/bin/env bash
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|1) ;;
    *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=cluster/euler_account.sh
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_relay_main_integration_20260815}"
FULL_BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_training_inputs_20260810/treatment_bank.tar.zst
FULL_BANK_ARCHIVE_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
FULL_BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
PARTIAL_BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_v8_relay_partial_bank_20260815/partial_bank.tar.zst
PARTIAL_BANK_ARCHIVE_SHA=eb200b151f6b47d9f2ea5f53f6b13cdb45b595a54029fd5d866ec732fea1c8b8
PARTIAL_BANK_SHA=fb73b1d12dfad98c9aa79680d4d3ac178bf84b537e1be1e822535c65473a23f5
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
SEED=20260815

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_v8_relay_partial_v1
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean" >&2; exit 3;
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "Terra runtime must be committed and clean" >&2; exit 3;
}
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$RUNTIME_TERRA_REVISION" = 25f855db3d913fd638c4e56b1740437a2b7122ca
test "$(sha256sum "$FULL_BANK_ARCHIVE" | awk '{print $1}')" = "$FULL_BANK_ARCHIVE_SHA"
test "$(tar --zstd -xOf "$FULL_BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$FULL_BANK_DATASET_SHA"
test "$(sha256sum "$PARTIAL_BANK_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_BANK_ARCHIVE_SHA"
[[ "$PARTIAL_BANK_SHA" =~ ^[0-9a-f]{64}$ ]]

LOCAL_PYTHON=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu "$LOCAL_PYTHON" - <<PY
from terra.maps_buffer import partial_reset_bank_sha256
observed = partial_reset_bank_sha256('/home/lorenzo/moleworks/.artifacts/terra_v8_relay_partial_bank_20260815/partial_bank')
assert observed == '$PARTIAL_BANK_SHA', observed
PY

echo "terra_revision=$RUNTIME_TERRA_REVISION"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "seed=$SEED target=200000 segments=100000+100000"
echo "partial_reset_bank_sha256=$PARTIAL_BANK_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
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
remote "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION' && test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
remote "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"

upload() {
    local source="$1" destination="$2" expected_sha="$3"
    if ! remote "test -f '$destination'"; then
        local partial="$destination.partial.$$"
        scp -q -o BatchMode=yes "$source" "$REMOTE_HOST:$partial"
        remote "test \"\$(sha256sum '$partial' | awk '{print \$1}')\" = '$expected_sha' && mv -T '$partial' '$destination'"
    fi
    remote "test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected_sha'"
}

REMOTE_BANK="$REMOTE_INPUTS/full-bank-$FULL_BANK_ARCHIVE_SHA.tar.zst"
REMOTE_PARTIAL="$REMOTE_INPUTS/partial-bank-$PARTIAL_BANK_ARCHIVE_SHA.tar.zst"
upload "$FULL_BANK_ARCHIVE" "$REMOTE_BANK" "$FULL_BANK_ARCHIVE_SHA"
upload "$PARTIAL_BANK_ARCHIVE" "$REMOTE_PARTIAL" "$PARTIAL_BANK_ARCHIVE_SHA"

PARTITION=gpuhe.120h
WALLTIME=119:45:00
GPU_TYPE=rtx_4090
GPU_COUNT=8
CPUS=8
ASSOCIATIONS="$(remote "sacctmgr -n -P show assoc where user='$TERRA_EULER_USER' format=Account")"
printf '%s\n' "$ASSOCIATIONS" | grep -Eq '^%?es_hutter$'
remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
remote "sinfo -h -p '$PARTITION' -o '%G' | grep -Eq 'gpu:nvidia_geforce_${GPU_TYPE}:([$GPU_COUNT-9]|[1-9][0-9]+)'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: exact code and inputs staged; scheduler gates passed; no job or W&B mutation"
    exit 0
fi

remote "python3 -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"
RUN_NAME="v8_relay_partial_${BASELINES_REVISION:0:12}_s${SEED}"
RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED/treatment"
remote "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"

COMMON_EXPORTS="ALL,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_TERRA_REVISION=$PROTOCOL_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$FULL_BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$FULL_BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,DISTANCE_SIDECAR_SHA=$DISTANCE_SIDECAR_SHA,PARTIAL_ARCHIVE=$REMOTE_PARTIAL,PARTIAL_ARCHIVE_SHA=$PARTIAL_BANK_ARCHIVE_SHA,PARTIAL_BANK_SHA=$PARTIAL_BANK_SHA"

JOB1_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_relay_partial_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:$GPU_COUNT' --cpus-per-task='$CPUS' --exclude='eu-g6-064' --job-name='terra-v8-relay-u100' --output='$RUN_DIR/slurm_scratch_%j.out' --export='$COMMON_EXPORTS,PHASE=scratch,TARGET_UPDATE=100000,RESUME_CHECKPOINT=none'")"
JOB1_ID="${JOB1_RAW%%;*}"
[[ "$JOB1_ID" =~ ^[0-9]+$ ]]

REMOTE_PHASE1_CHECKPOINT="$RUN_DIR/checkpoints/${RUN_NAME}_update_100000.pkl"
JOB2_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_relay_partial_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:$GPU_COUNT' --cpus-per-task='$CPUS' --exclude='eu-g6-064' --dependency=afterok:$JOB1_ID --kill-on-invalid-dep=yes --job-name='terra-v8-relay-u200' --output='$RUN_DIR/slurm_resume_%j.out' --export='$COMMON_EXPORTS,PHASE=resume,TARGET_UPDATE=200000,RESUME_CHECKPOINT=$REMOTE_PHASE1_CHECKPOINT'")"
JOB2_ID="${JOB2_RAW%%;*}"
[[ "$JOB2_ID" =~ ^[0-9]+$ ]] || { remote "scancel -- '$JOB1_ID'"; exit 3; }

printf '%s\n' \
    "scratch_job_id=$JOB1_ID" \
    "resume_job_id=$JOB2_ID" \
    "dependency=afterok:$JOB1_ID" \
    "final_target_update=200000"
