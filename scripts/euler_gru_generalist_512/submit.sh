#!/usr/bin/env bash
# Stage and submit the GRU generalist on the 512-maps-per-condition bank.
#   SUBMIT=0      local checks only (default)
#   SUBMIT=stage  upload exact source, Terra runtime, bank and teacher
#   SUBMIT=smoke  3-update finite smoke, W&B disabled, 4 h queue
#   SUBMIT=1      production segment (TERRA_RESUME_FROM=<ckpt> to continue)
# Outputs live on project storage: lterenzi scratch is over its file quota.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|stage|smoke|1) ;; *) echo "SUBMIT must be 0, stage, smoke or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/cluster/euler_account.sh"
terra_euler_configure lterenzi
TERRA_REPO="${TERRA_REPO:-$(dirname "$REPO")/terra}"

# Environment code must equal the corrected-geometry runtime; tool-only
# commits on the Terra branch are allowed.
TERRA_ENV_REVISION=2122b2dfc97bb04facc100e0e51905e4aea9d1b7
BANK_ARCHIVE="${BANK_ARCHIVE:-/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/train_v3_generalist_512.tar.zst}"
BANK_ARCHIVE_SHA="${BANK_ARCHIVE_SHA:-3ddaaf538f1e7a091123ca1491c7f305e03835822d4f4c83c17d1f5916f6fc74}"
BANK_MAPS_PATH=train_v3_generalist_512
BANK_DATASET_SIZE="${BANK_DATASET_SIZE:-20480}"
BANK_DISTANCE_SIDECAR_SHA="${BANK_DISTANCE_SIDECAR_SHA:-d9a8c9e61346e319b3ffead150ca44c7e85d016cb5853997cf1ee721286d4506}"
TEACHER_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_instance_efficiency_20260922/inputs/generalist_u110000.pkl
TEACHER_SHA=3fd74795794dbf27a3c2fce01c91414bfb211b913dc9b018bc540d8a7acd7057
EXPECTED_PARAMETERS=2359445
SEED="${TERRA_SEED:-20260923}"
NUM_DEVICES=4
ENVS_PER_DEVICE="${TERRA_ENVS_PER_DEVICE:-512}"
GPU_TYPE="${GPU_TYPE:-rtx_4090}"
# Teacher guidance covers 655,360,000 transitions, as in the 2026-09-15 restart.
KL_ANNEAL_UPDATES=$((655360000 / (NUM_DEVICES * ENVS_PER_DEVICE * 32)))
RESUME_FROM="${TERRA_RESUME_FROM:-none}"
DEPENDENCY="${TERRA_SLURM_DEPENDENCY:-none}"
if [ "$SUBMIT" = smoke ]; then
    TARGET_UPDATE=3 PARTITION=gpuhe.4h WALLTIME=01:30:00 WANDB_MODE=disabled
else
    TARGET_UPDATE="${TERRA_TARGET_UPDATE:-100000}"
    PARTITION="${TERRA_PARTITION:-gpuhe.120h}"
    WALLTIME="${TERRA_WALLTIME:-119:45:00}"
    WANDB_MODE="${WANDB_MODE:-online}"
fi
[[ "$DEPENDENCY" == none || "$DEPENDENCY" =~ ^after(ok|any):[0-9]+$ ]]
REMOTE_HOST=euler-lterenzi
REMOTE_VENV=/cluster/project/rsl/lterenzi/terra_runtime/terra_jax0433_cuda126_cudnn950_20260903
RUNTIME_LOCK_SHA=36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189
ROOT="$TERRA_EULER_PROJECT_ROOT/terra_experiments/terra_gru_bigbank_20260923"
WANDB_ENTITY=aless-weber-eth
WANDB_PROJECT=mixed-agents

test -z "$(git -C "$REPO" status --porcelain -- . ':(exclude)isaac_sim')"
test -z "$(git -C "$TERRA_REPO" status --porcelain)"
test -z "$(git -C "$TERRA_REPO" diff --name-only "$TERRA_ENV_REVISION" HEAD -- terra)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$TEACHER_LOCAL" | awk '{print $1}')" = "$TEACHER_SHA"
grep -q "path: &gen512_bank $BANK_MAPS_PATH\$" "$REPO/configs/training_configs.yaml"
echo "baselines=$BASELINES_REVISION terra=$RUNTIME_TERRA_REVISION seed=$SEED"
echo "shape=${NUM_DEVICES}x$GPU_TYPE envs/device=$ENVS_PER_DEVICE kl_anneal_updates=$KL_ANNEAL_UPDATES"
echo "target=$TARGET_UPDATE partition=$PARTITION walltime=$WALLTIME wandb=$WANDB_MODE resume=$RESUME_FROM"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local checks passed; nothing staged"
    exit 0
fi

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -x '$REMOTE_VENV/bin/python' && test \"\$(sha256sum '$REMOTE_VENV/requirements.lock.txt' | awk '{print \$1}')\" = '$RUNTIME_LOCK_SHA'"
REMOTE_SOURCE="$ROOT/src/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$ROOT/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
remote "mkdir -p '$ROOT/src' '$ROOT/runtime-terra' '$ROOT/inputs' '$ROOT/runs'"
if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$ROOT/src/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$ROOT/src/$BASELINES_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    PARTIAL="$ROOT/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | remote "tar -xf - -C '$PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv -T '$PARTIAL' '$ROOT/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
upload() {  # local, remote, sha
    if ! remote "test -f '$2'"; then
        scp -q -o BatchMode=yes "$1" "$REMOTE_HOST:$2.partial.$$"
        remote "mv -T '$2.partial.$$' '$2'"
    fi
    remote "test \"\$(sha256sum '$2' | awk '{print \$1}')\" = '$3'"
}
REMOTE_BANK="$ROOT/inputs/bank-$BANK_ARCHIVE_SHA.tar.zst"
REMOTE_TEACHER="$ROOT/inputs/teacher-$TEACHER_SHA.pkl"
upload "$BANK_ARCHIVE" "$REMOTE_BANK" "$BANK_ARCHIVE_SHA"
upload "$TEACHER_LOCAL" "$REMOTE_TEACHER" "$TEACHER_SHA"
remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: source, runtime, bank and teacher staged under $ROOT"
    exit 0
fi

if [ "$SUBMIT" = smoke ]; then
    RUN_DIR="$ROOT/runs/smoke/${BASELINES_REVISION:0:12}_$(date +%Y%m%d_%H%M%S)"
    RUN_NAME="gru_gen512_smoke_${BASELINES_REVISION:0:10}"
else
    RUN_DIR="$ROOT/runs/${BASELINES_REVISION:0:12}/s$SEED"
    RUN_NAME="gru_gen512_${BASELINES_REVISION:0:10}_s$SEED"
fi
if [ "$RESUME_FROM" = none ]; then
    remote "test ! -e '$RUN_DIR' && mkdir -p '$RUN_DIR'"
else
    remote "test -d '$RUN_DIR' && test -r '$RESUME_FROM'"
fi
EXPORTS="ALL,TARGET_UPDATE=$TARGET_UPDATE,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE"
EXPORTS+=",BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA"
EXPORTS+=",RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV"
EXPORTS+=",RUNTIME_LOCK_SHA=$RUNTIME_LOCK_SHA,TERRA_EULER_USER=$TERRA_EULER_USER"
EXPORTS+=",TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY"
EXPORTS+=",WANDB_PROJECT=$WANDB_PROJECT,WANDB_MODE=$WANDB_MODE,BANK_ARCHIVE=$REMOTE_BANK"
EXPORTS+=",BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_MAPS_PATH=$BANK_MAPS_PATH"
EXPORTS+=",BANK_DATASET_SIZE=$BANK_DATASET_SIZE,BANK_DISTANCE_SIDECAR_SHA=$BANK_DISTANCE_SIDECAR_SHA"
EXPORTS+=",TEACHER=$REMOTE_TEACHER,TEACHER_SHA=$TEACHER_SHA,NUM_DEVICES=$NUM_DEVICES"
EXPORTS+=",ENVS_PER_DEVICE=$ENVS_PER_DEVICE,KL_ANNEAL_UPDATES=$KL_ANNEAL_UPDATES,GPU_TYPE=$GPU_TYPE"
EXPORTS+=",EXPECTED_PARAMETERS=$EXPECTED_PARAMETERS,RESUME_FROM=$RESUME_FROM"
DEPENDENCY_OPTION=""
[ "$DEPENDENCY" = none ] || DEPENDENCY_OPTION="--dependency=$DEPENDENCY"
JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_gru_generalist_512/run.sbatch' | sbatch --parsable --account=es_hutter --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:$NUM_DEVICES' --cpus-per-task=8 --mem-per-cpu=8G --tmp=20G --exclude=eu-g6-064 --job-name='terra-gru-gen512' --output='$RUN_DIR/slurm_%j.out' $DEPENDENCY_OPTION --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
echo "job_id=$JOB_ID run_dir=$RUN_DIR"
