#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "usage: SUBMIT=0|stage|1 scripts/euler_v8_v61_u40_continuation/submit.sh" >&2
    exit 2
fi
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|stage|1) ;; *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;; esac

TRAINING_REVISION=dddc691c93ee21488cd7eeb8e01b067bf1f9733c
RUNTIME_TERRA_REVISION=c2d2a94a124759e9f21c2b37930f717e299f0c46
RESUME_UPDATE=40000
TARGET_UPDATE=70000
RESUME_SHA=17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8
PARENT_WANDB_RUN_ID=v8_v61_stall_age_v3_dddc691c93_phase2_10625259
PARENT_WANDB_MAX_UPDATE=39991

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH="$REPO/scripts/euler_v8_v61_u40_continuation/run.sbatch"
VALIDATOR="$REPO/scripts/euler_v8_v61_u40_continuation/verify_resume.py"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_v8_v61_stall_age_20260813}"
LOCAL_PYTHON="${LOCAL_TERRA_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}"
RESUME_LOCAL="${TERRA_U40_CHECKPOINT_LOCAL:-/home/lorenzo/moleworks/.artifacts/terra_v61_stall_age_continuation_20260814/final_u40000/v8_v61_stall_age_v3_u40000_FINAL_17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8.pkl}"
BANK_LOCAL="${TERRA_V8_BANK_LOCAL:-/home/lorenzo/moleworks/.artifacts/terra_v8_r2_training_inputs_20260810/treatment_bank.tar.zst}"

CAMPAIGN=terra_v8_v6_yolo_rv2
PHASE=phase3
ARM_NAME=v6_1_rv2_stall_age_v3
SEED=20260807
R2_HORIZON=450
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
DISTANCE_ARTIFACT_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980

# Keep the exact historical training source independent of this launcher commit.
git -C "$REPO" cat-file -e "$TRAINING_REVISION^{commit}"
git -C "$TERRA_REPO" cat-file -e "$RUNTIME_TERRA_REVISION^{commit}"
test -x "$LOCAL_PYTHON" -a -f "$SBATCH" -a -f "$VALIDATOR"
test "$(sha256sum "$RESUME_LOCAL" | awk '{print $1}')" = "$RESUME_SHA"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"

VALIDATION_OUTPUT="$(mktemp)"
trap 'rm -f -- "$VALIDATION_OUTPUT"' EXIT
PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONDONTWRITEBYTECODE=1 "$LOCAL_PYTHON" "$VALIDATOR" \
    --checkpoint "$RESUME_LOCAL" --sha256 "$RESUME_SHA" --output "$VALIDATION_OUTPUT"

LAUNCHER_REVISION="$(git -C "$REPO" rev-parse HEAD)"
LAUNCHER_SBATCH_SHA="$(sha256sum "$SBATCH" | awk '{print $1}')"
RESUME_VALIDATOR_SHA="$(sha256sum "$VALIDATOR" | awk '{print $1}')"
echo "resume=u${RESUME_UPDATE}->$TARGET_UPDATE checkpoint_sha256=$RESUME_SHA"
echo "training_revision=$TRAINING_REVISION runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "launcher_revision=$LAUNCHER_REVISION"
echo "wandb_run_id=$PARENT_WANDB_RUN_ID last_train_update=$PARENT_WANDB_MAX_UPDATE resume=must"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: exact local checkpoint, bank, and source contracts passed; no SSH, W&B, or Slurm mutation"
    exit 0
fi

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "launcher worktree must be committed and clean before staging" >&2
    exit 3
}

# shellcheck source=cluster/euler_account.sh
# shellcheck disable=SC1091
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"
REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
REMOTE_SOURCE="$REMOTE_WORK/$TRAINING_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

case "$REMOTE_HOST" in ''|-*|*[!a-zA-Z0-9._@-]*) echo "invalid REMOTE_HOST" >&2; exit 2 ;; esac
for REMOTE_PATH in "$REMOTE_WORK_ROOT" "$REMOTE_RUN_ROOT" "$REMOTE_VENV"; do
    case "$REMOTE_PATH" in /cluster/*) ;; *) echo "non-cluster remote path: $REMOTE_PATH" >&2; exit 2 ;; esac
    case "$REMOTE_PATH" in *[!a-zA-Z0-9_./-]*) echo "unsafe remote path: $REMOTE_PATH" >&2; exit 2 ;; esac
done

remote() {
    ssh -o BatchMode=yes "$REMOTE_HOST" "$@"
}

test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"
HOME_USED_GB="$(remote lquota | "$REPO/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$REMOTE_WORK/.${TRAINING_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$TRAINING_REVISION" \
        | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$TRAINING_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/$TRAINING_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" \
        | remote "tar -xf - -C '$PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
remote "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$TRAINING_REVISION' && test -s '$REMOTE_SOURCE/train_mixed.py' && test -s '$REMOTE_SOURCE/scripts/run_v8_v6_yolo_rv2.sh'"
remote "test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION' && test -s '$REMOTE_TERRA/terra/state.py'"
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

REMOTE_BANK="$REMOTE_INPUTS/treatment-bank-$BANK_SHA.tar.zst"
REMOTE_RESUME="$REMOTE_INPUTS/v8-v61-stall-age-v3-u40000-$RESUME_SHA.pkl"
REMOTE_VALIDATOR="$REMOTE_INPUTS/verify-u40000-$RESUME_VALIDATOR_SHA.py"
upload "$BANK_LOCAL" "$REMOTE_BANK" "$BANK_SHA"
upload "$RESUME_LOCAL" "$REMOTE_RESUME" "$RESUME_SHA"
upload "$VALIDATOR" "$REMOTE_VALIDATOR" "$RESUME_VALIDATOR_SHA"

PARTITION=gpuhe.24h
WALLTIME=23:45:00
if [ "$SUBMIT" = stage ]; then
    remote "sacctmgr -n -P show assoc where user='$TERRA_EULER_USER' format=Account | grep -Eq '^%?es_hutter$'"
    remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
    remote "sinfo -h -p '$PARTITION' -o '%G' | grep -q 'gpu:nvidia_geforce_rtx_4090:'"
    echo "SUBMIT=stage: exact code and content-addressed inputs staged; scheduler gates passed; no W&B or Slurm mutation"
    exit 0
fi

remote "python3 -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"
WANDB_MAX_OBSERVED="$(
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" \
    PARENT_WANDB_RUN_ID="$PARENT_WANDB_RUN_ID" "$LOCAL_PYTHON" - <<'PY'
import os

import wandb

path = f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/{os.environ["PARENT_WANDB_RUN_ID"]}'
run = wandb.Api(timeout=30).run(path)
if run.state != "finished":
    raise RuntimeError(f"parent W&B run state is {run.state!r}, expected 'finished'")
updates = [
    row["train/update"]
    for row in run.scan_history(keys=["train/update"], page_size=1000)
    if row.get("train/update") is not None
]
print(int(max(updates)))
PY
)"
test "$WANDB_MAX_OBSERVED" = "$PARENT_WANDB_MAX_UPDATE"
test "$WANDB_MAX_OBSERVED" -le "$RESUME_UPDATE"

RUN_DIR="$REMOTE_RUNS/$TRAINING_REVISION/$PHASE/s$SEED/$ARM_NAME"
remote "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"
JOB_ID=""
cleanup_new_job() {
    local status="$1"
    trap - ERR INT TERM
    set +e
    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        remote "scancel -- '$JOB_ID'"
    fi
    remote "rmdir -- '$RUN_DIR'" || true
    exit "$status"
}
trap 'cleanup_new_job $?' ERR
trap 'cleanup_new_job 130' INT TERM

EXPORTS="ALL,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,VENV=$REMOTE_VENV,RUN_DIR=$RUN_DIR,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,PARENT_WANDB_RUN_ID=$PARENT_WANDB_RUN_ID,PARENT_WANDB_MAX_UPDATE=$PARENT_WANDB_MAX_UPDATE,PHASE=$PHASE,ARM_NAME=$ARM_NAME,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$TRAINING_REVISION,LAUNCHER_REVISION=$LAUNCHER_REVISION,LAUNCHER_SBATCH_SHA=$LAUNCHER_SBATCH_SHA,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,R2_HORIZON=$R2_HORIZON,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,DISTANCE_ARTIFACT_SHA=$DISTANCE_ARTIFACT_SHA,RESUME_CHECKPOINT=$REMOTE_RESUME,RESUME_CHECKPOINT_SHA=$RESUME_SHA,RESUME_VALIDATOR=$REMOTE_VALIDATOR,RESUME_VALIDATOR_SHA=$RESUME_VALIDATOR_SHA"
JOB_RAW="$(remote "sbatch --parsable --account=es_hutter --partition='$PARTITION' --time='$WALLTIME' --gpus=rtx_4090:8 --cpus-per-task=8 --exclude=eu-g6-064 --job-name=terra-v61-u40-plus24h --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'" < "$SBATCH")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
trap - ERR INT TERM
echo "$PHASE $ARM_NAME $JOB_ID run_dir=$RUN_DIR wandb_resume=must"
