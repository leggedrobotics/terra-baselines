#!/usr/bin/env bash
# Stage both repositories and submit or validate one four-GPU Daint job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  submit.sh --dataset-path REMOTE_PATH --dataset-size N [options] [-- TRAIN_ARGS...]

Options:
  --profile smoke|production  Training defaults (default: smoke)
  --time HH:MM:SS             Wall time (smoke default: 00:20:00; production: 24:00:00)
  --partition NAME            Slurm partition (default from config.env)
  --account NAME              Slurm account (default from config.env)
  --run-id ID                 Stable run identifier
  --test-only                 Run sbatch --test-only; this is the default
  --submit                    Submit the billable job
  --no-sync                   Reuse an existing --run-id snapshot
EOF
}

PROFILE=smoke
JOB_TIME=""
PARTITION="$CSCS_PARTITION"
ACCOUNT="$CSCS_ACCOUNT"
RUN_ID=""
DATASET_PATH=""
DATASET_SIZE=""
MODE=test-only
SYNC_CODE=1
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --time) JOB_TIME="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --dataset-size) DATASET_SIZE="$2"; shift 2 ;;
        --test-only) MODE=test-only; shift ;;
        --submit) MODE=submit; shift ;;
        --no-sync) SYNC_CODE=0; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) cscs_die "unknown argument: $1" ;;
    esac
done

[[ "$PROFILE" == smoke || "$PROFILE" == production ]] || cscs_die "profile must be smoke or production"
[[ -n "$DATASET_PATH" ]] || cscs_die "--dataset-path is required"
cscs_validate_absolute_path "dataset path" "$DATASET_PATH"
[[ "$DATASET_SIZE" =~ ^[1-9][0-9]*$ ]] || cscs_die "--dataset-size must be a positive integer"
cscs_validate_token "partition" "$PARTITION"
cscs_validate_token "account" "$ACCOUNT"
JOB_TIME="${JOB_TIME:-$([[ "$PROFILE" == smoke ]] && echo 00:20:00 || echo 24:00:00)}"
[[ "$JOB_TIME" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || cscs_die "--time must use HH:MM:SS"

cscs_resolve_root
RUN_ID="${RUN_ID:-$(cscs_default_run_id)}"
cscs_validate_token "run id" "$RUN_ID"
SNAPSHOT_ROOT="${CSCS_ROOT}/snapshots/${RUN_ID}"
RUN_ROOT="${CSCS_ROOT}/runs/${RUN_ID}"
IMAGE_PATH="${CSCS_ROOT}/images/${CSCS_IMAGE_NAME}+${CSCS_IMAGE_TAG}.sqsh"
EDF_PATH="${RUN_ROOT}/terra.edf.toml"
SBATCH_PATH="${RUN_ROOT}/job.sbatch"

if [[ "$SYNC_CODE" -eq 1 ]]; then
    SNAPSHOT_ROOT="$("${SCRIPT_DIR}/sync_code.sh" --run-id "$RUN_ID")"
else
    ssh -T "$CSCS_SSH_TARGET" "test -f $(printf '%q' "$SNAPSHOT_ROOT/.ready")" \
        || cscs_die "snapshot is not ready: $SNAPSHOT_ROOT"
fi

ssh -T "$CSCS_SSH_TARGET" \
    env RUN_ROOT="$RUN_ROOT" SNAPSHOT_ROOT="$SNAPSHOT_ROOT" IMAGE_PATH="$IMAGE_PATH" \
        EDF_PATH="$EDF_PATH" SBATCH_PATH="$SBATCH_PATH" DATASET_PATH="$DATASET_PATH" \
        DATASET_SIZE="$DATASET_SIZE" RUN_ID="$RUN_ID" PROFILE="$PROFILE" \
        ACCOUNT="$ACCOUNT" PARTITION="$PARTITION" JOB_TIME="$JOB_TIME" \
        CSCS_WANDB_ENTITY="$CSCS_WANDB_ENTITY" \
    bash -s <<'REMOTE'
set -euo pipefail
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb" "$RUN_ROOT/work"

cat > "$EDF_PATH" <<EOF
image = "${IMAGE_PATH}"
mounts = [
    "/capstor/",
    "/iopsstor/",
    "/users/",
    "${SNAPSHOT_ROOT}/terra:/workspace/terra",
    "${SNAPSHOT_ROOT}/terra-baselines:/workspace/terra-baselines",
]
workdir = "${RUN_ROOT}/work"
entrypoint = true

[annotations]
com.pyxis.entrypoint_log = "true"
com.hooks.aws_ofi_nccl.enabled = "false"

[env]
HOME = "/users/\${USER}"
PYTHONUNBUFFERED = "1"
DATASET_PATH = "${DATASET_PATH}"
DATASET_SIZE = "${DATASET_SIZE}"
TERRA_RUN_DIR = "${RUN_ROOT}"
WANDB_DIR = "${RUN_ROOT}/wandb"
WANDB_ENTITY = "${CSCS_WANDB_ENTITY}"
MPLBACKEND = "Agg"
SDL_VIDEODRIVER = "dummy"
EOF

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=${JOB_TIME}
#SBATCH --job-name=terra-${PROFILE}
#SBATCH --output=${RUN_ROOT}/logs/slurm-%j.out
#SBATCH --error=${RUN_ROOT}/logs/slurm-%j.err

set -euo pipefail
srun --mpi=pmix --network=disable_rdzv_get --environment=${EDF_PATH} \
    /workspace/terra-baselines/cluster/cscs/run_training.sh ${PROFILE} __TRAIN_ARGS__
EOF
REMOTE

TRAIN_ARGS_ESCAPED=""
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    printf -v TRAIN_ARGS_ESCAPED ' %q' "${TRAIN_ARGS[@]}"
fi
ssh -T "$CSCS_SSH_TARGET" \
    env SBATCH_PATH="$SBATCH_PATH" TRAIN_ARGS_ESCAPED="$TRAIN_ARGS_ESCAPED" \
    bash -s <<'REMOTE'
set -euo pipefail
python3 - "$SBATCH_PATH" "$TRAIN_ARGS_ESCAPED" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
content = path.read_text()
content = content.replace(" __TRAIN_ARGS__", sys.argv[2])
path.write_text(content)
PY
chmod 700 "$SBATCH_PATH"
REMOTE

echo "Run: ${RUN_ID}"
echo "Snapshot: ${SNAPSHOT_ROOT}"
echo "Artifacts: ${RUN_ROOT}"
echo "Image: ${IMAGE_PATH}"

if [[ "$MODE" == test-only ]]; then
    ssh -T "$CSCS_SSH_TARGET" "test -f $(printf '%q' "$IMAGE_PATH")" \
        || echo "Warning: image is not built yet: ${IMAGE_PATH}" >&2
    ssh -T "$CSCS_SSH_TARGET" "sbatch --test-only $(printf '%q' "$SBATCH_PATH")"
else
    ssh -T "$CSCS_SSH_TARGET" "test -f $(printf '%q' "$IMAGE_PATH")" \
        || cscs_die "image is not built: ${IMAGE_PATH}"
    ssh -T "$CSCS_SSH_TARGET" "test -d $(printf '%q' "$DATASET_PATH")" \
        || cscs_die "remote dataset does not exist: ${DATASET_PATH}"
    ssh -T "$CSCS_SSH_TARGET" "sbatch $(printf '%q' "$SBATCH_PATH")"
fi
