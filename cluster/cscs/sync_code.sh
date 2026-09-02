#!/usr/bin/env bash
# Stage an immutable source snapshot containing both Terra repositories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    echo "Usage: $0 [--run-id ID]"
}

RUN_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            [[ $# -ge 2 ]] || cscs_die "--run-id requires a value"
            RUN_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            cscs_die "unknown argument: $1"
            ;;
    esac
done

RUN_ID="${RUN_ID:-$(cscs_default_run_id)}"
cscs_validate_token "run id" "$RUN_ID"
cscs_require_command git
cscs_require_command rsync
cscs_require_command ssh
[[ -e "${CSCS_TERRA_DIR}/.git" ]] || cscs_die "Terra repository not found: ${CSCS_TERRA_DIR}"
[[ -e "${CSCS_BASELINES_DIR}/.git" ]] || cscs_die "terra-baselines repository not found: ${CSCS_BASELINES_DIR}"

cscs_resolve_root
SNAPSHOT_ROOT="${CSCS_ROOT}/snapshots/${RUN_ID}"
cscs_validate_absolute_path "snapshot root" "$SNAPSHOT_ROOT"

if ssh -T "${CSCS_SSH_TARGET}" "test -e $(printf '%q' "$SNAPSHOT_ROOT")"; then
    cscs_die "remote snapshot already exists: ${SNAPSHOT_ROOT}"
fi

echo "Staging source snapshot ${RUN_ID} on ${CSCS_SSH_TARGET}..." >&2
ssh -T "${CSCS_SSH_TARGET}" "mkdir -p $(printf '%q' "$SNAPSHOT_ROOT/terra") $(printf '%q' "$SNAPSHOT_ROOT/terra-baselines")"

RSYNC_EXCLUDES=(
    --exclude=.git
    --exclude=.venv
    --exclude='.venv-*'
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=.pytest_cache
    --exclude=.ruff_cache
    --exclude=.mypy_cache
    --exclude=.vscode
    --exclude=wandb
    --exclude=logs
    --exclude=outputs
    --exclude=checkpoints
    --exclude=data
    --exclude=cluster/cscs/config.env
)

rsync -az --info=stats1 "${RSYNC_EXCLUDES[@]}" \
    "${CSCS_TERRA_DIR}/" "${CSCS_SSH_TARGET}:${SNAPSHOT_ROOT}/terra/" >&2
rsync -az --info=stats1 "${RSYNC_EXCLUDES[@]}" \
    "${CSCS_BASELINES_DIR}/" "${CSCS_SSH_TARGET}:${SNAPSHOT_ROOT}/terra-baselines/" >&2

TERRA_LABEL="$(cscs_git_label "${CSCS_TERRA_DIR}")"
BASELINES_LABEL="$(cscs_git_label "${CSCS_BASELINES_DIR}")"
TERRA_DIRTY="$(git -C "${CSCS_TERRA_DIR}" status --porcelain | wc -l)"
BASELINES_DIRTY="$(git -C "${CSCS_BASELINES_DIR}" status --porcelain | wc -l)"

ssh -T "${CSCS_SSH_TARGET}" \
    env SNAPSHOT_ROOT="$SNAPSHOT_ROOT" RUN_ID="$RUN_ID" \
        TERRA_LABEL="$TERRA_LABEL" BASELINES_LABEL="$BASELINES_LABEL" \
        TERRA_DIRTY="$TERRA_DIRTY" BASELINES_DIRTY="$BASELINES_DIRTY" \
    bash -s <<'REMOTE'
set -euo pipefail
{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'terra=%s\n' "$TERRA_LABEL"
    printf 'terra_dirty_entries=%s\n' "$TERRA_DIRTY"
    printf 'terra_baselines=%s\n' "$BASELINES_LABEL"
    printf 'terra_baselines_dirty_entries=%s\n' "$BASELINES_DIRTY"
} > "${SNAPSHOT_ROOT}/SOURCE_REVISIONS.txt"
touch "${SNAPSHOT_ROOT}/.ready"
REMOTE

echo "Staged: ${SNAPSHOT_ROOT}" >&2
printf '%s\n' "$SNAPSHOT_ROOT"
