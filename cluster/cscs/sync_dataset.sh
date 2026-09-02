#!/usr/bin/env bash
# Upload a Terra dataset root once; training jobs reference the remote copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    echo "Usage: $0 LOCAL_DATASET_ROOT [REMOTE_NAME]"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
LOCAL_DATASET_ROOT="$(realpath "$1")"
REMOTE_NAME="${2:-$(basename "$LOCAL_DATASET_ROOT")}"
cscs_validate_token "remote dataset name" "$REMOTE_NAME"
[[ -d "$LOCAL_DATASET_ROOT" ]] || cscs_die "dataset directory not found: $LOCAL_DATASET_ROOT"

MAP_COUNT="$(find "$LOCAL_DATASET_ROOT" -type f -path '*/images/img_*.npy' | wc -l)"
[[ "$MAP_COUNT" -gt 0 ]] || cscs_die "no */images/img_*.npy files under $LOCAL_DATASET_ROOT"

cscs_require_command rsync
cscs_resolve_root
REMOTE_DATASET_ROOT="${CSCS_ROOT}/datasets/${REMOTE_NAME}"

echo "Syncing ${MAP_COUNT} map files to ${CSCS_SSH_TARGET}:${REMOTE_DATASET_ROOT}..." >&2
ssh -T "${CSCS_SSH_TARGET}" "mkdir -p $(printf '%q' "$REMOTE_DATASET_ROOT")"
rsync -az --info=stats1 \
    --exclude=__pycache__ --exclude='*.pyc' --exclude=preview \
    "${LOCAL_DATASET_ROOT}/" "${CSCS_SSH_TARGET}:${REMOTE_DATASET_ROOT}/" >&2

REMOTE_COUNT="$(ssh -T "${CSCS_SSH_TARGET}" "find $(printf '%q' "$REMOTE_DATASET_ROOT") -type f -path '*/images/img_*.npy' | wc -l")"
[[ "$REMOTE_COUNT" -eq "$MAP_COUNT" ]] || cscs_die "remote map count ${REMOTE_COUNT} does not match local count ${MAP_COUNT}"
echo "Synced dataset: ${REMOTE_DATASET_ROOT} (${REMOTE_COUNT} maps)" >&2
printf '%s\n' "$REMOTE_DATASET_ROOT"
