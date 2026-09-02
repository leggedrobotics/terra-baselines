#!/usr/bin/env bash

set -euo pipefail

CSCS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSCS_BASELINES_DIR="$(cd "${CSCS_SCRIPT_DIR}/../.." && pwd)"
CSCS_WORKSPACE_DIR="$(cd "${CSCS_BASELINES_DIR}/.." && pwd)"
CSCS_TERRA_DIR="${CSCS_WORKSPACE_DIR}/terra"

if [[ -f "${CSCS_SCRIPT_DIR}/config.env" ]]; then
    # shellcheck source=/dev/null
    source "${CSCS_SCRIPT_DIR}/config.env"
fi

CSCS_SSH_TARGET="${CSCS_SSH_TARGET:-daint}"
CSCS_ACCOUNT="${CSCS_ACCOUNT:-d130}"
CSCS_PARTITION="${CSCS_PARTITION:-normal}"
CSCS_IMAGE_NAME="${CSCS_IMAGE_NAME:-terra-jax}"
CSCS_IMAGE_TAG="${CSCS_IMAGE_TAG:-jax24.10-v1}"
CSCS_BASE_IMAGE="${CSCS_BASE_IMAGE:-nvcr.io/nvidia/jax:24.10-py3}"
CSCS_WANDB_ENTITY="${CSCS_WANDB_ENTITY:-aless-weber-eth}"

cscs_die() {
    echo "Error: $*" >&2
    exit 1
}

cscs_require_command() {
    command -v "$1" >/dev/null 2>&1 || cscs_die "required command not found: $1"
}

cscs_validate_token() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[A-Za-z0-9._+-]+$ ]] || cscs_die "$label contains unsupported characters: $value"
}

cscs_validate_absolute_path() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^/[A-Za-z0-9._/+:-]+$ ]] \
        || cscs_die "$label must be an absolute path without spaces or shell metacharacters: $value"
}

cscs_remote_scratch() {
    ssh -T "${CSCS_SSH_TARGET}" 'printf "%s\n" "${SCRATCH:-/capstor/scratch/cscs/$USER}"'
}

cscs_resolve_root() {
    if [[ -z "${CSCS_ROOT:-}" ]]; then
        CSCS_ROOT="$(cscs_remote_scratch)/terra-training"
    fi
    cscs_validate_absolute_path "CSCS_ROOT" "${CSCS_ROOT}"
    export CSCS_ROOT
}

cscs_git_revision() {
    local repo="$1"
    git -C "$repo" rev-parse HEAD
}

cscs_git_label() {
    local repo="$1"
    local branch
    branch="$(git -C "$repo" symbolic-ref --quiet --short HEAD || echo detached)"
    printf '%s@%s' "$branch" "$(cscs_git_revision "$repo")"
}

cscs_default_run_id() {
    local terra_short baselines_short
    terra_short="$(cscs_git_revision "${CSCS_TERRA_DIR}" | cut -c1-8)"
    baselines_short="$(cscs_git_revision "${CSCS_BASELINES_DIR}" | cut -c1-8)"
    printf 'terra-%s-%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$terra_short" "$baselines_short"
}
