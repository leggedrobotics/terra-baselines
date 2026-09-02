#!/usr/bin/env bash
# Build the aarch64 JAX image on Daint and import it as a SquashFS image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cscs_require_command rsync
cscs_require_command ssh
cscs_validate_token "image name" "$CSCS_IMAGE_NAME"
cscs_validate_token "image tag" "$CSCS_IMAGE_TAG"
cscs_resolve_root

BUILD_ID="build-$(date -u +%Y%m%dT%H%M%SZ)-${CSCS_IMAGE_TAG}"
REMOTE_CONTEXT="${CSCS_ROOT}/image-builds/${BUILD_ID}"
REMOTE_IMAGE_DIR="${CSCS_ROOT}/images"
REMOTE_SQSH="${REMOTE_IMAGE_DIR}/${CSCS_IMAGE_NAME}+${CSCS_IMAGE_TAG}.sqsh"

if ssh -T "${CSCS_SSH_TARGET}" "test -e $(printf '%q' "$REMOTE_SQSH")"; then
    cscs_die "image already exists: ${REMOTE_SQSH}; change CSCS_IMAGE_TAG to build a new immutable image"
fi

ssh -T "${CSCS_SSH_TARGET}" "mkdir -p $(printf '%q' "$REMOTE_CONTEXT") $(printf '%q' "$REMOTE_IMAGE_DIR")"
rsync -az "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}/requirements.txt" \
    "${CSCS_SSH_TARGET}:${REMOTE_CONTEXT}/"

echo "Building ${CSCS_IMAGE_NAME}:${CSCS_IMAGE_TAG} on ${CSCS_SSH_TARGET}..."
ssh -T "${CSCS_SSH_TARGET}" \
    env REMOTE_CONTEXT="$REMOTE_CONTEXT" REMOTE_SQSH="$REMOTE_SQSH" \
        CSCS_IMAGE_NAME="$CSCS_IMAGE_NAME" CSCS_IMAGE_TAG="$CSCS_IMAGE_TAG" \
        CSCS_BASE_IMAGE="$CSCS_BASE_IMAGE" \
    bash -s <<'REMOTE'
set -euo pipefail

export CONTAINERS_STORAGE_CONF="${HOME}/.config/containers/storage.conf"
mkdir -p "$(dirname "$CONTAINERS_STORAGE_CONF")" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/root"
if [[ ! -f "$CONTAINERS_STORAGE_CONF" ]]; then
    cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "overlay"
runroot = "/dev/shm/${USER}/runroot"
graphroot = "/dev/shm/${USER}/root"
EOF
fi

podman build --format docker \
    --build-arg "BASE_IMAGE=${CSCS_BASE_IMAGE}" \
    -f "${REMOTE_CONTEXT}/Dockerfile" \
    -t "${CSCS_IMAGE_NAME}:${CSCS_IMAGE_TAG}" \
    "$REMOTE_CONTEXT"

PARTIAL_IMAGE="${REMOTE_SQSH}.partial"
if [[ -e "$PARTIAL_IMAGE" ]]; then
    echo "Error: partial image already exists: $PARTIAL_IMAGE" >&2
    exit 1
fi
if ! enroot import -x mount -o "$PARTIAL_IMAGE" "podman://${CSCS_IMAGE_NAME}:${CSCS_IMAGE_TAG}"; then
    # Enroot can return non-zero when a temporary mount cleanup fails even
    # after it has finished writing a valid image. Accept only a verified
    # SquashFS artifact in that case.
    if ! unsquashfs -s "$PARTIAL_IMAGE" >/dev/null; then
        echo "Error: enroot import failed and did not produce a valid image" >&2
        exit 1
    fi
    echo "Warning: enroot reported a cleanup error; the SquashFS image is valid" >&2
fi
unsquashfs -s "$PARTIAL_IMAGE" >/dev/null
mv "$PARTIAL_IMAGE" "$REMOTE_SQSH"
printf 'Built %s\n' "$REMOTE_SQSH"
REMOTE
