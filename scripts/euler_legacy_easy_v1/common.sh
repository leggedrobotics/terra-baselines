#!/usr/bin/env bash

legacy_easy_fail() {
    echo "legacy-easy preflight: $*" >&2
    return 1
}

legacy_easy_require_sha256() {
    local value="$1"
    local name="$2"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] ||
        legacy_easy_fail "$name must be one lowercase SHA-256"
}

legacy_easy_require_canonical_path() {
    local value="$1"
    local kind="$2"
    local name="$3"
    local resolved

    case "$kind" in
        file) test -f "$value" || legacy_easy_fail "$name is not a file: $value" ;;
        dir) test -d "$value" || legacy_easy_fail "$name is not a directory: $value" ;;
        *) legacy_easy_fail "internal error: unsupported path kind $kind" ;;
    esac
    resolved="$(realpath -e -- "$value")"
    test "$value" = "$resolved" ||
        legacy_easy_fail "$name must be canonical: use $resolved"
    [[ "$value" =~ ^[A-Za-z0-9_./:+-]+$ ]] ||
        legacy_easy_fail "$name contains unsupported characters: $value"
}

legacy_easy_require_executable() {
    local value="$1"
    local name="$2"

    test -x "$value" || legacy_easy_fail "$name is not executable: $value"
    [[ "$value" = /* ]] || legacy_easy_fail "$name must be absolute"
    [[ "$value" =~ ^[A-Za-z0-9_./:+-]+$ ]] ||
        legacy_easy_fail "$name contains unsupported characters: $value"
}

legacy_easy_require_canonical_new_path() {
    local value="$1"
    local name="$2"
    local normalized

    [[ "$value" = /* ]] || legacy_easy_fail "$name must be absolute"
    [[ "$value" =~ ^[A-Za-z0-9_./:+-]+$ ]] ||
        legacy_easy_fail "$name contains unsupported characters: $value"
    normalized="$(realpath -m -- "$value")"
    test "$value" = "$normalized" ||
        legacy_easy_fail "$name must be canonical: use $normalized"
}

legacy_easy_validate_git_root() {
    local root="$1"
    local expected_revision="$2"
    local name="$3"
    local actual_revision

    legacy_easy_require_canonical_path "$root" dir "$name"
    [[ "$expected_revision" =~ ^[0-9a-f]{40}$ ]] ||
        legacy_easy_fail "$name revision must be one full lowercase Git SHA"
    actual_revision="$(GIT_OPTIONAL_LOCKS=0 git -C "$root" rev-parse --verify HEAD)"
    test "$actual_revision" = "$expected_revision" ||
        legacy_easy_fail \
            "$name revision mismatch: expected $expected_revision, got $actual_revision"
    test -z "$(GIT_OPTIONAL_LOCKS=0 git -C "$root" status --porcelain --untracked-files=all)" ||
        legacy_easy_fail "$name must be a clean immutable checkout: $root"
}

legacy_easy_validate_static_inputs() {
    : "${BASELINES_ROOT:?missing BASELINES_ROOT}"
    : "${BASELINES_REVISION:?missing BASELINES_REVISION}"
    : "${TERRA_ROOT:?missing TERRA_ROOT}"
    : "${TERRA_REVISION:?missing TERRA_REVISION}"
    : "${EPISODE_BANK_ROOT:?missing EPISODE_BANK_ROOT}"
    : "${EPISODE_BANK_JSON_SHA256:?missing EPISODE_BANK_JSON_SHA256}"
    : "${EPISODE_BANK_FILES_SHA256:?missing EPISODE_BANK_FILES_SHA256}"
    : "${PYTHON_BIN:?missing PYTHON_BIN}"

    legacy_easy_validate_git_root \
        "$BASELINES_ROOT" "$BASELINES_REVISION" BASELINES_ROOT
    legacy_easy_validate_git_root "$TERRA_ROOT" "$TERRA_REVISION" TERRA_ROOT
    legacy_easy_require_canonical_path \
        "$EPISODE_BANK_ROOT" dir EPISODE_BANK_ROOT
    legacy_easy_require_executable "$PYTHON_BIN" PYTHON_BIN
    legacy_easy_require_sha256 \
        "$EPISODE_BANK_JSON_SHA256" EPISODE_BANK_JSON_SHA256
    legacy_easy_require_sha256 \
        "$EPISODE_BANK_FILES_SHA256" EPISODE_BANK_FILES_SHA256

    test "$(sha256sum "$EPISODE_BANK_ROOT/episode_bank.json" | awk '{print $1}')" = \
        "$EPISODE_BANK_JSON_SHA256" ||
        legacy_easy_fail "episode_bank.json SHA-256 mismatch"
    test "$(sha256sum "$EPISODE_BANK_ROOT/files.sha256" | awk '{print $1}')" = \
        "$EPISODE_BANK_FILES_SHA256" ||
        legacy_easy_fail "episode-bank files.sha256 receipt mismatch"
    (
        cd "$EPISODE_BANK_ROOT"
        sha256sum --quiet -c files.sha256
    ) || legacy_easy_fail "episode-bank payload does not match files.sha256"

    "$PYTHON_BIN" - "$EPISODE_BANK_ROOT/episode_bank.json" "$TERRA_REVISION" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
terra_revision = sys.argv[2]
bank = json.loads(path.read_text())
assert bank["schema"] == "terra_legacy_easy_explicit_episode_bank_v1"
assert bank["diagnostic_only"] is True
assert bank["included_in_constrained_macro"] is False
assert bank["terra_revision"] == terra_revision
assert bank["max_steps_in_episode"] == 450
assert bank["foundation_border_alignment"] is False
assert bank["policy_modes"] == {
    "primary": "deterministic",
    "secondary": "sampled",
}
protocol = json.loads((path.parent / bank["environment_protocol"]).read_text())
assert protocol["accepted_dump_contract"] == "exact_visible_dump_v1"
assert protocol["episode"]["max_steps_in_episode"] == 450
assert protocol["episode"]["rewards_type"] == "DENSE"
assert protocol["episode"]["apply_trench_rewards"] is False
panels = bank["evaluation_panels"]
assert panels["promotion"]["slot_count"] == 48
assert panels["development"]["slot_count"] == 48
PY
}

legacy_easy_validate_policy_inputs() {
    local policy_label="$1"
    local checkpoint_path="$2"
    local checkpoint_sha256="$3"

    [[ "$policy_label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] ||
        legacy_easy_fail \
            "POLICY_LABEL must be 1-48 safe filename characters: $policy_label"
    legacy_easy_require_canonical_path "$checkpoint_path" file CHECKPOINT_PATH
    legacy_easy_require_sha256 "$checkpoint_sha256" CHECKPOINT_SHA256
    test "$(sha256sum "$checkpoint_path" | awk '{print $1}')" = \
        "$checkpoint_sha256" || legacy_easy_fail "checkpoint SHA-256 mismatch"
}
