#!/usr/bin/env bash
# Preflight and optionally submit one Slurm job per policy in a small TSV matrix.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: submit.sh POLICIES.tsv

Required environment:
  BASELINES_ROOT BASELINES_REVISION TERRA_ROOT TERRA_REVISION
  EPISODE_BANK_ROOT EPISODE_BANK_JSON_SHA256 EPISODE_BANK_FILES_SHA256
  PYTHON_BIN OUTPUT_BASE

Optional:
  SUBMIT=0  exact preflight and command preview (default)
  SUBMIT=1  submit after the same preflight
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi
MATRIX="$1"
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|1) ;;
    *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolved relative to this installed launcher.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
: "${OUTPUT_BASE:?missing OUTPUT_BASE}"
[[ "$OUTPUT_BASE" = /* ]] || legacy_easy_fail "OUTPUT_BASE must be absolute"
[[ "$OUTPUT_BASE" =~ ^[A-Za-z0-9_./:+-]+$ ]] ||
    legacy_easy_fail "OUTPUT_BASE contains unsupported characters"
legacy_easy_require_canonical_new_path "$OUTPUT_BASE" OUTPUT_BASE
legacy_easy_require_canonical_path "$MATRIX" file policy_matrix
legacy_easy_validate_static_inputs

HEADER="$(head -n 1 "$MATRIX")"
EXPECTED_HEADER=$'policy_label\tcheckpoint_path\tcheckpoint_sha256'
test "$HEADER" = "$EXPECTED_HEADER" ||
    legacy_easy_fail "policy matrix must use the exact three-column TSV header"
test "$(wc -l <"$MATRIX")" -ge 2 || legacy_easy_fail "policy matrix is empty"

declare -A SEEN_LABELS=()
declare -a LABELS=()
declare -a CHECKPOINTS=()
declare -a CHECKPOINT_SHAS=()
line_number=1
while IFS=$'\t' read -r label checkpoint checkpoint_sha extra || [ -n "$label" ]; do
    line_number=$((line_number + 1))
    test -n "$label" || legacy_easy_fail "blank row at line $line_number"
    test -z "${extra:-}" ||
        legacy_easy_fail "too many columns at line $line_number"
    legacy_easy_validate_policy_inputs "$label" "$checkpoint" "$checkpoint_sha"
    test -z "${SEEN_LABELS[$label]:-}" ||
        legacy_easy_fail "duplicate policy label: $label"
    SEEN_LABELS[$label]=1
    test ! -e "$OUTPUT_BASE/$label" ||
        legacy_easy_fail "output already exists: $OUTPUT_BASE/$label"
    LABELS+=("$label")
    CHECKPOINTS+=("$checkpoint")
    CHECKPOINT_SHAS+=("$checkpoint_sha")
done < <(tail -n +2 "$MATRIX")
test "${#LABELS[@]}" -gt 0 || legacy_easy_fail "policy matrix is empty"

if [ "$SUBMIT" = 1 ]; then
    mkdir -p "$OUTPUT_BASE"
fi
printf 'index\tjob_name\tpolicy_label\tjob_id\toutput_root\n'
for index in "${!LABELS[@]}"; do
    label="${LABELS[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    checkpoint_sha="${CHECKPOINT_SHAS[$index]}"
    job_name="tle-${index}-${label}"
    output_root="$OUTPUT_BASE/$label"
    exports="ALL,LEGACY_EASY_LAUNCHER_DIR=$SCRIPT_DIR,POLICY_LABEL=$label,CHECKPOINT_PATH=$checkpoint,CHECKPOINT_SHA256=$checkpoint_sha,BASELINES_ROOT=$BASELINES_ROOT,BASELINES_REVISION=$BASELINES_REVISION,TERRA_ROOT=$TERRA_ROOT,TERRA_REVISION=$TERRA_REVISION,EPISODE_BANK_ROOT=$EPISODE_BANK_ROOT,EPISODE_BANK_JSON_SHA256=$EPISODE_BANK_JSON_SHA256,EPISODE_BANK_FILES_SHA256=$EPISODE_BANK_FILES_SHA256,PYTHON_BIN=$PYTHON_BIN,OUTPUT_ROOT=$output_root"
    command=(
        sbatch --parsable
        --job-name="$job_name"
        --chdir="$OUTPUT_BASE"
        --output="$OUTPUT_BASE/slurm-%x-%j.out"
        --export="$exports"
        "$SCRIPT_DIR/run.sbatch"
    )
    if [ "$SUBMIT" = 1 ]; then
        job_id="$("${command[@]}")"
    else
        job_id=DRY_RUN
        printf '# '
        printf '%q ' "${command[@]}"
        printf '\n'
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$index" "$job_name" "$label" "$job_id" "$output_root"
done
