#!/usr/bin/env bash
# shellcheck disable=SC2029  # Intentional client expansion of validated paths.
# Build one immutable accepted-bank campaign and optionally submit one phase.
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1

usage() {
    echo "usage: prepare_submit.sh PHASE TERRA_REPO BANK_ROOT [SEED]" >&2
    echo "  PHASE: smoke | screen" >&2
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
    exit 2
fi
PHASE="$1"
SEED="${4:-20260730}"
case "$PHASE" in
    smoke|screen) ;;
    promote)
        echo "P6 promotion is fail-closed: the separate 256-train-maps/condition bank is not implemented" >&2
        exit 9
        ;;
    *) usage; exit 2 ;;
esac
TERRA_REPO="$(realpath "$2")"
BANK_ROOT="$(realpath "$3")"
[[ "$SEED" =~ ^[0-9]+$ ]] || {
    echo "SEED must be a nonnegative integer" >&2
    exit 2
}

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|1) ;;
    *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;;
esac
ALLOW_NON_ADMISSION_FOR_TESTS="${ALLOW_NON_ADMISSION_FOR_TESTS:-0}"
case "$ALLOW_NON_ADMISSION_FOR_TESTS" in
    0|1) ;;
    *) echo "ALLOW_NON_ADMISSION_FOR_TESTS must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$ALLOW_NON_ADMISSION_FOR_TESTS" = 1 ] && [ "$SUBMIT" != 0 ]; then
    echo "test-only/non-admission banks can never be submitted" >&2
    exit 3
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_CHECK="${RUNTIME_CHECK:-/home/lorenzo/git/codex_skills/skills/terra-rl/scripts/check_jax_runtime.py}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/home/lorenzo/moleworks/.artifacts/euler_accepted_bank_v1}"
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK_BASE="/cluster/work/rsl/lterenzi/terra_curriculum_campaigns"
REMOTE_RUN_BASE="/cluster/scratch/lterenzi/codex_terra_edge_runs/accepted_bank_v1"
VENV="/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426"
VENV_LEDGER="$VENV/provenance/artifact-hashes.sha256"
VENV_LEDGER_SHA="853871aef55efe34a64474660109673c0d48b9a34cba333e368600a11b126d5c"
EXCLUDED_NODES="eu-g6-064"
IDENTITY_CONTRACT="terra_reset_arrays_sha256_v1"
EXPECTED_TRAIN_MAPS_PER_CONDITION=64
ARMS=(
    F-ANCHOR
    F-SPECIALIST
    T-ANCHOR
    T-SPECIALIST
    G-UNIFORM
    G-ADAPTIVE
)
ARMS_CSV="$(IFS=,; printf '%s' "${ARMS[*]}")"

for repository in "$REPO" "$TERRA_REPO"; do
    git -C "$repository" rev-parse --is-inside-work-tree >/dev/null
    if [ -n "$(git -C "$repository" status --porcelain)" ]; then
        echo "source repository must be clean: $repository" >&2
        exit 4
    fi
done
for required in \
    "$BANK_ROOT/dataset.json" \
    "$BANK_ROOT/environment_protocol.json" \
    "$BANK_ROOT/review_admission.json" \
    "$BANK_ROOT/source_registry.jsonl" \
    "$RUNTIME_CHECK"; do
    test -f "$required" || {
        echo "missing campaign input: $required" >&2
        exit 4
    }
done

ADMISSION="frozen"
for marker in NON_ADMISSION.md REVIEW_ONLY.md; do
    if [ -e "$BANK_ROOT/$marker" ]; then
        if [ "$ALLOW_NON_ADMISSION_FOR_TESTS" != 1 ]; then
            echo "refusing non-admission bank marker: $BANK_ROOT/$marker" >&2
            exit 5
        fi
        ADMISSION="test_only"
    fi
done

TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
read_json_field() {
    python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$1" "$2"
}
mkdir -p "$ARCHIVE_DIR"
STAGE="$(mktemp -d "$ARCHIVE_DIR/.prepare.XXXXXX")"
cleanup() {
    case "$STAGE" in
        "$ARCHIVE_DIR"/.prepare.*) rm -rf -- "$STAGE" ;;
        *) echo "refusing to clean unexpected temporary path: $STAGE" >&2 ;;
    esac
}
trap cleanup EXIT
mkdir -p \
    "$STAGE/campaign/source/terra" \
    "$STAGE/campaign/source/terra-baselines" \
    "$STAGE/campaign/bank" \
    "$STAGE/campaign/runtime"

git -C "$TERRA_REPO" archive --format=tar "$TERRA_REVISION" \
    | tar -xf - -C "$STAGE/campaign/source/terra"
git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
    | tar -xf - -C "$STAGE/campaign/source/terra-baselines"
tar -cf - -C "$BANK_ROOT" . \
    | tar -xf - -C "$STAGE/campaign/bank"
install -m 0644 \
    "$RUNTIME_CHECK" \
    "$STAGE/campaign/runtime/check_jax_runtime.py"

BANK_STAGE="$STAGE/campaign/bank"
BANK_SCHEMA="$(read_json_field "$BANK_STAGE/dataset.json" schema)"
BANK_IDENTITY_CONTRACT="$(
    read_json_field "$BANK_STAGE/dataset.json" scenario_identity_contract
)"
BANK_TERRA_REVISION="$(
    read_json_field "$BANK_STAGE/environment_protocol.json" terra_revision
)"
test "$BANK_SCHEMA" = "terra_curriculum_loader_bank_v1" || {
    echo "staged bank is not a loader-ready accepted bank" >&2
    exit 6
}
test "$BANK_IDENTITY_CONTRACT" = "$IDENTITY_CONTRACT" || {
    echo "staged bank lacks the frozen reset-array scenario identity" >&2
    exit 6
}
test "$BANK_TERRA_REVISION" = "$TERRA_REVISION" || {
    echo "staged bank Terra revision does not match the archived source commit" >&2
    exit 6
}
for marker in NON_ADMISSION.md REVIEW_ONLY.md; do
    if [ -e "$BANK_STAGE/$marker" ]; then
        if [ "$ALLOW_NON_ADMISSION_FOR_TESTS" != 1 ]; then
            echo "refusing staged non-admission bank marker: $BANK_STAGE/$marker" >&2
            exit 5
        fi
        ADMISSION="test_only"
    fi
done

BANK_MAPS_PER_CONDITION="$(
    PYTHONPATH="$STAGE/campaign/source/terra-baselines" \
        python3 \
        "$STAGE/campaign/source/terra-baselines/scripts/euler_accepted_bank_v1/validate_training_bank.py" \
        "$BANK_STAGE"
)"
test "$BANK_MAPS_PER_CONDITION" = "$EXPECTED_TRAIN_MAPS_PER_CONDITION" || {
    echo "smoke/screen banks require exactly $EXPECTED_TRAIN_MAPS_PER_CONDITION train maps per condition; staged bank has $BANK_MAPS_PER_CONDITION" >&2
    exit 6
}

BANK_TREE_SHA="$(
    tar --sort=name --mtime='UTC 1970-01-01' \
        --owner=0 --group=0 --numeric-owner \
        -cf - -C "$BANK_STAGE" . \
        | sha256sum | awk '{print $1}'
)"
DATASET_SHA="$(sha256sum "$BANK_STAGE/dataset.json" | awk '{print $1}')"
REVIEW_ADMISSION_SHA="$(
    sha256sum "$BANK_STAGE/review_admission.json" | awk '{print $1}'
)"
PROTOCOL_SHA="$(
    sha256sum "$BANK_STAGE/environment_protocol.json" | awk '{print $1}'
)"
REGISTRY_SHA="$(
    sha256sum "$BANK_STAGE/source_registry.jsonl" | awk '{print $1}'
)"
RUNTIME_SHA="$(sha256sum "$RUNTIME_CHECK" | awk '{print $1}')"
python3 -c '
import json, pathlib, sys
(
    output, terra_revision, baselines_revision, bank_tree_sha,
    dataset_sha, review_admission_sha, protocol_sha, registry_sha, runtime_sha,
    venv, ledger, ledger_sha, identity_contract, admission,
    train_maps_per_condition, arms_csv, excluded_nodes
) = sys.argv[1:]
payload = {
    "schema": "terra_accepted_bank_euler_campaign_v2",
    "arms": arms_csv.split(","),
    "admission": admission,
    "terra_revision": terra_revision,
    "terra_baselines_revision": baselines_revision,
    "bank_tree_sha256": bank_tree_sha,
    "bank_dataset_sha256": dataset_sha,
    "bank_review_admission_sha256": review_admission_sha,
    "bank_environment_protocol_file_sha256": protocol_sha,
    "bank_source_registry_file_sha256": registry_sha,
    "scenario_identity_contract": identity_contract,
    "train_maps_per_condition": int(train_maps_per_condition),
    "runtime_check_sha256": runtime_sha,
    "excluded_nodes": excluded_nodes.split(",") if excluded_nodes else [],
    "venv": venv,
    "venv_ledger": ledger,
    "venv_ledger_sha256": ledger_sha,
    "frozen_shape": {
        "num_devices": 4,
        "num_envs_per_device": 1024,
        "num_steps": 32,
    },
    "phases": {
        "smoke": {"partition": "gpuhe.4h", "updates": 1},
        "screen": {"partition": "gpuhe.24h", "updates": 2000},
    },
    "future_p6": {
        "implemented": False,
        "partition": "gpuhe.120h",
        "updates": 20000,
        "train_maps_per_condition": 256,
    },
}
pathlib.Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
' \
    "$STAGE/campaign/manifest.json" \
    "$TERRA_REVISION" \
    "$BASELINES_REVISION" \
    "$BANK_TREE_SHA" \
    "$DATASET_SHA" \
    "$REVIEW_ADMISSION_SHA" \
    "$PROTOCOL_SHA" \
    "$REGISTRY_SHA" \
    "$RUNTIME_SHA" \
    "$VENV" \
    "$VENV_LEDGER" \
    "$VENV_LEDGER_SHA" \
    "$IDENTITY_CONTRACT" \
    "$ADMISSION" \
    "$BANK_MAPS_PER_CONDITION" \
    "$ARMS_CSV" \
    "$EXCLUDED_NODES"

ARCHIVE_TMP="$STAGE/campaign.tar.zst"
tar --sort=name --mtime='UTC 1970-01-01' \
    --owner=0 --group=0 --numeric-owner \
    --zstd -cf "$ARCHIVE_TMP" -C "$STAGE" campaign
CAMPAIGN_SHA="$(sha256sum "$ARCHIVE_TMP" | awk '{print $1}')"
ARCHIVE="$ARCHIVE_DIR/campaign-$CAMPAIGN_SHA.tar.zst"
if [ -e "$ARCHIVE" ]; then
    cmp "$ARCHIVE_TMP" "$ARCHIVE" || {
        echo "content-address collision at $ARCHIVE" >&2
        exit 7
    }
else
    mv "$ARCHIVE_TMP" "$ARCHIVE"
fi
printf '%s  %s\n' "$CAMPAIGN_SHA" "$(basename "$ARCHIVE")" \
    > "$ARCHIVE.sha256"

REMOTE_DIR="$REMOTE_WORK_BASE/sha256-$CAMPAIGN_SHA"
REMOTE_ARCHIVE="$REMOTE_DIR/campaign.tar.zst"
echo "campaign_sha256=$CAMPAIGN_SHA"
echo "local_archive=$ARCHIVE"
echo "remote_archive=$REMOTE_ARCHIVE"
echo "phase=$PHASE seed=$SEED admission=$ADMISSION"

if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local archive prepared; no ssh, upload, scratch, W&B, or sbatch mutation"
    printf 'future upload: scp %q %q\n' \
        "$ARCHIVE" "$REMOTE_HOST:$REMOTE_ARCHIVE"
    printf 'future phase: ssh %q sbatch --partition=%q --export=<immutable-contract> <archived-run.sbatch>\n' \
        "$REMOTE_HOST" \
        "$(
            case "$PHASE" in
                smoke) echo gpuhe.4h ;;
                screen) echo gpuhe.24h ;;
            esac
        )"
    exit 0
fi

test "$ADMISSION" = frozen
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_ARCHIVE'"; then
    REMOTE_PARTIAL="$REMOTE_ARCHIVE.partial.$$"
    scp -q "$ARCHIVE" "$REMOTE_HOST:$REMOTE_PARTIAL"
    ssh "$REMOTE_HOST" "
        set -e
        test \"\$(sha256sum '$REMOTE_PARTIAL' | awk '{print \$1}')\" = '$CAMPAIGN_SHA'
        mv '$REMOTE_PARTIAL' '$REMOTE_ARCHIVE'
    "
fi
ssh "$REMOTE_HOST" "
    set -e
    test \"\$(sha256sum '$REMOTE_ARCHIVE' | awk '{print \$1}')\" = '$CAMPAIGN_SHA'
    test \$(tar --zstd -xOf '$REMOTE_ARCHIVE' campaign/manifest.json |
        python3 -c 'import json,sys; print(json.load(sys.stdin)[\"admission\"])') = frozen
    mkdir -p '$REMOTE_RUN_BASE'
"

if [ "$PHASE" = screen ]; then
    for ARM in "${ARMS[@]}"; do
        SMOKE="$REMOTE_RUN_BASE/$CAMPAIGN_SHA/smoke/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "
            set -e
            grep -qx 'schema=terra_accepted_bank_euler_receipt_v2' '$SMOKE/receipt.env'
            grep -qx 'status=PASSED' '$SMOKE/receipt.env'
            grep -qx 'campaign_sha256=$CAMPAIGN_SHA' '$SMOKE/receipt.env'
            grep -qx 'campaign_arms=$ARMS_CSV' '$SMOKE/receipt.env'
            grep -qx 'phase=smoke' '$SMOKE/receipt.env'
            grep -qx 'arm=$ARM' '$SMOKE/receipt.env'
            grep -qx 'seed=$SEED' '$SMOKE/receipt.env'
            python3 -c 'import json; p=json.load(open(\"$SMOKE/smoke_validation.json\")); assert p[\"passed\"] is True and p[\"arm\"] == \"$ARM\"'
        "
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=08:00:00 ;;
esac
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUN_BASE/$CAMPAIGN_SHA/$PHASE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,CAMPAIGN_ARCHIVE=$REMOTE_ARCHIVE,CAMPAIGN_SHA=$CAMPAIGN_SHA,RUN_BASE=$REMOTE_RUN_BASE,VENV=$VENV,VENV_LEDGER=$VENV_LEDGER,VENV_LEDGER_SHA=$VENV_LEDGER_SHA,SEED=$SEED"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "
            tar --zstd -xOf '$REMOTE_ARCHIVE' \
                campaign/source/terra-baselines/scripts/euler_accepted_bank_v1/run.sbatch |
                sbatch --parsable \
                    --partition='$PARTITION' \
                    --time='$WALLTIME' \
                    --exclude='$EXCLUDED_NODES' \
                    --job-name='terra-ab-${PHASE}-${ARM}' \
                    --output='$RUN_DIR/slurm_%j.out' \
                    --export='$EXPORTS'
        "
    )"
    echo "$PHASE $ARM $JOB_ID"
done
