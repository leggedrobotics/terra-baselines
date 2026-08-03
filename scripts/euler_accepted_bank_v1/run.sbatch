#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-accepted-bank
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/accepted_bank_v1/slurm-%x-%j.out

set -euo pipefail
umask 002

: "${PHASE:?missing PHASE=smoke|screen|promote}"
: "${ARM:?missing ARM}"
: "${CAMPAIGN_ARCHIVE:?missing immutable campaign archive}"
: "${CAMPAIGN_SHA:?missing campaign archive SHA-256}"
: "${RUN_BASE:?missing scratch run base}"
: "${VENV:?missing dependency-only venv}"
: "${VENV_LEDGER:?missing venv artifact ledger}"
: "${VENV_LEDGER_SHA:?missing venv ledger SHA-256}"
: "${SEED:?missing paired training seed}"
: "${SLURM_JOB_ID:?this script must run inside Slurm}"
: "${TMPDIR:?Euler allocation did not provide TMPDIR}"

case "$ARM" in
    F-ANCHOR|F-SPECIALIST|T-ANCHOR|T-SPECIALIST|G-UNIFORM|G-ADAPTIVE) ;;
    *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac
case "$PHASE" in
    smoke)
        EXPECTED_PARTITION=gpuhe.4h
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    screen)
        EXPECTED_PARTITION=gpuhe.24h
        UPDATES=2000
        CHECKPOINT_INTERVAL=500
        LOG_TRAIN_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    promote)
        echo "P6 promotion is fail-closed: the separate 256-train-maps/condition bank is not implemented" >&2
        exit 9
        ;;
    *) echo "unsupported PHASE=$PHASE" >&2; exit 2 ;;
esac

find_single_checkpoint() {
    local directory="$1"
    local pattern="$2"
    local -a matches
    mapfile -t matches < <(
        find "$directory" -maxdepth 1 -type f -name "$pattern" -print | sort
    )
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "expected one checkpoint matching $pattern, found ${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

test "${SLURM_JOB_PARTITION:-}" = "$EXPECTED_PARTITION" || {
    echo "phase $PHASE requires $EXPECTED_PARTITION, got ${SLURM_JOB_PARTITION:-unset}" >&2
    exit 40
}
test "${SLURM_CPUS_PER_TASK:-}" = "4" || {
    echo "accepted-bank jobs require four allocated CPUs" >&2
    exit 41
}
HOME_USED_GB="$(
    lquota |
        awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {
            value=$4
            gsub(/[^0-9.]/, "", value)
            print value
        }'
)"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 42
}

ACTUAL_ARCHIVE_SHA="$(sha256sum "$CAMPAIGN_ARCHIVE" | awk '{print $1}')"
test "$ACTUAL_ARCHIVE_SHA" = "$CAMPAIGN_SHA" || {
    echo "campaign archive SHA mismatch" >&2
    exit 43
}
mkdir "$TMPDIR/campaign"
tar --zstd -xf "$CAMPAIGN_ARCHIVE" -C "$TMPDIR"
SOURCE="$TMPDIR/campaign/source"
BANK="$TMPDIR/campaign/bank"
BASELINES="$SOURCE/terra-baselines"
TERRA="$SOURCE/terra"
MANIFEST="$TMPDIR/campaign/manifest.json"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
GATE_TOOL="$BASELINES/scripts/euler_accepted_bank_v1/select_promotion.py"
test -f "$MANIFEST"
test -f "$BASELINES/train_mixed.py"
test -f "$TERRA/terra/benchmark_state.py"
test -f "$BANK/dataset.json"
test -f "$BANK/review_admission.json"
test -f "$RUNTIME_CHECK"
test -f "$GATE_TOOL"

manifest_value() {
    python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$MANIFEST" "$1"
}
TERRA_REVISION="$(manifest_value terra_revision)"
BASELINES_REVISION="$(manifest_value terra_baselines_revision)"
MANIFEST_SCHEMA="$(manifest_value schema)"
MANIFEST_ADMISSION="$(manifest_value admission)"
MANIFEST_ARMS="$(
    python3 -c \
        'import json,sys; print(",".join(json.load(open(sys.argv[1]))["arms"]))' \
        "$MANIFEST"
)"
MANIFEST_IDENTITY_CONTRACT="$(manifest_value scenario_identity_contract)"
MANIFEST_TRAIN_MAPS_PER_CONDITION="$(
    manifest_value train_maps_per_condition
)"
MANIFEST_REVIEW_ADMISSION_SHA="$(
    manifest_value bank_review_admission_sha256
)"
MANIFEST_VENV_LEDGER="$(manifest_value venv_ledger)"
MANIFEST_VENV_LEDGER_SHA="$(manifest_value venv_ledger_sha256)"
EXPECTED_ARMS="F-ANCHOR,F-SPECIALIST,T-ANCHOR,T-SPECIALIST,G-UNIFORM,G-ADAPTIVE"
test "$MANIFEST_SCHEMA" = terra_accepted_bank_euler_campaign_v2 || {
    echo "unsupported campaign schema: $MANIFEST_SCHEMA" >&2
    exit 49
}
test "$MANIFEST_ARMS" = "$EXPECTED_ARMS" || {
    echo "campaign arm matrix mismatch: $MANIFEST_ARMS" >&2
    exit 48
}
test "$MANIFEST_ADMISSION" = frozen
test "$MANIFEST_IDENTITY_CONTRACT" = terra_reset_arrays_sha256_v1
test "$MANIFEST_TRAIN_MAPS_PER_CONDITION" = 64 || {
    echo "smoke/screen campaign requires exactly 64 train maps per condition" >&2
    exit 46
}
test "$(sha256sum "$BANK/review_admission.json" | awk '{print $1}')" = \
    "$MANIFEST_REVIEW_ADMISSION_SHA" || {
    echo "review admission SHA mismatch" >&2
    exit 47
}
test ! -e "$BANK/NON_ADMISSION.md"
test ! -e "$BANK/REVIEW_ONLY.md"
test "$MANIFEST_VENV_LEDGER" = "$VENV_LEDGER"
test "$MANIFEST_VENV_LEDGER_SHA" = "$VENV_LEDGER_SHA"
test "$(sha256sum "$VENV_LEDGER" | awk '{print $1}')" = "$VENV_LEDGER_SHA"
(cd "$(dirname "$VENV_LEDGER")" && sha256sum -c "$(basename "$VENV_LEDGER")")

module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" != smoke ]; then
    module load eth_proxy
    test -r "$HOME/.netrc"
    grep -Eq \
        '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' \
        "$HOME/.netrc" || {
        echo "non-smoke jobs require api.wandb.ai credentials in ~/.netrc" >&2
        exit 45
    }
fi

PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported allocated GPU: $GPU_NAME" >&2
        exit 44
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
"$PYTHON" -c \
    'import pathlib,sys,terra; from utils import accepted_bank; roots=[pathlib.Path(terra.__file__).resolve().parents[1], pathlib.Path(accepted_bank.__file__).resolve().parents[1]]; expected=[pathlib.Path(p).resolve() for p in sys.argv[1:]]; assert roots == expected, (roots, expected)' \
    "$TERRA" "$BASELINES"
"$PYTHON" -c \
    'import sys; from utils.accepted_bank import load_accepted_bank; bank=load_accepted_bank(sys.argv[1], sys.argv[2], sys.argv[3]); expected=int(sys.argv[4]); assert bank.map_count_per_condition == expected, (bank.map_count_per_condition, expected); print(len(bank.levels), bank.map_count_per_condition)' \
    "$BANK" "$ARM" "$TERRA_REVISION" "$MANIFEST_TRAIN_MAPS_PER_CONDITION"

NUM_DEVICES=4
NUM_ENVS_PER_DEVICE=1024
NUM_STEPS=32
test "$NUM_DEVICES" -eq 4
test "$NUM_ENVS_PER_DEVICE" -eq 1024
test "$NUM_STEPS" -eq 32
GLOBAL_TRANSITIONS=$((NUM_DEVICES * NUM_ENVS_PER_DEVICE * NUM_STEPS * UPDATES))
ARM_TOKEN="${ARM,,}"
ARM_TOKEN="${ARM_TOKEN//-/_}"
RUN_NAME="accepted_${CAMPAIGN_SHA:0:12}_${PHASE}_${ARM_TOKEN}_s${SEED}"
RUN_DIR="$RUN_BASE/$CAMPAIGN_SHA/$PHASE/s$SEED/$ARM"
test -d "$RUN_DIR"
test ! -e "$RUN_DIR/receipt.env"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/parity"
if [ "$PHASE" = screen ]; then
    SMOKE_DIR="$RUN_BASE/$CAMPAIGN_SHA/smoke/s$SEED/$ARM"
    test -f "$SMOKE_DIR/receipt.env"
    grep -qx "schema=terra_accepted_bank_euler_receipt_v2" "$SMOKE_DIR/receipt.env"
    grep -qx "status=PASSED" "$SMOKE_DIR/receipt.env"
    grep -qx "campaign_sha256=$CAMPAIGN_SHA" "$SMOKE_DIR/receipt.env"
    grep -qx "campaign_arms=$MANIFEST_ARMS" "$SMOKE_DIR/receipt.env"
    grep -qx "phase=smoke" "$SMOKE_DIR/receipt.env"
    grep -qx "arm=$ARM" "$SMOKE_DIR/receipt.env"
    grep -qx "seed=$SEED" "$SMOKE_DIR/receipt.env"
    test -f "$SMOKE_DIR/smoke_validation.json"
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True and p["arm"] == sys.argv[2]' \
        "$SMOKE_DIR/smoke_validation.json" "$ARM"
fi
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="abv1_${CAMPAIGN_SHA:0:10}_${PHASE}_${ARM_TOKEN}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled
    export WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online
    export WANDB_DIR="$RUN_DIR/wandb"
fi

scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
cp "$MANIFEST" "$RUN_DIR/campaign_manifest.json"
printf '%s  %s\n' "$CAMPAIGN_SHA" "$CAMPAIGN_ARCHIVE" > "$RUN_DIR/campaign_archive.sha256"
printf '%s\n' \
    "campaign_schema=$MANIFEST_SCHEMA" \
    "campaign_arms=$MANIFEST_ARMS" \
    "phase=$PHASE" \
    "arm=$ARM" \
    "seed=$SEED" \
    "updates=$UPDATES" \
    "num_devices=$NUM_DEVICES" \
    "num_envs_per_device=$NUM_ENVS_PER_DEVICE" \
    "num_steps=$NUM_STEPS" \
    "train_maps_per_condition=$MANIFEST_TRAIN_MAPS_PER_CONDITION" \
    "global_transitions=$GLOBAL_TRANSITIONS" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "venv=$VENV" \
    "venv_ledger=$VENV_LEDGER" \
    "venv_ledger_sha256=$VENV_LEDGER_SHA" \
    "wandb_mode=$WANDB_MODE" \
    "wandb_run_id=$WANDB_RUN_ID" \
    > "$RUN_DIR/run_contract.env"

if [ "$PHASE" = screen ]; then
    mapfile -t SMOKE_FINALS < <(
        find "$SMOKE_DIR/checkpoints" -maxdepth 1 -type f -name '*_FINAL.pkl'
    )
    test "${#SMOKE_FINALS[@]}" -eq 1
    PARITY_CHECKPOINT="${SMOKE_FINALS[0]}"
    for PANEL in promotion development; do
        CPU_HASHES="$RUN_DIR/parity/${PANEL}_cpu.json"
        GPU_HASHES="$RUN_DIR/parity/${PANEL}_gpu.json"
        PARITY="$RUN_DIR/parity/${PANEL}_parity.json"
        JAX_PLATFORMS=cpu "$PYTHON" "$GATE_TOOL" reset-hashes \
            --checkpoint "$PARITY_CHECKPOINT" \
            --bank-root "$BANK" \
            --panel "$PANEL" \
            --terra-revision "$TERRA_REVISION" \
            --output "$CPU_HASHES"
        JAX_PLATFORMS=cuda "$PYTHON" "$GATE_TOOL" reset-hashes \
            --checkpoint "$PARITY_CHECKPOINT" \
            --bank-root "$BANK" \
            --panel "$PANEL" \
            --terra-revision "$TERRA_REVISION" \
            --output "$GPU_HASHES"
        "$PYTHON" "$GATE_TOOL" compare-reset-hashes \
            --cpu "$CPU_HASHES" \
            --gpu "$GPU_HASHES" \
            --output "$PARITY"
    done
fi

TRAIN_COMMAND=(
    "$BASELINES/scripts/run_accepted_bank_screen.sh"
    "$ARM"
    "$BANK"
    "$RUN_NAME"
    "$UPDATES"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.sh"
printf '\n' >> "$RUN_DIR/train_command.sh"

export TERRA_ROOT="$TERRA"
export TERRA_REVISION
export RUN_ROOT="$RUN_DIR"
export PYTHON_BIN="$PYTHON"
export SEED
export MACHINE=euler
export NUM_DEVICES NUM_ENVS_PER_DEVICE NUM_STEPS
export FINITE_CHECK_INTERVAL=10
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL CACHE_CLEAR_INTERVAL
if [ "$PHASE" = smoke ]; then
    export FINITE_CHECK_INTERVAL=1
fi
"${TRAIN_COMMAND[@]}"

if [ "$PHASE" = smoke ]; then
    PERIODIC="$(
        find_single_checkpoint \
            "$RUN_DIR/checkpoints" \
            "${RUN_NAME}-euler-*_update_000001.pkl"
    )"
    FINAL="$(
        find_single_checkpoint \
            "$RUN_DIR/checkpoints" \
            "${RUN_NAME}-euler-*_FINAL.pkl"
    )"
    test "${PERIODIC%_update_000001.pkl}_FINAL.pkl" = "$FINAL"
    JAX_PLATFORMS=cpu "$PYTHON" "$GATE_TOOL" verify-smoke \
        --arm "$ARM" \
        --periodic "$PERIODIC" \
        --final "$FINAL" \
        --output "$RUN_DIR/smoke_validation.json"
else
    EVAL_UPDATES=(500 1000 2000)
    CHECKPOINT_ARGS=()
    CHECKPOINT_PREFIX=""
    for UPDATE in "${EVAL_UPDATES[@]}"; do
        UPDATE_TOKEN="$(printf '%06d' "$UPDATE")"
        CHECKPOINT="$(
            find_single_checkpoint \
                "$RUN_DIR/checkpoints" \
                "${RUN_NAME}-euler-*_update_${UPDATE_TOKEN}.pkl"
        )"
        CHECKPOINT_SUFFIX="_update_${UPDATE_TOKEN}.pkl"
        THIS_PREFIX="${CHECKPOINT%"$CHECKPOINT_SUFFIX"}"
        if [ -z "$CHECKPOINT_PREFIX" ]; then
            CHECKPOINT_PREFIX="$THIS_PREFIX"
        else
            test "$CHECKPOINT_PREFIX" = "$THIS_PREFIX"
        fi
        CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT")
    done
    for PANEL in promotion development; do
        JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES/eval_fixed_bank.py" \
            "${CHECKPOINT_ARGS[@]}" \
            --bank-root "$BANK" \
            --accepted-panel "$PANEL" \
            --terra-revision "$TERRA_REVISION" \
            --horizon 450 \
            --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$RUN_DIR/eval/${PANEL}.json"
    done
    WANDB_RESUME=must JAX_PLATFORMS=cpu "$PYTHON" \
        "$BASELINES/scripts/log_fixed_eval_wandb.py" \
        --promotion "$RUN_DIR/eval/promotion.json" \
        --development "$RUN_DIR/eval/development.json" \
        --receipt "$RUN_DIR/eval/wandb_receipt.json"
fi

find "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/parity" \
    -type f -print0 | sort -z | xargs -0 -r sha256sum \
    > "$RUN_DIR/artifacts.sha256"
if [ -f "$RUN_DIR/smoke_validation.json" ]; then
    sha256sum "$RUN_DIR/smoke_validation.json" >> "$RUN_DIR/artifacts.sha256"
fi
RECEIPT_TMP="$RUN_DIR/.receipt.env.$SLURM_JOB_ID"
printf '%s\n' \
    "schema=terra_accepted_bank_euler_receipt_v2" \
    "status=PASSED" \
    "campaign_sha256=$CAMPAIGN_SHA" \
    "campaign_arms=$MANIFEST_ARMS" \
    "campaign_archive=$CAMPAIGN_ARCHIVE" \
    "terra_revision=$TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "phase=$PHASE" \
    "arm=$ARM" \
    "seed=$SEED" \
    "updates=$UPDATES" \
    "global_transitions=$GLOBAL_TRANSITIONS" \
    "train_maps_per_condition=$MANIFEST_TRAIN_MAPS_PER_CONDITION" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "slurm_partition=$SLURM_JOB_PARTITION" \
    "wandb_mode=$WANDB_MODE" \
    "wandb_run_id=$WANDB_RUN_ID" \
    > "$RECEIPT_TMP"
mv "$RECEIPT_TMP" "$RUN_DIR/receipt.env"
echo "ACCEPTED_BANK_${PHASE^^}_${ARM}_PASSED"
