#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.4h
#SBATCH --time=03:45:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-v8-tail-eval

set -euo pipefail
umask 002

: "${PARENT_JOB_ID:?missing PARENT_JOB_ID}"
: "${ARM:?missing ARM}"
: "${BASELINES_ROOT:?missing BASELINES_ROOT}"
: "${BASELINES_REVISION:?missing BASELINES_REVISION}"
: "${SEED:?missing SEED}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${BANK_ARCHIVE:?missing BANK_ARCHIVE}"
: "${BANK_SHA:?missing BANK_SHA}"
: "${BANK_DATASET_SHA:?missing BANK_DATASET_SHA}"
: "${BANK_RELEASE_ID:?missing BANK_RELEASE_ID}"
: "${PRIOR_GATE_RECEIPT:?missing PRIOR_GATE_RECEIPT}"
: "${PRIOR_GATE_SHA:?missing PRIOR_GATE_SHA}"
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${TMPDIR:?Euler did not provide TMPDIR}"

case "$ARM" in
    G-DEEP-V8-DENSE-WARM|G-DEEP-XATTN-V8-DENSE-WARM) ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac
[[ "$PARENT_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$SEED" =~ ^[0-9]+$ ]]
test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
EXPECTED_BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
EXPECTED_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1
EXPECTED_RUN_DIR="$RUN_BASE/$BASELINES_REVISION/screen/full/s$SEED/$ARM"
RUN_CONTRACT="$RUN_DIR/run_contract.env"
TAIL_DIR="$RUN_DIR/tail_eval"
PARENT_RECEIPT="$TAIL_DIR/parent_job_receipt.json"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$BANK_SHA" = "$EXPECTED_BANK_SHA"
test "$BANK_DATASET_SHA" = "$EXPECTED_DATASET_SHA"
test "$RUN_DIR" = "$EXPECTED_RUN_DIR"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$PRIOR_GATE_RECEIPT" | awk '{print $1}')" = "$PRIOR_GATE_SHA"
test -f "$RUN_CONTRACT"
test -d "$RUN_DIR/checkpoints"
test -d "$TAIL_DIR"
test ! -e "$PARENT_RECEIPT"
test "${SLURM_CPUS_PER_TASK:-}" = 4

mapfile -t PARENT_ROWS < <(
    sacct -n -X -P -j "$PARENT_JOB_ID" \
        --format=JobIDRaw,State,ExitCode,Partition |
        awk -F'|' -v id="$PARENT_JOB_ID" '$1 == id { print }'
)
test "${#PARENT_ROWS[@]}" -eq 1 || {
    echo "expected one parent accounting row, got ${#PARENT_ROWS[@]}" >&2
    exit 3
}
printf '%s\n' "${PARENT_ROWS[0]}" > "$TAIL_DIR/parent_sacct.txt"
IFS='|' read -r ACCOUNTED_JOB_ID PARENT_STATE PARENT_EXIT_CODE PARENT_PARTITION \
    <<< "${PARENT_ROWS[0]}"
PARENT_STATE="${PARENT_STATE%%+*}"
test "$ACCOUNTED_JOB_ID" = "$PARENT_JOB_ID"
case "$PARENT_STATE" in COMPLETED|TIMEOUT) ;; *) echo "parent ended $PARENT_STATE" >&2; exit 3 ;; esac
test "$PARENT_PARTITION" = gpuhe.24h

PYTHON="$VENV/bin/python"
"$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/tail_prefix.py" \
    --run-dir "$RUN_DIR" --run-contract "$RUN_CONTRACT" \
    --parent-job-id "$PARENT_JOB_ID" --parent-state "$PARENT_STATE" \
    --parent-exit-code "$PARENT_EXIT_CODE" --parent-partition "$PARENT_PARTITION" \
    --evaluator-job-id "$SLURM_JOB_ID" --arm "$ARM" --seed "$SEED" \
    --baselines-revision "$BASELINES_REVISION" --output "$PARENT_RECEIPT" \
    > "$TAIL_DIR/checkpoints.txt"
mapfile -t CHECKPOINTS < "$TAIL_DIR/checkpoints.txt"
test "${#CHECKPOINTS[@]}" -ge 2

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported GPU: $GPU_NAME" >&2
        exit 3
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$MAPS_PER_CONDITION" = 96

JAX_PLATFORMS=cpu "$PYTHON" - "$BANK" "$RUN_CONTRACT" "$PRIOR_GATE_RECEIPT" "$ARM" <<'PY'
import sys
from pathlib import Path

from scripts.euler_v8_deep_xattn_v1 import stage_gate

bank = stage_gate.load_bank_contract(Path(sys.argv[1]).resolve())
contract = stage_gate.parse_run_contract(Path(sys.argv[2]).resolve())
prior = stage_gate.validate_prior_receipt(Path(sys.argv[3]).resolve(), sys.argv[4], "nearby")
stage_gate.validate_run_contract(contract, "full", sys.argv[4], prior)
stage_gate.stage_sampling_contract(bank, "full")
PY

scontrol show job "$SLURM_JOB_ID" -o > "$TAIL_DIR/evaluator_slurm_job.txt"
nvidia-smi -q > "$TAIL_DIR/nvidia_smi.txt"
CHECKPOINT_ARGS=()
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT")
done
mkdir "$TAIL_DIR/eval"
for PANEL in promotion development; do
    JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$TAIL_DIR/eval/$PANEL.json"
    JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$TAIL_DIR/eval/capability_$PANEL.json"
done

JAX_PLATFORMS=cpu "$PYTHON" \
    "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/stage_gate.py" promote \
    --stage full --arm "$ARM" --bank-root "$BANK" \
    --promotion "$TAIL_DIR/eval/promotion.json" \
    --development "$TAIL_DIR/eval/development.json" \
    --capability "$TAIL_DIR/eval/capability_promotion.json" \
    --capability-development "$TAIL_DIR/eval/capability_development.json" \
    --prior-receipt "$PRIOR_GATE_RECEIPT" --run-contract "$RUN_CONTRACT" \
    --parent-job-receipt "$PARENT_RECEIPT" \
    --output "$TAIL_DIR/stage_gate.json"

GATE_PASSED="$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$TAIL_DIR/stage_gate.json")"
printf '%s\n' \
    "status=PASSED" \
    "parent_job_id=$PARENT_JOB_ID" \
    "parent_state=$PARENT_STATE" \
    "checkpoint_count=${#CHECKPOINTS[@]}" \
    "last_checkpoint_update=$((500 * ${#CHECKPOINTS[@]}))" \
    "stage_gate_passed=$GATE_PASSED" > "$TAIL_DIR/status.env"
echo "V8_TAIL_EVAL_PASSED arm=$ARM prefix=${#CHECKPOINTS[@]} gate=$GATE_PASSED"
