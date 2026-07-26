#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-b0-eval
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1/logs/%x_%j.out

set -euo pipefail
umask 002

: "${PANEL:?submit with a declared B0 panel}"
case "$PANEL" in
    foundation_geometry | trench_side)
        PANEL_COUNT=16
        ;;
    foundation_distance | trench_distance)
        PANEL_COUNT=32
        ;;
    trench_topology)
        PANEL_COUNT=48
        ;;
    *)
        echo "Unsupported B0 panel: $PANEL" >&2
        exit 2
        ;;
esac

RUN_ROOT="${RUN_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-$RUN_ROOT}"
SOURCE="$SOURCE_ROOT/source"
BANK="$RUN_ROOT/bank"
PANEL_ROOT="$RUN_ROOT/runs/$PANEL"
CHECKPOINT_DIR="$PANEL_ROOT/checkpoints"
OUTPUT="$PANEL_ROOT/eval.json"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="$SOURCE/terra:$SOURCE/terra-baselines"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_MODE=disabled
export DATASET_PATH="$BANK"
export DATASET_SIZE="$PANEL_COUNT"

test -f "$PANEL_ROOT/TRAINING_COMPLETE"
test -f "$PANEL_ROOT/TRAINING_GATE.json"
test ! -e "$OUTPUT"
(
    cd "$SOURCE_ROOT"
    sha256sum --check manifests/source_files.sha256
)
(
    cd "$RUN_ROOT"
    sha256sum --check manifests/bank_files.sha256
)

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

cd "$SOURCE/terra-baselines"
"$VENV/bin/python" -m unittest discover \
    -s tests -p 'test_b0_panel_eval.py' -q
"$VENV/bin/python" -u eval_b0_panel.py \
    --checkpoint-glob "$CHECKPOINT_DIR/*_update_*.pkl" \
    --bank-root "$BANK" \
    --panel "$PANEL" \
    --expected-checkpoints 5 \
    --planned-updates 500 \
    --output "$OUTPUT"

DECISION="$(
    "$VENV/bin/python" - "$OUTPUT" "$PANEL" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload["schema"] != "terra_b0_panel_eval_v1":
    raise SystemExit("unexpected B0 evaluation schema")
if payload["panel"] != sys.argv[2]:
    raise SystemExit("unexpected B0 panel")
if len(payload["records"]) != 5:
    raise SystemExit("B0 evaluation did not cover five checkpoints")
if not payload["reset_verification"]["passed"]:
    raise SystemExit("B0 reset verification failed")
for record in payload["records"]:
    if record["summary"]["integrity_failure_count"]:
        raise SystemExit("B0 rollout integrity failed")
decision = payload["adjudication"]["decision"]
if decision not in {
    "panel_witness_passed",
    "continue_same_panel",
    "conditional_cell_isolates",
    "stop_and_diagnose_panel",
}:
    raise SystemExit(f"unexpected B0 decision: {decision}")
print(decision)
PY
)"

case "$DECISION" in
    panel_witness_passed)
        touch "$PANEL_ROOT/WITNESS_PASSED"
        ;;
    continue_same_panel)
        touch "$PANEL_ROOT/CONTINUATION_AUTHORIZED"
        ;;
    conditional_cell_isolates)
        touch "$PANEL_ROOT/CELL_ISOLATES_REQUIRED"
        ;;
    stop_and_diagnose_panel)
        touch "$PANEL_ROOT/PANEL_DIAGNOSIS_REQUIRED"
        ;;
esac
touch "$PANEL_ROOT/EVALUATION_COMPLETE"
echo "B0_${PANEL^^}_EVALUATION_COMPLETE decision=$DECISION"
