#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-f0-eval
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0/logs/%x_%j.out

set -euo pipefail
umask 002

: "${ARM:?submit with ARM=foundation or ARM=trench}"
case "$ARM" in
    foundation|trench) ;;
    *)
        echo "Unsupported F0 ARM=$ARM" >&2
        exit 2
        ;;
esac

RUN_ROOT="${RUN_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0}"
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
ARM_ROOT="$RUN_ROOT/$ARM"
CHECKPOINT_DIR="$ARM_ROOT/checkpoints"
OUTPUT="$ARM_ROOT/eval.json"
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
export DATASET_SIZE=1

test -f "$ARM_ROOT/TRAINING_COMPLETE"
test ! -e "$OUTPUT"
(
    cd "$RUN_ROOT"
    sha256sum --check manifests/source_files.sha256
    sha256sum --check manifests/bank_files.sha256
)

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

cd "$SOURCE/terra-baselines"
"$VENV/bin/python" -u eval_f0_identity.py \
    --checkpoint-glob "$CHECKPOINT_DIR/*_update_*.pkl" \
    --dataset-root "$BANK" \
    --identity "$ARM" \
    --expected-checkpoints 10 \
    --output "$OUTPUT"

GATE_RESULT="$(
    "$VENV/bin/python" - "$OUTPUT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload["schema"] != "terra_f0_identity_eval_v1":
    raise SystemExit("unexpected F0 evaluation schema")
if len(payload["records"]) != 10:
    raise SystemExit("F0 evaluation did not cover ten checkpoints")
if not payload["reset_verification"]["passed"]:
    raise SystemExit("F0 reset verification failed")
for record in payload["records"]:
    if record["summary"]["integrity_failure_count"]:
        raise SystemExit("F0 rollout integrity failed")
print("passed" if payload["mastery_gate"]["passed"] else "failed")
PY
)"
case "$GATE_RESULT" in
    passed)
        touch "$ARM_ROOT/MASTERY_PASSED"
        ;;
    failed)
        touch "$ARM_ROOT/MASTERY_FAILED"
        ;;
    *)
        echo "Unexpected F0 gate result: $GATE_RESULT" >&2
        exit 3
        ;;
esac
touch "$ARM_ROOT/EVALUATION_COMPLETE"
echo "F0_${ARM^^}_EVALUATION_COMPLETE mastery=$GATE_RESULT"
