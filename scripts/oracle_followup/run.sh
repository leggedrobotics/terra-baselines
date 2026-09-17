#!/usr/bin/env bash
# One long combined run; sequential allocations preserve the native checkpoint.
set -euo pipefail
: "${TERRA_ROOT:?}" "${INPUTS_ROOT:?}" "${BANK_ROOT:?}" "${PARENT_CHECKPOINT:?}" "${EXPERIMENT_ROOT:?}" "${SLURM_JOB_ID:?}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_PYTHON:-python}"
OUTPUT_DIR="$EXPERIMENT_ROOT/segments/$SLURM_JOB_ID"
export PYTHONPATH="$TERRA_ROOT:$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=true
export JAX_ENABLE_COMPILATION_CACHE=true WANDB_MODE=offline
export PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export XLA_FLAGS='--xla_gpu_mlir_emitter_level=0'
unset WANDB_RUN_ID WANDB_RESUME
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
export JAX_COMPILATION_CACHE_DIR="$EXPERIMENT_ROOT/jax-cache"
PARENT_CHECKPOINT="$("$PYTHON" - "$EXPERIMENT_ROOT" "$PARENT_CHECKPOINT" <<'PY'
from pathlib import Path
import sys
root, parent = Path(sys.argv[1]), Path(sys.argv[2])
paths = list(root.glob('segments/*/training/checkpoints/*_update_*.pkl'))
if paths:
    parent = max(paths, key=lambda path: int(path.stem.rsplit('_', 1)[1]))
print(parent)
PY
)"
printf '%s\n' "$PARENT_CHECKPOINT" > "$OUTPUT_DIR/parent_checkpoint.txt"
"$PYTHON" "$REPO/cluster/cscs/check_jax_runtime.py" --min-devices 4 > "$OUTPUT_DIR/preflight.log" 2>&1
"$PYTHON" - <<'PY'
import jax
if len(jax.devices()) != 4 or not all('GH200' in d.device_kind for d in jax.devices()):
    raise RuntimeError(f'Expected four GH200 GPUs, got {jax.devices()}')
PY
exec "$PYTHON" \
    "$REPO/scripts/oracle_followup/run.py" --checkpoint "$PARENT_CHECKPOINT" \
    --inputs "$INPUTS_ROOT" --output "$OUTPUT_DIR/training" --target-update 100000 \
    --eval-bank "$BANK_ROOT" --eval-output "$EXPERIMENT_ROOT/evaluation" \
    > "$OUTPUT_DIR/training.log" 2>&1
