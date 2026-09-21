#!/usr/bin/env bash
# One combined run; optional imitation has a bounded target and allocation checks.
set -euo pipefail
: "${TERRA_ROOT:?}" "${INPUTS_ROOT:?}" "${BANK_ROOT:?}" "${PARENT_CHECKPOINT:?}" "${EXPERIMENT_ROOT:?}" "${SLURM_JOB_ID:?}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_PYTHON:-python}"
INITIAL_PARENT_CHECKPOINT="$PARENT_CHECKPOINT"
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
DEMONSTRATION_ARGS=()
if [[ -n "${DEMONSTRATION_NPZ:-}" ]]; then
    : "${TARGET_UPDATE:?Set the bounded absolute imitation target, e.g. 36500 for u35000 + 1500}"
    DEMONSTRATION_ARGS=(
        --demonstration-npz "$DEMONSTRATION_NPZ"
        --demonstration-coef "${DEMONSTRATION_COEF:-0.01}"
        --demonstration-fade-transitions "${DEMONSTRATION_FADE_TRANSITIONS:-24576000}"
        --demonstration-batch-size "${DEMONSTRATION_BATCH_SIZE:-16}"
        --demonstration-hold-updates "${DEMONSTRATION_HOLD_UPDATES:-750}"
    )
else
    TARGET_UPDATE="${TARGET_UPDATE:-100000}"
fi
"$PYTHON" "$REPO/cluster/cscs/check_jax_runtime.py" --min-devices 4 > "$OUTPUT_DIR/preflight.log" 2>&1
"$PYTHON" - <<'PY'
import jax
if len(jax.devices()) != 4 or not all('GH200' in d.device_kind for d in jax.devices()):
    raise RuntimeError(f'Expected four GH200 GPUs, got {jax.devices()}')
PY
if [[ -n "${DEMONSTRATION_NPZ:-}" ]]; then
    checkpoint_update() {
        JAX_PLATFORMS=cpu "$PYTHON" - "$1" <<'PY'
import pickle
import sys
from utils.helpers import register_checkpoint_config_classes
register_checkpoint_config_classes()
with open(sys.argv[1], 'rb') as stream:
    print(int(pickle.load(stream)['next_update']))
PY
    }
    CURRENT_UPDATE="$(checkpoint_update "$PARENT_CHECKPOINT")"
fi
if [[ -n "${DEMONSTRATION_NPZ:-}" ]] && (( CURRENT_UPDATE < TARGET_UPDATE )); then
    QUALIFICATION_START="$(checkpoint_update "$INITIAL_PARENT_CHECKPOINT")"
    # Every diagnostic uses the real bank, model, Adam and 4x256 PPO layout.
    # The production parent remains the original campaign checkpoint.
    "$PYTHON" "$REPO/scripts/oracle_followup/run.py" \
        --checkpoint "$INITIAL_PARENT_CHECKPOINT" --inputs "$INPUTS_ROOT" \
        --output "$OUTPUT_DIR/qualification-first" --target-update "$((QUALIFICATION_START + 1))" \
        --qualification native "${DEMONSTRATION_ARGS[@]}" \
        > "$OUTPUT_DIR/qualification-first.log" 2>&1
    "$PYTHON" "$REPO/scripts/oracle_followup/run.py" \
        --checkpoint "$OUTPUT_DIR/qualification-first/checkpoints/generalist-oracle-combined_FINAL.pkl" \
        --inputs "$INPUTS_ROOT" --output "$OUTPUT_DIR/qualification-resume" \
        --target-update "$((QUALIFICATION_START + 2))" --qualification native \
        "${DEMONSTRATION_ARGS[@]}" > "$OUTPUT_DIR/qualification-resume.log" 2>&1
    "$PYTHON" "$REPO/scripts/oracle_followup/run.py" \
        --checkpoint "$INITIAL_PARENT_CHECKPOINT" --inputs "$INPUTS_ROOT" \
        --output "$OUTPUT_DIR/qualification-boundary" --target-update "$((QUALIFICATION_START + 2))" \
        --qualification fade-boundary "${DEMONSTRATION_ARGS[@]}" \
        > "$OUTPUT_DIR/qualification-boundary.log" 2>&1
    "$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
first, resumed, boundary = [
    json.loads((root / f'qualification-{stage}' / 'result.json').read_text())
    for stage in ('first', 'resume', 'boundary')
]
assert all(result['status'] == 'PASS' for result in (first, resumed, boundary))
assert resumed['update'] == first['update'] + 1
assert resumed['adam_step'] == first['adam_step'] + 64
assert first['demonstration_state'] == resumed['demonstration_state']
assert boundary['qualification_coefficients'][0] > 0
assert boundary['qualification_coefficients'][1] == 0
log = (root / 'qualification-boundary.log').read_text()
assert f"Demonstration execution active: True at u{boundary['update'] - 2}" in log
assert f"Demonstration execution active: False at u{boundary['update'] - 1}" in log
print('PASS exact-layout finite update, native save/resume and auxiliary fade boundary', flush=True)
PY
fi
exec "$PYTHON" \
    "$REPO/scripts/oracle_followup/run.py" --checkpoint "$PARENT_CHECKPOINT" \
    --inputs "$INPUTS_ROOT" --output "$OUTPUT_DIR/training" --target-update "$TARGET_UPDATE" \
    --eval-bank "$BANK_ROOT" --eval-output "$EXPERIMENT_ROOT/evaluation" \
    "${DEMONSTRATION_ARGS[@]}" \
    > "$OUTPUT_DIR/training.log" 2>&1
