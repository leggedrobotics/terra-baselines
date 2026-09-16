#!/usr/bin/env bash
# One combined run; no duplicate learning control or automatic efficiency phase.
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
"$PYTHON" "$REPO/cluster/cscs/check_jax_runtime.py" --min-devices 4 > "$OUTPUT_DIR/preflight.log" 2>&1
"$PYTHON" - <<'PY'
import jax
if len(jax.devices()) != 4 or not all('GH200' in d.device_kind for d in jax.devices()):
    raise RuntimeError(f'Expected four GH200 GPUs, got {jax.devices()}')
PY
timeout --signal=TERM --kill-after=30s 9000 "$PYTHON" \
    "$REPO/scripts/oracle_followup/run.py" --checkpoint "$PARENT_CHECKPOINT" \
    --inputs "$INPUTS_ROOT" --output "$OUTPUT_DIR/training" --updates 2500 \
    > "$OUTPUT_DIR/training.log" 2>&1

# Both milestone checkpoints come from one uninterrupted training process.
mapfile -t milestones < <("$PYTHON" - "$OUTPUT_DIR/training/plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print(plan['start_update'] + 1250)
print(plan['target_update'])
PY
)
for milestone in "${milestones[@]}"; do
    printf -v suffix '%06d' "$milestone"
    checkpoint="$OUTPUT_DIR/training/checkpoints/generalist-oracle-combined_update_${suffix}.pkl"
    timeout --signal=TERM --kill-after=30s 1800 \
        bash "$REPO/scripts/excavation_reliability/eval.sh" "$checkpoint" "$OUTPUT_DIR/u${milestone}.json" \
        > "$OUTPUT_DIR/evaluation_u${milestone}.log" 2>&1
done
"$PYTHON" "$REPO/scripts/analysis/terra_efficiency_readiness.py" \
    --previous "$OUTPUT_DIR/u${milestones[0]}.json" --current "$OUTPUT_DIR/u${milestones[1]}.json" \
    --transitions-per-update 32768 --output "$OUTPUT_DIR/efficiency_readiness.json"
echo "Combined bounded run and both fixed panels complete. Added efficiency costs remain zero."
