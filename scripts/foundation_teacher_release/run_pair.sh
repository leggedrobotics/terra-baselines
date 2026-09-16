#!/usr/bin/env bash
# One sequential 4-GPU comparison, with a conditional bounded hold.
set -euo pipefail
: "${TERRA_ROOT:?}" "${INPUTS_ROOT:?}" "${PARENT_CHECKPOINT:?}" "${BANK_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${SLURM_JOB_ID:?}"
OUTPUT_DIR="$EXPERIMENT_ROOT/segments/$SLURM_JOB_ID"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_PYTHON:-python}"
export PYTHONPATH="$TERRA_ROOT:$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=true
export JAX_ENABLE_COMPILATION_CACHE=true WANDB_MODE=offline
export PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export XLA_FLAGS='--xla_gpu_mlir_emitter_level=0'
unset WANDB_RUN_ID WANDB_RESUME
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
export JAX_COMPILATION_CACHE_DIR="$OUTPUT_DIR/jax-cache"
"$PYTHON" "$REPO/cluster/cscs/check_jax_runtime.py" --min-devices 4 > "$OUTPUT_DIR/preflight.log" 2>&1
"$PYTHON" - <<'PY'
import jax
if len(jax.devices()) != 4 or not all('GH200' in d.device_kind for d in jax.devices()):
    raise RuntimeError(f'Expected four GH200 GPUs, got {jax.devices()}')
PY

for target in 6250 7500; do
    for arm in control foundation_release; do
        parent="$PARENT_CHECKPOINT"
        if [[ "$target" == 7500 ]]; then
            parent="$OUTPUT_DIR/$arm/u6250/checkpoints/foundation-teacher-${arm}_FINAL.pkl"
        fi
        mkdir -p "$OUTPUT_DIR/$arm"
        stage="$OUTPUT_DIR/$arm/u$target"
        timeout --signal=TERM --kill-after=30s 7200 "$PYTHON" \
            "$REPO/scripts/foundation_teacher_release/run.py" \
            --checkpoint "$parent" --inputs "$INPUTS_ROOT" --output "$stage" \
            --arm "$arm" --target-update "$target" > "$OUTPUT_DIR/$arm/train_u$target.log" 2>&1
        checkpoint="$stage/checkpoints/foundation-teacher-${arm}_FINAL.pkl"
        timeout --signal=TERM --kill-after=30s 1800 \
            bash "$REPO/scripts/excavation_reliability/eval.sh" "$checkpoint" "$stage/full.json" \
            > "$stage/evaluation.log" 2>&1
    done
    "$PYTHON" "$REPO/scripts/foundation_teacher_release/compare.py" \
        --control "$OUTPUT_DIR/control/u$target/full.json" \
        --treatment "$OUTPUT_DIR/foundation_release/u$target/full.json" \
        --output "$OUTPUT_DIR/comparison_u$target.json"
    if [[ "$target" == 6250 ]] && ! "$PYTHON" - "$OUTPUT_DIR/comparison_u6250.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['allow_bounded_hold'] else 1)
PY
    then
        echo "Stopped at u6250: the bounded hold criteria did not pass." | tee "$OUTPUT_DIR/STOPPED"
        exit 0
    fi
done
echo "Completed both milestones. No further training or promotion is automatic." | tee "$OUTPUT_DIR/COMPLETED"
