#!/usr/bin/env bash
# usage: eval_local.sh CHECKPOINT OUTPUT_DIR
# Same milestone evaluation as eval.sbatch, on the local RTX 4090 (forward only).
# DUMP_MAX_RADIUS_M=5.5 evaluates under that excavator dump reach; PANEL=promotion
# scores the promotion panel instead of development (no 32-start panels).
# Terra's machine working rules, each passed on like DUMP_MAX_RADIUS_M (unset =
# the checkpoint's own treatment): DIG_MIN_RADIUS_M, DUMP_MIN_RADIUS_M,
# DUG_CLEARANCE_M, DUMP_MIN_DUG_DISTANCE_M (metres) and CENTRE_CHASSIS_ON_BASE
# (1 or 0). TERRA_ROOT defaults to the terra-machine-rules worktree, which is
# the release environment while every rule is off. TERRA_EVAL_PYTHON replaces
# the default runtime, whose site-packages sit on the external T7 drive.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: eval_local.sh CHECKPOINT OUTPUT_DIR" >&2; exit 2; }
test -r "$1"
test ! -e "$2"
mkdir -p "$2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TERRA_EVAL_PYTHON:-/home/lorenzo/moleworks/.artifacts/terra_foundation_sweep_20260907/runtime/bin/python}"
export PYTHONPATH="${TERRA_ROOT:-/home/lorenzo/moleworks/.worktrees/terra_machine_rules_20260930/terra}:$REPO:/home/lorenzo/moleworks/.artifacts/terra_test_time_compute_20260921/adaptation"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy MPLBACKEND=Agg WANDB_MODE=disabled
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=true
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
CUDA_PATHS="$("$PYTHON" -c 'import glob,site; print(":".join(glob.glob(site.getsitepackages()[0]+"/nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$CUDA_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RULE_ARGS=()
[ -z "${DUMP_MAX_RADIUS_M:-}" ] || RULE_ARGS+=(--dump-max-radius-m "$DUMP_MAX_RADIUS_M")
[ -z "${DIG_MIN_RADIUS_M:-}" ] || RULE_ARGS+=(--dig-min-radius-m "$DIG_MIN_RADIUS_M")
[ -z "${DUMP_MIN_RADIUS_M:-}" ] || RULE_ARGS+=(--dump-min-radius-m "$DUMP_MIN_RADIUS_M")
[ -z "${DUG_CLEARANCE_M:-}" ] || RULE_ARGS+=(--dug-clearance-m "$DUG_CLEARANCE_M")
[ -z "${DUMP_MIN_DUG_DISTANCE_M:-}" ] || RULE_ARGS+=(--dump-min-dug-distance-m "$DUMP_MIN_DUG_DISTANCE_M")
case "${CENTRE_CHASSIS_ON_BASE:-}" in
    "") ;;
    1) RULE_ARGS+=(--centre-chassis-on-base) ;;
    0) RULE_ARGS+=(--no-centre-chassis-on-base) ;;
    *) echo "CENTRE_CHASSIS_ON_BASE must be 1 or 0" >&2; exit 2 ;;
esac
EVAL_FORWARD_CHUNK=120 "$PYTHON" -u "$REPO/eval_fixed_bank.py" \
    --checkpoint "$1" --bank-root /home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819 \
    --accepted-panel "${PANEL:-development}" --panel-family gate_main \
    --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 \
    --horizon 450 --seed 20260724 \
    --expect-completion-contract exact_visible_dump_v1 \
    --output "$2/full608.json" "${RULE_ARGS[@]}" > "$2/full608.log" 2>&1
[ "${PANEL:-development}" != development ] || \
EVAL_FORWARD_CHUNK=32 "$PYTHON" -u "$REPO/scripts/euler_gru_generalist_512/eval_known_starts.py" \
    --checkpoint "$1" --output "$2/known_starts" "${RULE_ARGS[@]}" > "$2/known_starts.log" 2>&1
touch "$2/EVAL_DONE"
