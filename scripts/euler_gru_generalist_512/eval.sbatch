#!/usr/bin/env bash
# One-GPU milestone evaluation: the 608-map gate_main development panel
# (greedy, 450 actions) and the 32-start road, straight and tee panels.
set -euo pipefail
umask 027
for REQUIRED in CHECKPOINT OUTPUT BASELINES_ROOT RUNTIME_TERRA_ROOT VENV EVAL_BANK \
    KNOWN_MAPS KNOWN_HELPER GPU_TYPE; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
case "$GPU_TYPE" in rtx_4090) GPU_MODEL="RTX 4090" ;; rtx_3090) GPU_MODEL="RTX 3090" ;; *) exit 2 ;; esac
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "$GPU_MODEL"
test -r "$CHECKPOINT"
mkdir -p "$OUTPUT"

module purge
module load stack/2024-06
PYTHON="$VENV/bin/python"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT:$(dirname "$KNOWN_HELPER")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy MPLBACKEND=Agg WANDB_MODE=disabled
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
"$PYTHON" "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices 1

EVAL_FORWARD_CHUNK=120 "$PYTHON" -u "$BASELINES_ROOT/eval_fixed_bank.py" \
    --checkpoint "$CHECKPOINT" --bank-root "$EVAL_BANK" \
    --accepted-panel development --panel-family gate_main \
    --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 \
    --horizon 450 --seed 20260724 \
    --expect-completion-contract exact_visible_dump_v1 \
    --output "$OUTPUT/full608.json" > "$OUTPUT/full608.log" 2>&1

EVAL_FORWARD_CHUNK=32 "$PYTHON" -u "$BASELINES_ROOT/scripts/euler_gru_generalist_512/eval_known_starts.py" \
    --checkpoint "$CHECKPOINT" --output "$OUTPUT/known_starts" \
    --maps-root "$KNOWN_MAPS" --helper "$KNOWN_HELPER" > "$OUTPUT/known_starts.log" 2>&1
touch "$OUTPUT/EVAL_DONE"
