#!/usr/bin/env bash
# Matched precision-only scratch experiment. Submission and its 24 h wall limit
# belong to the caller. Both interfaces use the same corrected native rules.
set -euo pipefail
PHASE="${1:?smoke, probe, production or eval}"
INTERFACE="${2:?primitive8 or structured}"
[[ "$PHASE" == smoke || "$PHASE" == probe || "$PHASE" == production || "$PHASE" == eval ]]
[[ "$INTERFACE" == primitive8 || "$INTERFACE" == structured ]]
: "${TERRA_RUN_DIR:?}" "${TERRA_AB_INPUTS:?}" "${TERRA_EXPERIMENT_INPUTS:?}" "${TERRA_STRUCTURED_INPUTS:?}" "${DATASET_PATH:?}"
BASELINES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAIR_ROOT="$(dirname "$BASELINES_ROOT")"
export PYTHONPATH="$PAIR_ROOT/terra:$BASELINES_ROOT:$TERRA_EXPERIMENT_INPUTS/geometry_dependencies"
export DATASET_SIZE=20480 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"
export WANDB_DIR="$TERRA_RUN_DIR/wandb"
export JAX_COMPILATION_CACHE_DIR="$TERRA_RUN_DIR/jax-cache"
export JAX_ENABLE_COMPILATION_CACHE=true
mkdir -p "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$TERRA_RUN_DIR"
NUM_DEVICES="${NUM_DEVICES:-4}"
python -u "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices "$NUM_DEVICES"
python - <<'PY'
import jax
devices = [device for device in jax.devices() if device.platform == "gpu"]
if not devices or any("GH200" not in device.device_kind for device in devices):
    raise SystemExit(f"This matched CSCS experiment requires GH200 devices: {devices}")
PY

if [[ "$PHASE" == eval ]]; then
    CHECKPOINT="${TERRA_EVAL_CHECKPOINT:-$TERRA_RUN_DIR/production/checkpoint.pkl}"
    OUT="${TERRA_EVAL_OUTPUT:-$TERRA_RUN_DIR/eval_latest}"
    mkdir -p "$OUT"
    for DECODER in greedy sampled; do
        DECODE_ARGS=()
        if [[ "$DECODER" == sampled ]]; then DECODE_ARGS=(--sampled); fi
        python -u "$BASELINES_ROOT/eval_structured.py" --checkpoint "$CHECKPOINT" \
            --initial-states "$TERRA_AB_INPUTS/eval_states.pkl" --action-interface "$INTERFACE" \
            --seed 20261013 --policy-snapshot-interval 32 --output "$OUT/$DECODER.json" "${DECODE_ARGS[@]}"
    done
    exit 0
fi

ARGS=(
    --action-interface "$INTERFACE"
    --maps-path train_v3_generalist_512
    --env-template "$TERRA_STRUCTURED_INPUTS/initial_states_game.pkl"
    --distance-protocol-id obstacle_geodesic_8_physical_global_v1
    --training-slots "$TERRA_AB_INPUTS/train_slots.json"
    --precision-episode-fraction 1.0
    --precision-safety --precision-observation-v2
    --terminate-loaded-dead-ends --terminate-unloaded-dead-ends
    --precision-edge-weight 0.5 --advantage-normalization per_mode
    --model-config "$BASELINES_ROOT/scripts/structured/campaign_model.json"
    --seed 20261013 --num-devices "$NUM_DEVICES" --num-envs 512 --num-steps 32
    --epochs 2 --minibatches 32 --learning-rate 3e-4 --clip-eps 0.2 --vf-coef 2 --no-value-clip
    --entropy-type 0.02 --entropy-move 0.02 --entropy-turn 0.02 --entropy-heading 0.02
    --gamma 1.0 --gae-lambda 0.95 --discount-reference-s 300
    --time-limit --decision-limit 2000 --decisions-per-dig-unit 8.0
    --timing-json "$BASELINES_ROOT/scripts/structured/timing_simple.json"
    --time-budget-s 3600 --time-budget-factor 3.0 --time-budget-offset-s 7960
    --success-time-bonus 2 --success-decision-bonus 0
    --save-initial-checkpoint --finite-check-interval 100
    --checkpoint-interval 100 --keep-checkpoint-every 500
)
case "$PHASE" in
smoke)
    export WANDB_MODE=disabled
    test -s "$TERRA_AB_INPUTS/eval_states.pkl"
    OUT="$TERRA_RUN_DIR/smoke"
    ARGS+=(--output "$OUT" --num-envs 8 --num-steps 4 --minibatches 2
           --decision-limit 4 --decisions-per-dig-unit 0 --finite-check-interval 1
           --checkpoint-interval 1 --keep-checkpoint-every 1)
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --updates 2
    cp "$OUT/checkpoint.pkl" "$OUT/before_resume.pkl"
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --updates 3 --resume-from "$OUT/checkpoint.pkl"
    python - "$TERRA_AB_INPUTS/eval_states.pkl" "$OUT/eval_states.pkl" <<'PY'
import pickle
import sys
with open(sys.argv[1], "rb") as stream:
    bank = pickle.load(stream)
bank["initial"] = bank["initial"][:4]
bank["episodes"] = bank["episodes"][:4]
with open(sys.argv[2], "wb") as stream:
    pickle.dump(bank, stream, protocol=pickle.HIGHEST_PROTOCOL)
PY
    python -u "$BASELINES_ROOT/eval_structured.py" --checkpoint "$OUT/checkpoint.pkl" \
        --initial-states "$OUT/eval_states.pkl" --action-interface "$INTERFACE" \
        --seed 20261013 --policy-snapshot-interval 1 --output "$OUT/eval_smoke.json"
    python "$BASELINES_ROOT/scripts/precision_action_ab/verify_smoke.py" \
        --before "$OUT/before_resume.pkl" --after "$OUT/checkpoint.pkl" --metrics "$OUT/metrics.jsonl" \
        --interface "$INTERFACE" --output "$OUT/qualification.json" --evaluation "$OUT/eval_smoke.json"
    ;;
probe)
    export WANDB_MODE=disabled
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --output "$TERRA_RUN_DIR/probe" \
        --updates 3 --checkpoint-interval 1000
    ;;
production)
    RESUME=()
    if [[ -f "$TERRA_RUN_DIR/production/checkpoint.pkl" ]]; then
        RESUME=(--resume-from "$TERRA_RUN_DIR/production/checkpoint.pkl")
    fi
    WANDB_ARGS=(--wandb-project mixed-agents --wandb-entity aless-weber-eth
                --wandb-name "precision_action_ab_${INTERFACE}_20261013_s20261013")
    if [[ -n "${WANDB_RUN_ID:-}" ]]; then WANDB_ARGS+=(--wandb-id "$WANDB_RUN_ID"); fi
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --output "$TERRA_RUN_DIR/production" \
        --updates 1000000 "${WANDB_ARGS[@]}" "${RESUME[@]}"
    ;;
esac
