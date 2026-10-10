#!/usr/bin/env bash
# Scratch structured-action PPO (terra_structured_v1) on the 20,480-map
# generalist bank under the manual game's native rules (1 m pull, +-30 degree
# cone, perpendicular precision edges, turn-keeping moves, native dump
# observation; no bucket-width gate). Half the lanes are precision episodes on
# the qualified slots. Campaign encoder and observations; no teacher (no
# mapping from the eight-way policy to heading-conditioned DO).
#
# Episodes end on success or at a per-map macro cap (300 + 1.2 per dig cell:
# trenches 300, precision slabs ~530-960, about 2x the October 8 oracle);
# modeled time never ends an episode. Reward: material progress (about 1.15
# over a whole map), +6 on success, -1 at the cap, and on success only
# 2 x (1 - modeled time / T_ref) + 1 x (1 - decisions / cap). No per-step time
# cost, so time can never make progress worse than idling. Modeled time
# (timing_simple.json): 30 s per 0.25 m^3 bucket (120 s/m^3), 278 s setup per
# dig (the interruption-filtered field fit; the workspace penalty), 0.5 m/s
# driving, 5 s/rad chassis turns. T_ref = 2.2 h + 3 x the map's dig-only time
# is twice the oracle fit, so an oracle-pace finish earns about half the time
# bonus. GAE lambda applies per 300 modeled seconds; gamma is 1.
#
# Phases: smoke (tiny lanes, 2 updates + resume to 3, W&B off), probe
# (production lanes, 3 updates, W&B off: throughput) and production.
set -euo pipefail
PHASE="${1:?smoke, probe or production}"
[[ "$PHASE" == smoke || "$PHASE" == probe || "$PHASE" == production ]]
: "${TERRA_RUN_DIR:?}" "${TERRA_EXPERIMENT_INPUTS:?}" "${TERRA_STRUCTURED_INPUTS:?}" "${DATASET_PATH:?}"
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
NAME="structured_scratch_s20261010"
ARGS=(
    --maps-path train_v3_generalist_512
    --env-template "$TERRA_STRUCTURED_INPUTS/initial_states_game.pkl"
    --distance-protocol-id obstacle_geodesic_8_physical_global_v1
    --training-slots "${TERRA_TRAINING_SLOTS:-$TERRA_STRUCTURED_INPUTS/training_slots.json}"
    --precision-episode-fraction 0.5
    --model-config "$BASELINES_ROOT/scripts/structured/campaign_model.json"
    --seed 20261009 --num-devices "$NUM_DEVICES" --num-envs 512 --num-steps 32
    --epochs 2 --minibatches 32 --learning-rate 3e-4 --clip-eps 0.2 --vf-coef 2 --no-value-clip
    --entropy-type 0.02 --entropy-move 0.02 --entropy-turn 0.02 --entropy-heading 0.02
    --gamma 1.0 --gae-lambda 0.95 --discount-reference-s 300
    --no-time-limit --decision-limit 300 --decisions-per-dig-unit 1.2
    --timing-json "$BASELINES_ROOT/scripts/structured/timing_simple.json"
    --time-budget-s 3600 --time-budget-factor 3.0 --time-budget-offset-s 7960
    --success-time-bonus 2 --success-decision-bonus 1
    --finite-check-interval 100 --checkpoint-interval 100 --keep-checkpoint-every 1000
)
case "$PHASE" in
smoke)
    export WANDB_MODE=disabled
    OUT="$TERRA_RUN_DIR/smoke"
    ARGS+=(--output "$OUT" --num-envs 8 --num-steps 4 --minibatches 2 --finite-check-interval 1
           --checkpoint-interval 1 --keep-checkpoint-every 2)
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --updates 2
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --updates 3 --resume-from "$OUT/checkpoint.pkl"
    test -f "$OUT/checkpoint_update_000002.pkl"
    test "$(wc -l < "$OUT/metrics.jsonl")" -eq 3
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
    python -u "$BASELINES_ROOT/train_structured.py" "${ARGS[@]}" --output "$TERRA_RUN_DIR/production" \
        --updates 1000000 --wandb-project mixed-agents --wandb-entity aless-weber-eth \
        --wandb-name "$NAME" ${WANDB_RUN_ID:+--wandb-id "$WANDB_RUN_ID"} "${RESUME[@]}"
    ;;
esac
