#!/usr/bin/env bash
# Scratch structured-action PPO (terra_structured_v1) on the 20,480-map
# generalist bank under the manual game's native rules (1 m pull, +-30 degree
# cone, perpendicular precision edges, turn-keeping moves, native dump
# observation; no bucket-width gate). Half the lanes are precision episodes on
# the qualified slots. Campaign encoder and observations; no teacher (no
# mapping from the eight-way policy to heading-conditioned DO).
#
# Each episode's time budget is 2 h plus 2.15 times its map's dig-only modeled
# time: no fixed budget fits both trenches and the 7-9 h precision slots. The
# October 8 oracle under these rules takes about 1.35 h + 1.43 x dig-only time,
# so every oracle finish has 1.33-1.5x slack. The decision limit is 0.75 per
# dig unit, at least 450 (the oracle needs ~0.5 on foundations; small trenches
# fit the floor). GAE lambda applies per 300 modeled seconds, about
# 0.75 across a typical workspace dig and ~1 across moves and turns (the
# legacy campaign's trace per dig cycle); gamma is 1, so the objective does
# not depend on it.
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
NAME="structured_scratch_s20261009"
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
    --time-budget-s 3600 --time-budget-factor 2.15 --time-budget-offset-s 7200
    --decision-limit 450 --decisions-per-dig-unit 0.75
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
