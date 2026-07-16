#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./isaac_sim/moleworks_plan_on_map.sh <MAP_DIR> [POLICY_PKL]

Runs isaac_sim/extract_map.py on a single-map folder and writes artifacts into the map folder:
  - terra_plan.pkl
  - terra_plan.json
  - terra_plan.gif
  - terra_plan_rollout.gif

Arguments:
  MAP_DIR     Path to a map folder containing images/img_1.npy (and occupancy/, dumpability/, distance/, actions/).
  POLICY_PKL  Optional path to a checkpoint .pkl. If omitted, uses the newest checkpoints/*.pkl.

Environment overrides (optional):
  N_STEPS=400           Max rollout steps (default: 400)
  SEED=0               RNG seed (default: 0)
  ROLLOUT_GIF_EVERY=2   Render every Nth step in rollout GIF (default: 2)
  TRENCH_ALIGN=1        Locally align saved dig pose/yaw to trench axes and match paired dump pose (default: 0)
  TRENCH_ALIGN_MAX_POS_DELTA_TILES=1.5  Max position snap for trench align
  TRENCH_ALIGN_MAX_YAW_DELTA_STEPS=2    Max base yaw bucket change for trench align
  CONDA_ENV=terra       Conda env name (default: terra)
  PYTHONNOUSERSITE=1    Recommended to avoid ~/.local leakage
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

MAP_DIR="$(realpath "$1")"
if [[ ! -d "$MAP_DIR" ]]; then
  echo "ERROR: MAP_DIR does not exist: $MAP_DIR" >&2
  exit 1
fi
if [[ ! -f "$MAP_DIR/images/img_1.npy" ]]; then
  echo "ERROR: expected $MAP_DIR/images/img_1.npy" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-terra}"
N_STEPS="${N_STEPS:-400}"
SEED="${SEED:-0}"
ROLLOUT_GIF_EVERY="${ROLLOUT_GIF_EVERY:-2}"
TRENCH_ALIGN="${TRENCH_ALIGN:-0}"
TRENCH_ALIGN_MAX_POS_DELTA_TILES="${TRENCH_ALIGN_MAX_POS_DELTA_TILES:-1.5}"
TRENCH_ALIGN_MAX_YAW_DELTA_STEPS="${TRENCH_ALIGN_MAX_YAW_DELTA_STEPS:-2}"

POLICY_PKL="${2:-}"
if [[ -z "$POLICY_PKL" ]]; then
  POLICY_PKL="$(ls -1t "$REPO_ROOT"/checkpoints/*.pkl 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$POLICY_PKL" ]]; then
  echo "ERROR: no checkpoint found under $REPO_ROOT/checkpoints/ and none provided." >&2
  exit 1
fi
POLICY_PKL="$(realpath "$POLICY_PKL")"
if [[ ! -f "$POLICY_PKL" ]]; then
  echo "ERROR: POLICY_PKL not found: $POLICY_PKL" >&2
  exit 1
fi

if [[ ! -f "$MAP_DIR/distance/img_1.npy" ]]; then
  # Terra requires distance/img_*.npy. Auto-generate if Terra repo is next to baselines.
  TERRA_ROOT="$(cd "$REPO_ROOT/../terra" 2>/dev/null && pwd || true)"
  GEN="$TERRA_ROOT/tools/generate_distance_maps.py"
  if [[ -f "$GEN" ]]; then
    echo "distance/img_1.npy missing; generating with $GEN"
    conda run -n "$CONDA_ENV" env PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
      python -u "$GEN" --dataset "$MAP_DIR"
  else
    echo "ERROR: $MAP_DIR/distance/img_1.npy missing and could not find terra/tools/generate_distance_maps.py next to this repo." >&2
    exit 1
  fi
fi

OUT_PKL="$MAP_DIR/terra_plan.pkl"

echo "Map:    $MAP_DIR"
echo "Policy: $POLICY_PKL"
echo "Out:    $OUT_PKL"

EXTRA_ARGS=()
if [[ "$TRENCH_ALIGN" == "1" || "$TRENCH_ALIGN" == "true" || "$TRENCH_ALIGN" == "TRUE" ]]; then
  EXTRA_ARGS+=(
    --trench_align
    --trench_align_max_pos_delta_tiles "$TRENCH_ALIGN_MAX_POS_DELTA_TILES"
    --trench_align_max_yaw_delta_steps "$TRENCH_ALIGN_MAX_YAW_DELTA_STEPS"
  )
fi

conda run -n "$CONDA_ENV" env PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
  python -u -s "$REPO_ROOT/isaac_sim/extract_map.py" \
    --policy_path "$POLICY_PKL" \
    --map_path "$MAP_DIR" \
    --n_steps "$N_STEPS" \
    --seed "$SEED" \
    --output_path "$OUT_PKL" \
    --serialize \
    --render_plan_gif \
    --render_rollout_gif \
    --rollout_gif_every "$ROLLOUT_GIF_EVERY" \
    "${EXTRA_ARGS[@]}"

echo "Done."
echo "Wrote:"
echo "  $OUT_PKL"
echo "  ${OUT_PKL%.pkl}.json"
echo "  ${OUT_PKL%.pkl}.gif"
echo "  ${OUT_PKL%.pkl}_rollout.gif"
