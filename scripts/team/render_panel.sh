#!/usr/bin/env bash
# Render several GIFs in parallel, one process per GPU. RENDER_SPECS lists
# "agents:checkpoint:name" entries; outputs go to RENDER_OUTPUT/name.gif.
# Optional: RENDER_TYPES (e.g. 2 for a skid steer; overrides the agent count),
# RENDER_MAPS_PATH (bank subdirectory) and RENDER_DATASET_SIZE.
set -euo pipefail
: "${TERRA_ROOT:?}" "${BASELINES_ROOT:?}" "${RENDER_BANK_ROOT:?}" "${RENDER_OUTPUT:?}" "${RENDER_SPECS:?}"
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
mkdir -p "$RENDER_OUTPUT"
common=(--bank "$RENDER_BANK_ROOT" --grid 2 --seed "${RENDER_SEED:-0}")
[[ -z "${RENDER_TYPES:-}" ]] || common+=(--types "$RENDER_TYPES")
[[ -z "${RENDER_MAPS_PATH:-}" ]] || common+=(--maps-path "$RENDER_MAPS_PATH")
[[ -z "${RENDER_DATASET_SIZE:-}" ]] || common+=(--dataset-size "$RENDER_DATASET_SIZE")
gpu=0
for spec in $RENDER_SPECS; do
    IFS=: read -r agents checkpoint name <<< "$spec"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$BASELINES_ROOT/scripts/team/render.py" \
        --checkpoint "$checkpoint" --agents "$agents" "${common[@]}" \
        --output "$RENDER_OUTPUT/$name.gif" > "$RENDER_OUTPUT/$name.log" 2>&1 &
    gpu=$((gpu + 1))
done
wait
