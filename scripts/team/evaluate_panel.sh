#!/usr/bin/env bash
# Evaluate checkpoints as several team sizes / sampling modes in parallel, one
# process per GPU. EVAL_SPECS lists "agents:mode[:idle]" entries, mode sampled
# or greedy; ":idle" keeps every agent but slot 0 doing nothing (default:
# single and two-agent team, sampled and greedy). EVAL_CHECKPOINT may list
# several checkpoints; outputs are then prefixed with the checkpoint name.
# Optional: EVAL_TYPES (e.g. 2 for a skid steer; overrides the agent count),
# EVAL_MAPS_PATH (bank subdirectory), EVAL_DATASET_SIZE, and EVAL_SEQUENTIAL=1
# to run every evaluation in turn on GPU 0.
set -euo pipefail
: "${TERRA_ROOT:?}" "${BASELINES_ROOT:?}" "${EVAL_CHECKPOINT:?}" "${EVAL_BANK_ROOT:?}" "${EVAL_OUTPUT:?}"
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
mkdir -p "$EVAL_OUTPUT"
read -r -a checkpoints <<< "$EVAL_CHECKPOINT"
common=(--bank "$EVAL_BANK_ROOT" --envs "${EVAL_ENVS:-512}" --seed "${EVAL_SEED:-0}")
[[ -z "${EVAL_TYPES:-}" ]] || common+=(--types "$EVAL_TYPES")
[[ -z "${EVAL_MAPS_PATH:-}" ]] || common+=(--maps-path "$EVAL_MAPS_PATH")
[[ -z "${EVAL_DATASET_SIZE:-}" ]] || common+=(--dataset-size "$EVAL_DATASET_SIZE")
gpu=0
for checkpoint in "${checkpoints[@]}"; do
    prefix=""
    (( ${#checkpoints[@]} == 1 )) || prefix="$(basename "$checkpoint" .pkl)_"
    for spec in ${EVAL_SPECS:-1:sampled 2:sampled 1:greedy 2:greedy}; do
        IFS=: read -r agents mode idle <<< "$spec"
        flags=()
        [[ "$mode" == greedy ]] && flags+=(--greedy)
        [[ "${idle:-}" == idle ]] && flags+=(--idle-teammates)
        name="${prefix}a${agents}_${mode}${idle:+_$idle}"
        if [[ "${EVAL_SEQUENTIAL:-0}" == 1 ]]; then
            python -u "$BASELINES_ROOT/scripts/team/evaluate.py" \
                --checkpoint "$checkpoint" --agents "$agents" "${common[@]}" \
                "${flags[@]}" --output "$EVAL_OUTPUT/$name.json" \
                > "$EVAL_OUTPUT/$name.log" 2>&1 || echo "evaluation $name failed" >&2
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" python -u "$BASELINES_ROOT/scripts/team/evaluate.py" \
            --checkpoint "$checkpoint" --agents "$agents" "${common[@]}" \
            "${flags[@]}" --output "$EVAL_OUTPUT/$name.json" \
            > "$EVAL_OUTPUT/$name.log" 2>&1 &
        gpu=$((gpu + 1))
    done
done
wait
