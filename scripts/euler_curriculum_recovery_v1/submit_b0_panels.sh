#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1}"
B0_UPDATES="${B0_UPDATES:-500}"
EXCLUDE_NODES="${EXCLUDE_NODES:-eu-g6-064}"
SCRIPT_ROOT="$RUN_ROOT/source/terra-baselines/scripts/euler_curriculum_recovery_v1"
TRAIN_SCRIPT="$SCRIPT_ROOT/run_b0_panel.sbatch"
EVAL_SCRIPT="$SCRIPT_ROOT/eval_b0_panel.sbatch"
RECEIPT="$RUN_ROOT/submitted_jobs.txt"
PENDING_RECEIPT="$RUN_ROOT/submitted_jobs.pending.$BASHPID"
case "$B0_UPDATES" in
    500 | 1000 | 2000)
        TRAIN_TIME_LIMIT=08:00:00
        ;;
    5000)
        TRAIN_TIME_LIMIT=16:00:00
        ;;
    *)
        echo "Unsupported bounded B0 update target: $B0_UPDATES" >&2
        exit 2
        ;;
esac
if (($#)); then
    PANELS=("$@")
else
    PANELS=(
        foundation_geometry
        foundation_distance
        trench_distance
        trench_side
        trench_topology
    )
fi
for PANEL in "${PANELS[@]}"; do
    case "$PANEL" in
        foundation_geometry | foundation_distance | trench_distance | trench_side | trench_topology) ;;
        *)
            echo "Unsupported B0 panel: $PANEL" >&2
            exit 3
            ;;
    esac
done

SBATCH_EXCLUDE_ARGS=()
if [[ -n "$EXCLUDE_NODES" ]]; then
    SBATCH_EXCLUDE_ARGS=(--exclude="$EXCLUDE_NODES")
fi

test -x "$TRAIN_SCRIPT"
test -x "$EVAL_SCRIPT"
test -f "$RUN_ROOT/manifests/source_files.sha256"
test -f "$RUN_ROOT/manifests/bank_files.sha256"
test ! -e "$RECEIPT"
test ! -e "$PENDING_RECEIPT"
mkdir -p "$RUN_ROOT/logs"

trap 'echo "Partial B0 submission receipt: $PENDING_RECEIPT" >&2' ERR
{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "b0_updates=$B0_UPDATES"
    echo "train_time_limit=$TRAIN_TIME_LIMIT"
    echo "excluded_nodes=${EXCLUDE_NODES:-none}"
    echo "panels=${PANELS[*]}"
} > "$PENDING_RECEIPT"
for PANEL in "${PANELS[@]}"; do
    TRAIN_JOB="$(
        sbatch --parsable \
            "${SBATCH_EXCLUDE_ARGS[@]}" \
            --output="$RUN_ROOT/logs/%x_%j.out" \
            --job-name="terra-b0-${PANEL//_/-}" \
            --time="$TRAIN_TIME_LIMIT" \
            --export="ALL,PANEL=$PANEL,RUN_ROOT=$RUN_ROOT,B0_UPDATES=$B0_UPDATES" \
            "$TRAIN_SCRIPT"
    )"
    echo "${PANEL}_train_job=$TRAIN_JOB" >> "$PENDING_RECEIPT"

    EVAL_JOB="$(
        sbatch --parsable \
            "${SBATCH_EXCLUDE_ARGS[@]}" \
            --output="$RUN_ROOT/logs/%x_%j.out" \
            --dependency="afterok:$TRAIN_JOB" \
            --job-name="terra-b0-${PANEL//_/-}-eval" \
            --export="ALL,PANEL=$PANEL,RUN_ROOT=$RUN_ROOT,B0_UPDATES=$B0_UPDATES" \
            "$EVAL_SCRIPT"
    )"
    echo "${PANEL}_eval_job=$EVAL_JOB" >> "$PENDING_RECEIPT"
done
mv "$PENDING_RECEIPT" "$RECEIPT"
trap - ERR
cat "$RECEIPT"
