#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <completed-or-running-E8-slurm-job-id>" >&2
    exit 2
fi
E8_JOB_ID="$1"
RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
SCRIPT_ROOT="$RUN_ROOT/source/terra-baselines/scripts/euler_map_curriculum_v1"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/eval"
test -f "$SCRIPT_ROOT/gate_map_curriculum_v1.sbatch"
test -f "$SCRIPT_ROOT/run_map_curriculum_v1.sbatch"
test -f "$SCRIPT_ROOT/eval_map_curriculum_v1.sbatch"

E8_STATE="$(
    sacct -n -X -j "$E8_JOB_ID" --format=State |
        awk 'NF {print $1; exit}'
)"
if [ -z "$E8_STATE" ]; then
    E8_STATE="$(
        squeue -h -j "$E8_JOB_ID" -o '%T' |
            awk 'NF {print $1; exit}'
    )"
fi
case "$E8_STATE" in
    RUNNING|PENDING|COMPLETED) ;;
    *)
        echo "E8 job $E8_JOB_ID is not launchable: state=$E8_STATE" >&2
        exit 3
        ;;
esac

GATE_JOB="$(
    sbatch --parsable \
        --dependency="afterok:$E8_JOB_ID" \
        "$SCRIPT_ROOT/gate_map_curriculum_v1.sbatch"
)"
FLAT_JOB="$(
    sbatch --parsable \
        --dependency="afterok:$GATE_JOB" \
        --job-name=terra-map-v1-flat \
        --export=ALL,ARM=flat \
        "$SCRIPT_ROOT/run_map_curriculum_v1.sbatch"
)"
STAGED_JOB="$(
    sbatch --parsable \
        --dependency="afterok:$GATE_JOB" \
        --job-name=terra-map-v1-staged \
        --export=ALL,ARM=staged \
        "$SCRIPT_ROOT/run_map_curriculum_v1.sbatch"
)"
FLAT_EVAL_JOB="$(
    sbatch --parsable \
        --dependency="afterok:$FLAT_JOB" \
        --job-name=terra-map-v1-flat-eval \
        --export=ALL,ARM=flat \
        "$SCRIPT_ROOT/eval_map_curriculum_v1.sbatch"
)"
STAGED_EVAL_JOB="$(
    sbatch --parsable \
        --dependency="afterok:$STAGED_JOB" \
        --job-name=terra-map-v1-staged-eval \
        --export=ALL,ARM=staged \
        "$SCRIPT_ROOT/eval_map_curriculum_v1.sbatch"
)"

{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "e8_job=$E8_JOB_ID"
    echo "e8_state_at_submission=$E8_STATE"
    echo "gate_job=$GATE_JOB"
    echo "flat_job=$FLAT_JOB"
    echo "staged_job=$STAGED_JOB"
    echo "flat_eval_job=$FLAT_EVAL_JOB"
    echo "staged_eval_job=$STAGED_EVAL_JOB"
} > "$RUN_ROOT/submitted_jobs.txt"

cat "$RUN_ROOT/submitted_jobs.txt"
