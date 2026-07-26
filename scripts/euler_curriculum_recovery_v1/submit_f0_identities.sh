#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0}"
SCRIPT_ROOT="$RUN_ROOT/source/terra-baselines/scripts/euler_curriculum_recovery_v1"
TRAIN_SCRIPT="$SCRIPT_ROOT/run_f0_identity.sbatch"
EVAL_SCRIPT="$SCRIPT_ROOT/eval_f0_identity.sbatch"

test -f "$TRAIN_SCRIPT"
test -f "$EVAL_SCRIPT"
test -f "$RUN_ROOT/manifests/source_files.sha256"
test -f "$RUN_ROOT/manifests/bank_files.sha256"
test ! -e "$RUN_ROOT/submitted_jobs.txt"
mkdir -p "$RUN_ROOT/logs"

FOUNDATION_TRAIN_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --job-name=terra-f0-foundation \
        --export="ALL,ARM=foundation,RUN_ROOT=$RUN_ROOT" \
        "$TRAIN_SCRIPT"
)"
TRENCH_TRAIN_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --job-name=terra-f0-trench \
        --export="ALL,ARM=trench,RUN_ROOT=$RUN_ROOT" \
        "$TRAIN_SCRIPT"
)"
FOUNDATION_EVAL_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --dependency="afterok:$FOUNDATION_TRAIN_JOB" \
        --job-name=terra-f0-foundation-eval \
        --export="ALL,ARM=foundation,RUN_ROOT=$RUN_ROOT" \
        "$EVAL_SCRIPT"
)"
TRENCH_EVAL_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --dependency="afterok:$TRENCH_TRAIN_JOB" \
        --job-name=terra-f0-trench-eval \
        --export="ALL,ARM=trench,RUN_ROOT=$RUN_ROOT" \
        "$EVAL_SCRIPT"
)"

{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "foundation_train_job=$FOUNDATION_TRAIN_JOB"
    echo "foundation_eval_job=$FOUNDATION_EVAL_JOB"
    echo "trench_train_job=$TRENCH_TRAIN_JOB"
    echo "trench_eval_job=$TRENCH_EVAL_JOB"
} > "$RUN_ROOT/submitted_jobs.txt"

cat "$RUN_ROOT/submitted_jobs.txt"
