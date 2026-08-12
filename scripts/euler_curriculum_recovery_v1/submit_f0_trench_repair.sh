#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?set the immutable F0R run root}"
TREATMENT=corrected_dense_v1_trench_absolute_off
SCRIPT_ROOT="$RUN_ROOT/source/terra-baselines/scripts/euler_curriculum_recovery_v1"
TRAIN_SCRIPT="$SCRIPT_ROOT/run_f0_identity.sbatch"
EVAL_SCRIPT="$SCRIPT_ROOT/eval_f0_identity.sbatch"

test -x "$TRAIN_SCRIPT"
test -x "$EVAL_SCRIPT"
test -f "$RUN_ROOT/manifests/source_files.sha256"
test -f "$RUN_ROOT/manifests/bank_files.sha256"
test ! -e "$RUN_ROOT/submitted_jobs.txt"
mkdir -p "$RUN_ROOT/logs"

TRAIN_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --job-name=terra-f0r-trench \
        --export="ALL,ARM=trench,F0_TREATMENT=$TREATMENT,RUN_ROOT=$RUN_ROOT" \
        "$TRAIN_SCRIPT"
)"
EVAL_JOB="$(
    sbatch --parsable \
        --output="$RUN_ROOT/logs/%x_%j.out" \
        --dependency="afterok:$TRAIN_JOB" \
        --job-name=terra-f0r-trench-eval \
        --export="ALL,ARM=trench,F0_TREATMENT=$TREATMENT,RUN_ROOT=$RUN_ROOT" \
        "$EVAL_SCRIPT"
)"

{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "treatment=$TREATMENT"
    echo "trench_train_job=$TRAIN_JOB"
    echo "trench_eval_job=$EVAL_JOB"
} > "$RUN_ROOT/submitted_jobs.txt"

cat "$RUN_ROOT/submitted_jobs.txt"
