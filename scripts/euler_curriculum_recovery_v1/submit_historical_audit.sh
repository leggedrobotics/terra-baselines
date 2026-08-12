#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725
AUDIT_ROOT="$RUN_ROOT/historical_audit"
SBATCH_SCRIPT="$AUDIT_ROOT/source/terra-baselines/scripts/euler_curriculum_recovery_v1/audit_historical_curriculum.sbatch"

test -f "$SBATCH_SCRIPT"
test ! -e "$AUDIT_ROOT/PREFLIGHT_PASSED"
test ! -e "$AUDIT_ROOT/DETERMINISTIC_PASSED"
test ! -e "$AUDIT_ROOT/SAMPLED_PASSED"

PREFLIGHT_JOB="$(
    sbatch \
        --parsable \
        --job-name=terra-recovery-D12-preflight \
        --export=ALL,AUDIT_KIND=preflight \
        "$SBATCH_SCRIPT"
)"
DETERMINISTIC_JOB="$(
    sbatch \
        --parsable \
        --dependency="afterok:$PREFLIGHT_JOB" \
        --job-name=terra-recovery-D12-deterministic \
        --export=ALL,AUDIT_KIND=deterministic \
        "$SBATCH_SCRIPT"
)"
SAMPLED_JOB="$(
    sbatch \
        --parsable \
        --dependency="afterok:$PREFLIGHT_JOB" \
        --job-name=terra-recovery-D2-sampled \
        --export=ALL,AUDIT_KIND=sampled \
        "$SBATCH_SCRIPT"
)"

{
    echo "preflight_job=$PREFLIGHT_JOB"
    echo "deterministic_job=$DETERMINISTIC_JOB"
    echo "sampled_job=$SAMPLED_JOB"
} > "$AUDIT_ROOT/submission_receipt.txt"

printf 'preflight=%s deterministic=%s sampled=%s\n' \
    "$PREFLIGHT_JOB" "$DETERMINISTIC_JOB" "$SAMPLED_JOB"
