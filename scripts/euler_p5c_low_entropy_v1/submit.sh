#!/usr/bin/env bash
# Launch the minimal low-entropy P5c star through the tested P5b machinery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAMPAIGN_ID=p5c_low_entropy_v1
export EXPERIMENT_PREFIX=p5c
export ENT_SCHEDULE_START=0.02
export ENT_SCHEDULE_END=0.005
export ENT_SCHEDULE_STEPS=10000
export SCREEN_UPDATES=4000
export ARMS_STRING="G-MEDIUM-ADAPTIVE-WARM G-MEDIUM-UNIFORM-WARM G-DEEP-UNIFORM-WARM F-MEDIUM-UNIFORM-WARM T-MEDIUM-UNIFORM-WARM"
export DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_archive_20260802/control.tar.zst
export DIAGNOSTIC_CONTROL_SHA=f802681feade9057cdfa8e2c186f093540459831ae2d83a48b661aa96cbc4289

exec "$SCRIPT_DIR/../euler_p5b_warm_v1/submit.sh" "$@"
