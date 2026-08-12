#!/usr/bin/env bash
# Launch the named Train-96 plus capability-floor five-arm treatment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAMPAIGN_ID=terra_v6main_capfloor34_train96_v1
export EXPERIMENT_PREFIX=p6t96
export ENT_SCHEDULE_START=0.02
export ENT_SCHEDULE_END=0.005
export ENT_SCHEDULE_STEPS=10000
export SCREEN_UPDATES=4000
export ARMS_STRING="G-MEDIUM-ADAPTIVE-WARM G-MEDIUM-UNIFORM-WARM G-DEEP-UNIFORM-WARM F-MEDIUM-UNIFORM-WARM T-MEDIUM-UNIFORM-WARM"
export TRAIN_BANK_RELEASE_ID=terra_v6main_capfloor34_train96_v1
export TRAIN_MAPS_PER_CONDITION=96
export TRAIN_BANK_ARCHIVE_LOCAL="${TRAIN_BANK_ARCHIVE_LOCAL:-/home/lorenzo/moleworks/.artifacts/terra_v6main_capfloor34_train96_v1_20260803_a14d8302.tar.zst}"
export TRAIN_BANK_SHA="${TRAIN_BANK_SHA:-c19b27c0771eddb09b8c1f1f09655ec3bf9a84858b3f23b19cd6eda619db21cb}"
export TRAIN_BANK_DATASET_SHA="${TRAIN_BANK_DATASET_SHA:-2a1d74eec0ff8115b0922c9f82f14ddb1589aecec2d63f26d8461339b2f66f45}"
export DIAGNOSTIC_CONTROL_ARCHIVE_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_archive_20260802/control.tar.zst
export DIAGNOSTIC_CONTROL_SHA=f802681feade9057cdfa8e2c186f093540459831ae2d83a48b661aa96cbc4289

if [ "${SUBMIT:-0}" = 1 ] && [ -z "$TRAIN_BANK_ARCHIVE_LOCAL" ]; then
    echo "set TRAIN_BANK_ARCHIVE_LOCAL, TRAIN_BANK_SHA, and TRAIN_BANK_DATASET_SHA before SUBMIT=1" >&2
    exit 3
fi

exec "$SCRIPT_DIR/../euler_p5b_warm_v1/submit.sh" "$@"
