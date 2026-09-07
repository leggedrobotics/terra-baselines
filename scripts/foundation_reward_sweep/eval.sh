#!/usr/bin/env bash
# Full initial episodes, deterministic policy, unchanged 450-step horizon.
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: BANK_ROOT=... bash eval.sh CHECKPOINT OUTPUT_JSON validation|test" >&2; exit 2; }
: "${BANK_ROOT:?Set the same held-out bank root for every arm}"
case "$3" in validation|test) ;; *) echo "split must be validation or test" >&2; exit 2 ;; esac
SWEEP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${TERRA_ROOT:-$(dirname "$SWEEP_REPO")/terra}:$SWEEP_REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export EVAL_FORWARD_CHUNK="${EVAL_FORWARD_CHUNK:-32}"
exec "${TERRA_PYTHON:-python}" -u "$SWEEP_REPO/eval_fixed_bank.py" \
    --checkpoint "$1" --bank-root "$BANK_ROOT" --split "$3" --strata all \
    --horizon 450 --seed 20260907 --expect-completion-contract exact_visible_dump_v1 \
    --output "$2"
