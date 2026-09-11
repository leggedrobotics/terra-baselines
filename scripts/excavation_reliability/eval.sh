#!/usr/bin/env bash
# Development panel only: keep checkpoint, runtime source and decoder distinct.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: eval.sh CHECKPOINT OUTPUT.json' >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${TERRA_ROOT:?}" "${BANK_ROOT:?}"
DECODER="${DECODER:-greedy}"
EXTRA=()
case "$DECODER" in greedy) ;; sampled) EXTRA+=(--stochastic) ;; *) exit 2 ;; esac
export PYTHONPATH="$TERRA_ROOT:$REPO" EVAL_FORWARD_CHUNK=120 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
exec "${TERRA_PYTHON:-python}" -u "$REPO/eval_fixed_bank.py" \
    --checkpoint "$1" --bank-root "$BANK_ROOT" \
    --accepted-panel development --panel-family gate_main \
    --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 \
    --horizon 450 --seed 20260724 \
    --expect-completion-contract exact_visible_dump_v1 \
    --output "$2" "${EXTRA[@]}"
