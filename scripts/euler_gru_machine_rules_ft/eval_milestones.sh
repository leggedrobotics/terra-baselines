#!/usr/bin/env bash
# usage: eval_milestones.sh DUG_CLEARANCE_M [UPDATE...]
# Milestone evaluation of the machine-rules fine-tune on the local RTX 4090
# (forward only), as the GRU campaign scored its milestones. For each update
# (default 100000 102500 105000 107500 110000; 100000 is the parent) it waits
# until the checkpoint exists in the Euler run directory, copies it to
# .artifacts/terra_gru_rules_ft_20260930/checkpoints/, and scores the
# development panel (608 maps and the 32-start panels) and the promotion panel
# with eval_local.sh under the fine-tune's rules: dig 4.0-6.5 m, dump 4.0-6.0
# m, chassis centred, clearance DUG_CLEARANCE_M. Outputs:
# .artifacts/terra_gru_rules_ft_20260930/evaluations/<run>/<panel>_uNNNNNN/.
# Run it in tmux; it polls Euler every 10 minutes.
set -euo pipefail
case "${1:-}" in
    0.57) TAG=c057 ;;
    0.6) TAG=c060 ;;
    *) echo "usage: eval_milestones.sh 0.57|0.6 [UPDATE...]" >&2; exit 2 ;;
esac
CLEARANCE=$1
shift
UPDATES=("$@")
[ ${#UPDATES[@]} -gt 0 ] || UPDATES=(100000 102500 105000 107500 110000)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="gru_rules_ft_${TAG}_s${TERRA_SEED:-20260930}"
REMOTE_HOST=euler-lterenzi
REMOTE_CHECKPOINTS="/cluster/project/rsl/lterenzi/terra_experiments/terra_gru_rules_ft_20260930/runs/$RUN_NAME/checkpoints"
LOCAL=/home/lorenzo/moleworks/.artifacts/terra_gru_rules_ft_20260930
PARENT=/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/checkpoints/gru_gen512_s20260923_update_100000.pkl
# The Euler runtime lock rebuilt locally (the release eval runtime sits on the T7 drive).
export TERRA_EVAL_PYTHON="${TERRA_EVAL_PYTHON:-$LOCAL/runtime_jax0433_cuda/bin/python}"
export DIG_MIN_RADIUS_M=4.0 DUMP_MAX_RADIUS_M=6.0 CENTRE_CHASSIS_ON_BASE=1 DUG_CLEARANCE_M="$CLEARANCE"
mkdir -p "$LOCAL/checkpoints" "$LOCAL/evaluations/$RUN_NAME"
for update in "${UPDATES[@]}"; do
    [[ "$update" =~ ^1[0-9]{5}$ ]]
    if [ "$update" = 100000 ]; then
        checkpoint="$PARENT"
    else
        name="$(printf '%s_update_%06d.pkl' "$RUN_NAME" "$update")"
        checkpoint="$LOCAL/checkpoints/$name"
        # The trainer writes checkpoints atomically; a present file is complete.
        until [ -f "$checkpoint" ]; do
            if ssh -o BatchMode=yes "$REMOTE_HOST" "test -f '$REMOTE_CHECKPOINTS/$name'"; then
                scp -q -o BatchMode=yes "$REMOTE_HOST:$REMOTE_CHECKPOINTS/$name" "$checkpoint.partial"
                mv -T "$checkpoint.partial" "$checkpoint"
            else
                sleep 600
            fi
        done
    fi
    for panel in development promotion; do
        out="$LOCAL/evaluations/$RUN_NAME/${panel}_u$(printf '%06d' "$update")"
        [ ! -e "$out/EVAL_DONE" ] || continue
        [ ! -e "$out" ] || { echo "$out exists without EVAL_DONE; move it aside and rerun" >&2; exit 3; }
        PANEL="$panel" bash "$REPO/scripts/euler_gru_generalist_512/eval_local.sh" "$checkpoint" "$out"
        echo "u$update $panel: $(tail -n 1 "$out/full608.log")"
    done
done
