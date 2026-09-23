#!/usr/bin/env bash
# One training allocation (DEVICES GH200s, default a whole node). Continues
# from the newest checkpoint of this experiment, otherwise starts from the
# parent. EXTRA_ARGS go to run.py verbatim (e.g. --types 2 --scratch).
set -euo pipefail
: "${TERRA_ROOT:?}" "${BASELINES_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${PARENT_CHECKPOINT:?}"
: "${TRAIN_BANK_ROOT:?}" "${TARGET_UPDATES:?}" "${SLURM_JOB_ID:?}"
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
OUTPUT="$EXPERIMENT_ROOT/segments/$SLURM_JOB_ID"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"
DEVICES="${DEVICES:-4}"
python "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices "$DEVICES" > "$OUTPUT/preflight.log" 2>&1
python -c 'import jax, sys; d = jax.local_devices(); print(d); assert len(d) == int(sys.argv[1]) and all(sys.argv[2] in x.device_kind for x in d)' \
    "$DEVICES" "${GPU_KIND:-GH200}" > "$OUTPUT/gpu-layout.log" 2>&1
LATEST="$(python - "$EXPERIMENT_ROOT" <<'PY'
from pathlib import Path
import sys
paths = list(Path(sys.argv[1]).glob("segments/*/checkpoints/*_update_*.pkl"))
print(max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1])) if paths else "")
PY
)"
RESUME=()
[[ -z "$LATEST" ]] || RESUME=(--resume "$LATEST")
printf '%s\n' "${LATEST:-$PARENT_CHECKPOINT}" > "$OUTPUT/start_checkpoint.txt"
unset WANDB_RUN_ID WANDB_RESUME
exec python -u "$BASELINES_ROOT/scripts/team/run.py" \
    --checkpoint "$PARENT_CHECKPOINT" --bank "$TRAIN_BANK_ROOT" \
    --output "$OUTPUT" --name "${RUN_NAME:-team-excavators}" \
    --agents "${TEAM_AGENTS:-2}" --devices "$DEVICES" --envs "${ENVS_PER_DEVICE:-256}" \
    --updates "$TARGET_UPDATES" "${RESUME[@]}" ${EXTRA_ARGS:-} > "$OUTPUT/training.log" 2>&1
