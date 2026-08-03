#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.4h
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-legacy-easy
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
umask 002

: "${LEGACY_EASY_LAUNCHER_DIR:?missing LEGACY_EASY_LAUNCHER_DIR}"
# Slurm copies this script into its spool directory. Resolve helper files from
# the immutable installed launcher path exported by submit.sh, never from
# BASH_SOURCE inside the allocation.
# shellcheck disable=SC1091
source "$LEGACY_EASY_LAUNCHER_DIR/common.sh"

: "${OUTPUT_ROOT:?missing OUTPUT_ROOT}"
: "${SLURM_JOB_ID:?run.sbatch must execute inside Slurm}"
: "${TMPDIR:?Euler allocation did not provide TMPDIR}"
[[ "$OUTPUT_ROOT" = /* ]] || legacy_easy_fail "OUTPUT_ROOT must be absolute"
[[ "$OUTPUT_ROOT" =~ ^[A-Za-z0-9_./:+-]+$ ]] ||
    legacy_easy_fail "OUTPUT_ROOT contains unsupported characters"
legacy_easy_require_canonical_new_path "$OUTPUT_ROOT" OUTPUT_ROOT
legacy_easy_require_canonical_path \
    "$LEGACY_EASY_LAUNCHER_DIR" dir LEGACY_EASY_LAUNCHER_DIR

legacy_easy_validate_static_inputs
legacy_easy_validate_policy_inputs \
    "$POLICY_LABEL" "$CHECKPOINT_PATH" "$CHECKPOINT_SHA256"
test "$(realpath -e "$LEGACY_EASY_LAUNCHER_DIR/../..")" = "$BASELINES_ROOT" ||
    legacy_easy_fail "run.sbatch is not executing from BASELINES_ROOT"

test ! -e "$OUTPUT_ROOT" || legacy_easy_fail "output already exists: $OUTPUT_ROOT"
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"

module load stack/2024-06 cuda/12.1.1
SITE_PACKAGES="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON_BIN" - "$TERRA_ROOT" "$BASELINES_ROOT" <<'PY'
import pathlib
import sys

import jax
import terra
from terra.state import CORRECTED_DENSE_CONTRACT
from utils import explicit_episode_bank

expected = [pathlib.Path(value).resolve() for value in sys.argv[1:]]
actual = [
    pathlib.Path(terra.__file__).resolve().parents[1],
    pathlib.Path(explicit_episode_bank.__file__).resolve().parents[1],
]
assert actual == expected, (actual, expected)
assert jax.default_backend() == "gpu", jax.default_backend()
assert len(jax.devices()) == 1, jax.devices()
assert str(CORRECTED_DENSE_CONTRACT) == "exact_visible_dump_v1"
PY

SAMPLED_SEED=20260803
run_eval() {
    local panel="$1"
    local mode="$2"
    local output="$OUTPUT_ROOT/${panel}_${mode}.json"
    local log="$OUTPUT_ROOT/${panel}_${mode}.log"
    local -a command=(
        "$PYTHON_BIN" -u -s "$BASELINES_ROOT/eval_fixed_bank.py"
        --checkpoint "$CHECKPOINT_PATH"
        --bank-root "$EPISODE_BANK_ROOT"
        --explicit-episode-panel "$panel"
        --terra-revision "$TERRA_REVISION"
        --horizon 450
        --seed "$SAMPLED_SEED"
        --expect-completion-contract exact_visible_dump_v1
        --output "$output"
    )
    if [ "$mode" = sampled ]; then
        command+=(--stochastic)
    fi
    "${command[@]}" >"$log" 2>&1
    "$PYTHON_BIN" "$LEGACY_EASY_LAUNCHER_DIR/verify_result.py" "$output" \
        --panel "$panel" \
        --mode "$mode" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        --episode-bank-sha256 "$EPISODE_BANK_JSON_SHA256"
    tail -n 1 "$log"
}

run_eval promotion deterministic
run_eval promotion sampled
run_eval development deterministic
run_eval development sampled

cat >"$OUTPUT_ROOT/receipt.env" <<EOF
schema=terra_legacy_easy_v1_euler_eval_receipt_v1
status=PASSED
policy_label=$POLICY_LABEL
checkpoint_path=$CHECKPOINT_PATH
checkpoint_sha256=$CHECKPOINT_SHA256
baselines_root=$BASELINES_ROOT
baselines_revision=$BASELINES_REVISION
terra_root=$TERRA_ROOT
terra_revision=$TERRA_REVISION
episode_bank_root=$EPISODE_BANK_ROOT
episode_bank_json_sha256=$EPISODE_BANK_JSON_SHA256
episode_bank_files_sha256=$EPISODE_BANK_FILES_SHA256
panels=promotion,development
policy_modes=deterministic,sampled
sampled_seed=$SAMPLED_SEED
horizon=450
completion_contract=exact_visible_dump_v1
slurm_job_id=$SLURM_JOB_ID
slurm_job_name=${SLURM_JOB_NAME:-unknown}
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
EOF

OUTPUT_MANIFEST_TMP="$TMPDIR/legacy-easy-output-$SLURM_JOB_ID.sha256"
(
    cd "$OUTPUT_ROOT"
    find . -type f ! -name files.sha256 -print0 |
        sort -z |
        xargs -0 sha256sum >"$OUTPUT_MANIFEST_TMP"
)
mv "$OUTPUT_MANIFEST_TMP" "$OUTPUT_ROOT/files.sha256"
echo "Legacy-Easy evaluation passed: $OUTPUT_ROOT"
