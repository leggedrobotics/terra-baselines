#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.4h
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in \
    OUTPUT_DIR CLAIM_DIR BASELINES_ROOT BASELINES_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION PROTOCOL_TERRA_REVISION \
    ENVIRONMENT_PROTOCOL_SHA DISTANCE_SIDECAR_SHA \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_DATASET_SHA PROMOTION_MANIFEST_SHA \
    GRU_U44000 GRU_U44000_SHA FIXED_GRU_RESULT FIXED_GRU_RESULT_SHA \
    RUNTIME_FINGERPRINT_SHA \
    EVAL_SEED VENV TERRA_EULER_USER TERRA_EULER_HOME_ROOT \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
test "${SLURM_CPUS_PER_TASK:-}" = 8
test "$EVAL_SEED" = 20260807
test "$VENV" = /cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test -x "$VENV/bin/python"
test -f "$BASELINES_ROOT/scripts/gru_recurrence_probe_v1/run_probe.py"
test -d "$CLAIM_DIR"
test ! -e "$OUTPUT_DIR"

OWNER_TMP="$CLAIM_DIR/.job_id.$SLURM_JOB_ID.$$"
printf '%s\n' "$SLURM_JOB_ID" > "$OWNER_TMP"
if ! ln "$OWNER_TMP" "$CLAIM_DIR/job_id" 2>/dev/null; then
    test "$(cat "$CLAIM_DIR/job_id")" = "$SLURM_JOB_ID"
fi
rm "$OWNER_TMP"

HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$GRU_U44000" | awk '{print $1}')" = "$GRU_U44000_SHA"
test "$(sha256sum "$FIXED_GRU_RESULT" | awk '{print $1}')" = "$FIXED_GRU_RESULT_SHA"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
test "$(sha256sum "$BANK/evaluation/main/promotion/manifest.jsonl" | awk '{print $1}')" = "$PROMOTION_MANIFEST_SHA"

"$VENV/bin/python" - "$BANK/dataset.json" "$ENVIRONMENT_PROTOCOL_SHA" "$DISTANCE_SIDECAR_SHA" <<'PY'
import json
import sys

dataset = json.load(open(sys.argv[1]))
assert dataset["environment_protocol_sha256"] == sys.argv[2]
assert dataset["canonical_distance_sidecar_dataset_sha256"] == sys.argv[3]
assert dataset["distance_protocol_id"] == "obstacle_geodesic_8_physical_global_v1"
PY

module load stack/2024-06 cuda/12.1.1
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_cudnn_frontend=false
export JAX_PLATFORMS=cuda,cpu WANDB_MODE=disabled
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUNTIME_FINGERPRINT_JSON="$("$PYTHON" - <<'PY'
import json
import platform

import flax
import jax
import jaxlib
import numpy

value = {
    "flax": flax.__version__,
    "implementation": platform.python_implementation(),
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "numpy": numpy.__version__,
    "python": platform.python_version(),
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
)"
test "$(printf '%s' "$RUNTIME_FINGERPRINT_JSON" | sha256sum | awk '{print $1}')" = "$RUNTIME_FINGERPRINT_SHA"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"

"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
from jax import lax
from eval_mcts import EVAL_FORWARD_CHUNK

assert jax.local_device_count() == 1, jax.devices()
assert EVAL_FORWARD_CHUNK == 120, EVAL_FORWARD_CHUNK

def loss(x, kernel):
    value = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return jnp.mean(value * value)

x = jnp.ones((1, 16, 16, 4), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = jax.jit(jax.grad(loss))(x, kernel)
assert gradient.shape == x.shape
assert bool(jnp.all(jnp.isfinite(gradient)))
print(jax.devices(), float(jnp.linalg.norm(gradient)), EVAL_FORWARD_CHUNK)
PY

"$PYTHON" "$BASELINES_ROOT/scripts/gru_recurrence_probe_v1/run_probe.py" \
    --checkpoint "$GRU_U44000" \
    --fixed-eval "$FIXED_GRU_RESULT" \
    --bank-root "$BANK" \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
    --output-dir "$OUTPUT_DIR"

scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT_DIR/slurm_job.txt"
nvidia-smi -q > "$OUTPUT_DIR/nvidia_smi.txt"
"$PYTHON" - "$OUTPUT_DIR/receipt.json" "$GRU_U44000_SHA" "$FIXED_GRU_RESULT_SHA" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
assert receipt["schema"] == "terra_v8_gru_recurrence_probe_v1"
assert receipt["status"] == "passed"
assert receipt["checkpoint_sha256"] == sys.argv[2]
assert receipt["canonical_fixed_evaluation"]["sha256"] == sys.argv[3]
assert receipt["canonical_fixed_evaluation"]["normal_target_parity_passed"] is True
assert receipt["paired_execution"]["policy_forward_chunk"] == 120
assert receipt["horizon"] == 450
assert receipt["seed"] == 20260807
assert receipt["integrity"]["maximum_mass_residual"] == 0
PY

CONTRACT_TMP="$OUTPUT_DIR/.run_contract.env.tmp"
printf '%s\n' \
    "status=PASSED" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "environment_protocol_sha256=$ENVIRONMENT_PROTOCOL_SHA" \
    "distance_sidecar_sha256=$DISTANCE_SIDECAR_SHA" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_dataset_sha256=$BANK_DATASET_SHA" \
    "promotion_manifest_sha256=$PROMOTION_MANIFEST_SHA" \
    "checkpoint_sha256=$GRU_U44000_SHA" \
    "fixed_gru_result_sha256=$FIXED_GRU_RESULT_SHA" \
    "runtime_venv=$VENV" \
    "runtime_fingerprint_sha256=$RUNTIME_FINGERPRINT_SHA" \
    "runtime_fingerprint_json=$RUNTIME_FINGERPRINT_JSON" \
    "submission_claim=$CLAIM_DIR" \
    "submission_claim_job_id=$(cat "$CLAIM_DIR/job_id")" \
    "seed=$EVAL_SEED" \
    "horizon=450" \
    "forward_chunk=120" \
    "wandb_mode=disabled" > "$CONTRACT_TMP"
mv "$CONTRACT_TMP" "$OUTPUT_DIR/run_contract.env"
