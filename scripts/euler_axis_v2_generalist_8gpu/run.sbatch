#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=rtx_4090:8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in RUN_ROLE TARGET_UPDATE RESUME_CHECKPOINT RESUME_UPDATE RUN_DIR \
    RUN_NAME BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT \
    RUNTIME_TERRA_REVISION PROTOCOL_SHA SEED VENV TERRA_EULER_USER \
    TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT BANK_ARCHIVE \
    BANK_ARCHIVE_SHA BANK_DATASET_SHA BANK_BUILD_SHA BANK_AUDIT_SHA \
    SOURCE_REGISTRY_SHA DISTANCE_ARTIFACT_SHA PARTIAL_ARCHIVE \
    PARTIAL_ARCHIVE_SHA PARTIAL_BANK_SHA PARTIAL_ALIGNMENT_AUDIT_SHA \
    EXPECTED_PARTIAL_CONDITIONS EXPECTED_PARTIAL_TRIPLETS \
    EXPECTED_PARTIAL_SIDECARS EXPECTED_TRENCH_AUDIT_SIDECARS \
    EXPECTED_PARAMETERS ALGORITHM_DENYLIST_SHA AUTOTUNE_CACHE \
    AUTOTUNE_CACHE_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$RUN_ROLE:$TARGET_UPDATE:$RESUME_CHECKPOINT:$RESUME_UPDATE" in
    bootstrap:1:none:0)
        test "$AUTOTUNE_CACHE" = none
        test "$AUTOTUNE_CACHE_SHA" = none
        WANDB_MODE=offline
        WANDB_RESUME=never
        ;;
    smoke:5:none:0)
        test "$AUTOTUNE_CACHE" != none
        WANDB_MODE=offline
        WANDB_RESUME=never
        ;;
    phase1:75000:none:0)
        test "$AUTOTUNE_CACHE" != none
        WANDB_MODE=online
        WANDB_RESUME=never
        ;;
    phase2:100000:*:75000)
        test "$RESUME_CHECKPOINT" != none
        test "$AUTOTUNE_CACHE" != none
        WANDB_MODE=online
        WANDB_RESUME=must
        ;;
    *) echo "unsupported run contract" >&2; exit 2 ;;
esac
case "$RUN_ROLE:${SLURM_JOB_PARTITION:-}" in
    bootstrap:gpuhe.4h|smoke:gpuhe.4h|phase1:gpuhe.120h|phase2:gpuhe.120h) ;;
    *) echo "unexpected partition for $RUN_ROLE" >&2; exit 2 ;;
esac

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_ARCHIVE_SHA"

DENYLIST="$BASELINES_ROOT/scripts/euler_axis_v2_generalist_8gpu/hlo_algorithm_denylist.pbtxt"
test "$(sha256sum "$DENYLIST" | awk '{print $1}')" = "$ALGORITHM_DENYLIST_SHA"
if [ "$AUTOTUNE_CACHE" != none ]; then
    test -f "$AUTOTUNE_CACHE"
    test "$(sha256sum "$AUTOTUNE_CACHE" | awk '{print $1}')" = "$AUTOTUNE_CACHE_SHA"
fi

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb" "$TMPDIR/full" "$TMPDIR/partial"
test ! -e "$RUN_DIR/completion.env"
test -z "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -print -quit)"
if [ "$RESUME_CHECKPOINT" != none ]; then
    test -f "$RESUME_CHECKPOINT"
fi
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR/full" bank
tar --zstd -xf "$PARTIAL_ARCHIVE" -C "$TMPDIR/partial" partial_bank
BANK="$TMPDIR/full/bank"
PARTIAL_BANK="$TMPDIR/partial/partial_bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
test "$(sha256sum "$BANK/build_receipt.json" | awk '{print $1}')" = "$BANK_BUILD_SHA"
test "$(sha256sum "$BANK/audit_receipt.json" | awk '{print $1}')" = "$BANK_AUDIT_SHA"
test "$(sha256sum "$PARTIAL_BANK/trench_alignment_audit.json" | awk '{print $1}')" = "$PARTIAL_ALIGNMENT_AUDIT_SHA"

module load stack/2024-06 cuda/12.1.1 eth_proxy
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ "$AUTOTUNE_CACHE" = none ]; then
    export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_algorithm_denylist_path=$DENYLIST --xla_gpu_dump_autotune_results_to=$RUN_DIR/autotune_results.pbtxt"
else
    export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_algorithm_denylist_path=$DENYLIST --xla_gpu_load_autotune_results_from=$AUTOTUNE_CACHE --xla_gpu_dump_autotune_results_to=$RUN_DIR/autotune_results.pbtxt"
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 8
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done

NUM_DEVICES=8 "$PYTHON" - <<'PY'
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

devices = int(os.environ["NUM_DEVICES"])
assert len(jax.devices()) == devices, jax.devices()

def loss(x, kernel):
    y = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ).astype(jnp.float32)
    return lax.psum(jnp.mean(y * y), "devices")

kernel_grad = jax.pmap(jax.grad(loss, argnums=1), axis_name="devices", in_axes=(0, None))
for spatial, channels in ((16, 64), (8, 96)):
    x = jnp.ones((devices, 256, spatial, spatial, channels), dtype=jnp.bfloat16)
    kernel = jnp.ones((3, 3, channels, channels), dtype=jnp.bfloat16)
    gradient = kernel_grad(x, kernel)
    assert gradient.shape == (devices, 3, 3, channels, channels)
    assert bool(jnp.all(jnp.isfinite(gradient)))

x = jnp.ones((devices, 256, 8, 8, 96), dtype=jnp.bfloat16)
kernel = jnp.ones((1, 1, 96, 32), dtype=jnp.bfloat16)
gradient = kernel_grad(x, kernel)
assert gradient.shape == (devices, 1, 1, 96, 32)
assert bool(jnp.all(jnp.isfinite(gradient)))
np.testing.assert_allclose(
    np.asarray(gradient[0], dtype=np.float32) / devices,
    6.0,
    rtol=2e-2,
    atol=2e-2,
)
PY

PROTOCOL_TERRA_REVISION="$RUNTIME_TERRA_REVISION" PROTOCOL_SHA="$PROTOCOL_SHA" \
SOURCE_REGISTRY_SHA="$SOURCE_REGISTRY_SHA" DISTANCE_ARTIFACT_SHA="$DISTANCE_ARTIFACT_SHA" \
BANK="$BANK" PARTIAL_BANK="$PARTIAL_BANK" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_PARTIAL_CONDITIONS" \
EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_PARTIAL_TRIPLETS" \
EXPECTED_PARTIAL_SIDECARS="$EXPECTED_PARTIAL_SIDECARS" \
EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_TRENCH_AUDIT_SIDECARS" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from terra.maps_buffer import partial_reset_bank_sha256
from utils.accepted_bank import load_accepted_bank, validate_staged_training_bank

root = Path(os.environ["BANK"])
assert validate_staged_training_bank(
    root,
    expected_maps_per_condition=96,
    expected_release_id="terra_axis_v2_v6_constraints_v7_foundations_train96_v1",
) == 96
bank = load_accepted_bank(
    root,
    "G-UNIFORM",
    os.environ["PROTOCOL_TERRA_REVISION"],
    curriculum_stage="full",
    sampler_profile="continuous_banded_v3",
    condition_profile="axis_v2_40_v1",
)
assert len(bank.levels) == 40
assert sum(level.family == "foundation" for level in bank.levels) == 25
assert sum(level.family == "trench" for level in bank.levels) == 15
assert all(panel.condition_count == 38 for panel in bank.evaluation_panels)
assert all(panel.condition_count == 2 for panel in bank.capability_floor_evaluation_panels)
assert tuple(bank.curriculum_depths).count(0) == 2
assert tuple(bank.curriculum_depths).count(1) == 6
assert tuple(bank.curriculum_depths).count(2) == 32
assert bank.environment_protocol_sha256 == os.environ["PROTOCOL_SHA"]
assert bank.source_registry_sha256 == os.environ["SOURCE_REGISTRY_SHA"]
index = json.loads((root / "dataset.json").read_text())
assert index["canonical_distance_artifact_sha256"] == os.environ["DISTANCE_ARTIFACT_SHA"]

partial_root = Path(os.environ["PARTIAL_BANK"])
assert partial_reset_bank_sha256(partial_root) == os.environ["PARTIAL_BANK_SHA"]
partial = json.loads((partial_root / "partial_reset_bank.json").read_text())
assert partial["canonical_loader_registry_sha256"] == hashlib.sha256(
    (root / "dataset.json").read_bytes()
).hexdigest()
assert len(partial["supported_maps_paths"]) == int(os.environ["EXPECTED_PARTIAL_CONDITIONS"])
rows = []
triplets = set()
for maps_path in partial["supported_maps_paths"]:
    for line in (partial_root / maps_path / "partial_completion_manifest.jsonl").read_text().splitlines():
        if line:
            row = json.loads(line)
            rows.append(row)
            triplets.add((maps_path, int(row["source_index"])))
assert len(rows) == int(os.environ["EXPECTED_PARTIAL_SIDECARS"])
assert len(triplets) == int(os.environ["EXPECTED_PARTIAL_TRIPLETS"])
audit = json.loads((partial_root / "trench_alignment_audit.json").read_text())
assert audit["schema"] == "terra_partial_trench_alignment_audit_v2"
assert audit["accepted"] is True and audit["failed_sidecars"] == 0
assert audit["partial_reset_bank_sha256"] == os.environ["PARTIAL_BANK_SHA"]
assert audit["audited_sidecars"] == int(os.environ["EXPECTED_TRENCH_AUDIT_SIDECARS"])
PY

export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_RESUME
export WANDB_RUN_ID="axis_v2_generalist_${BASELINES_REVISION:0:10}_s${SEED}"
if [ "$RUN_ROLE" = bootstrap ] || [ "$RUN_ROLE" = smoke ]; then
    export WANDB_RUN_ID="${WANDB_RUN_ID}_${RUN_ROLE}_${SLURM_JOB_ID}"
fi
export WANDB_DIR="$RUN_DIR/wandb"
if [ "$WANDB_MODE" = online ]; then
    "$PYTHON" -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"

printf '%s\n' \
    "status=RUNNING" \
    "run_role=$RUN_ROLE" \
    "target_update=$TARGET_UPDATE" \
    "resume_update=$RESUME_UPDATE" \
    "resume_checkpoint=$RESUME_CHECKPOINT" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "environment_protocol_sha256=$PROTOCOL_SHA" \
    "source_registry_sha256=$SOURCE_REGISTRY_SHA" \
    "distance_artifact_sha256=$DISTANCE_ARTIFACT_SHA" \
    "condition_profile=axis_v2_40_v1" \
    "training_conditions=40" \
    "foundation_conditions=25" \
    "trench_conditions=15" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "num_devices=8" \
    "num_envs_per_device=256" \
    "global_rollout_transitions=65536" \
    "xla_gpu_algorithm_denylist_sha256=$ALGORITHM_DENYLIST_SHA" \
    "xla_gpu_autotune_cache_sha256=$AUTOTUNE_CACHE_SHA" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PYTHON_BIN="$PYTHON" MACHINE=euler
export PROTOCOL_TERRA_REVISION="$RUNTIME_TERRA_REVISION"
if [ "$RUN_ROLE" = bootstrap ] || [ "$RUN_ROLE" = smoke ]; then
    export CHECKPOINT_INTERVAL=1 LOG_EVAL_INTERVAL=0
else
    export CHECKPOINT_INTERVAL=500 LOG_EVAL_INTERVAL=100
fi
"$BASELINES_ROOT/scripts/run_axis_v2_generalist_8gpu.sh" \
    "$BANK" "$PARTIAL_BANK" "$RUN_NAME" "$TARGET_UPDATE" "$RUN_DIR" \
    "$DISTANCE_ARTIFACT_SHA" "$RESUME_CHECKPOINT"

test -s "$RUN_DIR/autotune_results.pbtxt"
FINAL_CHECKPOINT="$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
test -f "$FINAL_CHECKPOINT"
FINAL_CHECKPOINT="$FINAL_CHECKPOINT" TARGET_UPDATE="$TARGET_UPDATE" \
EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
"$PYTHON" - <<'PY'
import json
import os
import jax
import numpy as np
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(os.environ["FINAL_CHECKPOINT"])
assert checkpoint["next_update"] == int(os.environ["TARGET_UPDATE"])
model = jax.tree_util.tree_leaves(checkpoint["model"])
optimizer = jax.tree_util.tree_leaves(checkpoint["optimizer_state"])
assert sum(int(np.asarray(x).size) for x in model) == int(os.environ["EXPECTED_PARAMETERS"])
assert all(np.all(np.isfinite(np.asarray(x))) for x in model)
assert all(np.all(np.isfinite(np.asarray(x))) for x in optimizer)
config = checkpoint["train_config"]
assert config.accepted_bank.condition_profile == "axis_v2_40_v1"
assert config.partial_reset_bank_sha256 == os.environ["PARTIAL_BANK_SHA"]
assert config.enforce_trench_dig_alignment is True
assert config.trench_alignment_observation is True
open(os.path.join(os.path.dirname(os.environ["FINAL_CHECKPOINT"]), "..", "checkpoint_validation.json"), "w").write(
    json.dumps(
        {
            "status": "passed",
            "next_update": checkpoint["next_update"],
            "parameter_count": sum(int(np.asarray(x).size) for x in model),
            "model_finite": True,
            "optimizer_finite": True,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
)
PY

if [ "$RUN_ROLE" = smoke ]; then
    THROUGHPUT_LOG="$RUN_DIR/slurm_${SLURM_JOB_ID}.out" \
    THROUGHPUT_RECEIPT="$RUN_DIR/throughput_validation.json" "$PYTHON" - <<'PY'
import json
import os
import re
import statistics

text = open(os.environ["THROUGHPUT_LOG"], errors="replace").read()
values = [float(value) for value in re.findall(r"Steps/s:\s*([0-9]+(?:\.[0-9]+)?)", text)]
steady = values[2:]
if len(steady) < 3:
    raise RuntimeError(f"expected three post-compile samples, got {values}")
median = statistics.median(steady)
receipt = {
    "schema": "terra_axis_v2_8gpu_throughput_smoke_v1",
    "num_devices": 8,
    "num_envs_per_device": 256,
    "all_steps_per_second": values,
    "post_compile_steps_per_second": steady,
    "post_compile_median_steps_per_second": median,
    "minimum_required_steps_per_second": 12000.0,
    "passed": median >= 12000.0,
}
open(os.environ["THROUGHPUT_RECEIPT"], "w").write(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
if not receipt["passed"]:
    raise RuntimeError(f"throughput gate failed: {median:.1f} steps/s")
PY
fi

printf '%s\n' \
    "status=COMPLETED" \
    "target_update=$TARGET_UPDATE" \
    "checkpoint=$FINAL_CHECKPOINT" \
    "autotune_results=$RUN_DIR/autotune_results.pbtxt" > "$RUN_DIR/completion.env"
