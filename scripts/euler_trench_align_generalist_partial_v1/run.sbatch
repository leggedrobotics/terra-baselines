#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in RUN_ROLE TARGET_UPDATE RESUME_CHECKPOINT RUN_DIR RUN_NAME \
    BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION \
    PROTOCOL_TERRA_REVISION PROTOCOL_SHA SEED VENV TERRA_EULER_USER \
    TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT BANK_ARCHIVE \
    BANK_ARCHIVE_SHA BANK_DATASET_SHA BANK_DERIVATION_SHA SOURCE_REGISTRY_SHA \
    DISTANCE_SIDECAR_SHA PARTIAL_ARCHIVE PARTIAL_ARCHIVE_SHA PARTIAL_BANK_SHA \
    PARTIAL_ALIGNMENT_AUDIT_SHA EXPECTED_PARTIAL_CONDITIONS EXPECTED_PARAMETERS \
    EXPECTED_PARTIAL_TRIPLETS EXPECTED_PARTIAL_SIDECARS \
    EXPECTED_RELAY_TRIPLETS EXPECTED_IN_ZONE_TRIPLETS \
    EXPECTED_TRENCH_AUDIT_SIDECARS \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$RUN_ROLE:$TARGET_UPDATE:$RESUME_CHECKPOINT" in
    smoke:1:none) WANDB_RESUME=never; WANDB_MODE=offline ;;
    phase1:75000:none) WANDB_RESUME=never; WANDB_MODE=online ;;
    phase2:100000:*) WANDB_RESUME=must; WANDB_MODE=online ;;
    *) echo "unsupported role/target/resume contract" >&2; exit 2 ;;
esac
case "$RUN_ROLE:${SLURM_JOB_PARTITION:-}" in
    smoke:gpuhe.4h|phase1:gpuhe.120h|phase2:gpuhe.120h) ;;
    *) echo "unexpected partition for $RUN_ROLE" >&2; exit 2 ;;
esac

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$PROTOCOL_TERRA_REVISION" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_ARCHIVE_SHA"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
test ! -e "$RUN_DIR/completion.env"
if [ "$RUN_ROLE" = phase2 ]; then
    test -f "$RESUME_CHECKPOINT"
else
    test "$RESUME_CHECKPOINT" = none
    test -z "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -print -quit)"
fi

mkdir -p "$TMPDIR/full" "$TMPDIR/partial"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR/full" bank
tar --zstd -xf "$PARTIAL_ARCHIVE" -C "$TMPDIR/partial" partial_bank
BANK="$TMPDIR/full/bank"
PARTIAL_BANK="$TMPDIR/partial/partial_bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
test "$(sha256sum "$BANK/trench_aligned_runtime_derivation.json" | awk '{print $1}')" = "$BANK_DERIVATION_SHA"
test "$(sha256sum "$PARTIAL_BANK/trench_alignment_audit.json" | awk '{print $1}')" = "$PARTIAL_ALIGNMENT_AUDIT_SHA"

module load stack/2024-06 cuda/12.1.1 eth_proxy
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cuda,cpu

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done
NUM_DEVICES=4 "$PYTHON" - <<'PY'
import os
import jax
import jax.numpy as jnp
from jax import lax

devices = int(os.environ["NUM_DEVICES"])
assert len(jax.devices()) == devices, jax.devices()
def loss(x, kernel):
    y = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return lax.psum(jnp.mean(y * y), "devices")
step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((devices, 1, 16, 16, 4), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, kernel)
assert gradient.shape == x.shape and bool(jnp.all(jnp.isfinite(gradient)))
PY

PROTOCOL_TERRA_REVISION="$PROTOCOL_TERRA_REVISION" PROTOCOL_SHA="$PROTOCOL_SHA" \
SOURCE_REGISTRY_SHA="$SOURCE_REGISTRY_SHA" BANK="$BANK" \
EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_PARTIAL_CONDITIONS" \
EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_PARTIAL_TRIPLETS" \
EXPECTED_PARTIAL_SIDECARS="$EXPECTED_PARTIAL_SIDECARS" \
EXPECTED_RELAY_TRIPLETS="$EXPECTED_RELAY_TRIPLETS" \
EXPECTED_IN_ZONE_TRIPLETS="$EXPECTED_IN_ZONE_TRIPLETS" \
EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_TRENCH_AUDIT_SIDECARS" \
PARTIAL_BANK="$PARTIAL_BANK" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" "$PYTHON" - <<'PY'
from collections import Counter
import json
import os
from pathlib import Path
from terra.maps_buffer import partial_reset_bank_sha256
from utils.accepted_bank import load_accepted_bank, validate_staged_training_bank

bank_root = Path(os.environ["BANK"])
partial_root = Path(os.environ["PARTIAL_BANK"])
assert validate_staged_training_bank(
    bank_root,
    expected_maps_per_condition=96,
    expected_release_id="terra_v8_v6_constraints_v7_adjacent_train96_v5",
) == 96
bank = load_accepted_bank(
    bank_root,
    "G-UNIFORM",
    os.environ["PROTOCOL_TERRA_REVISION"],
    curriculum_stage="full",
    sampler_profile="continuous_banded_v3",
    condition_profile="trench_aligned_37_v1",
)
assert len(bank.levels) == 37
assert sum(level.family == "foundation" for level in bank.levels) == 25
assert sum(level.family == "trench" for level in bank.levels) == 12
assert all(panel.condition_count == 35 for panel in bank.evaluation_panels)
assert all(
    panel.condition_count == 2
    for panel in bank.capability_floor_evaluation_panels
)
assert bank.environment_protocol_sha256 == os.environ["PROTOCOL_SHA"]
assert bank.source_registry_sha256 == os.environ["SOURCE_REGISTRY_SHA"]
assert partial_reset_bank_sha256(partial_root) == os.environ["PARTIAL_BANK_SHA"]
partial = json.loads((partial_root / "partial_reset_bank.json").read_text())
assert partial["canonical_loader_registry_sha256"] == __import__("hashlib").sha256(
    (bank_root / "dataset.json").read_bytes()
).hexdigest()
assert partial["pile_mode_policy"] == ["relay_corridor", "in_zone"]
assert len(partial["supported_maps_paths"]) == int(
    os.environ["EXPECTED_PARTIAL_CONDITIONS"]
)
modes = Counter()
triplets = set()
for maps_path in partial["supported_maps_paths"]:
    manifest = partial_root / maps_path / "partial_completion_manifest.jsonl"
    for line in manifest.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        modes[row["pile_mode"]] += 1
        triplets.add((maps_path, int(row["source_index"]), row["pile_mode"]))
assert len(triplets) == int(os.environ["EXPECTED_PARTIAL_TRIPLETS"])
assert sum(modes.values()) == int(os.environ["EXPECTED_PARTIAL_SIDECARS"])
assert modes["relay_corridor"] == 3 * int(
    os.environ["EXPECTED_RELAY_TRIPLETS"]
)
assert modes["in_zone"] == 3 * int(
    os.environ["EXPECTED_IN_ZONE_TRIPLETS"]
)
audit = json.loads((partial_root / "trench_alignment_audit.json").read_text())
assert audit["accepted"] is True and audit["failed_sidecars"] == 0
assert audit["partial_reset_bank_sha256"] == os.environ["PARTIAL_BANK_SHA"]
assert audit["audited_sidecars"] == int(
    os.environ["EXPECTED_TRENCH_AUDIT_SIDECARS"]
)
PY

EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import os
from types import SimpleNamespace
import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig
from utils.models import get_model_ready

class Config(dict):
    __getattr__ = dict.__getitem__

config = Config(
    clip_action_maps=True, loaded_max=100,
    local_map_normalization_bounds=(-16, 16),
    maps_net_normalization_bounds=(-10, 10), model_core="mlp",
    model_size="medium", num_prev_actions=5, critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16", attention_compute_dtype="float32",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(2, 2, 3, 3),
    token_mixer_residual_init_scale=0.1, flatten_reduce_channels=32,
    attn_latent_queries=8, aux_coef=0.0, carry_work_observation=True,
    stall_age_observation=True, reward_v2_reset_context_observation=True,
    trench_alignment_observation=True,
)
environment = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
_, params = get_model_ready(jax.random.PRNGKey(0), config, environment)
count = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params))
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
PY

if [ "$RUN_ROLE" = phase2 ]; then
    RESUME_CHECKPOINT="$RESUME_CHECKPOINT" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
    EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import os
import jax
import numpy as np
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(os.environ["RESUME_CHECKPOINT"])
assert checkpoint["next_update"] == 75000
config = checkpoint["train_config"]
assert config.partial_reset_bank_sha256 == os.environ["PARTIAL_BANK_SHA"]
assert config.reward_v2_reset_context_observation is True
assert config.stall_age_observation is True
assert config.trench_alignment_observation is True
assert config.enforce_trench_dig_alignment is True
assert config.accepted_bank.condition_profile == "trench_aligned_37_v1"
leaves = jax.tree_util.tree_leaves(checkpoint["model"])
assert sum(int(np.asarray(x).size) for x in leaves) == int(
    os.environ["EXPECTED_PARAMETERS"]
)
assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves)
assert all(
    np.all(np.isfinite(np.asarray(x)))
    for x in jax.tree_util.tree_leaves(checkpoint["optimizer_state"])
)
PY
fi

export WANDB_ENTITY WANDB_PROJECT WANDB_RESUME WANDB_MODE
export WANDB_RUN_ID="trench_generalist_partial_${BASELINES_REVISION:0:10}_s${SEED}"
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
    "final_goal_update=100000" \
    "resume_checkpoint=$RESUME_CHECKPOINT" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "environment_protocol_sha256=$PROTOCOL_SHA" \
    "source_registry_sha256=$SOURCE_REGISTRY_SHA" \
    "condition_profile=trench_aligned_37_v1" \
    "training_conditions=37" \
    "foundation_conditions=25" \
    "trench_conditions=12" \
    "partial_supported_conditions=$EXPECTED_PARTIAL_CONDITIONS" \
    "partial_source_triplets=$EXPECTED_PARTIAL_TRIPLETS" \
    "partial_sidecars=$EXPECTED_PARTIAL_SIDECARS" \
    "partial_relay_corridor_triplets=$EXPECTED_RELAY_TRIPLETS" \
    "partial_in_zone_triplets=$EXPECTED_IN_ZONE_TRIPLETS" \
    "trench_alignment_audited_sidecars=$EXPECTED_TRENCH_AUDIT_SIDECARS" \
    "full_bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "full_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "full_bank_derivation_sha256=$BANK_DERIVATION_SHA" \
    "partial_archive_sha256=$PARTIAL_ARCHIVE_SHA" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "partial_alignment_audit_sha256=$PARTIAL_ALIGNMENT_AUDIT_SHA" \
    "partial_reset_schedule=25pct_90_then_75_90_then_50_75_90_then_fade_zero_u10000" \
    "partial_mastery_updates=false" \
    "evaluation_resets=full_only" \
    "pile_mode_policy=relay_corridor_then_in_zone" \
    "enforce_trench_dig_alignment=true" \
    "trench_yaw_tolerance_deg=15" \
    "trench_standoff_m=3.5-7.0" \
    "reward_stage=reward_v2" \
    "reward_v2_reset_context_observation=true" \
    "stall_age_observation=true" \
    "sampler=continuous_banded_v3" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "wandb_run_id=$WANDB_RUN_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PYTHON_BIN="$PYTHON" MACHINE=euler
export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
if [ "$RUN_ROLE" = smoke ]; then
    export CHECKPOINT_INTERVAL=1 LOG_EVAL_INTERVAL=0
else
    export CHECKPOINT_INTERVAL=500 LOG_EVAL_INTERVAL=100
fi
"$BASELINES_ROOT/scripts/run_trench_align_generalist_partial_v1.sh" \
    "$BANK" "$PARTIAL_BANK" "$RUN_NAME" "$TARGET_UPDATE" "$RUN_DIR" \
    "$DISTANCE_SIDECAR_SHA" "$RESUME_CHECKPOINT"

FINAL_CHECKPOINT="$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
test -f "$FINAL_CHECKPOINT"
FINAL_CHECKPOINT="$FINAL_CHECKPOINT" TARGET_UPDATE="$TARGET_UPDATE" \
EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import hashlib
import json
import os
import jax
import numpy as np
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
path = os.environ["FINAL_CHECKPOINT"]
checkpoint = load_pkl_object(path)
assert checkpoint["next_update"] == int(os.environ["TARGET_UPDATE"])
model = jax.tree_util.tree_leaves(checkpoint["model"])
optimizer = jax.tree_util.tree_leaves(checkpoint["optimizer_state"])
assert sum(int(np.asarray(x).size) for x in model) == int(
    os.environ["EXPECTED_PARAMETERS"]
)
assert all(np.all(np.isfinite(np.asarray(x))) for x in model)
assert all(np.all(np.isfinite(np.asarray(x))) for x in optimizer)
receipt = {
    "schema": "terra_update_checkpoint_validation_v1",
    "checkpoint": path,
    "checkpoint_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    "next_update": checkpoint["next_update"],
    "model_parameter_count": sum(int(np.asarray(x).size) for x in model),
    "model_finite": True,
    "optimizer_finite": True,
}
open(os.path.join(os.path.dirname(os.path.dirname(path)), "checkpoint_validation.json"), "w").write(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY
printf '%s\n' "status=COMPLETE" "target_update=$TARGET_UPDATE" > "$RUN_DIR/completion.env"
echo "TRENCH_GENERALIST_PARTIAL_${RUN_ROLE^^}_COMPLETE"
