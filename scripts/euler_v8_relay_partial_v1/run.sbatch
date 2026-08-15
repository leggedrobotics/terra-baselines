#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.120h
#SBATCH --time=119:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in PHASE TARGET_UPDATE RESUME_CHECKPOINT RUN_DIR RUN_NAME \
    BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION \
    PROTOCOL_TERRA_REVISION SEED VENV TERRA_EULER_USER TERRA_EULER_HOME_ROOT \
    WANDB_ENTITY WANDB_PROJECT BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_DATASET_SHA \
    BANK_RELEASE_ID DISTANCE_SIDECAR_SHA PARTIAL_ARCHIVE PARTIAL_ARCHIVE_SHA \
    PARTIAL_BANK_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

case "$PHASE:$TARGET_UPDATE:$RESUME_CHECKPOINT" in
    scratch:100000:none) WANDB_RESUME=never ;;
    resume:200000:*) WANDB_RESUME=must ;;
    *) echo "unsupported phase/target/resume contract" >&2; exit 2 ;;
esac
test "${SLURM_JOB_PARTITION:-}" = gpuhe.120h
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_ARCHIVE_SHA"
test "$BANK_RELEASE_ID" = terra_v8_v6_constraints_v7_adjacent_train96_v5
test "$PROTOCOL_TERRA_REVISION" = a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
CONTRACT="$RUN_DIR/${PHASE}_contract.env"
test ! -e "$CONTRACT"
if [ "$PHASE" = scratch ]; then
    test -z "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -print -quit)"
else
    test -f "$RESUME_CHECKPOINT"
fi

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
tar --zstd -xf "$PARTIAL_ARCHIVE" -C "$TMPDIR" partial_bank
BANK="$TMPDIR/bank"
PARTIAL_BANK="$TMPDIR/partial_bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

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

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 8
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
done
export JAX_PLATFORMS=cuda,cpu
NUM_DEVICES=8 "$PYTHON" - <<'PY'
import os
import jax
import jax.numpy as jnp
from jax import lax

devices = int(os.environ["NUM_DEVICES"])
assert len(jax.devices()) == devices
def loss(x, kernel):
    y = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return lax.psum(jnp.mean(y * y), "devices")
step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((devices, 1, 16, 16, 4), dtype=jnp.float32)
k = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, k)
assert gradient.shape == x.shape and jnp.all(jnp.isfinite(gradient))
PY

OBSERVED_MAPS="$("$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$BANK_RELEASE_ID")"
test "$OBSERVED_MAPS" = 96
PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" PARTIAL_BANK="$PARTIAL_BANK" "$PYTHON" - <<'PY'
import os
from terra.maps_buffer import partial_reset_bank_sha256

observed = partial_reset_bank_sha256(os.environ["PARTIAL_BANK"])
assert observed == os.environ["PARTIAL_BANK_SHA"], (observed, os.environ["PARTIAL_BANK_SHA"])
PY

EXPECTED_PARAMETERS=2306237
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
    clip_action_maps=True,
    loaded_max=100,
    local_map_normalization_bounds=(-16, 16),
    maps_net_normalization_bounds=(-10, 10),
    model_core="mlp",
    model_size="medium",
    num_prev_actions=5,
    critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16",
    attention_compute_dtype="float32",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(2, 2, 3, 3),
    token_mixer_residual_init_scale=0.1,
    flatten_reduce_channels=32,
    attn_latent_queries=8,
    aux_coef=0.0,
    carry_work_observation=True,
    stall_age_observation=False,
    reward_v2_reset_context_observation=True,
)
environment = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
_, params = get_model_ready(jax.random.PRNGKey(0), config, environment)
count = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params))
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
PY

if [ "$PHASE" = resume ]; then
    RESUME_CHECKPOINT="$RESUME_CHECKPOINT" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
        EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" "$PYTHON" - <<'PY'
import json
import os
import numpy as np
import jax
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(os.environ["RESUME_CHECKPOINT"])
assert checkpoint["next_update"] == 100000
assert int(np.asarray(checkpoint["train_state_step"])) == 6400000
config = checkpoint["train_config"]
assert config.partial_reset_bank_sha256 == os.environ["PARTIAL_BANK_SHA"]
assert config.reward_v2_reset_context_observation is True
assert config.stall_age_observation is False
assert config.pooled_sampler["rule"] == "continuous_banded_v3"
assert checkpoint["partial_reset_curriculum"]["partial_reset_bank_sha256"] == os.environ["PARTIAL_BANK_SHA"]
leaves = jax.tree_util.tree_leaves(checkpoint["model"])
assert sum(int(np.asarray(x).size) for x in leaves) == int(os.environ["EXPECTED_PARAMETERS"])
assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves)
assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(checkpoint["optimizer_state"]))
print(json.dumps({"checkpoint_sha256": __import__("hashlib").sha256(open(os.environ["RESUME_CHECKPOINT"], "rb").read()).hexdigest(), "next_update": 100000}, sort_keys=True))
PY
fi

export WANDB_ENTITY WANDB_PROJECT WANDB_RESUME
export WANDB_RUN_ID="v8_relay_partial_${BASELINES_REVISION:0:10}_s${SEED}"
export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
"$PYTHON" -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/${PHASE}_slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/${PHASE}_nvidia_smi.txt"

printf '%s\n' \
    "status=RUNNING" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "absolute_target_update=$TARGET_UPDATE" \
    "final_goal_update=200000" \
    "initialization=$([ "$PHASE" = scratch ] && echo fresh || echo native_u100000_resume)" \
    "resume_checkpoint=$RESUME_CHECKPOINT" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "training_bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "partial_reset_archive_sha256=$PARTIAL_ARCHIVE_SHA" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "partial_reset_schedule=25pct_90_then_75_90_then_50_75_90_then_fade_to_zero_by_u10000" \
    "partial_mastery_updates=false" \
    "evaluation_resets=full_only" \
    "reward_stage=reward_v2" \
    "reward_v2_timing_variant=0" \
    "reward_v2_reset_context_observation=true" \
    "stall_age_observation=false" \
    "action_logit_masking=false" \
    "sampler_profile=continuous_banded_v3" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "num_devices=8" \
    "num_envs_per_device=256" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.0003" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "wandb_run_id=$WANDB_RUN_ID" > "$CONTRACT"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PYTHON_BIN="$PYTHON" MACHINE=euler
"$BASELINES_ROOT/scripts/run_v8_relay_partial_v1.sh" \
    "$BANK" "$PARTIAL_BANK" "$RUN_NAME" "$TARGET_UPDATE" "$RUN_DIR" \
    "$DISTANCE_SIDECAR_SHA" "$RESUME_CHECKPOINT"

TOKEN="$(printf '%06d' "$TARGET_UPDATE")"
test -f "$RUN_DIR/checkpoints/${RUN_NAME}_update_${TOKEN}.pkl"
test -f "$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
printf '%s\n' "status=TARGET_REACHED" >> "$CONTRACT"
echo "V8_RELAY_PARTIAL_${PHASE^^}_TARGET_REACHED"
