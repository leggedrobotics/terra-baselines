#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

# rv2p1_scratch: the reward_v2_scratch baseline with reward-v2.1 timing
# (undiscounted shaping + explicit step cost 3.6) and the capped
# continuous_banded_v3 sampler. Architecture, batch shape and every other
# contract flag are the compact baseline's.
ARM_NAME=rv2p1_scratch
RUN_PREFIX=v8_rv2p1
EXPECTED_RUNTIME_TERRA_REVISION=46b5a1ddcd3b0e3a0d9e637af2e4ea94af51b4c8
EXPECTED_PARAMETERS=2856701
EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK=2856685
EXPECTED_HORIZON=450
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
SAMPLER_PROFILE=continuous_banded_v3
REWARD_V2_TIMING_VARIANT=1

for REQUIRED in TERRA_EULER_USER TERRA_EULER_HOME_ROOT VENV RUN_BASE \
    PHASE BASELINES_ROOT BASELINES_REVISION RUNTIME_TERRA_ROOT \
    RUNTIME_TERRA_REVISION R2_HORIZON SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_TREE_SHA BANK_RELEASE_ID DISTANCE_ARTIFACT_SHA \
    MATERIALIZATION_RECEIPT MATERIALIZATION_RECEIPT_SHA SMOKE_JOB_ID \
    SMOKE_RUN SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "$SEED" = 20260807
test "$RUNTIME_TERRA_REVISION" = "$EXPECTED_RUNTIME_TERRA_REVISION"
test "$R2_HORIZON" -eq "$EXPECTED_HORIZON"
test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"

case "$PHASE" in
    smoke)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
        test "$SMOKE_JOB_ID" = none -a "$SMOKE_RUN" = none
        UPDATES=1
        CHECKPOINT_INTERVAL=1
        LOG_TRAIN_INTERVAL=1
        FINITE_CHECK_INTERVAL=1
        CACHE_CLEAR_INTERVAL=0
        ;;
    phase1)
        test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        [[ "$SMOKE_RUN" = /* ]]
        UPDATES=14000
        CHECKPOINT_INTERVAL=500
        EVAL_INTERVAL=1000
        LOG_TRAIN_INTERVAL=10
        FINITE_CHECK_INTERVAL=10
        CACHE_CLEAR_INTERVAL=1000
        ;;
    *) echo "unsupported phase '$PHASE'" >&2; exit 2 ;;
esac

RUN_DIR="$RUN_BASE/$BASELINES_REVISION/$PHASE/s$SEED/$ARM_NAME"
test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
for SPEC in \
    "$BANK_ARCHIVE:$BANK_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA"; do
    FILE="${SPEC%:*}" EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$FILE" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "${SLURM_CPUS_PER_TASK:-}" = 4
test -d "$RUN_DIR" -a ! -e "$RUN_DIR/run_contract.env"

if [ "$PHASE" = phase1 ]; then
    SMOKE_STATE="$(sacct -n -X -P -j "$SMOKE_JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$SMOKE_JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$SMOKE_STATE" = COMPLETED
    test -f "$SMOKE_RUN/smoke_validation.json" -a -f "$SMOKE_RUN/run_contract.env"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$SMOKE_RUN/smoke_validation.json"
    for KEY in runtime_terra_revision:"$RUNTIME_TERRA_REVISION" \
        terra_baselines_revision:"$BASELINES_REVISION" \
        distance_artifact_sha256:"$DISTANCE_ARTIFACT_SHA" \
        model_parameter_count:"$EXPECTED_PARAMETERS" \
        arm:"$ARM_NAME" \
        reward_v2_timing_variant:"$REWARD_V2_TIMING_VARIANT" \
        sampler_profile:"$SAMPLER_PROFILE"; do
        NAME="${KEY%%:*}" VALUE="${KEY#*:}"
        test "$(awk -F= -v key="$NAME" '$1==key {print $2}' "$SMOKE_RUN/run_contract.env")" = "$VALUE"
    done
fi

HOME_USED_GB="$(lquota | awk -F'|' -v home="$TERRA_EULER_HOME_ROOT" \
    'index($0, home) && /space/ {value=$4; gsub(/[^0-9.]/, "", value); print value}')"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
module load stack/2024-06 cuda/12.1.1
if [ "$PHASE" = phase1 ]; then
    module load eth_proxy
    test -r "$HOME/.netrc"
fi
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    if [ "$PHASE" = smoke ]; then
        test "$GPU_NAME" = "NVIDIA GeForce RTX 3090"
    else
        test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
    fi
done
# cuda first keeps every default on GPU; cpu must be reachable for the
# host-side model.init (see utils/models.py get_model_ready).
export JAX_PLATFORMS=cuda,cpu
"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
from jax import lax

assert len(jax.devices()) == 4
def loss(x, kernel):
    y = lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return lax.psum(jnp.mean(y * y), "devices")
step = jax.pmap(jax.grad(loss), axis_name="devices", in_axes=(0, None))
x = jnp.ones((4, 1, 16, 16, 4), dtype=jnp.float32)
k = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = step(x, k)
assert gradient.shape == x.shape and jnp.all(jnp.isfinite(gradient))
print(jax.devices(), float(jnp.linalg.norm(gradient)))
PY
# The reward receipt: everything reward_v2 froze is unchanged, and the two
# adopted timing constants are exactly the ones this arm claims.
"$PYTHON" - <<'PY'
from terra.config import (
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    REWARD_V2_SHAPING_WEIGHT,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SUCCESS_BONUS,
    REWARD_V2_TIMING_BASELINE,
    REWARD_V2_TIMING_V21,
    REWARD_V2_TIMING_V21_ID,
    REWARD_V2_V21_SHAPING_GAMMA,
    REWARD_V2_V21_STEP_COST_TOTAL,
    EnvConfig,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.state import CORRECTED_DENSE_CONTRACT

assert int(RewardStage.REWARD_V2) == 3
assert REWARD_V2_PROTOCOL_ID == "material_potential_v2"
assert REWARD_V2_DISTANCE_PROTOCOL_ID == "obstacle_geodesic_8_physical_global_v1"
assert CORRECTED_DENSE_CONTRACT == "exact_visible_dump_v1"
assert (
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_SUCCESS_BONUS,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SHAPING_WEIGHT,
) == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)
assert (REWARD_V2_TIMING_BASELINE, REWARD_V2_TIMING_V21) == (0, 1)
assert REWARD_V2_TIMING_V21_ID == "gamma1_stepcost_3.6"
assert (REWARD_V2_V21_SHAPING_GAMMA, REWARD_V2_V21_STEP_COST_TOTAL) == (1.0, 3.6)
assert REWARD_V2_V21_STEP_COST_TOTAL / 450.0 == 0.008
assert EnvConfig().reward_v2_timing_variant == REWARD_V2_TIMING_BASELINE
PY
# The architecture receipt: the compact baseline trunk, unchanged. 2,856,701
# under the carry-work channel and 16 fewer without it.
EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" \
    EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK="$EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK" \
    "$PYTHON" - <<'PY'
import os
from types import SimpleNamespace

import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import get_model_ready

EXPECTED = int(os.environ["EXPECTED_PARAMETERS"])
EXPECTED_WITHOUT_CARRY_WORK = int(os.environ["EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK"])


class Config(dict):
    __getattr__ = dict.__getitem__


def config(**extra):
    return Config(
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
        map_encoder="resnet_spatial_8x8_se_xattn",
        resnet_stage_channels=(24, 48, 64, 96),
        resnet_blocks_per_stage=(2, 2, 3, 3),
        **extra,
    )


environment = SimpleNamespace(
    batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
)
counts = {}
for carry in (False, True):
    _, params = get_model_ready(
        jax.random.PRNGKey(0), config(carry_work_observation=carry), environment
    )
    counts[carry] = sum(
        int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)
    )
assert counts[False] == EXPECTED_WITHOUT_CARRY_WORK, counts
assert counts[True] == EXPECTED, counts
print("compact baseline parameters", counts)
PY

OBSERVED_MAPS="$("$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$OBSERVED_MAPS" = 96
GRAPH_SHA="$("$PYTHON" -c 'import collections,sys; from utils.accepted_bank import load_accepted_bank; b=load_accepted_bank(sys.argv[1], "G-UNIFORM", sys.argv[2], curriculum_stage="full", sampler_profile=sys.argv[3]); assert len(b.levels)==47 and b.map_count_per_condition==96 and b.sampling_probabilities==(); assert collections.Counter(b.curriculum_depths)=={0:2,1:13,2:32}; print(b.curriculum_graph_sha256)' "$BANK" "$PROTOCOL_TERRA_REVISION" "$SAMPLER_PROFILE")"
[[ "$GRAPH_SHA" =~ ^[0-9a-f]{64}$ ]]
MAX_CONDITION_MASS="$("$PYTHON" -c 'from utils.pooled_sampler import CONTINUOUS_MAX_MASS; print(CONTINUOUS_MAX_MASS)')"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/eval" "$RUN_DIR/wandb"
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
export WANDB_RUN_ID="${RUN_PREFIX}_${BASELINES_REVISION:0:10}_${PHASE}_${SLURM_JOB_ID}"
export WANDB_RESUME=never
if [ "$PHASE" = smoke ]; then
    export WANDB_MODE=disabled WANDB_DIR="$TMPDIR/wandb"
else
    export WANDB_MODE=online WANDB_DIR="$RUN_DIR/wandb"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
RUN_NAME="${RUN_PREFIX}_${BASELINES_REVISION:0:12}_s${SEED}"
if [ "$PHASE" = smoke ]; then RUN_NAME="${RUN_NAME}_smoke"; fi
TRANSITIONS=$((4 * 512 * 32 * UPDATES))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=v8_rv2p1_timing_adoption" \
    "arm=$ARM_NAME" \
    "comparison_role=historical_descriptive_reference_to_reward_v2_scratch" \
    "adopted_changes=reward_v2p1_timing+continuous_banded_v3" \
    "inherited_from_baseline=reward_v2+global_distance_ledger+carry_input+compact_architecture+phase1_14000_of_40000" \
    "phase1_selection_updates=1000_to_${UPDATES}_promotion_only" \
    "promotion_selection_rule=combined_exact_then_47_condition_macro_then_worst_then_earliest" \
    "phase=$PHASE" \
    "seed=$SEED" \
    "absolute_start_update=0" \
    "absolute_target_update=$UPDATES" \
    "updates=$UPDATES" \
    "transitions=$TRANSITIONS" \
    "support_scope=all47_continuous_from_update0" \
    "condition_count=47" \
    "maps_per_condition=96" \
    "sampler_profile=$SAMPLER_PROFILE" \
    "sampler_max_condition_mass=$MAX_CONDITION_MASS" \
    "curriculum_graph_sha256=$GRAPH_SHA" \
    "reward_stage=reward_v2" \
    "reward_protocol_id=material_potential_v2" \
    "reward_v2_timing=gamma1_stepcost_3.6" \
    "reward_v2_timing_variant=$REWARD_V2_TIMING_VARIANT" \
    "shaping_gamma=1.0" \
    "step_cost_total=3.6" \
    "distance_protocol_id=obstacle_geodesic_8_physical_global_v1" \
    "distance_artifact_sha256=$DISTANCE_ARTIFACT_SHA" \
    "carry_channel=agent_states_8_normalized_carry_work" \
    "horizon=$R2_HORIZON" \
    "full_resets=true" \
    "initialization=random_no_teacher" \
    "warm_start_from=none" \
    "prepared_fork_from=none" \
    "optimizer_initial_local_step=0" \
    "entropy_schedule=0.15_to_0.02_over_20000" \
    "ppo_gamma=0.9984" \
    "reward_v2_distance_ref_m=16.0" \
    "reward_v2_distance_bound=2.5" \
    "reward_v2_success_bonus=6.0" \
    "reward_v2_horizon_failure_penalty=1.0" \
    "reward_v2_alpha=1.0" \
    "reward_v2_beta=1.5" \
    "reward_v2_step_cost_total=3.6" \
    "reward_v2_shaping_weight=1.0" \
    "resume_supported=phase_boundaries_only" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "euler_user=$TERRA_EULER_USER" \
    "euler_home_root=$TERRA_EULER_HOME_ROOT" \
    "runtime_venv=$VENV" \
    "training_bank_release_id=$EXPECTED_RELEASE" \
    "training_bank_archive_sha256=$BANK_SHA" \
    "training_bank_dataset_sha256=$BANK_DATASET_SHA" \
    "training_bank_tree_sha256=$BANK_TREE_SHA" \
    "materialization_receipt_sha256=$MATERIALIZATION_RECEIPT_SHA" \
    "map_encoder=resnet_spatial_8x8_se_xattn" \
    "resnet_stage_channels=24,48,64,96" \
    "resnet_blocks_per_stage=2,2,3,3" \
    "critic_hidden_dims=512,256" \
    "model_parameter_count=$EXPECTED_PARAMETERS" \
    "model_parameter_count_without_carry_work=$EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK" \
    "num_devices=4" \
    "num_envs_per_device=512" \
    "num_steps=32" \
    "num_minibatches=32" \
    "update_epochs=2" \
    "learning_rate=0.0003" \
    "checkpoint_interval=$CHECKPOINT_INTERVAL" \
    "planned_fixed_panel_spacing_updates=${EVAL_INTERVAL:-none}" \
    "smoke_job_id=$SMOKE_JOB_ID" \
    "slurm_job_id=$SLURM_JOB_ID" > "$RUN_DIR/run_contract.env"

export TERRA_ROOT="$RUNTIME_TERRA_ROOT" PROTOCOL_TERRA_REVISION PYTHON_BIN="$PYTHON" MACHINE=euler
export REWARD_V2_TIMING_VARIANT
export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32
export CHECKPOINT_INTERVAL LOG_TRAIN_INTERVAL FINITE_CHECK_INTERVAL CACHE_CLEAR_INTERVAL
export ENTROPY_SCHEDULE_STEPS=20000
"$BASELINES_ROOT/scripts/run_v8_reward_timing_pilot.sh" \
    "$BANK" "$RUN_NAME" "$UPDATES" "$RUN_DIR" "$DISTANCE_ARTIFACT_SHA" none

find_one() {
    local pattern="$1"
    local -a matches
    mapfile -t matches < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name "$pattern" -print | sort)
    test "${#matches[@]}" -eq 1
    printf '%s\n' "${matches[0]}"
}

if [ "$PHASE" = smoke ]; then
    UPDATE_ONE="$(find_one "${RUN_NAME}_update_000001.pkl")"
    find_one "${RUN_NAME}_FINAL.pkl" >/dev/null
    JAX_PLATFORMS=cpu EXPECTED_PARAMETERS="$EXPECTED_PARAMETERS" \
        SAMPLER_PROFILE="$SAMPLER_PROFILE" \
        REWARD_V2_TIMING_VARIANT="$REWARD_V2_TIMING_VARIANT" \
        DISTANCE_ARTIFACT_SHA="$DISTANCE_ARTIFACT_SHA" \
        "$PYTHON" - "$UPDATE_ONE" "$RUN_DIR/smoke_validation.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np

from scripts.verify_continuous_sampler_checkpoint import verify_sampler_state
from terra.config import RewardStage
from utils import helpers

path = Path(sys.argv[1])
helpers.register_checkpoint_config_classes()
checkpoint = helpers.load_pkl_object(str(path))
variant = int(os.environ["REWARD_V2_TIMING_VARIANT"])
profile = os.environ["SAMPLER_PROFILE"]


def value(config, name):
    return config[name] if isinstance(config, dict) else getattr(config, name)


assert checkpoint.get("update") == 0 and checkpoint.get("next_update") == 1
assert int(np.asarray(checkpoint["train_state_step"]).reshape(())) == 64
count = sum(
    int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(checkpoint["model"])
)
assert count == int(os.environ["EXPECTED_PARAMETERS"]), count
for label in ("model", "optimizer_state"):
    for leaf in jax.tree_util.tree_leaves(checkpoint[label]):
        leaf = np.asarray(leaf)
        if np.issubdtype(leaf.dtype, np.number):
            assert np.all(np.isfinite(leaf)), label
integrity = checkpoint.get("transition_integrity")
assert isinstance(integrity, dict) and all(int(v) == 0 for v in integrity.values())

config = checkpoint["train_config"]
expected = {
    "config_name": "G-V8-CONTINUOUS-V3",
    "seed": 20260807,
    "reward_stage": "reward_v2",
    "reward_v2_timing_variant": variant,
    "carry_work_observation": True,
    "distance_sidecar_sha256": os.environ["DISTANCE_ARTIFACT_SHA"],
    "map_encoder": "resnet_spatial_8x8_se_xattn",
    "num_devices": 4,
    "num_envs_per_device": 512,
    "num_steps": 32,
    "num_minibatches": 32,
}
for name, want in expected.items():
    got = value(config, name)
    assert got == want, f"{name}={got!r}, expected {want!r}"
assert value(config, "pooled_sampler")["rule"] == profile
assert value(config, "accepted_bank").sampler_profile == profile

env_config = checkpoint["env_config"]
assert np.all(np.asarray(env_config.reward_stage) == int(RewardStage.REWARD_V2))
assert np.all(np.asarray(env_config.reward_v2_timing_variant) == variant)

receipt = checkpoint["r2_protocol_receipt"]
assert receipt["reward_v2_timing"] == "gamma1_stepcost_3.6"
assert receipt["reward_v2_timing_variant"] == variant
assert receipt["constants"]["shaping_gamma"] == 1.0
assert receipt["constants"]["step_cost_total"] == 3.6

sampler = verify_sampler_state(checkpoint["pooled_sampler_state"])
assert sampler["sampler_rule"] == profile, sampler
state = checkpoint["pooled_sampler_state"]
probabilities = np.asarray(state["probabilities"], dtype=np.float64)
assert probabilities.size == 47
assert abs(float(probabilities.sum()) - 1.0) < 1e-12
assert float(probabilities.max()) <= state["settings"]["max_mass"] + 1e-12
labels, conditions = state["labels"], state["conditions"]
sampler["family_mass"] = {
    family: float(
        probabilities[[labels[n]["family"] == family for n in conditions]].sum()
    )
    for family in ("foundation", "trench")
}
sampler["max_condition_mass"] = float(probabilities.max())
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "passed": True,
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameter_count": count,
            "reward_protocol": receipt,
            "sampler": sampler,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print("RV2P1_SMOKE_CHECKS_PASSED")
PY
    printf '%s\n' status=PASSED >> "$RUN_DIR/run_contract.env"
    echo "V8_RV2P1_SMOKE_PASSED"
    exit 0
fi

find_one "${RUN_NAME}_update_$(printf '%06d' "$UPDATES").pkl" >/dev/null
printf '%s\n' status=TRAINING_TARGET_REACHED >> "$RUN_DIR/run_contract.env"
CHECKPOINT_ARGS=()
for UPDATE in $(seq "$EVAL_INTERVAL" "$EVAL_INTERVAL" "$UPDATES"); do
    TOKEN="$(printf '%06d' "$UPDATE")"
    CHECKPOINT_ARGS+=(--checkpoint "$(find_one "${RUN_NAME}_update_${TOKEN}.pkl")")
done
"$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
    "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel promotion \
    --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
    --expect-completion-contract exact_visible_dump_v1 \
    --expect-reward-protocol-id material_potential_v2 \
    --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
    --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
    --require-productive-workspace-cycles \
    --output "$RUN_DIR/eval/promotion.json"
"$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
    "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel promotion \
    --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
    --expect-completion-contract exact_visible_dump_v1 \
    --expect-reward-protocol-id material_potential_v2 \
    --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
    --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
    --require-productive-workspace-cycles \
    --output "$RUN_DIR/eval/capability_promotion.json"
"$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_r2_reward_v2/select_promotion.py" \
    --promotion "$RUN_DIR/eval/promotion.json" \
    --capability-promotion "$RUN_DIR/eval/capability_promotion.json" \
    --output "$RUN_DIR/eval/promotion_selection.json"
SELECTED_CHECKPOINT="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$RUN_DIR/eval/promotion_selection.json")"
SELECTED_UPDATE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_update"])' "$RUN_DIR/eval/promotion_selection.json")"
SELECTED_SHA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_sha256"])' "$RUN_DIR/eval/promotion_selection.json")"
case "$SELECTED_CHECKPOINT" in "$RUN_DIR/checkpoints/"*) ;; *) exit 3 ;; esac
test "$(sha256sum "$SELECTED_CHECKPOINT" | awk '{print $1}')" = "$SELECTED_SHA"

for PANEL in development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$SELECTED_CHECKPOINT" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --expect-reward-protocol-id material_potential_v2 \
        --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
        --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/${PANEL}.json"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$SELECTED_CHECKPOINT" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$PROTOCOL_TERRA_REVISION" --horizon "$R2_HORIZON" --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --expect-reward-protocol-id material_potential_v2 \
        --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
        --expect-distance-sidecar-sha256 "$DISTANCE_ARTIFACT_SHA" \
        --require-productive-workspace-cycles \
        --output "$RUN_DIR/eval/capability_${PANEL}.json"
done
SELECTION_SHA="$(sha256sum "$RUN_DIR/eval/promotion_selection.json" | awk '{print $1}')"
printf '%s\n' \
    status=EVALUATED \
    "selected_checkpoint_update=$SELECTED_UPDATE" \
    "selected_checkpoint_sha256=$SELECTED_SHA" \
    "promotion_selection_receipt_sha256=$SELECTION_SHA" >> "$RUN_DIR/run_contract.env"
echo "V8_RV2P1_EVALUATED"
