#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-map-v1-gate
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724/logs/%x_%j.out

set -euo pipefail
umask 002

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
PARENT_SOURCE=/cluster/home/lterenzi/codex_terra_edge_validation/snapshots/spatial-v5-bbacfde/terra-baselines/checkpoints/terra-sv3-E8-multitask-ks-euler-2026-07-22-13-54-23_FINAL.pkl
PARENT="$RUN_ROOT/parent/E8_FINAL.pkl"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="$SOURCE/terra:$SOURCE/terra-baselines"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_DIR="$RUN_ROOT/wandb"
export DATASET_PATH="$BANK"
export DATASET_SIZE=256

mkdir -p \
    "$RUN_ROOT/logs" \
    "$RUN_ROOT/parent" \
    "$RUN_ROOT/eval" \
    "$RUN_ROOT/smoke/flat" \
    "$RUN_ROOT/smoke/staged" \
    "$RUN_ROOT/wandb"

HOME_USED_GB="$(
    lquota |
        awk -F'|' '/\\/cluster\\/home\\/lterenzi/ && /space/ {
            value=$4
            gsub(/[^0-9.]/, "", value)
            print value
        }'
)"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "Home quota launch gate failed: ${HOME_USED_GB} GB used (must be <45 GB)" >&2
    exit 40
}

test -x "$VENV/bin/python"
test -f "$RUNTIME_CHECK"
test -f "$PARENT_SOURCE"
test -f "$BANK/validation.json"
test -f "$BANK/manifests/identities.jsonl"
test -f "$SOURCE/terra-baselines/eval_fixed_bank.py"
test -s "$SOURCE/terra/REVISION"
test -s "$SOURCE/terra-baselines/REVISION"
"$VENV/bin/python" - "$BANK/validation.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "passed":
    raise SystemExit(f"bank validation did not pass: {payload}")
PY

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "Unsupported GPU: $GPU_NAME" >&2
        exit 41
    }
done
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 4

cp "$PARENT_SOURCE" "$PARENT.tmp"
test "$(
    sha256sum "$PARENT_SOURCE" | awk '{print $1}'
)" = "$(
    sha256sum "$PARENT.tmp" | awk '{print $1}'
)"
mv "$PARENT.tmp" "$PARENT"

"$VENV/bin/python" - "$PARENT" <<'PY'
import sys
import jax
import numpy as np
sys.path.insert(0, "/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724/source/terra-baselines")
from utils import helpers
helpers.register_checkpoint_config_classes()
checkpoint = helpers.load_pkl_object(sys.argv[1])
if int(checkpoint.get("next_update", -1)) != 20000:
    raise SystemExit(
        f"E8 parent is not the completed update-20000 checkpoint: "
        f"next_update={checkpoint.get('next_update')}"
    )
for name in ("model", "optimizer_state"):
    leaves = jax.tree_util.tree_leaves(checkpoint[name])
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise SystemExit(f"E8 parent has non-finite {name}")
print("E8_PARENT_AUDIT_PASSED")
PY

{
    echo "gate_job_id=$SLURM_JOB_ID"
    echo "terra_revision=$(cat "$SOURCE/terra/REVISION")"
    echo "terra_baselines_revision=$(cat "$SOURCE/terra-baselines/REVISION")"
    echo "parent_sha256=$(sha256sum "$PARENT" | awk '{print $1}')"
    echo "identity_manifest_sha256=$(sha256sum "$BANK/manifests/identities.jsonl" | awk '{print $1}')"
    echo "bank_provenance_sha256=$(sha256sum "$BANK/provenance.json" | awk '{print $1}')"
    echo "flat_path=train/local_M2_terminal"
    echo "staged_paths=train/local_M0,train/local_M1,train/local_M2_terminal"
    echo "horizon=450"
    echo "curriculum_thresholds=3/3/none"
} > "$RUN_ROOT/launch_manifest.txt"

cd "$SOURCE/terra-baselines"
export WANDB_MODE=disabled

echo "=== E8 parameters-only fixed M0 zero-shot evaluation ==="
"$VENV/bin/python" -u eval_fixed_bank.py \
    --checkpoint "$PARENT" \
    --bank-root "$BANK" \
    --split development \
    --strata M0 \
    --horizon 450 \
    --output "$RUN_ROOT/eval/E8_zero_shot_development_M0.json"

run_smoke() {
    local arm="$1"
    local preset="$2"
    local checkpoint_dir="$RUN_ROOT/smoke/$arm"
    echo "=== 10-update parameters-only smoke: $arm ==="
    "$VENV/bin/python" -u train_mixed.py \
        --config "$preset" \
        --name "terra-map-v1-${arm}-smoke" \
        --machine euler \
        --num_devices 4 \
        --num_envs_per_device 1024 \
        --num_steps 32 \
        --update_epochs 2 \
        --num_minibatches 32 \
        --total_timesteps 1310720 \
        --log_train_interval 1 \
        --log_eval_interval 0 \
        --checkpoint_interval 0 \
        --checkpoint_dir "$checkpoint_dir" \
        --cache_clear_interval 0 \
        --eval_episodes 4 \
        --ent_schedule_start 0.02 \
        --ent_schedule_end 0.005 \
        --ent_schedule_steps 10000 \
        --model_size medium \
        --model_core mlp \
        --map_encoder resnet_spatial_8x8_se \
        --encoder_compute_dtype bfloat16 \
        --critic_hidden_dims 512,256 \
        --warm_start_from "$PARENT" \
        --no_value_clip \
        --flat_minibatch_shuffle \
        --fail_on_nonfinite \
        --finite_check_interval 1
}

run_smoke flat map_curriculum_flat_v1
run_smoke staged map_curriculum_staged_v1

"$VENV/bin/python" - "$RUN_ROOT" "$PARENT" <<'PY'
import glob
import sys
import jax
import numpy as np
sys.path.insert(0, sys.argv[1] + "/source/terra-baselines")
from utils import helpers
helpers.register_checkpoint_config_classes()
for arm in ("flat", "staged"):
    paths = glob.glob(sys.argv[1] + f"/smoke/{arm}/*_FINAL.pkl")
    if len(paths) != 1:
        raise SystemExit(f"{arm}: expected one final smoke checkpoint, got {paths}")
    checkpoint = helpers.load_pkl_object(paths[0])
    config = checkpoint["train_config"]
    if int(checkpoint.get("next_update", -1)) != 10:
        raise SystemExit(f"{arm}: smoke did not finish 10 updates")
    if getattr(config, "warm_start_from", None) != sys.argv[2]:
        raise SystemExit(f"{arm}: wrong parameters-only parent")
    if getattr(config, "resume_from", None) is not None:
        raise SystemExit(f"{arm}: resume state was configured")
    for name in ("model", "optimizer_state", "loss_info"):
        leaves = jax.tree_util.tree_leaves(checkpoint[name])
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise SystemExit(f"{arm}: non-finite {name}")
print("PAIRED_10_UPDATE_SMOKE_AUDIT_PASSED")
PY

echo "=== Paired smoke fixed-bank M0 health audit ==="
"$VENV/bin/python" -u eval_fixed_bank.py \
    --checkpoint-glob "$RUN_ROOT/smoke/flat/*_FINAL.pkl" \
    --checkpoint-glob "$RUN_ROOT/smoke/staged/*_FINAL.pkl" \
    --bank-root "$BANK" \
    --split development \
    --strata M0 \
    --horizon 450 \
    --output "$RUN_ROOT/eval/smoke_development_M0.json"

touch "$RUN_ROOT/SMOKE_GATE_PASSED"
echo "SMOKE_GATE_PASSED"
