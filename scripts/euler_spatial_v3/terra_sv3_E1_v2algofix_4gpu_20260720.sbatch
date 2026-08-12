#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-sv3-E1
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%x_%j.out

set -euo pipefail
umask 002

TERRA_WORK=/cluster/home/lterenzi/codex_terra_edge_validation
TERRA_SOURCE="$TERRA_WORK/snapshots/spatial-v3-3a21cd6"
TERRA_VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
TERRA_DATASET=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train
TERRA_RUNTIME_CHECK="$TERRA_WORK/scripts/check_jax_runtime.py"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

TERRA_SITE_PACKAGES="$("$TERRA_VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export PYTHONPATH="$TERRA_SOURCE/terra:$TERRA_SOURCE/terra-baselines${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$TERRA_SITE_PACKAGES/nvidia/cudnn/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_cupti/lib:$TERRA_SITE_PACKAGES/nvidia/cublas/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$TERRA_SITE_PACKAGES/nvidia/nccl/lib:$TERRA_SITE_PACKAGES/nvidia/cusolver/lib:$TERRA_SITE_PACKAGES/nvidia/cusparse/lib:$TERRA_SITE_PACKAGES/nvidia/cufft/lib:$TERRA_SITE_PACKAGES/nvidia/nvjitlink/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$TERRA_WORK/wandb"
export DATASET_PATH="$TERRA_DATASET"
export DATASET_SIZE=600

mkdir -p "$TERRA_WORK/logs" "$TERRA_WORK/wandb" "$TERRA_SOURCE/terra-baselines/checkpoints"

echo "=== Terra spatial-v3 E1 (v2 encoder + algo fixes) launch ==="
echo "job_id=$SLURM_JOB_ID"
echo "terra_revision=34b5d39d"
echo "terra_baselines_revision=3a21cd6 (agent/spatial-v3-improvements)"
echo "preset=solo_excavator"
echo "map_encoder=resnet_spatial_8x8"
echo "model_size=base"
echo "devices=4 envs_per_device=1024 num_steps=32 update_epochs=2 num_minibatches=32"
echo "total_timesteps=2621440000 (20000 updates)"
echo "ent_schedule=0.15->0.005 over 19000 updates (sed-patched default; no CLI flag exists)"
echo "new_flags=--no_value_clip --flat_minibatch_shuffle"
echo "source=$TERRA_SOURCE"
echo "venv=$TERRA_VENV"
scontrol show job "$SLURM_JOB_ID" | grep -E 'JobId=|Partition=|ReqTRES=|AllocTRES=' || true

test -x "$TERRA_VENV/bin/python"
test -f "$TERRA_RUNTIME_CHECK"
test -d "$TERRA_DATASET/foundations"

test "$(sha256sum "$TERRA_SOURCE/terra/terra/env.py" | awk '{print $1}')" = "628b897fcbd372e86e7ee5c2c3cb41f7b9c9300539c388452085bf66acea9685"
test "$(sha256sum "$TERRA_SOURCE/terra/terra/state.py" | awk '{print $1}')" = "ef53459f7177f994b1dd54375a7a9be479e3391d334bc278e6409c9395f2fafa"
test "$(sha256sum "$TERRA_SOURCE/terra-baselines/train_mixed.py" | awk '{print $1}')" = "2a0cd6edba53a87e73dd94f8e807e3858f41ced88c0d646a4b854b2497a31f1b"
test "$(sha256sum "$TERRA_SOURCE/terra-baselines/utils/models.py" | awk '{print $1}')" = "d9c0818bc2890436750ea06312d726af57bc64d75e1365126e78ed2ac75381ef"

mapfile -t TERRA_GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "Allocated GPUs (${#TERRA_GPU_NAMES[@]}):"
printf '  %s\n' "${TERRA_GPU_NAMES[@]}"
if [ "${#TERRA_GPU_NAMES[@]}" -ne 4 ]; then
    echo "Expected exactly 4 visible GPUs" >&2
    exit 42
fi
for TERRA_GPU_NAME in "${TERRA_GPU_NAMES[@]}"; do
    if [ "$TERRA_GPU_NAME" != "NVIDIA GeForce RTX 4090" ]; then
        echo "Unsupported GPU: $TERRA_GPU_NAME (RTX 4090 only)" >&2
        exit 43
    fi
done

"$TERRA_VENV/bin/python" "$TERRA_RUNTIME_CHECK" --min-devices 4

cd "$TERRA_SOURCE/terra-baselines"

# --- ent-schedule patch -----------------------------------------------------
# ent_schedule_* are MixedAgentTrainConfig fields with no argparse flags
# (parse_known_args hard-fails on unknown args). Derive a per-run trainer from
# the sha-pinned train_mixed.py with ONLY the ent_schedule_steps default
# changed 9500 -> 19000, and prove the derivation is exactly that one line.
TRAIN_ENTRY=train_mixed_sv3_e1.py
test "$(grep -c '^    ent_schedule_steps: int = 9500$' train_mixed.py)" = 1
sed -e 's/^    ent_schedule_steps: int = 9500$/    ent_schedule_steps: int = 19000/' train_mixed.py > "$TRAIN_ENTRY"
test "$(grep -c '^    ent_schedule_steps: int = 19000$' "$TRAIN_ENTRY")" = 1
sed -e 's/^    ent_schedule_steps: int = 19000$/    ent_schedule_steps: int = 9500/' "$TRAIN_ENTRY" | cmp - train_mixed.py || { echo "ent patch produced unexpected diff" >&2; exit 45; }
echo "ENT_PATCH_VERIFIED: ent_schedule_steps 9500 -> 19000 in $TRAIN_ENTRY"

echo "=== First-update smoke gate (W&B disabled) ==="
export WANDB_MODE=disabled
"$TERRA_VENV/bin/python" -u "$TRAIN_ENTRY" \
    --config solo_excavator \
    --name terra-sv3-E1-v2algofix-smoke \
    --machine euler \
    --num_devices 4 \
    --num_envs_per_device 1024 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 131072 \
    --log_train_interval 1 \
    --log_eval_interval 0 \
    --checkpoint_interval 0 \
    --cache_clear_interval 0 \
    --eval_episodes 4 \
    --model_size base \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8 \
    --no_value_clip \
    --flat_minibatch_shuffle
echo "SMOKE_GATE_PASSED"

echo "=== Production training (W&B online) ==="
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME
"$TERRA_VENV/bin/python" -u "$TRAIN_ENTRY" \
    --config solo_excavator \
    --name terra-sv3-E1-v2algofix \
    --machine euler \
    --num_devices 4 \
    --num_envs_per_device 1024 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 2621440000 \
    --log_train_interval 1 \
    --log_eval_interval 100 \
    --checkpoint_interval 100 \
    --cache_clear_interval 1000 \
    --eval_episodes 100 \
    --model_size base \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8 \
    --no_value_clip \
    --flat_minibatch_shuffle
