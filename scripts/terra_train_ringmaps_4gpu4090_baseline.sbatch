#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --partition=gpuhe.120h
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-ring-4x4090-base
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%j_ring_4gpu4090_baseline.out

set -euo pipefail

WORK=/cluster/home/lterenzi/codex_terra_edge_validation
VENV=/cluster/scratch/lterenzi/codex_terra_edge_venv
DATASET=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES=$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export PYTHONPATH="$WORK/terra:$WORK/terra-baselines:${PYTHONPATH:-}"
export DATASET_PATH="$DATASET"
export DATASET_SIZE=600
export WANDB_MODE=online
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$WORK/wandb"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$WANDB_DIR" "$WORK/logs"
cd "$WORK/terra-baselines"

"$VENV/bin/python" "$WORK/scripts/check_jax_runtime.py" --min-devices 4

"$VENV/bin/python" train_mixed.py \
  --config solo_excavator \
  --name "terra-ringmaps-4gpu4090-nomask-baseline-online" \
  --machine euler-pr \
  --num_devices 4 \
  --num_envs_per_device 1024 \
  --num_steps 32 \
  --update_epochs 2 \
  --num_minibatches 16 \
  --total_timesteps 50000000000 \
  --log_train_interval 1 \
  --log_eval_interval 100 \
  --checkpoint_interval 100 \
  --eval_episodes 100 \
  --disable_action_mask
