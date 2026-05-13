#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=gpu:nvidia_geforce_rtx_4090:2
#SBATCH --partition=gpuhe.120h
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-1
#SBATCH --job-name=terra-ring-2x4090-full
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%A_%a_ring_2gpu4090_full.out

set -euo pipefail

WORK=/cluster/home/lterenzi/codex_terra_edge_validation
VENV=/cluster/scratch/lterenzi/codex_terra_edge_venv
DATASET=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train

MODES=(mask nomask)
FLAGS=(--enable_action_mask --disable_action_mask)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
MASK_FLAG=${FLAGS[$SLURM_ARRAY_TASK_ID]}

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

"$VENV/bin/python" "$WORK/scripts/check_jax_runtime.py" --min-devices 2

"$VENV/bin/python" train_mixed.py \
  --config solo_excavator \
  --name "terra-ringmaps-2gpu4090-${MODE}-full-online" \
  --machine euler-pr \
  --num_devices 2 \
  --num_envs_per_device 1024 \
  --num_steps 32 \
  --update_epochs 2 \
  --num_minibatches 16 \
  --total_timesteps 50000000000 \
  --log_train_interval 1 \
  --log_eval_interval 100 \
  --checkpoint_interval 100 \
  --eval_episodes 100 \
  $MASK_FLAG
