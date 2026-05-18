#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --partition=gpuhe.120h
#SBATCH --nodelist=eu-g4-[001-032],eu-g6-[001-080]
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-resmap64-mb32
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%j_resmap64_phase_4gpu.out

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
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$WORK/wandb"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$WANDB_DIR" "$WORK/logs"
cd "$WORK/terra-baselines"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
printf 'Allocated GPUs:\n'
printf '  %s\n' "${GPU_NAMES[@]}"
if [ "${#GPU_NAMES[@]}" -ne 4 ]; then
  echo "Expected exactly 4 GPUs, got ${#GPU_NAMES[@]}" >&2
  exit 2
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  case "$gpu_name" in
    "NVIDIA GeForce RTX 3090"|"NVIDIA GeForce RTX 4090") ;;
    *)
      echo "Refusing non-3090/4090 GPU: $gpu_name" >&2
      exit 2
      ;;
  esac
done

"$VENV/bin/python" "$WORK/terra-baselines/scripts/euler/check_jax_runtime.py" --min-devices 4

COMMON_ARGS=(
  --config solo_excavator
  --num_devices 4
  --num_steps 32
  --map_encoder resnet_delayed
  --map_feature_dim 128
  --use_map_derived_channels
  --separate_actor_critic_trunks
  --use_critic_affordances
  --include_episode_progress
  --edge_features_dim 10
)

WANDB_MODE=disabled "$VENV/bin/python" train_mixed.py \
  "${COMMON_ARGS[@]}" \
  --name "terra-solo-resmap64-r1r2-mb32-unmasked-smoke" \
  --machine euler-smoke-fullshape \
  --num_envs_per_device 1024 \
  --update_epochs 2 \
  --num_minibatches 32 \
  --total_timesteps 131072 \
  --log_train_interval 0 \
  --log_eval_interval 0 \
  --checkpoint_interval 0 \
  --eval_episodes 4 \
  --num_rollouts_eval 8

WANDB_MODE=online "$VENV/bin/python" train_mixed.py \
  "${COMMON_ARGS[@]}" \
  --name "terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516" \
  --machine euler-4gpu \
  --num_envs_per_device 1024 \
  --update_epochs 2 \
  --num_minibatches 32 \
  --total_timesteps 50000000000 \
  --log_train_interval 1 \
  --log_eval_interval 100 \
  --checkpoint_interval 100 \
  --eval_episodes 100 \
  --num_rollouts_eval 550
