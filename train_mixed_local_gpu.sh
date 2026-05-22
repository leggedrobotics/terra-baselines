#!/usr/bin/env bash
set -euo pipefail

# Local single-GPU equivalent of cluster/train_cluster.sh.
# Extra train_mixed.py args can be appended at the end, for example:
#   ./train_mixed_local_gpu.sh --total_timesteps 1000000
#   ./train_mixed_local_gpu.sh --resume_from checkpoints/model.pkl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate terra
fi

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export DATASET_PATH="${REPO_ROOT}/terra/data/terra/train"
export DATASET_SIZE=600
export PYTHONPATH="${REPO_ROOT}/terra:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${SCRIPT_DIR}"

echo "Running local mixed-agent training on GPU ${CUDA_VISIBLE_DEVICES}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "DATASET_SIZE=${DATASET_SIZE}"

python -u train_mixed.py \
    --config solo_excavator_rectangles \
    --model_size medium \
    --num_devices 1 \
    --num_envs_per_device 1024 \
    "$@"
