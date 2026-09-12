#!/bin/bash

set -euo pipefail
: "${CONDA_ROOT:?Set CONDA_ROOT to the selected account runtime installation}"
: "${TERRA_BASELINES_ROOT:?Set TERRA_BASELINES_ROOT to the staged terra-baselines checkout}"
: "${DATASET_PATH:?Set DATASET_PATH to the selected readable map bank}"

# Recommended srun command:
# srun --cpus-per-task=8 --mem-per-cpu=8G --gres=gpu:rtx_4090:8 --time=12:00:00 terra-baselines/cluster/train_srun.sh
#
# Or for shorter test runs:
# srun --cpus-per-task=1 --mem-per-cpu=4G --gres=gpu:rtx_4090:1 --time=2:00:00 terra-baselines/cluster/train_srun.sh

# Set up environment
module load eth_proxy
module load stack/2024 cuda/12.1.1

# Set paths to conda and initialize properly
CONDA_ENV=terra

# Initialize conda properly
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

export DATASET_PATH
export DATASET_SIZE=200


# Change to the correct directory
cd "$TERRA_BASELINES_ROOT"
python train_mixed.py --config solo_excavator
