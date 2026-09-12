#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="visualization-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_visualization.out

set -euo pipefail
: "${CONDA_ROOT:?Set CONDA_ROOT to the selected account runtime installation}"
: "${TERRA_BASELINES_ROOT:?Set TERRA_BASELINES_ROOT to the staged terra-baselines checkout}"
: "${DATASET_PATH:?Set DATASET_PATH to the selected readable map bank}"
: "${CHECKPOINT:?Set CHECKPOINT to the selected policy checkpoint}"


# Disable audio and set dummy display for cluster nodes
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
export DISPLAY=:0

# Load required modules
module load eth_proxy
module load stack/2024 cuda/12.1.1

# Set paths to conda
CONDA_ENV=terra

# Activate conda environment properly for batch jobs
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

# Set environment variables and run visualization
export DATASET_PATH
export DATASET_SIZE=25


# Change to the directory containing visualize.py
cd "$TERRA_BASELINES_ROOT"

python visualize_mixed.py --config solo_excavator_dumpzone --run_name "$CHECKPOINT"

#JAX_PLATFORMS=cpu 