#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="inference"
#SBATCH --output=%j_inference.out

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
#export DATASET_SIZE=1


# Change to the directory containing inference script
cd "$TERRA_BASELINES_ROOT/inference"
python inference_single_map.py --policy "$CHECKPOINT" --config trench_excavator_double --map_name map_2wide --render_plan_gif --trench_align
#--use-mcts
#mixed-agents-skidsteer-skidsteer-local-2025-08-07-16-58-21_FINAL.pkl

#JAX_PLATFORMS=cpu 