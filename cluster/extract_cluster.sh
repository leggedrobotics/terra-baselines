#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="extract"
#SBATCH --output=%j_extract.out

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
export DATASET_SIZE=1


# Change to the directory containing visualize.py
cd "$TERRA_BASELINES_ROOT/isaac_sim"
python extract_map.py --policy "$CHECKPOINT" --config trench_excavator_double --map map_2wide --render_plan_gif --trench_align
#--postprocess_base_position
#--foundation_dump_min_free_fraction 0
#--postprocess_base_position

#  --trench_align
#--use-mcts
#mixed-agents-skidsteer-skidsteer-local-2025-08-07-16-58-21_FINAL.pkl

#JAX_PLATFORMS=cpu
