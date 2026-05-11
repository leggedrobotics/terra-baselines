#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="visualization-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_visualization.out

# Disable audio and set dummy display for cluster nodes
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
export DISPLAY=:0

# Load required modules
module load eth_proxy
module load stack/2024 cuda/12.1.1

# Set paths to conda
CONDA_ROOT=/cluster/home/alesweber/miniconda3
CONDA_ENV=terra

# Activate conda environment properly for batch jobs
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

# Set environment variables and run visualization
export DATASET_PATH=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train/
#export DATASET_PATH=/cluster/project/rsl/alesweber/TerraProject/terra-baselines/inference/maps
export DATASET_SIZE=25


# Change to the directory containing visualize.py
cd /cluster/project/rsl/alesweber/TerraProject/terra-baselines

python visualize_mixed.py --config solo_excavator --run_name /cluster/project/rsl/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2026-05-11-12-23-30.pkl

#JAX_PLATFORMS=cpu 