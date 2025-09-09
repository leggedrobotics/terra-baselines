#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:1
#SBATCH --time=2:00:00
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
export DATASET_PATH=/cluster/home/alesweber/TerraProject/terra/data/terra/train/
export DATASET_SIZE=20


# Change to the directory containing visualize.py
cd /cluster/home/alesweber/TerraProject/terra-baselines
#JAX_PLATFORMS=cpu  python visualize_mixed.py --run_name /cluster/home/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2025-08-08-13-52-00.pkl
python visualize_mixed.py --run_name /cluster/home/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2025-08-19-02-58-56.pkl

#mixed-agents-skidsteer-skidsteer-local-2025-08-07-16-58-21_FINAL.pkl

#JAX_PLATFORMS=cpu 