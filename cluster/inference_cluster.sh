#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="inference"
#SBATCH --output=%j_inference.out

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
#export DATASET_PATH=/cluster/project/rsl/alesweber/TerraProject/terra-baselines/isaac_sim/
#export DATASET_SIZE=1


# Change to the directory containing inference script
cd /cluster/project/rsl/alesweber/TerraProject/terra-baselines/inference
#JAX_PLATFORMS=cpu  python visualize_mixed.py --run_name /cluster/home/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2025-08-08-13-52-00.pkl
python inference_single_map.py --policy /cluster/project/rsl/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2026-07-14-17-30-39.pkl --config solo_excavator --map_name map_16
#--use-mcts
#mixed-agents-skidsteer-skidsteer-local-2025-08-07-16-58-21_FINAL.pkl

#JAX_PLATFORMS=cpu 