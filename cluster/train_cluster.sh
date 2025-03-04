#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Load required modules
module load eth_proxy

# Activate your conda environment (adjust the path and env name as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate terra

# Set environment variables and run training
export DATASET_PATH=/cluster/home/lterenzi/terra_jax/terra/data/terra/train
export DATASET_SIZE=200

python train.py
