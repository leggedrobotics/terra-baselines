#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_training.out

# Load required modules
module load eth_proxy

# Set paths to conda
CONDA_ROOT=/cluster/home/lterenzi/miniconda3
CONDA_ENV=terra                 # Your conda environment name

# Activate conda environment properly for batch jobs
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

# Set environment variables and run training
export DATASET_PATH=/cluster/home/lterenzi/terra_jax/terra/data/terra/train
export DATASET_SIZE=200

# Add the terra package to the Python path
export PYTHONPATH=$PYTHONPATH:/cluster/home/lterenzi/terra_jax

# Change to the directory containing train.py or use the full path
cd /cluster/home/lterenzi/terra_jax/terra-baselines
python train.py
