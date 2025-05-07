#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:2
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_training.out

# Load required modules
module load eth_proxy
module load stack/2024-06 cuda/12.1.1

# Set paths to conda
CONDA_ROOT=/cluster/home/spiasecki/miniconda3
CONDA_ENV=terra

# Activate conda environment properly for batch jobs
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

# Set environment variables and run training
export DATASET_PATH=/cluster/home/spiasecki/terra/data/
export DATASET_SIZE=1000

# Add the terra package to the Python path
export PYTHONPATH=$PYTHONPATH:/cluster/home/lterenzi/terra_jax

# Change to the directory containing train.py or use the full path
cd /cluster/home/spiasecki/terra-baselines
python train.py
