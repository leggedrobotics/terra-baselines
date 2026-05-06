#!/bin/bash
# Recommended srun command:
# srun --cpus-per-task=8 --mem-per-cpu=8G --gres=gpu:rtx_4090:8 --time=12:00:00 terra-baselines/cluster/train_srun.sh
#
# Or for shorter test runs:
# srun --cpus-per-task=1 --mem-per-cpu=4G --gres=gpu:rtx_4090:1 --time=2:00:00 terra-baselines/cluster/train_srun.sh

# Set up environment
module load eth_proxy
module load stack/2024 cuda/12.1.1

# Set paths to conda and initialize properly
CONDA_ROOT=/cluster/home/alesweber/miniconda3
CONDA_ENV=terra

# Initialize conda properly
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

export DATASET_PATH=/cluster/home/alesweber/TerraProject/terra/data/terra/train/
export DATASET_SIZE=200


# Change to the correct directory
cd /cluster/home/alesweber/TerraProject/terra-baselines
python train_mixed.py --config solo_excavator \