#!/bin/bash
# Recommended srun command (if running from login node):
# srun --cpus-per-task=1 --mem-per-cpu=4G --gres=gpu:rtx_4090:2 --time=2:00:00 terra-baselines/cluster/visualize_srun.sh --run_name experiment-local-2025-06-28-21-43-50.pkl
#
# Or use sbatch for full resource allocation:
# sbatch terra-baselines/cluster/train_cluster.sh
#
# Or request interactive session first:
# salloc --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:rtx_4090:2 --time=2:00:00
# srun terra-baselines/cluster/visualize_srun.sh --run_name experiment-local-2025-06-28-21-43-50.pkl

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
python visualize.py