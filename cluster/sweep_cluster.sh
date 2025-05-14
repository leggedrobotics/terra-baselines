#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="sweep-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_sweep.out

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

CONDA_ROOT=/cluster/home/spiasecki/miniconda3
CONDA_ENV=terra

eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

export DATASET_PATH=/cluster/home/spiasecki/terra/data/
export DATASET_SIZE=1000

cd /cluster/home/spiasecki/terra-baselines

# Create the sweep and capture the sweep ID
SWEEP_ID=$(python -c "import sweep; print(sweep.create_sweep_and_return_id())")

# Run two agents in parallel
wandb agent $SWEEP_ID &  # Agent 1
wandb agent $SWEEP_ID &  # Agent 2
wandb agent $SWEEP_ID &  # Agent 3
wandb agent $SWEEP_ID &  # Agent 4

wait
