#!/bin/bash
#SBATCH --job-name=random_parallel_level5
#SBATCH --array=0-99            
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs_level5/job_%A_%a.out
#SBATCH --error=logs_level5/job_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gioelemo@ethz.ch

# Load necessary modules
module load eth_proxy
mkdir -p logs

export GOOGLE_API_KEY=...

# Set paths to conda
CONDA_ROOT=/cluster/home/gioelemo/miniconda3
CONDA_ENV=terra

# Activate conda environment
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

# Set environment variables
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu

export CUDA_VISIBLE_DEVICES=""

export DATASET_PATH=/cluster/home/gioelemo/terra_jax/terra/train4/train
export DATASET_SIZE=500

# Run the Python script for the given level

python -m main_llm_random \
  --model_name gemini-2.5-pro \
  --model_key gemini \
  --num_timesteps 400 \
  -s 1 --n_maps 100 --n_partitions_per_map 20  \
  -run /cluster/home/gioelemo/terra_jax/terra/tracked-dense.pkl \
  --level_index 5 \
  --map_idx $SLURM_ARRAY_TASK_ID \
  --use_rendering

