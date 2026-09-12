#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:rtx_4090:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_training.out

set -euo pipefail
: "${CONDA_ROOT:?Set CONDA_ROOT to the selected account runtime installation}"
: "${TERRA_BASELINES_ROOT:?Set TERRA_BASELINES_ROOT to the staged terra-baselines checkout}"
: "${DATASET_PATH:?Set DATASET_PATH to the selected readable map bank}"



#--gpus=gpu:rtx_4090:8

# Load required modules
module load eth_proxy
module load stack/2024-06 cuda/12.1.1

# Set paths to conda
CONDA_ENV=terra

# Activate conda environment properly for batch jobs
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

# Set environment variables and run training
export DATASET_PATH
export DATASET_SIZE=600


# Change to the directory containing train.py or use the full path
cd "$TERRA_BASELINES_ROOT"

python train_mixed.py \
    --config solo_excavator_direct_dig_reward \
    --model_size base \
    --model_core mlp
    #--map_encoder resnet_spatial_v2 \
    #--num_envs_per_device 512
    # --replay_map_count 15 --target_map_repeat 10 \
    # --total_timesteps 5_000_000_000 \
    #--map_encoder resnet_spatial_v2 \

    #--map_encoder resnet_spatial_8x8
    
    
    #--no-load-env-from-checkpoint
    





#WANDB_MODE=offline 
# solo_excavator solo_excavator_dumpzone  
# model_size: base, medium, large
