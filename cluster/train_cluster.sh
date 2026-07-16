#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:rtx_3090:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_training.out


#--gpus=gpu:rtx_4090:8

# Load required modules
module load eth_proxy
module load stack/2024-06 cuda/12.1.1

# Set paths to conda
CONDA_ROOT=/cluster/home/alesweber/miniconda3
CONDA_ENV=terra

# Activate conda environment properly for batch jobs
eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

# Set environment variables and run training
export DATASET_PATH=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train/
export DATASET_SIZE=600


# Change to the directory containing train.py or use the full path
cd /cluster/project/rsl/alesweber/TerraProject/terra-baselines

python train_mixed.py \
    --config solo_excavator \
    --model_size medium \
    --model_core mlp \
    --resume_from /cluster/project/rsl/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2026-07-14-17-30-39.pkl
    # --num_envs_per_device 128 \
    # --map_path /cluster/project/rsl/alesweber/TerraProject/terra-baselines/inference/maps/map_13 \
    # --replay_map_count 15 --target_map_repeat 10 \
    # --total_timesteps 5_000_000_000 \
    # --resume_from /cluster/project/rsl/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2026-07-07-12-56-06.pkl
    #--resume_from /cluster/project/rsl/alesweber/TerraProject/terra-baselines/checkpoints/mixed-agents-skidsteer-skidsteer-local-2026-05-21-17-38-51-best_rectangles.pkl \
    #--no-load-env-from-checkpoint
    





#WANDB_MODE=offline 
# solo_excavator solo_excavator_dumpzone  
# model_size: base, medium, large
