#!/bin/bash
# Recommended srun command (if running from login node):
# srun --x11 --cpus-per-task=1 --mem-per-cpu=4G /cluster/home/alesweber/TerraProject/terra-baselines/cluster/main_manual_srun.sh
# Or use sbatch for full resource allocation:
# sbatch terra-baselines/cluster/train_cluster.sh
#
# Or request interactive session first:
# salloc --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:rtx_4090:2 --time=2:00:00
# srun terra-baselines/cluster/visualize_srun.sh --run_name experiment-local-2025-06-28-21-43-50.pkl

# Option 1: X11 forwarding (connect with: ssh -Y alesweber@eu-login-07.ethz.ch)
# Option 2: VNC (run: vncserver :1, then connect with VNC client to eu-login-07.ethz.ch:5901)

# Configure X11 forwarding for -Y session
export DISPLAY=localhost:10.0
# Disable audio only (keep video for X11)
export SDL_AUDIODRIVER=dummy
# Suppress Mesa/OpenGL warnings and improve compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
# Use X11 driver for display
export SDL_VIDEODRIVER=x11
export SDL_RENDER_DRIVER=software

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
export DATASET_SIZE=10

export JAX_PLATFORMS=cpu


# Change to the correct directory
cd /cluster/home/alesweber/TerraProject/terra/terra/viz
python main_manual.py