#!/bin/bash

set -euo pipefail
: "${CONDA_ROOT:?Set CONDA_ROOT to the selected account runtime installation}"
: "${TERRA_BASELINES_ROOT:?Set TERRA_BASELINES_ROOT to the staged terra-baselines checkout}"
: "${DATASET_PATH:?Set DATASET_PATH to the selected readable map bank}"

# Legacy 2D policy GIF wrapper. Run inside an allocation selected through
# cluster/README.md; this script does not request resources or submit a job.
# Example inside that allocation:
# srun cluster/visualize_srun.sh --run_name checkpoint.pkl -nx 1 -ny 1 -o rollout.gif
# Saved 3D recordings use terra-postprocess render, without policy inference.

# Set up environment
module load eth_proxy
module load stack/2024 cuda/12.1.1

# Set paths to conda and initialize properly
CONDA_ENV=terra

# Initialize conda properly
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

export DATASET_PATH
export DATASET_SIZE="${DATASET_SIZE:-200}"


# Change to the correct directory
cd "$TERRA_BASELINES_ROOT"
exec python visualize_mixed.py "$@"
