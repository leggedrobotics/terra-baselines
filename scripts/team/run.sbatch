#!/usr/bin/env bash
#SBATCH --account=d130
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-team
# usage: sbatch run.sbatch CONTAINER.edf.toml   (environment: see scripts/team/README.md)
set -euo pipefail
[[ "$(id -un)" == lterenzi ]] || exit 2
[[ $# == 1 ]] || { echo "usage: run.sbatch CONTAINER.edf.toml" >&2; exit 2; }
exec srun --mpi=pmix --network=disable_rdzv_get --kill-on-bad-exit=1 \
    --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 --cpu-bind=cores \
    --environment="$1" bash -c 'bash "$BASELINES_ROOT/scripts/team/run.sh"'
