#!/usr/bin/env bash
#SBATCH --account=d130
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-pair
# usage: sbatch run_pair.sbatch ARM_A.edf.toml ARM_B.edf.toml
# Two independent two-GPU runs of run.sh on one node; both EDFs set DEVICES=2.
# A failing arm does not stop the other.
set -euo pipefail
[[ "$(id -un)" == lterenzi ]] || exit 2
[[ $# == 2 ]] || { echo "usage: run_pair.sbatch ARM_A.edf.toml ARM_B.edf.toml" >&2; exit 2; }
for edf in "$1" "$2"; do
    srun --mpi=pmix --network=disable_rdzv_get --exact --ntasks=1 --gpus-per-task=2 \
        --cpus-per-task=64 --mem=400G --cpu-bind=cores \
        --environment="$edf" bash -c 'bash "$BASELINES_ROOT/scripts/team/run.sh"' &
done
wait
