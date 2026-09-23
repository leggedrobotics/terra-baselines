#!/usr/bin/env bash
#SBATCH --account=d130
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-team-eval
# usage: sbatch evaluate_panel.sbatch EVAL.edf.toml   (EVAL_* variables: see evaluate_panel.sh)
set -euo pipefail
[[ "$(id -un)" == lterenzi ]] || exit 2
[[ $# == 1 ]] || { echo "usage: evaluate_panel.sbatch EVAL.edf.toml" >&2; exit 2; }
exec srun --mpi=pmix --network=disable_rdzv_get --kill-on-bad-exit=1 \
    --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 --cpu-bind=cores \
    --environment="$1" bash -c 'bash "$BASELINES_ROOT/scripts/team/evaluate_panel.sh"'
