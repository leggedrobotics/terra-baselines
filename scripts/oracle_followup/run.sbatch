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
#SBATCH --job-name=terra-oracle-long
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: run.sbatch CONTAINER.edf.toml BASELINES_ROOT" >&2; exit 2; }
[[ "$(id -un)" == lterenzi ]] || exit 2
if [[ -n "${PREVIOUS_JOB_ID:-}" ]]; then
    previous_state="$(sacct -j "$PREVIOUS_JOB_ID" -X -n -o State | xargs)"
    case "$previous_state" in
        COMPLETED|TIMEOUT) ;;
        *) echo "Predecessor $PREVIOUS_JOB_ID ended $previous_state; stopping continuation." >&2; exit 2 ;;
    esac
fi
exec srun --mpi=pmix --network=disable_rdzv_get --kill-on-bad-exit=1 \
    --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 --cpu-bind=cores \
    --environment="$1" bash "$2/scripts/oracle_followup/run.sh"
