#!/usr/bin/env bash
#SBATCH --account=d130
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-foundation-scratch
set -euo pipefail
CAMPAIGN_ROOT="${1:?campaign root required}"
test "$(id -un)" = lterenzi
test -r "$CAMPAIGN_ROOT/launch/terra.edf.toml"
printf '%s\n' "job=$SLURM_JOB_ID" "host=$(hostname)" "started=$(date -Is)"
scontrol show job "$SLURM_JOB_ID" -o
# Check the complete node's runtime once before binding one GPU to each policy.
srun --mpi=pmix --network=disable_rdzv_get --ntasks=1 --gpus-per-task=4 \
    --cpus-per-task=64 --gpu-bind=none --cpu-bind=cores \
    --environment="$CAMPAIGN_ROOT/launch/terra.edf.toml" \
    python -u /workspace/terra-baselines/cluster/cscs/check_jax_runtime.py --min-devices 4
exec srun --mpi=pmix --network=disable_rdzv_get --kill-on-bad-exit=1 \
    --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
    --environment="$CAMPAIGN_ROOT/launch/terra.edf.toml" \
    bash /workspace/terra-baselines/scripts/foundation_reward_sweep/run_cscs_scratch_arm.sh
