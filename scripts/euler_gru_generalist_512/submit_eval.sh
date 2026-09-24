#!/usr/bin/env bash
# usage: submit_eval.sh REMOTE_CHECKPOINT OUTPUT_NAME [rtx_4090|rtx_3090]
# Submits eval.sbatch from the staged source of HEAD (run SUBMIT=stage first).
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: submit_eval.sh REMOTE_CHECKPOINT OUTPUT_NAME [GPU_TYPE]" >&2; exit 2; }
CHECKPOINT="$1"
NAME="$2"
GPU_TYPE="${3:-rtx_4090}"
[[ "$NAME" =~ ^[a-zA-Z0-9_.-]+$ ]]
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="$(dirname "$REPO")/terra"
REVISION="$(git -C "$REPO" rev-parse HEAD)"
TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
ROOT=/cluster/project/rsl/lterenzi/terra_experiments/terra_gru_bigbank_20260923
SOURCE="$ROOT/src/$REVISION/terra-baselines"
OUTPUT="$ROOT/evaluations/$NAME"
EXPORTS="ALL,CHECKPOINT=$CHECKPOINT,OUTPUT=$OUTPUT,BASELINES_ROOT=$SOURCE"
EXPORTS+=",RUNTIME_TERRA_ROOT=$ROOT/runtime-terra/$TERRA_REVISION/terra"
EXPORTS+=",VENV=/cluster/project/rsl/lterenzi/terra_runtime/terra_jax0433_cuda126_cudnn950_20260903"
EXPORTS+=",EVAL_BANK=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_latest_geometry_20260921/inputs/evaluation_bank"
EXPORTS+=",KNOWN_MAPS=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_test_time_compute_20260921/adaptation/maps"
EXPORTS+=",KNOWN_HELPER=/cluster/scratch/lterenzi/codex_terra_edge_validation/terra_test_time_compute_20260922/helpers/adaptation/evaluate.py"
EXPORTS+=",GPU_TYPE=$GPU_TYPE"
ssh -o BatchMode=yes euler-lterenzi "test -e '$SOURCE' && test -r '$CHECKPOINT' && test ! -e '$OUTPUT' && mkdir -p '$OUTPUT' && cat '$SOURCE/scripts/euler_gru_generalist_512/eval.sbatch' | sbatch --parsable --account=es_hutter --partition=gpuhe.4h --time=04:00:00 --gpus='$GPU_TYPE:1' --cpus-per-task=8 --mem-per-cpu=8G --job-name='terra-gru-eval' --output='$OUTPUT/slurm_%j.out' --export='$EXPORTS'"
