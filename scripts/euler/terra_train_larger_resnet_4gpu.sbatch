#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --partition=gpuhe.120h
#SBATCH --nodelist=eu-g4-[001-032],eu-g6-[001-080]
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-bigresnet
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%x_%j.out

set -euo pipefail

WORK=/cluster/home/lterenzi/codex_terra_edge_validation
VENV=/cluster/scratch/lterenzi/codex_terra_edge_venv
DATASET=/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train
TEACHER_CHECKPOINT="$WORK/checkpoints/terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516-euler-4gpu-2026-05-16-12-18-34.pkl"
RUN_KIND="${RUN_KIND:-medium_distill}"
WANDB_GROUP="solo-resnet-warmstart-20260518"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES=$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export PYTHONPATH="$WORK/terra:$WORK/terra-baselines:${PYTHONPATH:-}"
export DATASET_PATH="$DATASET"
export DATASET_SIZE=600
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$WORK/wandb"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$WANDB_DIR" "$WORK/logs" "$WORK/checkpoints"
cd "$WORK/terra-baselines"

case "$RUN_KIND" in
  medium_distill)
    MODEL_SIZE=medium
    MAP_FEATURE_DIM=192
    NUM_MINIBATCHES=64
    IMITATION_UPDATES="${IMITATION_UPDATES:-200}"
    IMITATION_NUM_MINIBATCHES=64
    USE_TEACHER=1
    RUN_NAME="terra-solo-resmap192-medium-distill${IMITATION_UPDATES}-mb64-unmasked-4gpu-50B-20260518"
    WANDB_TAGS="solo_excavator,resnet_delayed,model_size:medium,distill,teacher:resmap64"
    ;;
  medium_scratch)
    MODEL_SIZE=medium
    MAP_FEATURE_DIM=192
    NUM_MINIBATCHES=64
    IMITATION_UPDATES=0
    IMITATION_NUM_MINIBATCHES=64
    USE_TEACHER=0
    RUN_NAME="terra-solo-resmap192-medium-scratch-mb64-unmasked-4gpu-50B-20260518"
    WANDB_TAGS="solo_excavator,resnet_delayed,model_size:medium,scratch"
    ;;
  large_distill)
    MODEL_SIZE=large
    MAP_FEATURE_DIM=256
    NUM_MINIBATCHES=128
    IMITATION_UPDATES="${IMITATION_UPDATES:-100}"
    IMITATION_NUM_MINIBATCHES=128
    USE_TEACHER=1
    RUN_NAME="terra-solo-resmap256-large-distill${IMITATION_UPDATES}-mb128-autotune0-unmasked-4gpu-50B-20260518"
    WANDB_TAGS="solo_excavator,resnet_delayed,model_size:large,distill,teacher:resmap64,autotune0"
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
    ;;
  large_scratch)
    MODEL_SIZE=large
    MAP_FEATURE_DIM=256
    NUM_MINIBATCHES=128
    IMITATION_UPDATES=0
    IMITATION_NUM_MINIBATCHES=128
    USE_TEACHER=0
    RUN_NAME="terra-solo-resmap256-large-scratch-mb128-autotune0-unmasked-4gpu-50B-20260518"
    WANDB_TAGS="solo_excavator,resnet_delayed,model_size:large,scratch,autotune0"
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
    ;;
  *)
    echo "Unknown RUN_KIND: $RUN_KIND" >&2
    exit 2
    ;;
esac

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
printf 'Allocated GPUs:\n'
printf '  %s\n' "${GPU_NAMES[@]}"
if [ "${#GPU_NAMES[@]}" -ne 4 ]; then
  echo "Expected exactly 4 GPUs, got ${#GPU_NAMES[@]}" >&2
  exit 42
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  case "$gpu_name" in
    "NVIDIA GeForce RTX 3090"|"NVIDIA GeForce RTX 4090") ;;
    *)
      echo "Refusing non-3090/4090 GPU: $gpu_name" >&2
      exit 43
      ;;
  esac
done

if [ "$USE_TEACHER" -eq 1 ] && [ ! -f "$TEACHER_CHECKPOINT" ]; then
  echo "Missing teacher checkpoint: $TEACHER_CHECKPOINT" >&2
  exit 44
fi

"$VENV/bin/python" "$WORK/terra-baselines/scripts/euler/check_jax_runtime.py" --min-devices 4

COMMON_ARGS=(
  --config solo_excavator
  --project mixed-agents
  --group "$WANDB_GROUP"
  --tags "$WANDB_TAGS"
  --num_devices 4
  --num_envs_per_device 1024
  --num_steps 32
  --update_epochs 2
  --num_minibatches "$NUM_MINIBATCHES"
  --map_encoder resnet_delayed
  --map_feature_dim "$MAP_FEATURE_DIM"
  --use_map_derived_channels
  --separate_actor_critic_trunks
  --use_critic_affordances
  --include_episode_progress
  --edge_features_dim 10
  --model_size "$MODEL_SIZE"
)

TEACHER_ARGS=()
if [ "$USE_TEACHER" -eq 1 ]; then
  TEACHER_ARGS=(
    --teacher_checkpoint "$TEACHER_CHECKPOINT"
    --imitation_num_steps 32
    --imitation_num_minibatches "$IMITATION_NUM_MINIBATCHES"
    --imitation_lr 0.0001
    --imitation_temperature 1.0
    --imitation_value_coef 0.25
  )
fi

SMOKE_IMITATION_UPDATES=0
if [ "$USE_TEACHER" -eq 1 ]; then
  SMOKE_IMITATION_UPDATES=1
fi

WANDB_MODE=disabled "$VENV/bin/python" train_mixed.py \
  "${COMMON_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --imitation_updates "$SMOKE_IMITATION_UPDATES" \
  --name "${RUN_NAME}-smoke" \
  --machine euler-smoke-fullshape \
  --total_timesteps 131072 \
  --log_train_interval 0 \
  --log_eval_interval 1 \
  --checkpoint_interval 1 \
  --eval_episodes 4 \
  --num_rollouts_eval 8

WANDB_MODE=online "$VENV/bin/python" train_mixed.py \
  "${COMMON_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --imitation_updates "$IMITATION_UPDATES" \
  --name "$RUN_NAME" \
  --machine euler-4gpu \
  --total_timesteps 50000000000 \
  --log_train_interval 1 \
  --log_eval_interval 100 \
  --checkpoint_interval 100 \
  --eval_episodes 100 \
  --num_rollouts_eval 550
