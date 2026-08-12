#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-sv3-E9b
#SBATCH --output=/cluster/home/lterenzi/codex_terra_edge_validation/logs/%x_%j.out

set -euo pipefail
umask 002

TERRA_WORK=/cluster/home/lterenzi/codex_terra_edge_validation
TERRA_SOURCE="$TERRA_WORK/snapshots/f16-e9e10-20260722"
TERRA_VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
TERRA_DATASET=/cluster/scratch/lterenzi/codex_terra_edge_runs/datasets/train_128
TERRA_RUNTIME_CHECK="$TERRA_WORK/scripts/check_jax_runtime.py"
TEACHER_CKPT="$TERRA_WORK/snapshots/spatial-v3-3a21cd6/terra-baselines/checkpoints/terra-sv3-E3-kickstart-med-euler-2026-07-20-20-45-27_FINAL.pkl"
STUDENT_INIT="$TERRA_SOURCE/terra-baselines/checkpoints/grown_medium_se_5stage_128.pkl"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

TERRA_SITE_PACKAGES="$("$TERRA_VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export PYTHONPATH="$TERRA_SOURCE/terra:$TERRA_SOURCE/terra-baselines${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$TERRA_SITE_PACKAGES/nvidia/cudnn/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_cupti/lib:$TERRA_SITE_PACKAGES/nvidia/cublas/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$TERRA_SITE_PACKAGES/nvidia/nccl/lib:$TERRA_SITE_PACKAGES/nvidia/cusolver/lib:$TERRA_SITE_PACKAGES/nvidia/cusparse/lib:$TERRA_SITE_PACKAGES/nvidia/cufft/lib:$TERRA_SITE_PACKAGES/nvidia/nvjitlink/lib:$TERRA_SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$TERRA_WORK/wandb"
export DATASET_PATH="$TERRA_DATASET"
export DATASET_SIZE=600

mkdir -p "$TERRA_WORK/logs" "$TERRA_WORK/wandb" "$TERRA_SOURCE/terra-baselines/checkpoints"

echo "=== Terra spatial-v3 E9b (128x128 relaunch, teacher-env fix) ==="
echo "job_id=$SLURM_JOB_ID"
echo "source=$TERRA_SOURCE"
echo "terra_revision=34b5d39d"
echo "terra_baselines_base_revision=88ccaba-dirty (agent/spatial-v3-improvements snapshot)"
echo "config=solo_excavator_128 dataset=foundations_128"
echo "map_encoder=resnet_spatial_8x8_se model_size=medium encoder_compute_dtype=bfloat16 critic_hidden_dims=512,256"
echo "resume_from=$STUDENT_INIT"
echo "teacher_checkpoint=$TEACHER_CKPT teacher_obs_downsample=2"
echo "devices=4 envs_per_device=512 num_steps=32 update_epochs=2 num_minibatches=32"
echo "total_timesteps=655360000 (10000 updates)"
echo "ent_schedule=0.02->0.005 over 10000 updates (sed-patched defaults; no CLI flags exist)"
echo "new_flags=--no_value_clip --flat_minibatch_shuffle --resnet_stage_channels 24,48,64,96,96 --resnet_blocks_per_stage 1,2,2,2,2 --reward_normalizer 280 --agent_move_tiles 10 --dig_radius_tiles 10"
scontrol show job "$SLURM_JOB_ID" | grep -E 'JobId=|Partition=|ReqTRES=|AllocTRES=' || true

test -x "$TERRA_VENV/bin/python"
test -f "$TERRA_RUNTIME_CHECK"
test -d "$TERRA_DATASET/foundations_128"
for i in 1 2 3 4 5; do test -f "$TEACHER_CKPT" && break || sleep 20; done; test -f "$TEACHER_CKPT"
for i in 1 2 3 4 5; do test -f "$STUDENT_INIT" && break || sleep 20; done; test -f "$STUDENT_INIT"

test "$(sha256sum "$TERRA_SOURCE/terra/terra/env.py" | awk '{print $1}')" = "628b897fcbd372e86e7ee5c2c3cb41f7b9c9300539c388452085bf66acea9685"
test "$(sha256sum "$TERRA_SOURCE/terra/terra/state.py" | awk '{print $1}')" = "ef53459f7177f994b1dd54375a7a9be479e3391d334bc278e6409c9395f2fafa"
test "$(sha256sum "$TERRA_SOURCE/terra-baselines/train_mixed.py" | awk '{print $1}')" = "2efc18f88862738984852f2d55aaeff1d37f95a3f6e704692d432636560e420b"
test "$(sha256sum "$TERRA_SOURCE/terra-baselines/utils/models.py" | awk '{print $1}')" = "2205cab36694f90d56bf65a5511449eaccf1d75d2749a23507ec1f0b62c39f18"
test "$(sha256sum "$TERRA_SOURCE/terra-baselines/scripts/grow_checkpoint.py" | awk '{print $1}')" = "32e8a3e4fadcafd5286e8adcd69ab7ac4d47995df2ba7e2f6c6dfb828ebd7847"

mapfile -t TERRA_GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "Allocated GPUs (${#TERRA_GPU_NAMES[@]}):"
printf '  %s\n' "${TERRA_GPU_NAMES[@]}"
if [ "${#TERRA_GPU_NAMES[@]}" -ne 4 ]; then
    echo "Expected exactly 4 visible GPUs" >&2
    exit 42
fi
for TERRA_GPU_NAME in "${TERRA_GPU_NAMES[@]}"; do
    if [ "$TERRA_GPU_NAME" != "NVIDIA GeForce RTX 4090" ]; then
        echo "Unsupported GPU: $TERRA_GPU_NAME (RTX 4090 only)" >&2
        exit 43
    fi
done

"$TERRA_VENV/bin/python" "$TERRA_RUNTIME_CHECK" --min-devices 4

cd "$TERRA_SOURCE/terra-baselines"

TRAIN_ENTRY=train_mixed_sv3_e9b.py
test "$(grep -c '^    ent_schedule_start: float = 0.15$' train_mixed.py)" = 1
test "$(grep -c '^    ent_schedule_steps: int = 9500$' train_mixed.py)" = 1
sed -e 's/^    ent_schedule_start: float = 0.15$/    ent_schedule_start: float = 0.02/' \
    -e 's/^    ent_schedule_steps: int = 9500$/    ent_schedule_steps: int = 10000/' \
    train_mixed.py > "$TRAIN_ENTRY"
test "$(grep -c '^    ent_schedule_start: float = 0.02$' "$TRAIN_ENTRY")" = 1
test "$(grep -c '^    ent_schedule_steps: int = 10000$' "$TRAIN_ENTRY")" = 1
sed -e 's/^    ent_schedule_start: float = 0.02$/    ent_schedule_start: float = 0.15/' \
    -e 's/^    ent_schedule_steps: int = 10000$/    ent_schedule_steps: int = 9500/' \
    "$TRAIN_ENTRY" | cmp - train_mixed.py || { echo "ent patch produced unexpected diff" >&2; exit 45; }
echo "ENT_PATCH_VERIFIED: ent_schedule_start 0.15 -> 0.02, ent_schedule_steps 9500 -> 10000 in $TRAIN_ENTRY"

echo "=== First-update smoke gate (W&B disabled) ==="
export WANDB_MODE=disabled
"$TERRA_VENV/bin/python" -u "$TRAIN_ENTRY" \
    --config solo_excavator_128 \
    --name terra-sv3-E9b-res128-ks-smoke \
    --machine euler \
    --num_devices 4 \
    --num_envs_per_device 512 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 65536 \
    --log_train_interval 1 \
    --log_eval_interval 0 \
    --checkpoint_interval 0 \
    --cache_clear_interval 0 \
    --eval_episodes 4 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --resume_from "$STUDENT_INIT" \
    --no-load-env-from-checkpoint \
    --teacher_checkpoint "$TEACHER_CKPT" \
    --kickstart_kl_coef 1.0 \
    --kickstart_kl_anneal_updates 1000 \
    --kickstart_value_coef 0.5 \
    --kickstart_value_anneal_updates 500 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --resnet_stage_channels 24,48,64,96,96 \
    --resnet_blocks_per_stage 1,2,2,2,2 \
    --reward_normalizer 280 \
    --agent_move_tiles 10 \
    --dig_radius_tiles 10 \
    --teacher_obs_downsample 2
echo "SMOKE_GATE_PASSED"

echo "=== Production training (W&B online) ==="
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME
"$TERRA_VENV/bin/python" -u "$TRAIN_ENTRY" \
    --config solo_excavator_128 \
    --name terra-sv3-E9b-res128-ks \
    --machine euler \
    --num_devices 4 \
    --num_envs_per_device 512 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 655360000 \
    --log_train_interval 1 \
    --log_eval_interval 100 \
    --checkpoint_interval 100 \
    --cache_clear_interval 1000 \
    --eval_episodes 100 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --resume_from "$STUDENT_INIT" \
    --no-load-env-from-checkpoint \
    --teacher_checkpoint "$TEACHER_CKPT" \
    --kickstart_kl_coef 1.0 \
    --kickstart_kl_anneal_updates 1000 \
    --kickstart_value_coef 0.5 \
    --kickstart_value_anneal_updates 500 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --resnet_stage_channels 24,48,64,96,96 \
    --resnet_blocks_per_stage 1,2,2,2,2 \
    --reward_normalizer 280 \
    --agent_move_tiles 10 \
    --dig_radius_tiles 10 \
    --teacher_obs_downsample 2
