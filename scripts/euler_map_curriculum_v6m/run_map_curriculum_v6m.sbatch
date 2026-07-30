#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-v6m
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v6m_20260730/logs/%x_%j.out

# M2 dose-curriculum campaign (CURRICULUM_SPEC_V6.md section 4). Three arms,
# 8,000 updates each, all FROM SCRATCH, reward-v2 ACTIVE on all three
# (--require_reward_v2 makes a reward-v1 config a launch failure, not a readout
# surprise).
#
#   sbatch --job-name=terra-v6m-wave-long --export=ALL,ARM=WAVE run_map_curriculum_v6m.sbatch
#   sbatch --job-name=terra-v6m-dose      --export=ALL,ARM=DOSE run_map_curriculum_v6m.sbatch
#   sbatch --job-name=terra-v6m-dose-fast --export=ALL,ARM=FAST run_map_curriculum_v6m.sbatch
#
# --time=20:00:00 is set from MEASURED M1 elapsed (sacct -j 9046454/9046455/
# 9046456: 06:46:51 / 05:33:11 / 06:11:22 for 4,000 updates), i.e. 11-14 h
# projected for 8,000 updates, plus margin inside the gpuhe.24h ceiling.
#
# DO NOT REQUEUE. REVIEW_V6 F5-3: `cond_cum/*` is rebuilt at update 0 and resets
# to zero on resume, and `_write_episode_aggregate_receipt` raises
# FileExistsError when resuming from a checkpoint older than the last written
# receipt. If an arm dies, fix F5-3 first and restart the arm from scratch, or
# accept post-resume-only cumulative counters and say so in the readout.

set -euo pipefail
umask 002

: "${ARM:?submit with --export=ALL,ARM=WAVE|DOSE|FAST}"
case "$ARM" in
    WAVE) PRESET=m2_wave_long; RUN_NAME=terra-v6m-wave-long; BANK_SUBDIR=v5m; SEED=20260730 ;;
    DOSE) PRESET=m2_dose;      RUN_NAME=terra-v6m-dose;      BANK_SUBDIR=v6m; SEED=20260731 ;;
    FAST) PRESET=m2_dose_fast; RUN_NAME=terra-v6m-dose-fast; BANK_SUBDIR=v6m; SEED=20260732 ;;
    *) echo "Unsupported ARM=$ARM" >&2; exit 2 ;;
esac

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v6m_20260730
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank/$BANK_SUBDIR"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$ARM"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="$SOURCE/terra:$SOURCE/terra-baselines"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_MODE=online
export DATASET_PATH="$BANK"
export DATASET_SIZE=864

mkdir -p "$CHECKPOINT_DIR" "$RUN_ROOT/wandb" "$RUN_ROOT/logs"
test -f "$RUN_ROOT/LOCAL_SMOKE_PASSED"
test -f "$RUN_ROOT/SOURCE_SYNC_VERIFIED"
test -f "$BANK/held_out/all/manifest.jsonl"
if [ "$BANK_SUBDIR" = v6m ]; then
    test -f "$BANK/curriculum_v6m_provenance.json"
    for L in L0p L1p L2p L3p; do test -f "$BANK/train/$L/dataset.json"; done
else
    test -f "$BANK/curriculum_v5m_provenance.json"
    for L in L0 L1 L2 L3; do test -f "$BANK/train/$L/dataset.json"; done
fi

HOME_USED_GB="$(
    lquota |
        awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {
            value=$4
            gsub(/[^0-9.]/, "", value)
            print value
        }'
)"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "Home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 40
}

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "Unsupported GPU: $GPU_NAME" >&2
        exit 41
    }
done
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 4

echo "=== Terra M2 dose curriculum: arm $ARM ($RUN_NAME) ==="
echo "job_id=$SLURM_JOB_ID"
echo "preset=$PRESET  seed=$SEED  init=scratch (no warm start, no teacher)"
echo "bank=$BANK  DATASET_SIZE=$DATASET_SIZE"
echo "reward=v2 (excavator_relocate_dumped_mult 0.2 < dug 1.5), enforced by --require_reward_v2"
echo "arch=medium + mlp + resnet_spatial_8x8_se (bf16 encoder, critic 512,256) = 2,441,223 params"
echo "shape=4 devices x 1024 envs x 32 steps = 131,072 transitions/update, 8000 updates"
echo "horizon=450 (set on every curriculum level; curriculum.py rewrites env_cfg from that table)"
cat "$RUN_ROOT/source_sync_manifest.txt" || true
scontrol show job "$SLURM_JOB_ID" | grep -E 'JobId=|Partition=|ReqTRES=|AllocTRES=' || true

cd "$SOURCE/terra-baselines"
unset WANDB_RUN_ID WANDB_RESUME
"$VENV/bin/python" -u train_mixed.py \
    --config "$PRESET" \
    --name "$RUN_NAME" \
    --machine euler \
    --seed "$SEED" \
    --num_devices 4 \
    --num_envs_per_device 1024 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 1048576000 \
    --lr 3e-4 \
    --require_reward_v2 \
    --log_train_interval 1 \
    --log_eval_interval 0 \
    --checkpoint_interval 250 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --keep_checkpoint_history \
    --cache_clear_interval 1000 \
    --eval_episodes 64 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 7600 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval 10

FINAL_COUNT="$(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*_FINAL.pkl' | wc -l
)"
test "$FINAL_COUNT" -eq 1
touch "$RUN_ROOT/${ARM}_TRAINING_COMPLETE"
echo "ARM_${ARM}_TRAINING_COMPLETE"
