#!/usr/bin/env bash
# Stage and submit the GRU fine-tune under Terra's machine working rules: a
# native resume of the released GRU u100000 (model, Adam state and update
# counter) with dig 4.0-6.5 m, dump 4.0-6.0 m, the chassis raster centred on
# the base cell (preset gru_generalist_512_machine_rules) and a clearance of
# DUG_CLEARANCE_M from dug ground, for 10,000 updates to u110000 on 4 x RTX
# 4090. Release recipe otherwise (reward_v2, lr 3e-4, entropy 0.02, 2 x 32
# minibatches, 512 envs per GPU, bank train_v3_generalist_512). The teacher is
# passed only because a native resume of a teacher run must keep it; its KL
# weight reached 0 at u10000, so the trainer runs teacher-free.
#   DUG_CLEARANCE_M=0.57|0.6   required: one free cell, or two along an axis
#   SUBMIT=0      local checks only (default)
#   SUBMIT=stage  upload exact source, Terra runtime, bank, teacher and parent
#   SUBMIT=smoke  3-update finite smoke from u100000, W&B disabled, 4 h queue
#   SUBMIT=1      24 h segment of the fine-tune, resuming its newest checkpoint
#                 (the parent u100000 on the first segment); rerun after a
#                 timeout to continue
# Milestones are scored locally: eval_milestones.sh.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|stage|smoke|1) ;; *) echo "SUBMIT must be 0, stage, smoke or 1" >&2; exit 2 ;; esac
case "${DUG_CLEARANCE_M:-}" in
    0.57) CLEARANCE_TAG=c057 ;;
    0.6) CLEARANCE_TAG=c060 ;;
    *) echo "DUG_CLEARANCE_M must be 0.57 or 0.6" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/cluster/euler_account.sh"
terra_euler_configure lterenzi
# The Terra worktree with the machine working rules (branch terra-machine-rules).
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_machine_rules_20260930/terra}"

# Environment code must equal the dump-reach release env (9b96e9a4) plus the
# opt-in machine working rules (4c49f274, 15f7cc47) and the chassis centring
# (9300543d); tool-only commits on the Terra branch are allowed.
TERRA_ENV_REVISION=9300543dafee6b1a7fe104cbf964401b5737fa04
PRESET=gru_generalist_512_machine_rules
BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/train_v3_generalist_512.tar.zst
BANK_ARCHIVE_SHA=3ddaaf538f1e7a091123ca1491c7f305e03835822d4f4c83c17d1f5916f6fc74
BANK_MAPS_PATH=train_v3_generalist_512
BANK_DATASET_SIZE=20480
BANK_DISTANCE_SIDECAR_SHA=d9a8c9e61346e319b3ffead150ca44c7e85d016cb5853997cf1ee721286d4506
PARENT_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/checkpoints/gru_gen512_s20260923_update_100000.pkl
PARENT_SHA=ba5fdceac3be3ad6729da63400df72fa71c1ad527fa39b555299773dc7520c2d
PARENT_UPDATE=100000
TEACHER_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_instance_efficiency_20260922/inputs/generalist_u110000.pkl
TEACHER_SHA=3fd74795794dbf27a3c2fce01c91414bfb211b913dc9b018bc540d8a7acd7057
# The parent's kickstart schedule (655,360,000 transitions at 4 x 512 x 32);
# a native resume must repeat it.
KL_ANNEAL_UPDATES=10000
SEED="${TERRA_SEED:-20260930}"
NUM_DEVICES="${TERRA_NUM_DEVICES:-4}"
ENVS_PER_DEVICE=512
GPU_TYPE=rtx_4090
if [ "$SUBMIT" = smoke ]; then
    TARGET_UPDATE=$((PARENT_UPDATE + 3)) PARTITION=gpuhe.4h WALLTIME=01:30:00 WANDB_MODE=disabled
else
    TARGET_UPDATE=$((PARENT_UPDATE + 10000))
    WANDB_MODE="${WANDB_MODE:-online}"
    PARTITION="${TERRA_PARTITION:-gpuhe.24h}"
    WALLTIME="${TERRA_WALLTIME:-24:00:00}"
fi
[[ "$NUM_DEVICES" =~ ^[1-4]$ ]]
[ "$SUBMIT" = smoke ] || [ "$NUM_DEVICES" = 4 ]
REMOTE_HOST=euler-lterenzi
REMOTE_VENV=/cluster/project/rsl/lterenzi/terra_runtime/terra_jax0433_cuda126_cudnn950_20260903
RUNTIME_LOCK_SHA=36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189
ROOT="$TERRA_EULER_PROJECT_ROOT/terra_experiments/terra_gru_rules_ft_20260930"
RUN_NAME="gru_rules_ft_${CLEARANCE_TAG}_s$SEED"
WANDB_ENTITY=aless-weber-eth
WANDB_PROJECT=mixed-agents

test -z "$(git -C "$REPO" status --porcelain -- . ':(exclude)isaac_sim')"
test -z "$(git -C "$TERRA_REPO" status --porcelain)"
test -z "$(git -C "$TERRA_REPO" diff --name-only "$TERRA_ENV_REVISION" HEAD -- terra)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARENT_LOCAL" | awk '{print $1}')" = "$PARENT_SHA"
test "$(sha256sum "$TEACHER_LOCAL" | awk '{print $1}')" = "$TEACHER_SHA"
grep -q "path: &gen512_bank $BANK_MAPS_PATH\$" "$REPO/configs/training_configs.yaml"
# The preset's rules, as its YAML entry states them.
PRESET_RULES="$(python3 - "$REPO/configs/training_configs.yaml" "$PRESET" <<'PY'
import sys, yaml
preset = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
rules = {k: preset.get(k) for k in ("dig_min_radius_m", "dump_min_radius_m", "dump_max_radius_m",
                                    "dug_clearance_m", "dump_min_dug_distance_m", "centre_chassis_on_base")}
assert rules == dict(dig_min_radius_m=4.0, dump_min_radius_m=None, dump_max_radius_m=6.0, dug_clearance_m=None,
                     dump_min_dug_distance_m=None, centre_chassis_on_base=True), rules
assert preset["maps"][0]["path"] == "train_v3_generalist_512", preset["maps"]
print("dig_min_radius_m=4.0 dump_max_radius_m=6.0 centre_chassis_on_base=true")
PY
)"
echo "baselines=$BASELINES_REVISION terra=$RUNTIME_TERRA_REVISION seed=$SEED run=$RUN_NAME"
echo "parent=u$PARENT_UPDATE ($PARENT_SHA) preset=$PRESET $PRESET_RULES dug_clearance_m=$DUG_CLEARANCE_M"
echo "shape=${NUM_DEVICES}x$GPU_TYPE envs/device=$ENVS_PER_DEVICE target=u$TARGET_UPDATE partition=$PARTITION walltime=$WALLTIME wandb=$WANDB_MODE"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local checks passed; nothing staged"
    exit 0
fi

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -x '$REMOTE_VENV/bin/python' && test \"\$(sha256sum '$REMOTE_VENV/requirements.lock.txt' | awk '{print \$1}')\" = '$RUNTIME_LOCK_SHA'"
REMOTE_SOURCE="$ROOT/src/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$ROOT/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
remote "mkdir -p '$ROOT/src' '$ROOT/runtime-terra' '$ROOT/inputs'"
if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$ROOT/src/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$ROOT/src/$BASELINES_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    PARTIAL="$ROOT/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | remote "tar -xf - -C '$PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv -T '$PARTIAL' '$ROOT/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
upload() {  # local, remote, sha
    if ! remote "test -f '$2'"; then
        scp -q -o BatchMode=yes "$1" "$REMOTE_HOST:$2.partial.$$"
        remote "mv -T '$2.partial.$$' '$2'"
    fi
    remote "test \"\$(sha256sum '$2' | awk '{print \$1}')\" = '$3'"
}
REMOTE_BANK="$ROOT/inputs/bank-$BANK_ARCHIVE_SHA.tar.zst"
REMOTE_PARENT="$ROOT/inputs/parent-$PARENT_SHA.pkl"
REMOTE_TEACHER="$ROOT/inputs/teacher-$TEACHER_SHA.pkl"
upload "$BANK_ARCHIVE" "$REMOTE_BANK" "$BANK_ARCHIVE_SHA"
upload "$PARENT_LOCAL" "$REMOTE_PARENT" "$PARENT_SHA"
upload "$TEACHER_LOCAL" "$REMOTE_TEACHER" "$TEACHER_SHA"
remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: source, runtime, bank, parent and teacher staged under $ROOT"
    exit 0
fi

if [ "$SUBMIT" = smoke ]; then
    RUN_DIR="$ROOT/runs/smoke/${RUN_NAME}_${BASELINES_REVISION:0:12}_$(date +%Y%m%d_%H%M%S)"
    RUN_NAME="${RUN_NAME}_smoke"
    remote "test ! -e '$RUN_DIR' && mkdir -p '$RUN_DIR'"
else
    # Every segment of this run shares one run directory.
    RUN_DIR="$ROOT/runs/$RUN_NAME"
    remote "mkdir -p '$RUN_DIR'"
fi
EXPORTS="ALL,TARGET_UPDATE=$TARGET_UPDATE,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME,BASELINES_ROOT=$REMOTE_SOURCE"
EXPORTS+=",BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA"
EXPORTS+=",RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,SEED=$SEED,VENV=$REMOTE_VENV"
EXPORTS+=",RUNTIME_LOCK_SHA=$RUNTIME_LOCK_SHA,TERRA_EULER_USER=$TERRA_EULER_USER"
EXPORTS+=",TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY"
EXPORTS+=",WANDB_PROJECT=$WANDB_PROJECT,WANDB_MODE=$WANDB_MODE,BANK_ARCHIVE=$REMOTE_BANK"
EXPORTS+=",BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_MAPS_PATH=$BANK_MAPS_PATH"
EXPORTS+=",BANK_DATASET_SIZE=$BANK_DATASET_SIZE,BANK_DISTANCE_SIDECAR_SHA=$BANK_DISTANCE_SIDECAR_SHA"
EXPORTS+=",PARENT=$REMOTE_PARENT,PARENT_SHA=$PARENT_SHA,TEACHER=$REMOTE_TEACHER,TEACHER_SHA=$TEACHER_SHA"
EXPORTS+=",NUM_DEVICES=$NUM_DEVICES,ENVS_PER_DEVICE=$ENVS_PER_DEVICE,KL_ANNEAL_UPDATES=$KL_ANNEAL_UPDATES"
EXPORTS+=",GPU_TYPE=$GPU_TYPE,PRESET=$PRESET"
EXPORTS+=",DUG_CLEARANCE_M=$DUG_CLEARANCE_M"
raw="$(remote "cat '$REMOTE_SOURCE/scripts/euler_gru_machine_rules_ft/run.sbatch' | sbatch --parsable --account=es_hutter --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:$NUM_DEVICES' --cpus-per-task=8 --mem-per-cpu=8G --tmp=20G --exclude=eu-g6-064 --job-name='terra-gru-rules-ft' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${raw%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
echo "job_id=$JOB_ID gpu=${NUM_DEVICES}x$GPU_TYPE partition=$PARTITION run_dir=$RUN_DIR"
