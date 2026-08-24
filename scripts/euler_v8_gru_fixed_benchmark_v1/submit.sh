#!/usr/bin/env bash
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|1) ;;
    *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

RUNTIME_TERRA_REVISION=25f855db3d913fd638c4e56b1740437a2b7122ca
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
BANK_ARCHIVE_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
PROMOTION_MANIFEST_SHA=dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826
GRU_SOURCE_REVISION=33d26213327d66921b66753a5a6018a37d6f2e81
FF_SOURCE_REVISION=2778766683fb8a0a53a761385fae05cf9396dda9
GRU_U40000_SHA=9eb032308b07a8bb43a44bb01993f8e1aaa439d70eb8e14c2047c6469d6091fd
GRU_U44000_SHA=0985b6338fb02f866b7aadbf065431cd667954a6f9b1a457e3eae9213533569d
FF_U44000_SHA=64ea0270dba0faf744eb15066232f1f137f9391c5aaf166ccbd57f00e329c623
FF_U86000_SHA=2fe5d23c86cc7702b188d33ca1ca9a42066a9a2515150e8795f8c640bbbeb4af
EVAL_SEED=20260807

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_v8_gru_fixed_benchmark_v1

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
GRU_RUN="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/runs/$GRU_SOURCE_REVISION/s20260817"
FF_RUN="$REMOTE_RUN_ROOT/terra_v8_relay_partial_v1/runs/$FF_SOURCE_REVISION/s20260815/treatment"
GRU_U40000="$GRU_RUN/checkpoints/v8_relay_gru64r_33d26213327d_s20260817_update_040000.pkl"
GRU_U44000="$GRU_RUN/checkpoints/v8_relay_gru64r_33d26213327d_s20260817_update_044000.pkl"
FF_U44000="$FF_RUN/checkpoints/v8_relay_partial_2778766683fb_s20260815_update_044000.pkl"
FF_U86000="$FF_RUN/checkpoints/v8_relay_partial_2778766683fb_s20260815_update_086000.pkl"
GRU_TRAINING_SOURCE="$REMOTE_WORK_ROOT/terra_v8_relay_gru_v2/$GRU_SOURCE_REVISION/terra-baselines"
FF_TRAINING_SOURCE="$REMOTE_WORK_ROOT/terra_v8_relay_partial_v1/$FF_SOURCE_REVISION/terra-baselines"
RUNTIME_TERRA_ROOT="$REMOTE_WORK_ROOT/terra_v8_relay_gru_v2/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
BANK_ARCHIVE="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/inputs/full-bank-$BANK_ARCHIVE_SHA.tar.zst"
REMOTE_SOURCE="$REMOTE_WORK_ROOT/$CAMPAIGN/$BASELINES_REVISION/terra-baselines"
RESULT_PARENT="$REMOTE_RUN_ROOT/$CAMPAIGN/$BASELINES_REVISION"
OUTPUT_DIR="$RESULT_PARENT/fixed_promotion_u44_ff86"

printf '%s\n' \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "gru_training_source=$GRU_TRAINING_SOURCE" \
    "gru_training_source_revision=$GRU_SOURCE_REVISION" \
    "ff_training_source=$FF_TRAINING_SOURCE" \
    "ff_training_source_revision=$FF_SOURCE_REVISION" \
    "gru_u40000=$GRU_U40000" \
    "gru_u40000_sha256=$GRU_U40000_SHA" \
    "gru_u44000=$GRU_U44000" \
    "gru_u44000_sha256=$GRU_U44000_SHA" \
    "ff_u44000=$FF_U44000" \
    "ff_u44000_sha256=$FF_U44000_SHA" \
    "ff_u86000=$FF_U86000" \
    "ff_u86000_sha256=$FF_U86000_SHA" \
    "bank_archive=$BANK_ARCHIVE" \
    "output_dir=$OUTPUT_DIR"

if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: contract printed; no SSH, staging, or Slurm mutation"
    exit 0
fi

test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" || {
    echo "terra-baselines must be committed and clean before staging" >&2
    exit 3
}
test -f "$REPO/scripts/euler_v8_gru_fixed_benchmark_v1/run.sbatch"
git -C "$REPO" cat-file -e "$BASELINES_REVISION:scripts/euler_v8_gru_fixed_benchmark_v1/run.sbatch"

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"
HOME_USED_GB="$(remote lquota | "$REPO/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}
remote "sacctmgr -n -P show assoc user='$TERRA_EULER_USER' account='es_hutter' format=User,Account | grep -Fx '$TERRA_EULER_USER|es_hutter' >/dev/null"
remote "scontrol show partition gpuhe.4h -o | grep -q ' State=UP '"

if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$REMOTE_WORK_ROOT/$CAMPAIGN/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK_ROOT/$CAMPAIGN/$BASELINES_REVISION'"
fi

remote "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
remote "test \"\$(cat '$GRU_TRAINING_SOURCE/REVISION')\" = '$GRU_SOURCE_REVISION'"
remote "test \"\$(cat '$FF_TRAINING_SOURCE/REVISION')\" = '$FF_SOURCE_REVISION'"
remote "test \"\$(cat '$RUNTIME_TERRA_ROOT/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
remote "test \"\$(sha256sum '$BANK_ARCHIVE' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA'"
for CHECKPOINT_AND_SHA in \
    "$GRU_U40000:$GRU_U40000_SHA" \
    "$GRU_U44000:$GRU_U44000_SHA" \
    "$FF_U44000:$FF_U44000_SHA" \
    "$FF_U86000:$FF_U86000_SHA"; do
    CHECKPOINT="${CHECKPOINT_AND_SHA%%:*}"
    EXPECTED_SHA="${CHECKPOINT_AND_SHA##*:}"
    remote "test \"\$(sha256sum '$CHECKPOINT' | awk '{print \$1}')\" = '$EXPECTED_SHA'"
done

remote "test ! -e '$OUTPUT_DIR'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: evaluator source and immutable inputs verified; no Slurm mutation"
    exit 0
fi
remote "mkdir -p '$RESULT_PARENT'"

EXPORTS="ALL,OUTPUT_DIR=$OUTPUT_DIR,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,GRU_TRAINING_SOURCE=$GRU_TRAINING_SOURCE,GRU_SOURCE_REVISION=$GRU_SOURCE_REVISION,FF_TRAINING_SOURCE=$FF_TRAINING_SOURCE,FF_SOURCE_REVISION=$FF_SOURCE_REVISION,RUNTIME_TERRA_ROOT=$RUNTIME_TERRA_ROOT,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,BANK_ARCHIVE=$BANK_ARCHIVE,BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,PROMOTION_MANIFEST_SHA=$PROMOTION_MANIFEST_SHA,PROTOCOL_TERRA_REVISION=$PROTOCOL_TERRA_REVISION,DISTANCE_SIDECAR_SHA=$DISTANCE_SIDECAR_SHA,GRU_U40000=$GRU_U40000,GRU_U40000_SHA=$GRU_U40000_SHA,GRU_U44000=$GRU_U44000,GRU_U44000_SHA=$GRU_U44000_SHA,FF_U44000=$FF_U44000,FF_U44000_SHA=$FF_U44000_SHA,FF_U86000=$FF_U86000,FF_U86000_SHA=$FF_U86000_SHA,EVAL_SEED=$EVAL_SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT"
JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_gru_fixed_benchmark_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.4h' --time='03:45:00' --gpus='rtx_4090:1' --cpus-per-task='8' --exclude='eu-g6-064' --job-name='terra-v8-gru-fixed' --output='$RESULT_PARENT/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
printf '%s\n' "job_id=$JOB_ID" "output_dir=$OUTPUT_DIR"
