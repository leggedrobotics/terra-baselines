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
BANK_ARCHIVE_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
PROMOTION_MANIFEST_SHA=dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826
PARENT_JOB_ID=11303967
PARENT_SOURCE_REVISION=74f72a65659353a6b4b2d163904dcbf60987805c
PARENT_CONTRACT_SHA=264624ba56d56fc9f322d646ff3015a30b8f27213e072cf1cd6a0e08e6e89d5a
GRU_FIXED_SHA=83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2
FF_FIXED_SHA=c0c53f54ee2d282c8cd5e4151e52ac3910c449cc909c5ee70c16a20965e800e5
SELECTION_SHA=675436d00ed6a156bfa1a00a325141c6fde98f52f09da85b02e21c7df9f93070
GRU_SOURCE_REVISION=33d26213327d66921b66753a5a6018a37d6f2e81
FF_SOURCE_REVISION=2778766683fb8a0a53a761385fae05cf9396dda9
GRU_U44000_SHA=0985b6338fb02f866b7aadbf065431cd667954a6f9b1a457e3eae9213533569d
FF_U44000_SHA=64ea0270dba0faf744eb15066232f1f137f9391c5aaf166ccbd57f00e329c623

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_v8_gru_media_recovery_v1

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
GRU_RUN="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/runs/$GRU_SOURCE_REVISION/s20260817"
FF_RUN="$REMOTE_RUN_ROOT/terra_v8_relay_partial_v1/runs/$FF_SOURCE_REVISION/s20260815/treatment"
GRU_U44000="$GRU_RUN/checkpoints/v8_relay_gru64r_33d26213327d_s20260817_update_044000.pkl"
FF_U44000="$FF_RUN/checkpoints/v8_relay_partial_2778766683fb_s20260815_update_044000.pkl"
RUNTIME_TERRA_ROOT="$REMOTE_WORK_ROOT/terra_v8_relay_gru_v2/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
BANK_ARCHIVE="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/inputs/full-bank-$BANK_ARCHIVE_SHA.tar.zst"
PARENT_FIXED_ROOT="$REMOTE_RUN_ROOT/terra_v8_gru_fixed_benchmark_v1/$PARENT_SOURCE_REVISION/fixed_promotion_u44_ff86"
REMOTE_SOURCE="$REMOTE_WORK_ROOT/$CAMPAIGN/$BASELINES_REVISION/terra-baselines"
RESULT_PARENT="$REMOTE_RUN_ROOT/$CAMPAIGN/$BASELINES_REVISION"
OUTPUT_DIR="$RESULT_PARENT/media_dashboard_recovery_$PARENT_JOB_ID"

printf '%s\n' \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "parent_job_id=$PARENT_JOB_ID" \
    "parent_source_revision=$PARENT_SOURCE_REVISION" \
    "parent_fixed_root=$PARENT_FIXED_ROOT" \
    "gru_fixed_json_sha256=$GRU_FIXED_SHA" \
    "ff_fixed_json_sha256=$FF_FIXED_SHA" \
    "selection_sha256=$SELECTION_SHA" \
    "output_dir=$OUTPUT_DIR"

if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: recovery contract printed; no SSH, staging, or Slurm mutation"
    exit 0
fi

test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" || {
    echo "terra-baselines must be committed and clean before staging" >&2
    exit 3
}
test -f "$REPO/scripts/euler_v8_gru_media_recovery_v1/run.sbatch"
git -C "$REPO" cat-file -e "$BASELINES_REVISION:scripts/euler_v8_gru_media_recovery_v1/run.sbatch"

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
remote "test \"\$(cat '$RUNTIME_TERRA_ROOT/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
remote "test \"\$(sha256sum '$BANK_ARCHIVE' | awk '{print \$1}')\" = '$BANK_ARCHIVE_SHA'"
remote "test \"\$(sha256sum '$PARENT_FIXED_ROOT/run_contract.env' | awk '{print \$1}')\" = '$PARENT_CONTRACT_SHA'"
remote "test \"\$(sha256sum '$PARENT_FIXED_ROOT/gru_u40000_u44000.json' | awk '{print \$1}')\" = '$GRU_FIXED_SHA'"
remote "test \"\$(sha256sum '$PARENT_FIXED_ROOT/ff_u44000_u86000.json' | awk '{print \$1}')\" = '$FF_FIXED_SHA'"
remote "test \"\$(sha256sum '$PARENT_FIXED_ROOT/selection_ff_u44_vs_gru_u44/review_selection.json' | awk '{print \$1}')\" = '$SELECTION_SHA'"
remote "test \"\$(sha256sum '$GRU_U44000' | awk '{print \$1}')\" = '$GRU_U44000_SHA'"
remote "test \"\$(sha256sum '$FF_U44000' | awk '{print \$1}')\" = '$FF_U44000_SHA'"
remote "test ! -e '$OUTPUT_DIR'"

if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: recovery source and immutable inputs verified; no Slurm mutation"
    exit 0
fi
remote "mkdir -p '$RESULT_PARENT'"

EXPORTS="ALL,OUTPUT_DIR=$OUTPUT_DIR,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$RUNTIME_TERRA_ROOT,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,BANK_ARCHIVE=$BANK_ARCHIVE,BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,PROMOTION_MANIFEST_SHA=$PROMOTION_MANIFEST_SHA,PROTOCOL_TERRA_REVISION=$PROTOCOL_TERRA_REVISION,PARENT_JOB_ID=$PARENT_JOB_ID,PARENT_SOURCE_REVISION=$PARENT_SOURCE_REVISION,PARENT_FIXED_ROOT=$PARENT_FIXED_ROOT,PARENT_CONTRACT_SHA=$PARENT_CONTRACT_SHA,GRU_FIXED_SHA=$GRU_FIXED_SHA,FF_FIXED_SHA=$FF_FIXED_SHA,SELECTION_SHA=$SELECTION_SHA,GRU_U44000=$GRU_U44000,GRU_U44000_SHA=$GRU_U44000_SHA,FF_U44000=$FF_U44000,FF_U44000_SHA=$FF_U44000_SHA,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT"
JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_gru_media_recovery_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.4h' --time='03:45:00' --gpus='rtx_4090:1' --cpus-per-task='8' --exclude='eu-g6-064' --job-name='terra-v8-media-recover' --output='$RESULT_PARENT/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
printf '%s\n' "job_id=$JOB_ID" "output_dir=$OUTPUT_DIR"
