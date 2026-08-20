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
ENVIRONMENT_PROTOCOL_SHA=9917b9238e9e6e844377e6d4a8ca18d1f0defbbacf887642743e579243109367
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
BANK_ARCHIVE_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
PROMOTION_MANIFEST_SHA=dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826
GRU_SOURCE_REVISION=33d26213327d66921b66753a5a6018a37d6f2e81
GRU_U44000_SHA=0985b6338fb02f866b7aadbf065431cd667954a6f9b1a457e3eae9213533569d
FIXED_SOURCE_REVISION=74f72a65659353a6b4b2d163904dcbf60987805c
FIXED_GRU_RESULT_SHA=83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2
RUNTIME_FINGERPRINT_SHA=73c80e3dd483e3202679844228b422f416bbf48b49d6ce35056f3afff91d9b7e
EVAL_SEED=20260807

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_v8_gru_recurrence_probe_v1

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK_ROOT/$CAMPAIGN/$BASELINES_REVISION/terra-baselines"
RUNTIME_TERRA_ROOT="$REMOTE_WORK_ROOT/terra_v8_relay_gru_v2/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
GRU_RUN="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/runs/$GRU_SOURCE_REVISION/s20260817"
GRU_U44000="$GRU_RUN/checkpoints/v8_relay_gru64r_33d26213327d_s20260817_update_044000.pkl"
BANK_ARCHIVE="$REMOTE_RUN_ROOT/terra_v8_relay_gru_v2/inputs/full-bank-$BANK_ARCHIVE_SHA.tar.zst"
FIXED_GRU_RESULT="$REMOTE_RUN_ROOT/terra_v8_gru_fixed_benchmark_v1/$FIXED_SOURCE_REVISION/fixed_promotion_u44_ff86/gru_u40000_u44000.json"
RESULT_PARENT="$REMOTE_RUN_ROOT/$CAMPAIGN/$BASELINES_REVISION"
OUTPUT_DIR="$RESULT_PARENT/gru_u44000_carry_vs_zero"
CLAIM_DIR="$RESULT_PARENT/.gru_u44000_carry_vs_zero.submit_claim"

printf '%s\n' \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "environment_protocol_sha256=$ENVIRONMENT_PROTOCOL_SHA" \
    "distance_sidecar_sha256=$DISTANCE_SIDECAR_SHA" \
    "checkpoint=$GRU_U44000" \
    "checkpoint_sha256=$GRU_U44000_SHA" \
    "fixed_gru_result=$FIXED_GRU_RESULT" \
    "fixed_gru_result_sha256=$FIXED_GRU_RESULT_SHA" \
    "runtime_venv=$REMOTE_VENV" \
    "runtime_fingerprint_sha256=$RUNTIME_FINGERPRINT_SHA" \
    "bank_archive=$BANK_ARCHIVE" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "output_dir=$OUTPUT_DIR"

if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: contract printed; no SSH, staging, or Slurm mutation"
    exit 0
fi

test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" || {
    echo "terra-baselines must be committed and clean before staging" >&2
    exit 3
}
test -f "$REPO/scripts/euler_v8_gru_recurrence_probe_v1/run.sbatch"
test -f "$REPO/scripts/gru_recurrence_probe_v1/run_probe.py"
git -C "$REPO" cat-file -e "$BASELINES_REVISION:scripts/euler_v8_gru_recurrence_probe_v1/run.sbatch"
git -C "$REPO" cat-file -e "$BASELINES_REVISION:scripts/gru_recurrence_probe_v1/run_probe.py"

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
remote "test \"\$(sha256sum '$GRU_U44000' | awk '{print \$1}')\" = '$GRU_U44000_SHA'"
remote "test \"\$(sha256sum '$FIXED_GRU_RESULT' | awk '{print \$1}')\" = '$FIXED_GRU_RESULT_SHA'"
remote "test ! -e '$OUTPUT_DIR' && test ! -e '$CLAIM_DIR'"

if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: evaluator source and immutable inputs verified; no Slurm mutation"
    exit 0
fi

remote "mkdir -p '$RESULT_PARENT'"
remote "mkdir '$CLAIM_DIR'"
remote "printf '%s\n' 'state=submitting' 'output_dir=$OUTPUT_DIR' 'terra_baselines_revision=$BASELINES_REVISION' > '$CLAIM_DIR/submission.env'"

EXPORTS="ALL,OUTPUT_DIR=$OUTPUT_DIR,CLAIM_DIR=$CLAIM_DIR,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$RUNTIME_TERRA_ROOT,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_TERRA_REVISION=$PROTOCOL_TERRA_REVISION,ENVIRONMENT_PROTOCOL_SHA=$ENVIRONMENT_PROTOCOL_SHA,DISTANCE_SIDECAR_SHA=$DISTANCE_SIDECAR_SHA,BANK_ARCHIVE=$BANK_ARCHIVE,BANK_ARCHIVE_SHA=$BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,PROMOTION_MANIFEST_SHA=$PROMOTION_MANIFEST_SHA,GRU_U44000=$GRU_U44000,GRU_U44000_SHA=$GRU_U44000_SHA,FIXED_GRU_RESULT=$FIXED_GRU_RESULT,FIXED_GRU_RESULT_SHA=$FIXED_GRU_RESULT_SHA,RUNTIME_FINGERPRINT_SHA=$RUNTIME_FINGERPRINT_SHA,EVAL_SEED=$EVAL_SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT"
JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_gru_recurrence_probe_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.4h' --time='02:00:00' --gpus='rtx_4090:1' --cpus-per-task='8' --exclude='eu-g6-064' --job-name='terra-v8-gru-probe' --output='$RESULT_PARENT/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
remote "tmp='$CLAIM_DIR/.submission.env.$JOB_ID'; printf '%s\n' 'state=submitted' 'job_id=$JOB_ID' 'output_dir=$OUTPUT_DIR' 'terra_baselines_revision=$BASELINES_REVISION' > \"\$tmp\" && mv \"\$tmp\" '$CLAIM_DIR/submission.env'"
printf '%s\n' "job_id=$JOB_ID" "output_dir=$OUTPUT_DIR"
