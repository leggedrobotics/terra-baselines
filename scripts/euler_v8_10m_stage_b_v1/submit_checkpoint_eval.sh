#!/usr/bin/env bash
# shellcheck disable=SC2029  # Intentional local interpolation into quoted SSH commands.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: submit_checkpoint_eval.sh ARM UPDATE" >&2
    exit 2
fi

ARM="$1"
UPDATE="$2"
case "$ARM" in
    G-V8-XATTN-REWARM-CONTROL|G-V8-10M-XATTN-WARM) ;;
    *) echo "invalid Stage-B arm '$ARM'" >&2; exit 2 ;;
esac
[[ "$UPDATE" =~ ^[0-9]+$ ]] || { echo "UPDATE must be an integer" >&2; exit 2; }
if [ "$UPDATE" -lt 1000 ] || [ "$UPDATE" -gt 20000 ] || [ $((UPDATE % 1000)) -ne 0 ]; then
    echo "UPDATE must be a retained Stage-B update in {1000,2000,...,20000}" >&2
    exit 2
fi

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REMOTE_HOST="${REMOTE_HOST:-euler}"
BASELINES_REVISION=f682f37d6a856c779b2c52e9e2d02a56cb04c15c
EVALUATOR_SHA=1a5e21e6356ff0e2820eb6fa928cb7bfd567da7f5b1369e0aeb288af9b0fd700
SEED=20260730
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
CAMPAIGN_ID=terra_v8_10m_nearby_long_v1
REMOTE_SOURCE=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID/$BASELINES_REVISION/terra-baselines
REMOTE_RUNS=/cluster/work/rsl/lterenzi/$CAMPAIGN_ID
REMOTE_BANK=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs/bank-$BANK_SHA.tar.zst
RUN_DIR=$REMOTE_RUNS/$BASELINES_REVISION/screen/nearby/s$SEED/$ARM
TOKEN="$(printf '%06d' "$UPDATE")"
CHECKPOINT_PATTERN="*_update_${TOKEN}.pkl"
OUTPUT_PARENT=$REMOTE_RUNS/$BASELINES_REVISION/checkpoint_eval/nearby/s$SEED/$ARM/update_$TOKEN
OUTPUT_ROOT=$OUTPUT_PARENT/result

echo "arm=$ARM update=$UPDATE"
echo "training_revision=$BASELINES_REVISION evaluator_revision=$BASELINES_REVISION"
echo "run_dir=$RUN_DIR"
echo "checkpoint_pattern=$CHECKPOINT_PATTERN"
echo "output_root=$OUTPUT_ROOT"
echo "panels=main_promotion,main_development,capability_promotion,capability_development"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH or Slurm mutation"
    exit 0
fi

ssh "$REMOTE_HOST" "set -e
test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'
test \"\$(sha256sum '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/teacher_full_v8_eval.sbatch' | awk '{print \$1}')\" = '$EVALUATOR_SHA'
test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'
test -f '$RUN_DIR/run_contract.env'
test \"\$(awk -F= '\$1==\"terra_baselines_revision\" {print \$2}' '$RUN_DIR/run_contract.env')\" = '$BASELINES_REVISION'
test \"\$(awk -F= '\$1==\"reward_type\" {print \$2}' '$RUN_DIR/run_contract.env')\" = DENSE
test \"\$(awk -F= '\$1==\"reward_transition_launched\" {print \$2}' '$RUN_DIR/run_contract.env')\" = false"

CHECKPOINT="$(
    ssh "$REMOTE_HOST" "find '$RUN_DIR/checkpoints' -maxdepth 1 -type f -name '$CHECKPOINT_PATTERN' -print" \
        | awk 'NF {count += 1; value = $0} END {if (count != 1) exit 1; print value}'
)"
CHECKPOINT_SHA="$(ssh "$REMOTE_HOST" "sha256sum '$CHECKPOINT' | awk '{print \$1}'")"
[[ "$CHECKPOINT_SHA" =~ ^[0-9a-f]{64}$ ]]

ssh "$REMOTE_HOST" "mkdir -p '$OUTPUT_PARENT' && test ! -e '$OUTPUT_ROOT'"
EXPORTS="ALL,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,TEACHER_CHECKPOINT=$CHECKPOINT,TEACHER_SHA=$CHECKPOINT_SHA,OUTPUT_ROOT=$OUTPUT_ROOT"
JOB_ID="$(
    ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/teacher_full_v8_eval.sbatch' | sbatch --parsable --partition=gpuhe.4h --time=03:45:00 --gpus=rtx_4090:1 --exclude='eu-g6-064' --job-name='terra-v8-b-${ARM}-u${TOKEN}-eval' --output='$OUTPUT_PARENT/eval_%j.out' --export='$EXPORTS'"
)"

echo "checkpoint=$CHECKPOINT"
echo "checkpoint_sha256=$CHECKPOINT_SHA"
echo "checkpoint_eval_job=$JOB_ID"
