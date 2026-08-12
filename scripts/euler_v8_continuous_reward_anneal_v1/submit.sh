#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: submit.sh smoke|screen RUNTIME_TERRA_REVISION" >&2
    exit 2
fi
PHASE="$1"
RUNTIME_TERRA_REVISION="$2"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
[[ "$RUNTIME_TERRA_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo "RUNTIME_TERRA_REVISION must be a full 40-character commit SHA" >&2
    exit 2
}
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_v8_continuous_reward_anneal_20260807}"
REMOTE_HOST="${REMOTE_HOST:-euler}"
SEED=20260807
CAMPAIGN_ID=terra_v8_continuous_reward_anneal_v1
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/work/rsl/lterenzi/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
ARMS=(constant_dense dense_to_terminal)

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "runtime Terra must be committed and clean before launch" >&2
    exit 3
}
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(git -C "$TERRA_REPO" rev-parse "$RUNTIME_TERRA_REVISION^{commit}")" = "$RUNTIME_TERRA_REVISION"
git -C "$TERRA_REPO" show "$RUNTIME_TERRA_REVISION:terra/config.py" | grep -q 'ANNEALED_OBJECTIVE'
git -C "$TERRA_REPO" show "$RUNTIME_TERRA_REVISION:terra/config.py" | grep -q 'terminal_reward_mix:'

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"

echo "phase=$PHASE support=all47_continuous seed=$SEED updates=$([ "$PHASE" = smoke ] && echo 1 || echo 20000)"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "protocol_terra_revision=$PROTOCOL_TERRA_REVISION"
echo "runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "arms=${ARMS[*]}"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, storage, W&B, Slurm, or job cancellation mutation"
    exit 0
fi

if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_TERRA/REVISION'"; then
    PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    screen) PARTITION=gpuhe.120h; WALLTIME=119:45:00; GPU_TYPE=rtx_4090 ;;
esac
declare -A SMOKE_JOB_IDS=()
declare -A SMOKE_RUNS=()
PAIR_SMOKE_RECEIPT=none
if [ "$PHASE" = screen ]; then
    PAIR_ID="v8_continuous_reward_anneal_v1_${BASELINES_REVISION:0:12}_${RUNTIME_TERRA_REVISION:0:12}_s${SEED}"
    for ARM in "${ARMS[@]}"; do
        case "$ARM" in
            constant_dense) REWARD_STAGE=dense_skill ;;
            dense_to_terminal) REWARD_STAGE=annealed_objective ;;
        esac
        SMOKE_RUN="$REMOTE_RUNS/$BASELINES_REVISION/smoke/all47/s$SEED/$ARM"
        SMOKE_RUNS[$ARM]="$SMOKE_RUN"
        ssh "$REMOTE_HOST" "test -f '$SMOKE_RUN/run_contract.env' && test -f '$SMOKE_RUN/smoke_validation.json' && test -f '$SMOKE_RUN/smoke_sampler_validation.json'"
        ssh "$REMOTE_HOST" "python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))[\"passed\"] is True; assert json.load(open(sys.argv[2]))[\"passed\"] is True' '$SMOKE_RUN/smoke_validation.json' '$SMOKE_RUN/smoke_sampler_validation.json'"
        for EXPECTED in \
            "pair_id=$PAIR_ID" \
            "arm=$ARM" \
            "reward_stage=$REWARD_STAGE" \
            "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
            "terra_baselines_revision=$BASELINES_REVISION" \
            "training_bank_archive_sha256=$BANK_SHA" \
            "training_bank_dataset_sha256=$BANK_DATASET_SHA"; do
            KEY="${EXPECTED%%=*}"
            VALUE="${EXPECTED#*=}"
            ssh "$REMOTE_HOST" "test \"\$(awk -F= -v key='$KEY' '\$1==key {print \$2}' '$SMOKE_RUN/run_contract.env')\" = '$VALUE'"
        done
        SMOKE_JOB_ID="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        SMOKE_JOB_IDS[$ARM]="$SMOKE_JOB_ID"
        SMOKE_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
        test "$SMOKE_STATE" = COMPLETED
    done
    DENSE_GRAPH="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"curriculum_graph_sha256\" {print \$2}' '${SMOKE_RUNS[constant_dense]}/run_contract.env'")"
    ANNEALED_GRAPH="$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"curriculum_graph_sha256\" {print \$2}' '${SMOKE_RUNS[dense_to_terminal]}/run_contract.env'")"
    [[ "$DENSE_GRAPH" =~ ^[0-9a-f]{64}$ ]]
    test "$DENSE_GRAPH" = "$ANNEALED_GRAPH"
    PAIR_SMOKE_RECEIPT="$REMOTE_RUNS/$BASELINES_REVISION/smoke/all47/s$SEED/pair_smoke_validation.json"
    ssh "$REMOTE_HOST" "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH='$REMOTE_TERRA:$REMOTE_SOURCE' /cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426/bin/python '$REMOTE_SOURCE/scripts/euler_v8_continuous_reward_anneal_v1/verify_pair_smoke.py' --dense-run '${SMOKE_RUNS[constant_dense]}' --annealed-run '${SMOKE_RUNS[dense_to_terminal]}' --output '$PAIR_SMOKE_RECEIPT'"
    ssh "$REMOTE_HOST" "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d[\"passed\"] is True; assert d[\"pretrigger_dense_parity\"] is True; assert d[\"schema\"]==\"terra_continuous_reward_pair_smoke_v1\"' '$PAIR_SMOKE_RECEIPT'"
fi

RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/all47/s$SEED"
ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT'"
for ARM in "${ARMS[@]}"; do
    ssh "$REMOTE_HOST" "test ! -e '$RUN_PARENT/$ARM'"
done
for ARM in "${ARMS[@]}"; do
    ssh "$REMOTE_HOST" "mkdir '$RUN_PARENT/$ARM'"
done
for ARM in "${ARMS[@]}"; do
    case "$ARM" in
        constant_dense) REWARD_STAGE=dense_skill ;;
        dense_to_terminal) REWARD_STAGE=annealed_objective ;;
    esac
    RUN_DIR="$RUN_PARENT/$ARM"
    SMOKE_JOB_ID="${SMOKE_JOB_IDS[$ARM]:-none}"
    SMOKE_RUN="${SMOKE_RUNS[$ARM]:-none}"
    EXPORTS="ALL,PHASE=$PHASE,ARM=$ARM,REWARD_STAGE=$REWARD_STAGE,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN,PAIR_SMOKE_RECEIPT=$PAIR_SMOKE_RECEIPT"
    JOB_ID="$(ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_continuous_reward_anneal_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' --exclude='eu-g6-064' --job-name='terra-v8-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
    echo "$PHASE all47 $ARM $JOB_ID"
done
