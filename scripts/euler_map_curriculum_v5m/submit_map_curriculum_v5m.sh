#!/usr/bin/env bash
# Sync the M1 v5-main screen to Euler and submit the three 4-GPU arms.
# Run from this machine (needs the `euler` ssh host and the materialized bank).
#
#   scripts/euler_map_curriculum_v5m/submit_map_curriculum_v5m.sh          # sync + submit
#   SUBMIT=0 scripts/euler_map_curriculum_v5m/submit_map_curriculum_v5m.sh # sync only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA=/home/lorenzo/moleworks/.worktrees/terra_v5m_screen_20260730
BANK="$REPO/terra_data/curriculum_v5m"
RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v5m_20260730
SUBMIT="${SUBMIT:-1}"

test -f "$BANK/curriculum_v5m_provenance.json"
test -f "$REPO/logs/smoke_v5m/LOCAL_SMOKE_PASSED"

TERRA_REV="$(git -C "$TERRA" rev-parse HEAD)"
BASELINES_REV="$(git -C "$REPO" rev-parse HEAD)"
TERRA_DIRTY="$(git -C "$TERRA" status --porcelain | wc -l)"
BASELINES_DIRTY="$(git -C "$REPO" status --porcelain | wc -l)"
BANK_SHA="$(python3 -c "
import json,sys
d=json.load(open('$BANK/curriculum_v5m_provenance.json'))
print(d['bank_manifest_csv_sha256'])
")"
# Both trees carry uncommitted screen work, so pin the files that define the run
# instead of trusting the branch tips alone.
sha() { sha256sum "$1" | awk '{print $1}'; }
SHA_MAPS_BUFFER="$(sha "$TERRA/terra/maps_buffer.py")"
SHA_CURRICULUM="$(sha "$TERRA/terra/curriculum.py")"
SHA_STATE="$(sha "$TERRA/terra/state.py")"
SHA_TRAIN_MIXED="$(sha "$REPO/train_mixed.py")"
SHA_MODELS="$(sha "$REPO/utils/models.py")"
SHA_CONFIGS="$(sha "$REPO/configs/training_configs.yaml")"

echo "terra            = $TERRA_REV (dirty files: $TERRA_DIRTY)"
echo "terra-baselines  = $BASELINES_REV (dirty files: $BASELINES_DIRTY)"
echo "bank manifest    = $BANK_SHA"

ssh euler "mkdir -p '$RUN_ROOT'/{source,bank,logs,checkpoints}"

rsync -a --delete --exclude '__pycache__' --exclude '.git' --exclude 'data' \
    "$TERRA/" "euler:$RUN_ROOT/source/terra/"
rsync -a --delete --exclude '__pycache__' --exclude '.git' --exclude 'logs' \
    --exclude 'checkpoints' --exclude 'terra_data' \
    "$REPO/" "euler:$RUN_ROOT/source/terra-baselines/"
# -H keeps the converter's intra-level hardlinks so the bank costs ~6k inodes
# instead of ~21k on the shared scratch file quota.
rsync -aH --delete "$BANK/" "euler:$RUN_ROOT/bank/"

ssh euler "cat > '$RUN_ROOT/source_sync_manifest.txt'" <<EOF
terra_revision=$TERRA_REV
terra_uncommitted_files=$TERRA_DIRTY
terra_baselines_revision=$BASELINES_REV
terra_baselines_uncommitted_files=$BASELINES_DIRTY
terra_maps_buffer_sha256=$SHA_MAPS_BUFFER
terra_curriculum_sha256=$SHA_CURRICULUM
terra_state_sha256=$SHA_STATE
train_mixed_sha256=$SHA_TRAIN_MIXED
models_sha256=$SHA_MODELS
training_configs_yaml_sha256=$SHA_CONFIGS
bank_manifest_csv_sha256=$BANK_SHA
holdout_map_indices=3,7,11,14
level_size=864
init=scratch
synced_at=$(date -Is)
EOF

ssh euler "
    set -e
    cd '$RUN_ROOT'
    test -f source/terra/terra/curriculum.py
    test -f source/terra-baselines/train_mixed.py
    test -f bank/train/L0/dataset.json
    test -f bank/train/L3/dataset.json
    test -f bank/held_out/all/manifest.jsonl
    for L in L0 L1 L2 L3; do
        n=\$(ls bank/train/\$L/images | wc -l)
        test \"\$n\" -eq 864 || { echo \"level \$L has \$n slots\" >&2; exit 3; }
    done
    test \$(ls bank/held_out/all/images | wc -l) -eq 116
    touch SOURCE_SYNC_VERIFIED
    echo SOURCE_SYNC_VERIFIED
"
scp -q "$REPO/logs/smoke_v5m/LOCAL_SMOKE_PASSED" "euler:$RUN_ROOT/LOCAL_SMOKE_PASSED"

if [ "$SUBMIT" != "1" ]; then
    echo "SUBMIT=0, stopping after sync"
    exit 0
fi

ssh euler "
    set -e
    cd '$RUN_ROOT/source/terra-baselines/scripts/euler_map_curriculum_v5m'
    : > '$RUN_ROOT/submitted_jobs.txt'
    for spec in 'A:terra-v5m-A-t0base' 'B:terra-v5m-B-uniform' 'C:terra-v5m-C-curriculum'; do
        arm=\${spec%%:*}
        name=\${spec##*:}
        jid=\$(sbatch --parsable --job-name=\"\$name\" --export=ALL,ARM=\$arm run_map_curriculum_v5m.sbatch)
        echo \"\$arm \$name \$jid\" | tee -a '$RUN_ROOT/submitted_jobs.txt'
    done
    squeue -u lterenzi -o '%.10i %.24j %.9P %.2t %.11M %R' | grep -E 'JOBID|terra-v5m'
"
