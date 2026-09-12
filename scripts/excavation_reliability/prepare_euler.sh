#!/usr/bin/env bash
# SUBMIT=0 validates locally; stage uploads inputs only; 1 additionally submits.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_ROOT="${TERRA_ROOT:-$(dirname "$REPO")/terra}"
# shellcheck disable=SC1091
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-lterenzi}"
REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_runtime/terra_jax0433_cuda126_cudnn950_20260903}"
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|stage|1) ;; *) exit 2 ;; esac
: "${PACKAGE_DIR:?local directory for the reviewable launch package}"
: "${PARENT:?local native checkpoint}"
: "${BANK_ARCHIVE:?local terra_v2_generalist_pooled_bank_20260901.tar.zst}"
for value in "$REMOTE_HOST" "$VENV"; do [[ "$value" =~ ^[a-zA-Z0-9_./-]+$ ]] || exit 2; done
test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(git -C "$TERRA_ROOT" status --porcelain)"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
TERRA_REVISION="$(git -C "$TERRA_ROOT" rev-parse HEAD)"
BANK_SHA=1125177d322df6097f8da9f67ec95fe48762e16327f83dc157ec282b24993fb3
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
PARENT_SHA="$(sha256sum "$PARENT" | awk '{print $1}')"
mkdir -p "$PACKAGE_DIR"
JAX_PLATFORMS=cpu PYTHONPATH="$TERRA_ROOT:$REPO" PYGAME_HIDE_SUPPORT_PROMPT=1 \
    "${LOCAL_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}" - "$PARENT" "$PACKAGE_DIR/parent.json" <<'PY'
import json, sys
from utils.helpers import load_pkl_object, register_checkpoint_config_classes, checkpoint_foundation_behavior
register_checkpoint_config_classes()
p = load_pkl_object(sys.argv[1])
assert p['train_config'].config_name == 'trench_align_v2_generalist_gen'
u = int(p['next_update'])
assert 5000 <= u < 99998
assert p['r2_protocol_receipt']['distance_sidecar_sha256'] == 'f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980'
if u > 5000:
    import numpy as np
    settings = checkpoint_foundation_behavior(p)
    for k, v in {'lateral_dig_cost': .5, 'base_travel_cost': .01, 'base_turn_cost': .04, 'executable_dig_observation': True}.items():
        assert np.isclose(settings[k], v), (k, settings[k])
with open(sys.argv[2], 'w') as f:
    json.dump({'next_update': u, 'adam_step': int(p['train_state_step']), 'behavior_transfer': int(u == 5000)}, f)
PY
read -r PARENT_UPDATE BEHAVIOR_TRANSFER < <(python3 - "$PACKAGE_DIR/parent.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(p['next_update'], p['behavior_transfer'])
PY
)
CAMPAIGN=terra_excavation_reliability_20260909
REMOTE_WORK="$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation/$CAMPAIGN"
RUN_ROOT="$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs/$CAMPAIGN"
REMOTE_TERRA="$REMOTE_WORK/terra-$TERRA_REVISION"
REMOTE_BASELINES="$REMOTE_WORK/baselines-$BASELINES_REVISION"
REMOTE_BANK="$RUN_ROOT/inputs/bank-$BANK_SHA.tar.zst"
REMOTE_PARENT="$RUN_ROOT/inputs/parent-$PARENT_SHA.pkl"
REMOTE_CONFIG="$REMOTE_WORK/launch-${BASELINES_REVISION:0:12}-${PARENT_SHA:0:12}.env"
{
    printf 'TERRA_EULER_USER=%q\nTERRA_ROOT=%q\nBASELINES_ROOT=%q\n' "$TERRA_EULER_USER" "$REMOTE_TERRA" "$REMOTE_BASELINES"
    printf 'TERRA_REVISION=%q\nBASELINES_REVISION=%q\nVENV=%q\n' "$TERRA_REVISION" "$BASELINES_REVISION" "$VENV"
    printf 'BANK_ARCHIVE=%q\nBANK_SHA=%q\nPARENT=%q\nPARENT_SHA=%q\n' "$REMOTE_BANK" "$BANK_SHA" "$REMOTE_PARENT" "$PARENT_SHA"
    printf 'RUN_ROOT=%q\nPARENT_UPDATE=%q\nBEHAVIOR_TRANSFER=%q\n' "$RUN_ROOT" "$PARENT_UPDATE" "$BEHAVIOR_TRANSFER"
} > "$PACKAGE_DIR/launch.env"
printf '%s\n' "account=$TERRA_EULER_USER" "terra=$TERRA_REVISION" "baselines=$BASELINES_REVISION" \
    "parent_sha256=$PARENT_SHA" 'allocation=one RTX4090, 24 hours' "mode=$SUBMIT"
[[ "$SUBMIT" != 0 ]] || exit 0
remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "$@"; }
remote "test \"\$(id -un)\" = '$TERRA_EULER_USER' && test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$VENV/bin/python'"
remote lquota > "$PACKAGE_DIR/lquota.txt"
used_gb=$(bash "$REPO/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT" < "$PACKAGE_DIR/lquota.txt")
awk -v used="$used_gb" 'BEGIN {exit !(used <= 45)}'
remote "test \"\$(sha256sum '$VENV/requirements.lock.txt' | awk '{print \$1}')\" = 36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189"
remote "scontrol show partition gpuhe.24h -o" > "$PACKAGE_DIR/partition.txt"
remote "sacctmgr -nP show assoc where user=$TERRA_EULER_USER format=User,Account,Partition,QOS" > "$PACKAGE_DIR/association.txt"
remote "sinfo -N -p gpuhe.24h -o '%N %G %t'" > "$PACKAGE_DIR/gpu_inventory.txt"
remote "mkdir -p '$REMOTE_WORK' '$RUN_ROOT/inputs'"
for pair in "$TERRA_ROOT|$TERRA_REVISION|$REMOTE_TERRA" "$REPO|$BASELINES_REVISION|$REMOTE_BASELINES"; do
    IFS='|' read -r local_repo revision destination <<< "$pair"
    if ! remote "test -d '$destination'"; then
        remote "mkdir '$destination.partial'"
        git -C "$local_repo" archive "$revision" | remote "tar -xf - -C '$destination.partial'"
        remote "printf '%s\n' '$revision' > '$destination.partial/REVISION' && mv -T '$destination.partial' '$destination'"
    fi
    remote "test \"\$(cat '$destination/REVISION')\" = '$revision'"
done
for pair in "$BANK_ARCHIVE|$REMOTE_BANK|$BANK_SHA" "$PARENT|$REMOTE_PARENT|$PARENT_SHA"; do
    IFS='|' read -r local_file destination expected <<< "$pair"
    if ! remote "test -f '$destination'"; then
        scp -q -o BatchMode=yes "$local_file" "$REMOTE_HOST:$destination.partial"
        remote "test \"\$(sha256sum '$destination.partial' | awk '{print \$1}')\" = '$expected' && mv -T '$destination.partial' '$destination'"
    fi
    remote "test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected'"
done
scp -q -o BatchMode=yes "$PACKAGE_DIR/launch.env" "$REMOTE_HOST:$REMOTE_CONFIG"
printf '%s\n' "staged_config=$REMOTE_CONFIG" > "$PACKAGE_DIR/staged.txt"
if [[ "$SUBMIT" == stage ]]; then echo 'STAGED: no job submitted'; exit 0; fi
remote "mkdir -p '$RUN_ROOT/slurm'"
remote "sbatch --parsable --export=NONE --job-name=terra-generalist-2x --output='$RUN_ROOT/slurm/%j.out' '$REMOTE_BASELINES/scripts/excavation_reliability/run_euler.sbatch' '$REMOTE_CONFIG'" | tee "$PACKAGE_DIR/submission.txt"
