# Full-bank 2x generalist preparation

This recipe continues the mixed generalist with repaired chassis occupancy,
eligible-soil selection and shorter tracked movement. It is a capability
experiment; one arm does not isolate the reward effect. The unchanged specialist
loses held-out completion under the new dynamics, so existing checkpoint weights
are not promoted for deployment by these environment changes.

- Data: all 3,840 maps in `train_v2_pooled_generalist`, 25 foundation and 15
  finite-metadata trench conditions. Uniform full resets; no partial-reset or
  held-out-state training. The seven unsupported V8 trench conditions stay out.
- Parent: mixed generalist u5000, Adam step 320,000, 2,311,701 parameters. Same-bank
  native resume retains Adam and the absolute schedule. Executable fresh-dig
  observation replaces the existing affordance semantics without adding inputs.
- Costs: lateral fresh digging 0.5, actual base travel 0.01/m and turn 0.04/rad.
  Loose-soil pickup and dumping have no lateral digging cost. PPO and model
  settings follow the parent and the foundation 2x screen.
- Shape: one RTX4090, 512 environments x 32 steps, two epochs, 32 minibatches.
  Each update adds 16,384 transitions and 64 Adam steps. Parent u5000 used four
  GPUs; do not infer total sample exposure by multiplying its update count by
  the new batch size.
- Allocation: one 24-hour Euler job, at most 24 GPU-hours. A two-update,
  W&B-disabled smoke runs first inside the same allocation. Finite checks must
  pass before native continuation to the absolute u100000 ceiling. Checkpoints
  are retained every 500 updates. No CSCS duplicate is part of this recipe.
- Every Slurm job has a separate segment directory and linked W&B run. This
  avoids overwriting post-checkpoint episode receipts or logging backward into
  an old run. Resumes restart live environments and RNG; they are not bit exact.
  Persistent compilation cache is shared across the campaign; cache clearing
  inside training is disabled.

The full-panel initializer screen retains this mixed-bank parent. Under the
repaired environment it completes 32/608 cases; the reviewed easy-foundation
2x u15000 checkpoint completes 0/608. Neither completes any of the 384
foundations. The next run must establish learning and completion before
workspace efficiency can be assessed. See the
[readiness report](../../../../../.artifacts/terra_excavation_reliability_20260909/SUBMISSION_READINESS.md)
for paired environment results and the descriptive initializer comparison.

`prepare_euler.sh` defaults to a local contract check. It requires clean paired
source commits and verifies the exact native checkpoint and bank archive. Use
the explicit `lterenzi` account for this campaign: the locked cuDNN9 runtime is
accessible there. Its accessibility through `alesweber` failed on September 9.

```bash
export TERRA_EULER_USER=lterenzi REMOTE_HOST=euler-lterenzi
export TERRA_ROOT=/home/lorenzo/moleworks/.worktrees/terra_excavation_reliability_20260909/terra
export PACKAGE_DIR=/home/lorenzo/moleworks/.artifacts/terra_excavation_reliability_20260909/euler_package
export PARENT=/home/lorenzo/moleworks/.artifacts/terra_foundation_sweep_20260907/checkpoints/generalist_parent_u5000.pkl
export BANK_ARCHIVE=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819/terra_v2_generalist_pooled_bank_20260901.tar.zst

SUBMIT=0 bash scripts/excavation_reliability/prepare_euler.sh
SUBMIT=stage bash scripts/excavation_reliability/prepare_euler.sh
# This final command creates the allocation; staging does not.
SUBMIT=1 bash scripts/excavation_reliability/prepare_euler.sh
```

For a later wall-time continuation, retrieve and verify the latest complete
checkpoint, set `PARENT` to that local file, and use a new `PACKAGE_DIR`. The
launcher reads its native update, requires the same 2x treatment after u5000,
and omits the behavior-transfer flag. A finite checkpoint at a timeout is
continuable. Copy final checkpoints to persistent project storage and locally
before scratch retention expires; source and input archives are rebuildable.

## Evaluation

Use `eval.sh CHECKPOINT OUTPUT.json` with `TERRA_ROOT`, `BANK_ROOT` and
`TERRA_PYTHON` set, under the verified CUDA library environment. It evaluates
only the development panel: 608 frozen episodes, 450 decisions, seed 20260724,
greedy inference in chunks 120/120/120/120/120/8. `DECODER=sampled` is a separate
treatment. The bank-binding revision passed to the evaluator is not the runtime
source revision; retain both source commits alongside each result.

Evaluate the parent and saved u7000/u10000/u15000/u25000 checkpoints. Report
foundations and trenches separately, per-condition success, and efficiency on
the same common-success episodes. The normal benchmark includes raw navigation,
fresh digging/workspace area, lateral digging, new versus rehandled material,
accepted-disposal progress and unproductive tails. It also includes:

- Effective DO operations and retained work setups, including dump/relift poses.
  Consecutive work at the same chassis position and heading is one setup even
  after an intervening navigation excursion. Unsupported embodiments or missing
  action history produce unavailable values.
- Straight-line approach and inter-setup distances, maximum transfer, heading
  changes, exact pose revisits and A-B-A returns. These are lower bounds through
  work poses; no Nav2 feasibility or actual travel duration is inferred.
- Unique new area per retained productive setup and edge/corner adjacency of
  successive productive setups' fresh-cell unions. Deepening existing cells
  adds volume, but no new area or invented adjacency. These are excavated-cell
  relations, not geometric cone overlap or serialized requested workspaces.

Run milestone evaluation separately from training on retrieved checkpoints, so
the 24-hour training process retains its compiled state and GPU memory. The
training-bank online evaluation remains diagnostic. Recovery decoding,
observation expansion and stronger penalties remain separate treatments; none
is enabled here.
