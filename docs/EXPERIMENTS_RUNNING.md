# Experiments — current state (updated 2026-08-17 CEST)

## v6.1 reward-v2 + stall age + Continuous Banded v3

The first capability segment is complete.  Slurm job `10625259` continued the
selected v6.1 policy from absolute update 14,000 to update 40,000 on eight RTX
4090 GPUs and exited successfully after `22:48:04`.  The final held-out
promotion result is 657/720 exact, versus 407/720 at the u14 source.  This is a
combined stall-age plus final-v3 treatment, not a component ablation.

The exact u40 source is frozen as:

- terra-baselines:
  `dddc691c93ee21488cd7eeb8e01b067bf1f9733c`;
- Terra:
  `c2d2a94a124759e9f21c2b37930f717e299f0c46`;
- final checkpoint:
  `v8_v61_stall_age_v3_u40000_FINAL_17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8.pkl`;
- checkpoint SHA-256:
  `17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8`;
- checkpoint clocks: `next_update=40000`, optimizer step `2560000`; and
- W&B run:
  `v8_v61_stall_age_v3_dddc691c93_phase2_10625259`.

The same annotated tag, `v8-v61-stall-age-u40-20260814`, identifies the paired
source commit in each repository.

### Direct one-day extension

A second 23:45 segment resumes the native u40 checkpoint without changing the
treatment.  Its absolute target is u70,000, deliberately beyond the roughly
27,100 updates expected to fit in one allocation.  A wall-time exit near u67k
with a verified rolling checkpoint is therefore `CONTINUABLE`, not a failed
run.

The extension preserves:

- reward-v2 and its timing;
- material stall age and Continuous Banded v3;
- the v6.1 spatial MLP architecture and no-action-mask contract;
- 8 x 256 environments x 32 steps, 32 minibatches, and two PPO epochs;
- 65,536 transitions per absolute update;
- learning rate, entropy schedule, horizon 450, bank, and seed 20260807;
- the complete optimizer, sampler, and absolute update clocks; and
- the original W&B lineage with `resume=must` because its last logged
  `train/update=39991` does not exceed the u40 checkpoint.

It does **not** include later Terra commits `88c0099e` or `30ad500f`, the relay
partial-reset generator `67c72d09`/`794d4759`, new outcome observations, a DO
affordance, reward changes, a GRU, or action masking.  Those remain separate
fresh-treatment arms.

Checkpoints remain every 500 absolute updates.  Fixed source-disjoint
evaluation—not online return or mastery—is the decision evidence.  Because the
u39-to-u40 comparison had 38 conversions and 32 regressions for only +6 net,
the extended line must be evaluated at multiple retained checkpoints rather
than only at its final wall-time checkpoint.

Slurm job `10752100` was submitted at 2026-08-14 23:47 CEST with account
`gpuhe/es_hutter`, QOS `es_hutter/gpuhe/24`, partition `gpuhe.24h`, and an
exact request for eight RTX 4090 GPUs, eight CPUs, and 64 GB RAM.  It started
on `eu-g6-077` at 2026-08-15 02:51 CEST and was still running at the latest
ledger refresh.  Its rolling checkpoints and later fixed-panel evaluations,
not allocation alone, determine whether the u40 treatment continues to improve.

The launcher is commit
`bbaebc04c2ddc7c3ae667e434e223e1d01b95f84` on branch
`experiment/v8-v61-u40-phase3-20260814`.  Its run directory is
`/cluster/scratch/alesweber/codex_terra_edge_runs/terra_v8_v6_yolo_rv2/runs/dddc691c93ee21488cd7eeb8e01b067bf1f9733c/phase3/s20260807/v6_1_rv2_stall_age_v3`.

## Fresh relay partial-reset run to u200,000

The new environment/runtime and the Backplay-inspired relay curriculum are now
merged to the default branches.  The exact training sources are:

- Terra base `25f855db3d913fd638c4e56b1740437a2b7122ca`, with the
  pre-allocation runtime patch
  `ebdc3ad7b0e7ef505bb6d442a97d18d986cced44` described below;
- terra-baselines `2778766683fb8a0a53a761385fae05cf9396dda9`;
- seed `20260815`; and
- partial-reset bank digest
  `fb73b1d12dfad98c9aa79680d4d3ac178bf84b537e1be1e822535c65473a23f5`.

This is fresh training, not a u40 continuation.  It uses reward-v2 timing
variant zero, Continuous Banded v3, the v6.1 spatial actor/critic with
2,306,237 parameters, eight RTX 4090 GPUs, 8 x 256 environments, 32 rollout
steps, 32 minibatches, two PPO epochs, no action mask, no stall-age scalar, and
the reset-context observation required by the partial-start treatment.

The runtime sidecar contains all three validated tiers for 96 independent
`fnd-slab-apron-d16` training maps, with zero rejected triplets.  Partial starts
occupy at most 25% of lanes during the first 10,000 updates and anneal to zero;
all later training and all fixed evaluation use ordinary full starts.  Partial
episode outcomes cannot update full-start v3 mastery.  The merged generator
also contains trench leaf-first ordering, but this first immutable sidecar does
not claim broad 47-condition synthetic coverage.

One 200,000-update job would exceed the observed throughput budget of a
119:45 allocation, so the run is split into two native segments with one W&B
identity and absolute clocks:

- job `10777230`: fresh u0 -> u100,000, currently `PENDING (Priority)`;
- job `10777232`: u100,000 -> u200,000, currently
  `PENDING (Dependency)` on `afterok:10777230`.

Both jobs request one node, eight RTX 4090 GPUs, eight CPUs, 64 GB RAM,
`gpuhe.120h`, and 119:45 wall time under `gpuhe/es_hutter`.  The resume job
fails closed unless the u100,000 checkpoint contains the same partial-bank,
architecture, reward, sampler, optimizer, and absolute-update contracts.
Submission is execution evidence only; promotion still requires the untouched
720-map full-start panel and the relay/recurrence diagnostics.

Before either job was allocated, the shared staged Terra runtime was patched
in place with Terra `ebdc3ad7` to prevent dig/relift under the active base and
to preserve underlying terrain blockers in the visible traversability channel.
The same job IDs and queue positions were retained.  The immutable Slurm guard
still names the base revision `25f855db`; therefore
`RUNTIME_PATCH_EBDC3AD7.env` in both the staged runtime and run directory is
the authoritative addendum for the effective source, file hashes, backup, and
validation.  The commit is pushed on
`origin/experiment/v8-relay-corridor-resets-20260814` but is not yet merged to
Terra `main`.  Local validation passed 47 focused tests, and the two exact
regressions passed again from the staged Euler source.  At the 2026-08-17
09:50 CEST snapshot, `10777230` remained `PENDING (Priority)` and `10777232`
remained `PENDING (Dependency)`; neither job was resubmitted.

## Fresh-trench dig-alignment C0/T1 pilot (launched 2026-08-19)

Two matched-seed 4x RTX 4090 arms test the empty-excavator fresh-trench dig
gate causally.  The ONLY difference is `enforce_trench_dig_alignment`
(C0 off / T1 on); both arms carry the finite-metadata requirement and the
width-3 alignment observation via zero-init (3, 704) actor/critic embeddings
(2,307,645 parameters, feed-forward v6.1 spatial encoder, no GRU, no partial
resets, no stall age, no reset context, no action mask).

Sources and inputs:

- terra-baselines `f64694a569fbeb1353f2f908c46b9baab5f7e22b`
  (branch `experiment/trench-pose-alignment-20260818`);
- Terra `a4b838b6cb894fdf982b614d4deea96f778fd7b0`
  (branch `experiment/trench-fresh-dig-alignment-20260818`);
- bank archive
  `terra_v8_trench_pilot_pooled_bank_20260819.tar.zst`, SHA-256
  `e88370a5314bad189e75dbacd706cd12d08d9d5f920d6c877bbace7aca55d48c`:
  the pooled 12-condition slice (1,152 maps) of the finite-enriched V8 R2
  release (`.artifacts/terra_v8_trench_finite_enriched_20260819`); net4 and
  v7-trn conditions are excluded per the 2,400-map strict-gate preflight
  (61 net4 maps fail structurally; receipts in Terra `tools/`);
- reward stage `reward_v2` timing variant 0, distance protocol
  `obstacle_geodesic_8_physical_global_v1`, sidecar SHA `f0c43065…`;
- seed `20260818` both arms, 4 x 512 envs x 32 steps, 32 minibatches, two PPO
  epochs, 65,536 transitions/update, target u100,000, checkpoints every 500.

Jobs (submitted 2026-08-19, `gpuhe.120h`, es_hutter, 119:45, one node,
4 x RTX 4090, 8 CPUs, 64 GB each):

- C0 `11152229`, T1 `11152230`; run dirs
  `…/codex_terra_edge_runs/terra_trench_align_v1/runs/f64694a5…/s20260818/{c0,t1}`;
- W&B `trench_align_{c0,t1}_f64694a569_s20260818`.

Both arms passed a local 4090 first-update smoke (3/3 updates, finite model,
optimizer, and loss tensors asserted; gate prints False/True respectively).
Preregistration, endpoints, and stop rules:
Terra `TRENCH_ALIGNMENT_PILOT_PREREGISTRATION_20260819.md` (pilot decision at
u10,000; T1 invalid fresh-`DO` fraction must fall by half; stop if T1 exact
completion trails C0 by more than 5 pp at two successive evaluations).
Known debt before endpoint measurement: the fixed-bank evaluators do not yet
set `trench_alignment_observation`; wire it before evaluating pilot
checkpoints.

## Trench-aligned 37-condition partial-reset generalist recovery

The named `trench_align_generalist_partial_v1` capability recipe uses 25
foundation and 12 strict-gate trench conditions, with partial resets on by
default for this recipe only. The frozen full/partial bank identities and
complete design are recorded in
`research/TRENCH_ALIGNED_GENERALIST_PARTIAL_RESET_DESIGN_20260822.md`.

Recovery phase 1 job `11626135` is running on `eu-g6-044`; phase 2 job
`11626137` is pending on its `afterok` dependency. At the 2026-08-25 throughput
audit it had crossed u3,700 with finite checkpoints but sustained only
3,124.6 steps/s. Matched 4 x RTX 4090 controls with the same 65,536
transitions/update sustained 16,771.1 steps/s (C0 `11152229`), 16,503.0
(T1 `11152230`), and 15,800.5 (GRU control `11364188`). The 5.0--5.4x
regression is therefore real; partial resets and the strict trench gate are
not its cause.

The regression was introduced by the recovery's global
`--xla_gpu_autotune_level=0`. A frontend-off deterministic candidate was also
rejected on an exclusive Supercluster RTX 3060: after two compilation samples,
its first two steady updates reached only 591.69 and 579.32 steps/s. The current
replacement restores the default level-4 frontend on the eligible node pool
while retaining the `eu-g6-064,eu-g6-065` exclusions. It also fixes native
stall-age checkpoint resume, checks all three exact bf16 backward-filter shapes
against closed-form gradients, and requires five resumed updates with a
post-compile median of at least 12,000 steps/s before production can start. A one-GPU Euler compiler
canary using the same per-device batch and convolution shapes runs first; it is
only a fast correctness gate, not the aggregate-throughput measurement, and
its checkpoint is discarded. The pinned recovery source is u3,500 checkpoint
SHA-256 `f84a6cdfcb4aba0ca55abf1a658e4d57d21c6dffff9c4c2f61263733cd4f4790`.
The rejected Euler chain `11722865/11722918/11722925/11722935` was cancelled
before allocation. The slow jobs remain live until the replacement passes on
Euler; no new policy or curriculum claim follows from this compiler repair.

## Current issue checklist

The living status ledger, exact u40 readout, and bounded next actions are in
[`research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`](research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md).
The archived Oracle response remains unchanged in
[`research/ORACLE_TERRA_STAGING_REVIEW_20260814.md`](research/ORACLE_TERRA_STAGING_REVIEW_20260814.md).

Completed historical runs remain in [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md).
