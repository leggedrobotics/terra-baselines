# Experiments — current state (updated 2026-08-21 00:56 CEST)

## V8 paired movement-feedback GRU pilot

Two independent fresh-scratch jobs were submitted on 2026-08-21. At the
recorded snapshot both are `PENDING (Priority)` with no allocated node, no
W&B history, and no training evidence:

| Arm | Slurm | W&B ID | Requested resources |
| --- | ---: | --- | --- |
| repaired-runtime control | `11364188` | `v8_movefb_c_5d7284f6ca_s20260821` | 4 x RTX 4090, 8 CPUs, 64 GB, 71:45 |
| six-bit feedback | `11364189` | `v8_movefb_f_5d7284f6ca_s20260821` | 4 x RTX 4090, 8 CPUs, 64 GB, 71:45 |

There is no dependency between the jobs. They compare equal transition counts,
not wall time. Each allocation must pass exact GPU/TRES, CUDA convolution,
NCCL, bank identity, u0 parity, and a finite W&B-disabled update-1 smoke before
starting its fresh production run. Do not call either arm healthy while it is
only queued, allocated, compiling, or running the smoke.

Frozen training source:

- terra-baselines `5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14`;
- Terra `c8ab920504e09173760c8beba71589102d54ed21`;
- paired seed `20260821`, terminal update `50,000`;
- full-bank archive `b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725`;
- partial-bank archive `eb200b151f6b47d9f2ea5f53f6b13cdb45b595a54029fd5d866ec732fea1c8b8`; and
- run root
  `/cluster/scratch/alesweber/codex_terra_edge_runs/terra_v8_movement_feedback_v1/runs/5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14/c8ab920504e09173760c8beba71589102d54ed21/s20260821`.

The preregistered question, scope, and u50 decision gate are in
[`research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md`](research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md).
The older sections below are retained as historical lineage and are superseded
where their live scheduler wording conflicts with this timestamp.

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
exact request for eight RTX 4090 GPUs, eight CPUs, and 64 GB RAM.  At the
recorded snapshot it is `PENDING (Priority)`, with no allocated node; Slurm's
current estimated start is 2026-08-15 07:15 CEST.  The phase-3 run directory is
reserved but contains no training evidence yet.  W&B remains in its completed
u40 state until the allocation
passes the in-job GPU/CUDA/NCCL/checkpoint gates and resumes it.

The launcher is commit
`bbaebc04c2ddc7c3ae667e434e223e1d01b95f84` on branch
`experiment/v8-v61-u40-phase3-20260814`.  Its run directory is
`/cluster/scratch/alesweber/codex_terra_edge_runs/terra_v8_v6_yolo_rv2/runs/dddc691c93ee21488cd7eeb8e01b067bf1f9733c/phase3/s20260807/v6_1_rv2_stall_age_v3`.

## Current issue checklist

The living status ledger, exact u40 readout, and bounded next actions are in
[`research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`](research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md).
The archived Oracle response remains unchanged in
[`research/ORACLE_TERRA_STAGING_REVIEW_20260814.md`](research/ORACLE_TERRA_STAGING_REVIEW_20260814.md).

Completed historical runs remain in [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md).
