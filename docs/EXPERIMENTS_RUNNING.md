# Experiments — current state (updated 2026-08-24)

## No live V8 movement-feedback jobs

The paired fresh-scratch jobs are complete; there is nothing left to cancel:

| Arm | Slurm | Terminal update | State | Final checkpoint SHA-256 |
| --- | ---: | ---: | --- | --- |
| repaired-runtime control | `11364188` | 50,000 | `COMPLETED 0:0` | `5459bd5347dbdf64431cd78df5f61f22b75ee56bc2b15662d9751fb2959a7f84` |
| six-bit feedback | `11364189` | 50,000 | `COMPLETED 0:0` | `8cde5ccd4fd4ef5b1ed716a9c5c3a4c4b43f69d44db66d29ed7db86f2ad7d7df` |

Both passed their startup gates, finished W&B, and wrote rolling plus `FINAL`
checkpoints. The final online aggregate shows tied success (0.99019 control,
0.99037 feedback) and a lower feedback no-effect rate (0.01450 versus
0.03152). This is diagnostic training evidence only: the preregistered
development-720 panel has not been run, so no policy is selected and feedback
remains disabled by default.

Frozen training source:

- terra-baselines `5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14`;
- Terra `c8ab920504e09173760c8beba71589102d54ed21`;
- paired seed `20260821`, terminal update `50,000`;
- full-bank archive `b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725`;
- partial-bank archive `eb200b151f6b47d9f2ea5f53f6b13cdb45b595a54029fd5d866ec732fea1c8b8`; and
- run root
  `/cluster/scratch/alesweber/codex_terra_edge_runs/terra_v8_movement_feedback_v1/runs/5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14/c8ab920504e09173760c8beba71589102d54ed21/s20260821`.

The completed online readout, exact checkpoint provenance, preregistered
question, and pending u50 decision gate are in
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

## Trench-aligned 37-condition partial-reset generalist recovery

The named `trench_align_generalist_partial_v1` capability recipe uses 25
foundation and 12 strict-gate trench conditions, with partial resets on by
default for this recipe only. The frozen full/partial bank identities and
complete design are recorded in
`research/TRENCH_ALIGNED_GENERALIST_PARTIAL_RESET_DESIGN_20260822.md`.

The 2026-08-25 audit measured only 3,124.6 steps/s in the original recovery,
versus 16,771.1 in C0 `11152229`, 16,503.0 in T1 `11152230`, and 15,800.5 in
GRU control `11364188`, all with the same 65,536 transitions/update on four
RTX 4090s. The regression came from the recovery's global
`--xla_gpu_autotune_level=0`, not from partial resets or the strict trench
gate. A frontend-off deterministic candidate was also rejected at 591.69 and
579.32 steady steps/s on an exclusive RTX 3060.

The level-4 bf16 repair first reached 4,944.71 steps/s on one GPU in job
`11735195`, but the first four-GPU attempt `11735196` failed before update 1
with `CUDNN_STATUS_EXECUTION_FAILED`. An identical traced rerun `11738360`
then completed u3,500--u3,505 with finite checkpoints and samples/s
`155.39, 150.27, 17546.44, 7820.19, 17454.15`; its post-compile median is
17,454.15 and passes the 12,000 gate. This pair proves that level-4 restores
matched historical speed but that unconstrained cuDNN plan selection is not
repeatable enough for production.

Revision `58e26fc969b9b0d42477c7ce8151dc7318be4fd4` therefore uses one direct
four-GPU path: bf16 level 4, the exact engine-20 denylist, and the successful
four-GPU autotune cache, SHA-256
`698e856cae464e5fea93e0b2121fc8de4d9cb691135571ca4b5d56f3259d16a3`.
The redundant one-GPU gate was removed because it cannot establish four-GPU
execution or scaling. Pinned-cache replay `11740651` is queued; fresh u0 smoke
and production remain conditional on it.

After `11738360` passed, slow jobs `11626135/11626137` were cancelled. The
latest preserved slow-run checkpoint is u4,000, SHA-256
`1a977ffca984458699c6b9ef3940bd3f3815699c876de6b58704e21f31484e7c`;
the run stopped at u4,442. The repair changes compiler selection only and is
not policy or curriculum evidence.

## v2 generalist with the corrected fresh-trench gate (submitted 2026-09-02)

Single gate-on arm, foundation + trench, launched from `main` after the gate's
standoff semantics were corrected (see Terra
`TRENCH_GATE_STANDOFF_SEMANTICS_BUG_20260901.md`): a dig is admitted iff the
chassis is parallel to the section axis (<= 0.2619 rad) AND the base centre is
within 2.0 m of the line (on top of the trench); the retired v1 band is off;
working distance is the dig cone's.

- terra-baselines `445ad79662eb0863a1588762074ec99bfbc18d28` (main);
  Terra `facc44e66aa36e6132267afaa4e3b9e0f38722f7` (main), which also carries
  the corrected footprint raster and contained dig-side soil relaxation;
- preset `trench_align_v2_generalist_gen`; bank = pooled 40-condition slice
  `train_v2_pooled_generalist` (25 foundation + 15 trench incl. net4, 3,840
  maps) of the finite-enriched V8 R2 release, archive
  `terra_v2_generalist_pooled_bank_20260901.tar.zst` SHA-256
  `1125177d322df6097f8da9f67ec95fe48762e16327f83dc157ec282b24993fb3`;
- reward_v2 timing 0, R2 distance protocol, seed `20260901`, 4 x 512 envs x 32
  steps, 32 minibatches, two PPO epochs, target u100,000 (beyond one
  allocation; wall exit with a checkpoint is CONTINUABLE), checkpoints every 500;
- job `12505854`, account `lterenzi` (`/cluster/project/rsl` became group-only
  on 2026-09-01 and `alesweber` can no longer read the pinned venv), Slurm
  account es_hutter, `gpuhe.120h`, 4 x RTX 4090; run dir
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_generalist/runs/445ad79…/s20260901/gen`;
  W&B `trench_align_v2gen_gen_445ad79662_s20260901`.

**Job `12505854` FAILED before update 1** (10 min on `eu-g6-071`, 4 x RTX
4090, driver 580.178.4, cuDNN 8.9.7): repeated
`conv_algorithm_picker: Results mismatch between different convolution
algorithms` on the bf16 3x3 backward-filter convs, then
`XlaRuntimeError: CUDNN_STATUS_EXECUTION_FAILED`. Same defect the
`generalist_partial_v1` campaign root-caused to cuDNN frontend engine 20 on
cc 8.9. Launcher `20b846c` ports that repair for `GPU_TYPE=rtx_4090` only:
`--xla_gpu_autotune_level=4`, the exact engine-20 denylist
(`scripts/euler_trench_align_v2/hlo_algorithm_denylist.pbtxt`), and the
pinned four-GPU autotune cache (SHA-256 `698e856c…`, group-readable under
`/cluster/project/rsl/alesweber/terra_runtime/autotune/`); the run contract
records `cudnn_repair`. Level 0 is not used (8x slow path in that campaign).
RTX 3090 (cc 8.6) path unchanged. Relaunched as **job `12508156`**
(terra-baselines `e2a020e`, Terra `502c80b2` = doc-only ahead of `c383b0b1`),
run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_generalist/runs/e2a020e…/s20260901/gen`;
W&B `trench_align_v2gen_gen_e2a020efc7_s20260901`. Fallback if it dies the
same way: `GPU_TYPE=rtx_3090`.

Launch gates: Terra suites 51 passed on the merged tree, baselines 44; local
GPU first-update smoke on the exact pooled bank (gate on, v2, bound 2.0 m,
checkpoint finite); archive round-tripped through the fail-closed loader.
Solvability under the corrected gate: every panel and pooled trench cell is
admissibly diggable from an aligned on-the-line station (zero loss at 2.0 m);
all 2,400 trench maps have complete covers under v2 (net4 re-admitted).

No matched control was launched; a clean causal claim needs a C0 pair.

### v2 trench specialist (submitted 2026-09-02)

Second arm on the same launcher: trench only, all 15 trench conditions
including net4 (re-admitted under v2), pooled bank `train_v2_pooled_trench15`
(1,440 maps; archive `terra_v2_trench15_pooled_bank_20260902.tar.zst`,
SHA-256 `788e47444d51a0281c1dbddfaea12683a90890afe2ee889cee6bc254ea002a72`),
preset `trench_align_v2_specialist_spec`, same gate semantics, seed, PPO
config and target as the generalist. terra-baselines `2a5716e` (main), Terra
`c383b0b1` (main; doc-only ahead of `facc44e6`). Job `12506562`, account
`lterenzi`, `gpuhe.120h`, **4 x RTX 3090** (in-job guard refuses any other
model); run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_specialist/runs/2a5716e…/s20260901/spec`;
W&B `trench_align_v2_spec_2a5716ee50_s20260901`. Local first-update smoke on
the exact bank: gate on, v2, bound 2.0 m, checkpoint finite.

**Job `12506562` FAILED before update 1** (15 min on `eu-g4-013`, 4 x RTX
3090): `CUDNN_STATUS_EXECUTION_FAILED` on every replica at the first
`_update_step`, with no autotuner mismatch warnings in the log. So the defect
is not 4090-specific. Evidence gathered: the August pilot (C0/T1) ran on the
same 580-series driver (580.173.2 vs 580.178.4 now) with the same three
"Results mismatch" warnings and survived five days; W&B system metrics show
the pilot and the failed generalist both peaking at 36% / 22% GPU memory, so
memory pressure is excluded; the pinned venv pairs cuDNN 8.9.7.29 with CUDA
12.9 cuBLAS/NVRTC/runtime wheels under the cuda/12.1.1 module. The failure is
therefore a flaky autotuner pick of a faulty cuDNN frontend engine (timing
decides which engine wins). The cc-8.9 denylist and cache do not apply on
3090, so the specialist was resubmitted on **4 x RTX 4090** with the same
repaired path as the generalist: **job `12508490`** (terra-baselines
`23297f6`, Terra `502c80b2`), run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_specialist/runs/23297f6…/s20260901/spec`;
W&B `trench_align_v2_spec_23297f63fd_s20260901`. A local RTX 4090 battery
(frontend off / denylist / default / level 0 / float32 convs at the exact
per-device shapes) is measuring a class-independent fix in parallel.
Evaluate checkpoints with `eval_fixed_bank.py --panel-family gate_main`
(the pilot's v1 checkpoints need `--gate-v1`).

## Current issue checklist

The living status ledger, exact u40 readout, and bounded next actions are in
[`research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`](research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md).
The archived Oracle response remains unchanged in
[`research/ORACLE_TERRA_STAGING_REVIEW_20260814.md`](research/ORACLE_TERRA_STAGING_REVIEW_20260814.md).

Completed historical runs remain in [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md).
