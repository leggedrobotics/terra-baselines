# Experiments — current correction (2026-09-15 12:40 CEST)

**4670716 is held with `JobHeldUser`; it never started.** Lorenzo clarified
that the mixed generalist should use all foundations and a teacher trained on
the broad dataset. The earlier teacher selection missed the mature V8 policies.
Recovered FF u86000 completed341/384 foundations and670/720 promotion maps;
GRU u40000 completed343/384 foundations and677/720. Those are legacy Terra
results, with checkpoints present locally. The FF policy is now being replayed
under current Terra46738cde and the current trench gate on608development cases.

Correction artifacts and full3840-map bank qualification are in
`/home/lorenzo/moleworks/.artifacts/terra_generalist_broad_teachers_20260915/`.
The existing foundation pair4665916 is untouched. No replacement training job
has been submitted. The paragraphs below preserve earlier status snapshots.


The new CSCS generalist **4670716 is queued**; no training update has run there
yet. One node, four GPUs, account `lterenzi`/`d130`. The original 24-hour request
was shortened to **18 hours** because CSCS maintenance covers September 16,
07:00–19:00 CEST. Its maintenance blockage cleared; at 11:03:26 CEST the scheduler
reported `PENDING`, reason `Priority`, with no assigned start time or node.

It uses a fresh student with separate frozen foundation/trench teachers,
easy foundations plus all 15 trench training conditions, and zero added behavior
costs. CPU routing/gradient/resume tests, bank validation, local finite GPU
u1/u2/FINAL plus native u3, and independent source/submission reviews pass.
Four-GH200 runtime, NCCL, production-size startup and the first production
checkpoint remain unverified until the allocation starts. See
[the campaign design](research/GENERALIST_TASK_TEACHERS_20260915.md).

The pinned training source is baselines `a80fe8bfe14ac4b26cdcd8f306e54cfd97e5fca9`
and Terra `46738cde28e455da7c466fc0a2cb64f677d86401`; follow-up documentation
commits do not change that immutable snapshot. Campaign receipts and handoff:
`/home/lorenzo/moleworks/.artifacts/terra_generalist_teachers_20260915/`.
W&B ID `terra-generalist-teachers-4670716` is reserved for offline logging once
production starts; no online history is claimed yet.

Next recommended check: **2026-09-15 10:00 UTC (12:00 CEST)**, to resolve queue
admission, actual four-GPU startup, and the first save if due. This is a
recommendation; no new polling worker or scheduled follow-up was created.
No automatic continuation is queued. A finite checkpoint can continue after
maintenance; an 18-hour first segment alone is not a negative learning verdict.

The existing foundation comparison **4665916** is unchanged. Last independently
checked at 10:43:46 CEST: RUNNING on `nid005694`, elapsed 6:33:14. All older
authentication/status paragraphs below are historical snapshots.

Foundation teacher-KL recovery is prepared and passes local finite/native-resume
checks. **CSCS authentication blocks submission; no new job is queued.** See
the final entry. Earlier scheduler snapshots retain their original timestamps.

Fresh historical foundation replay: **old control 62/64, old 2× 63/64 under
current rules**, versus new global-normalization u41000 at 11/64. See the final
entry and foundation regression analysis. The live scheduler snapshot below
remains from 12:54 CEST; no jobs changed during this audit.

Both overnight Euler jobs are now training with full startup gates passed.
Completed fixed evaluations: **trench recovery 192/224 (roads 24/32)**;
**foundation global normalization 11/64**. The milestone watcher finished
normally. Penalties remain zero. See the final entry for today's live status
and comparisons; preceding September 13 observations are historical.


Two new overnight Euler recovery screens are submitted: foundation **14055215**
and old-trench recovery **14055463**, four RTX4090s and24hours each. Both local
finite resume tests pass; remote startup is not yet verified. Added penalties
remain zero. See the last entry for source, contracts and the automatic
milestone-evaluation monitor. CSCS observations below remain auth-limited.


Live CSCS job state is **unverified**: SSH authentication expired at 15:52 CEST
and was rejected at 22:22. Last confirmed training was foundation4655350 near
u41113 and trench4652918 near u85992 at 11:44. Scheduled allocation ends were
trench September13 17:10 and foundation September14 09:00; these are not fresh
scheduler observations.

A fresh local evaluation of the retained foundation **u41000 regresses to
3/64 exact**, versus8/64 at u33000; mean excavation falls93.3% to79.4%.
Trench u85500 is nearly flat at 162/224, but road completion remains 0/32.
Old u83500 weights replayed in the current environment achieve 179/224 and
20/32 roads. Manual replays show avoidable policy stalls.
Identity, numerical state and integrity checks pass. Added penalties remain
zero. This is a result for the saved morning checkpoint, not the latest
remote weights. See the final entry for evaluation evidence and the
[recipe](../scripts/foundation_reward_sweep/README.md); earlier observations
below are historical.

## 2026-09-07 local pipeline maintenance

Junction/continuation/compilation fixes passed local CPU checks and a one-GPU
checkpoint-replay smoke through absolute update 5, with finite model/optimizer
state and persistent-cache reuse. W&B was disabled and no production or Slurm
job was launched. See the 2026-09-07 entry in `EXPERIMENTS_LOG.md`. Historical
cluster entries below were not refreshed by this local maintenance task.

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

**Job `12508490` FAILED too** (8 min on `eu-g6-072`, 4 x RTX 4090, with the
denylist + pinned cache active, contract `cudnn_repair=rtx4090_engine20_
denylist+autotune_cache_698e856c`): all four replicas
`CUDNN_STATUS_EXECUTION_FAILED` at the first `_update_step`, no autotuner
output at all (plans came from the cache). The generalist `12508156` runs the
identical pinned plans on `eu-g6-071` and cleared update 1. So the failure is
nondeterministic at execution even with a fixed plan; plan pinning is not a
reliable repair. Local RTX 4090 battery (exact per-device shapes, 30 updates
each, shared card): frontend off, denylist+level 4 and default all pass at
the same steady-state ~6 s/update (the local card does not reproduce the
Euler failure, so the battery ranks throughput only). Launcher `c0e06d1`
adds `TERRA_CUDNN_REPAIR` = auto | denylist_cache | frontend_off | none
(auto = denylist_cache on 4090, frontend_off elsewhere). Specialist
relaunched with **`--xla_gpu_enable_cudnn_frontend=false`** (legacy cuDNN
algorithm API, no frontend engines) on **4 x RTX 3090**: **job `12511685`**
(terra-baselines `c0e06d1`, Terra `502c80b2`), run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_specialist/runs/c0e06d1…/s20260901/spec`;
W&B `trench_align_v2_spec_c0e06d1527_s20260901`. If the generalist dies the
same way, it is relaunched with frontend_off as well.

Local battery final (RTX 4090 shared with another job, 512 envs x 32 steps x
32 minibatches, 30 updates, seconds per update over updates 10--30):
frontend_off 6.0, denylist+level 4 6.4, default 5.2, level 0 30.0 (5x, dead
end, matches the partial_v1 audit); float32 convs OOM on the shared card
(6 GB extra activations) and would change numerics, not pursued. The local
card never reproduced the Euler execution failure, so the battery ranks
throughput only. Generalist `12508156`: u100 at 19 min, 3.69 s/update =
17.8k steps/s, pilot speed.

**Generalist early readout (2026-09-02, u2000/u3000 checkpoints, local
`eval_fixed_bank.py` gate_main/development panel, 608 slots / 38 conditions,
deterministic, horizon 450, seed 20260724, v2 gate on so admissible =
raw).** W&B `online_eval/success_within_horizon_rate` 0 -> 0.055 by u3100,
return -4.36 -> -3.80, episode length 446 -> 430, entropy 1.9-2.0 (no
collapse), KL 0.003-0.006, no nonfinite; 12x the pilot T1 at matched
updates, about half the gate-off C0. Panel: ALL of the aggregate is trench
straights: side1 / side1-tight 7/16 (43.75%), altsides / side2 4/16 (25%),
each roughly doubled from u2000; every tee / seg / net3 / net4 condition
0/16; ALL 24 foundation conditions 0/384 at both checkpoints (graded
terminal_absolute macro 0.28 -> 0.41, dig_fraction 0.52 -> 0.71, but no
episode closes). Trench-minus-net4 endpoint 22/176 = 12.5% at u3000
(pilot T1 reached 38.6% at u10000). Watch item: failing straight episodes
regress (median dig_fraction 0.875 -> 0.381) while successes grow, i.e.
finish-or-stall sharpening. W&B logs no per-family or per-condition eval;
the panel eval is the only family-resolved instrument. Throughput drifts
3.67 -> 4.0 s/update (ETA 108 h vs 115 h wall); a manual continuation via
`TERRA_RESUME_FROM` may be needed near the end. Receipts:
scratchpad `gen_u3000_panel/` (session-local).

**Generalist u10000 panel (2026-09-03, same gate_main/development recipe,
matched update with the pilot U10000 readout).** Whole panel 279/608 =
45.9% (pilot T1 73/608, C0 raw 204/608). Foundation 226/384 = 58.9%, every
one of the 24 foundation conditions closes episodes (best
`v7-fnd-pads-adjacent` 16/16, `fnd-slab-ring3x` 15/16; T1 4/384, C0
12/384). Trench straights 53/64 = 82.8% (T1 41/64), closing in 113 steps
mean. Junctions 0/160: tee 0/32, seg 0/32, net3 0/48, net4 0/48 (T1 had
27/112 across tee/seg/net3); graded terminal_absolute 0.19-0.28 there, so
material moves but episodes stall at the 450 cap. Pilot endpoint (trench
minus net4, admissible): 53/176 = 30.1% vs T1 68/176 = 38.6%; the whole
gap is the junction gap. u5000 -> u10000: 31/608 -> 279/608, foundation
0 -> 226, straights 31 -> 53, junctions flat at 0. Receipts clean
(integrity passed, no horizon censoring, manifest 1216bee3be9f). Caveat:
v2 does not enforce the standoff band, so admissibility is marginally
looser than T1's (about 0.8% of T1's blocked attempts were standoff-only).
Watch item for u20000: any junction completion; if tee/seg/net stay at
zero, the junction veto under v2 needs a look before calling it an RL
difficulty. Receipts: scratchpad `gen_u10000_panel/`.

**Junction diagnosis at u10000 (rollout probe, 224 trench slots, patched to
v2 semantics with per-step clause records; scratchpad
`gen_u10000_junction_probe/`).** Explicit gate refusals are a non-event:
61 in 224 episodes, confined to 2 episodes. The gate binds as DETERRENCE:
the policy presses DO when the exported valid bit is 0 at 0.0-0.02% of
steps. On junction maps the machine reaches a dig-opportunity pose (empty,
fresh cell in cone) as often as on straights (26-34% of steps) but the gate
marks 71 / 86 / 83 / 97% of those poses invalid (tee / seg / net3 / net4;
straights 62%). Clause shares of the invalid poses: junction
all-or-nothing 41 / 45 / 43 / 22% (straights 0% by construction), yaw-only
54 / 54 / 47 / 60% (straights 56%), on-the-line 4 / 2 / 10 / 17%
(straights 44%). Where the gate says valid, DO is pressed 12 / 23 / 13 /
97% (straights 80%). Three classes among the 160 junction episodes: A (19)
parked the whole horizon against the junction veto (dig 0.24; e.g. seg2
slot 389: 0/68 cells, 437/438 opportunities vetoed because the +-30 deg
cone straddles both oblique sections and no yaw is parallel to both);
B (36) parked against the yaw clause (dig 0.25); C (105) rarely
gate-blocked (dig 0.43) but 67% of post-stall moves refused by
traversability (dug cells non-traversable, 7x11 chassis). All stop digging
by step 40-60 and spend ~400 steps in 5-8 base cells; 90-96% of undug
cells were inside cone reach of a visited pose. Illegal spoil (8-15 units)
lands on neutral ground, not on trench cells (secondary). Verdict: mix of
a structural junction veto (class A), yaw deterrence (B), an RL deficit
(DO pressed at only 12-23% of valid junction poses) and a traversability
deadlock (C). Next: per-cell admission variant probe.

**Per-cell admission A/B (scratch Terra copy, same probe, same 224 slots,
u10000 checkpoint; straight control 0/64 episodes changed).** Replacing the
all-or-nothing veto by admitting the aligned fresh cells in the cone (valid
bit on when at least one is admissible): junction gate-invalid poses
18,101 -> 9,693, all-or-nothing share 36% -> 0%, yaw share 54% -> 95%;
junction dig fraction 0.366 -> 0.515 (tee 0.45 -> 0.61, seg 0.39 -> 0.55,
net3 0.35 -> 0.52, net4 0.31 -> 0.43); episodes at dig fraction >= 0.8
0 -> 14 of 160; completions 0 -> 2. Class A (parked against the veto)
0.24 -> 0.57 and its DO-when-valid rate 5.8% -> 87.9%. Residual: yaw
deterrence (95% of remaining invalid poses), RL competence (DO at 16-35%
of admitted junction poses vs 80% on straights), traversability deadlock
(blocked moves rise to 180-240/episode as episodes keep working). Scratch
A/B only: no EnvConfig field, fingerprint identical to baseline, not
poolable with panel receipts. Decision pending (user): adopt per-cell
admission as the gate semantics and restart both arms, or continue under
the veto. Artifacts: scratchpad `gen_u10000_junction_probe/` and
`terra_percell_variant/`.

**Geometric confirmation (2026-09-03, axis-sweep feasibility tool, same
code/config for both rules, self-checked against Terra's gate, 0
mismatches).** On-axis lane (stand on the line, dig straight ahead or
behind, back up, one axis at a time), maps complete at tolerance
0.5 / 1.0 / 2.0 tiles: straight 64/64 both rules; tee veto 2/4/5 of 32 vs
per-cell 32/32/32; segmented veto 0/2/5 of 32 vs 32/32/32; net3 veto 0/0/0
of 16 vs 16/16/16. Under the all-or-nothing veto a junction cannot be dug
by approaching along one of its axes; per-cell admission restores the
intended semantics (junction diggable from any of its axes, order left to
the policy) with every straight map unchanged. Terra note
TRENCH_JUNCTION_PER_CELL_ADMISSION_20260903.md on branch
epoch/trench-per-cell-admission-20260903.

**Job `12511685` (frontend off, 3090, `eu-g4-007`) FAILED the same way** at
the first update, and its log shows even the legacy algorithms disagreeing by
~50% on the bf16 backward-filter convs. Verdict: the failure is independent
of plan selection and GPU class; it is the same cuDNN 8.9.7 defect the
13 August v6.1 audit bisected (needs the token mixer + flatten-reduce
readout, shape dependent, float32 does not help) and that killed job
`10569391` mid-run at u14,001 on 8 x RTX 3090. It predates the driver patch
and the pilot's two clean starts were luck (5 of 7 v2 starts failed today).
The venv is untouched since 30 July (identical to the pilot's).

Operational fix, launcher `c239d12`: a failed attempt greps its own log for
`CUDNN_STATUS_EXECUTION_FAILED` and sbatch-es itself again into the same
RUN_DIR (`ATTEMPT+1`, up to `MAX_ATTEMPTS`=6, same sbatch options via
`RESUBMIT_SBATCH_ARGS`), resuming from the newest `*_update_*.pkl` when one
exists (model, optimizer, absolute update clock through `--resume_from`;
W&B `resume=allow`). `run_contract.env` records attempt/resume_from and the
terminal status; each attempt keeps `run_contract.attempt<N>.job<id>.env`.
`TERRA_RESUME_FROM` allows a manual continuation. The running generalist
`12508156` predates this launcher; if it dies, continue it manually with
`TERRA_RESUME_FROM=<its newest checkpoint>`.

Specialist attempt chain: **job `12517301`** (4 x RTX 4090, `auto` =
denylist + pinned cache, terra-baselines `c239d12`, Terra `502c80b2`), run
dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_specialist/runs/c239d12…/s20260901/spec`;
W&B `trench_align_v2_spec_c239d124d5_s20260901`; children (if any) are
listed in that run dir's contract.

**Euler specialist chain CANCELLED 2026-09-02 (user decision: one
generalist on Euler + one specialist on CSCS is enough).** Attempt 0
`12517301` and attempt 1 `12561591` both died at the first update with the
cuDNN error on `eu-g6-047` and resubmitted themselves as designed (child
pointers recorded); attempt 2 `12562373` was cancelled by hand at 5 min
while compiling. Euler v2 start tally today: 2 of 8 survived. The
specialist endpoint is now carried by CSCS job `4586880` and its chain.
Evaluate checkpoints with `eval_fixed_bank.py --panel-family gate_main`
(the pilot's v1 checkpoints need `--gate-v1`).

### v2 trench specialist on CSCS Daint (submitted 2026-09-02)

Third venue for the same arm, opened because every Euler attempt died on the
cuDNN 8.9.7 defect. Daint's GH200 nodes are a different stack (aarch64,
NVIDIA JAX 24.10 image `terra-jax+jax24.10-v1`, cuDNN 9, CUDA 12.6, NCCL
2.22.3, driver 590.48.01), so that defect does not apply. Launcher
`cluster/cscs/submit.sh` (`--profile production`), one JAX process with all
four GH200 GPUs on one node.

Dataset uploaded once as
`/capstor/scratch/cscs/lterenzi/terra-training/datasets/terra_v2_trench15_pooled_bank_20260902`
(**1,440 maps**; archive `terra_v2_trench15_pooled_bank_20260902.tar.zst`,
SHA-256 `788e4744…a002a72`; remote `dataset.json` byte-identical to the local
copy, so the R2 sidecar receipt checks out). `DATASET_PATH` is the bank root
and the preset selects `train_v2_pooled_trench15` itself.

Run id `terra-v2spec-145a94c-s20260901`, **Slurm job `4586880`**, account
`d130`, partition `normal`, 24 h, node `nid005954`, started 2026-09-02
16:43 UTC. Snapshot revisions: terra-baselines `145a94c`, Terra `502c80b2`
(both clean detached worktrees; the one dirty entry is the `submit.sh` change
below, which the job does not execute). Image tag `terra-jax+jax24.10-v1`. Run
root `/capstor/scratch/cscs/lterenzi/terra-training/runs/terra-v2spec-145a94c-s20260901`,
checkpoints in `.../checkpoints`; run name
`trench_align_v2_spec_cscs_145a94c_s20260901`.

Trainer flags are the Euler launcher's `train_mixed.py` line verbatim (medium
mlp core, `resnet_spatial_8x8_se_sa_xattn`, bf16 encoder / f32 attention,
critic 512,256, stages 24,48,64,96, blocks 2,2,3,3, mixer init 0.1,
flatten_reduce 32, latent queries 8, aux 0, vf_coef 2.0, entropy 0.15 to 0.02
over 20,000, `--no_value_clip`, `--carry_work_observation`, `--lr 3e-4`,
`--reward_stage reward_v2`, `--reward_v2_timing_variant 0`,
`--distance_protocol_id obstacle_geodesic_8_physical_global_v1`,
`--distance_sidecar_sha256 f0c43065…6c58980`, `--fail_on_nonfinite`,
`--finite_check_interval 10`, `--eval_episodes 100`,
`--log_eval_interval 100`), plus the CSCS-specific
`--config trench_align_v2_specialist_spec`,
`--name trench_align_v2_spec_cscs_145a94c_s20260901 --exact_run_name`,
`--seed 20260901`, `--num_devices 4`, `--num_envs_per_device 512`,
`--num_steps 32`, `--num_minibatches 32`, `--update_epochs 2`,
`--total_timesteps 6553600000` (= 4 x 512 x 32 x 100,000),
`--checkpoint_interval 500`, `--cache_clear_interval 1000`,
`--log_train_interval 10`, `--keep_checkpoint_history` and the CSCS
`--checkpoint_dir`. These follow `run_training.sh`'s production defaults
(`solo_excavator`, 1024 envs/device, 16 minibatches, 5e10 timesteps,
checkpoint 100, log_train 1) and argparse takes the last occurrence; the
generated `job.sbatch` and the trainer's own configuration banner were both
inspected and every override is in effect (2,307,645 parameters, obs_len 23).
The profile's `--machine daint` only feeds run-name composition and is inert
under `--exact_run_name`.

W&B: Daint has no credential, so the job runs `WANDB_MODE=offline` through the
new `submit.sh --wandb-mode online|offline` option (written into the generated
EDF `[env]` block). The Slurm log and the checkpoints are the record; the
offline directory under `runs/…/wandb` can be `wandb sync`ed later.

Runtime check at start: 4 x NVIDIA GH200 120GB, JAX
`0.4.33.dev20241023+85f5076f1` (jaxlib 0.4.33 from `nvcr.io/nvidia/jax:24.10-py3`),
cuDNN 9, CUPTI/cuBLAS/NVRTC from CUDA 12.6, NCCL 2.22.3; jitted convolution
backward and the four-GPU `pmap` all-reduce both passed. Loading 1,440 maps
took 43 s. **Two XLA compilations** precede steady state: iteration 0 took
266 s and iteration 1 took 272 s (GPUs at 0% during both, host-side
compilation of the finite-check and plain variants of the update step);
from iteration 2 the run is steady at **2.19 s/update** (83 updates in 182 s;
tqdm 2.18 s/it; ~29,900 env steps/s, versus 3.69 s/update = 17.8k steps/s for
the generalist on 4 x RTX 4090). No cuDNN execution failure.

`normal` caps at 24 h and 100,000 updates would need ~61 h, so this job reaches
roughly 39,000 updates. It writes a checkpoint every 500 updates and is
continuable with the trainer's `--resume_from`.

Continuation is now wired into the launcher. `submit.sh --resume-latest` writes
`TERRA_RESUME_LATEST = "1"` into the EDF `[env]` block; `run_training.sh` then
picks the newest `*_update_*.pkl` under `runs/.../checkpoints` and appends
`--resume_from <path>` after the caller's arguments, so the model, the
optimizer state and the absolute update counter carry over and the job
continues towards the same `--total_timesteps`. `submit.sh --dependency
afterany:JOBID` adds the matching `#SBATCH --dependency` directive, so the
whole chain is queued in advance. Environment, RNG and action-history state
restart at each hand-off, so the continuation is not bit-exact.

Two follow-ups were submitted on 2026-09-02 with the same run id, the same
snapshot (`--no-sync`), `--wandb-mode offline` and the trainer arguments of
`4586880` verbatim (the generated `job.sbatch` differs from the running job's
only by the dependency line, and the EDF only by `TERRA_RESUME_LATEST`):

- **`4586997`** (job B), `--dependency afterany:4586880`
- **`4586999`** (job C), `--dependency afterany:4586997`

Each job costs about 9 minutes of start-up before steady state (two XLA
compilations, ~266 s and ~272 s, plus the 43 s map load) and then covers
roughly 39,000 updates in 24 h, so the three jobs together reach the 100,000
update target with margin. Because `--no-sync` reuses the immutable snapshot
staged for `4586880`, the snapshot's own
`terra-baselines/cluster/cscs/run_training.sh` was replaced in place with the
continuation-aware version (original kept beside it as
`run_training.sh.attempt0`, the substitution done by atomic rename so the
running job keeps its open inode, and the patch recorded in the snapshot's
`SOURCE_REVISIONS.txt`). The running job's original `job.sbatch` and
`terra.edf.toml` are kept in the run root as `*.attempt0`.

The hand-off path was exercised end to end before it is needed. Job
**`4587176`** (partition `debug`, 30 min, node `nid006553`, run id
`terra-v2spec-resumesmoke-20260902`) took a copy of `4586880`'s
`..._update_000500.pkl` as the only checkpoint of a fresh run root and was
submitted with `--resume-latest` and `4586880`'s trainer arguments verbatim
except `--total_timesteps 33095680` (= 4 x 512 x 32 x 505), a smoke `--name`
and its own `--checkpoint_dir`. The launcher printed
`resume_from=.../terra-v2spec-resumesmoke-20260902/checkpoints/trench_align_v2_spec_cscs_145a94c_s20260901_update_000500.pkl`,
the trainer printed `Loaded resume checkpoint`, `Replaced model parameters from
checkpoint.` and `Restored optimizer state from checkpoint (next_update=500).`,
ran only the 5 remaining updates (the eval fired at absolute update 500) and
exited 0; `sacct` reports COMPLETED, 15:31 wall (4.7 min start-up and map load,
9:18 to the first update through both XLA compilations, 2.2 s/update after).
Both runtime files of the reused `4586880` snapshot are byte-identical to the
freshly staged ones, so the smoke ran the same code the continuations will
(`run_training.sh` SHA-256 `a7274e46…`, `train_mixed.py` `e370c5bf…`). One
launcher note: a second `submit.sh` call for the same run id needs `--no-sync`,
because `sync_code.sh` refuses an existing snapshot.

### v2 trench specialist + relocation/admissible observations on CSCS (submitted 2026-09-02)

Paired arm for the running specialist above: same bank, seed, recipe, and
trainer arguments verbatim, plus exactly two observation flags,
`--relocation_distance_observation` (Terra's static geodesic dump-zone
distance map as a twelfth encoder channel) and `--admissible_dig_observation`
(width-12 fresh digs a DO would be admitted per cabin angle from the current
base pose, LocalMapNet's tenth map). Parameters 2,307,645 -> **2,311,701**
(+216 stem conv, +3,840 local-map MLP), `obs_len` 23 -> 25. Sources are the
merged mains: Terra `09712ad5` and terra-baselines `ad9ee96` (snapshot
`SOURCE_REVISIONS.txt`: both clean, 0 dirty entries), staged from the sibling
layout `.worktrees/cscs_stage_obs_v2/{terra,terra-baselines}`.

Gates before submission: local RTX 4090 smoke (128 envs, 3 updates, nonfinite
guard every update, finite FINAL checkpoint, flags recorded in `train_config`);
CSCS debug smoke **job `4588162`** (run id `terra-v2spec-obs-smoke-20260902`,
smoke profile with the specialist config and both flags, `COMPLETED 0:0` in
8:55, `obs_len = 25`, 2,311,701 parameters, one finite update, FINAL
checkpoint written).

Run id `terra-v2spec-obs-ad9ee96-s20260901`, run name
`trench_align_v2_spec_obs_cscs_ad9ee96_s20260901`, account `d130`, partition
`normal`, 24 h each, `WANDB_MODE=offline`, run root
`/capstor/scratch/cscs/lterenzi/terra-training/runs/terra-v2spec-obs-ad9ee96-s20260901`:

- **`4588229`** (job A), started 2026-09-02 19:54 UTC;
- **`4588230`** (job B), `--no-sync --resume-latest --dependency afterany:4588229`;
- **`4588231`** (job C), `--no-sync --resume-latest --dependency afterany:4588230`.

Each submission rewrites `job.sbatch` and `terra.edf.toml` in the run root, so
the files there are job C's; Slurm holds each job's own batch script. The
control chain `4586880 -> 4586997 -> 4586999` is untouched and is the matched
comparison. The rollout buffer carries the float32 distance map, about +16 KB
per env-step; irrelevant on GH200 120 GB.

## Current issue checklist

The living status ledger, exact u40 readout, and bounded next actions are in
[`research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`](research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md).
The archived Oracle response remains unchanged in
[`research/ORACLE_TERRA_STAGING_REVIEW_20260814.md`](research/ORACLE_TERRA_STAGING_REVIEW_20260814.md).

Completed historical runs remain in [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md).

## Per-cell junction admission epoch (2026-09-03)

User decision after the geometric check: the all-or-nothing junction veto is
removed from Terra (per-cell admission is the only behaviour; no switch).
Terra `a4da127a` = branch `launch/trench-per-cell-20260903` (veto removal
on top of `c703c4eb`; Terra main additionally received the unrelated
`09712ad5` relocation-distance / admissible-dig-map export from another
session, which is NOT part of this launch). terra-baselines `7989a9b`
(flag threading dropped, arms `genpc` / `specpc` map to the v2 presets,
run contract `gate_semantics=v2_yaw_parallel_on_the_line_per_cell_admission`,
Terra pin a4da127a). Local gates: Terra alignment + dump-contract suites
48/48 on a4da127a; local first-update smoke of the specialist preset on
a4da127a (3/3 updates, finite checkpoint).

Veto-era runs cancelled at 08:30 UTC: Euler generalist `12508156`
(u14,000, 28 checkpoints kept on disk) and the CSCS specialist chain
`4586880` / `4586997` / `4586999` (u18,000, 34 checkpoints kept). Their
u10000 panel and probe receipts remain the veto baseline.

### v2 generalist, per-cell (Euler, submitted 2026-09-03)

Job `12640873`, arm `genpc`, preset `trench_align_v2_generalist_gen`,
bank `train_v2_pooled_generalist` (3,840 maps, same archive), seed
20260901, 4 x RTX 4090, cuDNN repair `auto` (denylist + pinned cache) with
the self-resubmit chain (`MAX_ATTEMPTS`=6); run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_generalist/runs/7989a9b…/s20260901/genpc`;
W&B `trench_align_v2_genpc_7989a9bea7_s20260901`. Attempt chain recorded in
that run dir's `run_contract.attempt<N>.job<id>.env` files.

**Job `12640873` FAILED at start (3 min):** `relocation_distance_observation
requires Terra obs['relocation_distance_map']`. Cause: baselines `e18f7f1`
(other session, 2026-09-02) made the v2 launcher pass
`--relocation_distance_observation --admissible_dig_observation` (parameter
count 2,311,701, obs length 25) for Terra `09712ad5`, and my re-pin to
`a4da127a` (veto removal on top of `c383b0b1`-era main) kept the flags but
not the Terra that exports the maps. Not a cuDNN failure, so no
self-resubmit. Resolution: the epoch is per-cell admission AND the two new
observations. Terra main `171cf116` = `09712ad5` + veto removal (alignment,
dump-contract and admissible-dig-map suites 52/52); baselines `ba08192`
pins it and points `TERRA_REPO` back to the main worktree. Local smoke with
the exact launcher flags on 171cf116: 2/2 updates, finite checkpoints,
2,311,701 parameters. The per-cell-only branch
`launch/trench-per-cell-20260903` (a4da127a) is superseded.

### v2 generalist, per-cell + relocation/admissible-dig observations (Euler, submitted 2026-09-03)

Job `12643491`, arm `genpc`, preset `trench_align_v2_generalist_gen`,
Terra `171cf116`, terra-baselines `ba08192`, 4 x RTX 4090, cuDNN repair
`auto`, self-resubmit chain; run dir
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_align_v2_generalist/runs/ba08192…/s20260901/genpc`;
W&B `trench_align_v2_genpc_ba08192ca3_s20260901`.

**Chain exhausted: 7 of 7 starts died of `CUDNN_STATUS_EXECUTION_FAILED`
at the first update** (attempts 0-6: jobs `12643491`, `12644464`,
`12646060`, `12647121`, `12648578`, `12649756`, `12650797`; nodes eu-g6-042
/ 027 / 050 / 050 / 056 / 057 / 057; 9-10 min each; no autotuner mismatch
lines, no other error). Versus 2 of 3 failing starts yesterday, the new
observation layout appears to raise the failure probability on the cuDNN
8.9.7 stack. Fallback chain in **frontend-off mode** (legacy cuDNN API, no
frontend engines): root job **`12686930`**, arm `genpc`,
`TERRA_CUDNN_REPAIR=frontend_off`, same revisions (Terra 171cf116,
baselines 20cf553 = ba08192 plus ledger commits), run dir
`runs/20cf553…/s20260901/genpc`. Open decision: move the generalist to CSCS
(no cuDNN failures there, third daint node) if this chain also exhausts.

**Superseded 2026-09-03 17:18 UTC:** the frontend-off fallback `12686930`
was cancelled while pending by the other session, which is replacing the
Euler runtime instead: a cuDNN 9 runtime smoke campaign
(`terra_cudnn9_runtime_smoke`; 1-GPU smoke `12690597` PASSED with three
cold update-1 starts and ten continuous updates; 4-GPU smoke `12696174`
queued) gates the Euler generalist `12688665` (`afterok:12696174`, same
revisions ba08192 / 171cf116, 4 x RTX 4090). This session no longer
submits Euler generalist jobs; the CSCS specialist chain stays here.

### v2 trench specialist on CSCS Daint, per-cell + relocation/admissible-dig observations (submitted 2026-09-03)

Same epoch as the Euler generalist: Terra `171cf116`, terra-baselines
`ba08192`, preset `trench_align_v2_specialist_spec` with
`--relocation_distance_observation --admissible_dig_observation` (obs length
25, 2,311,701 parameters), bank `terra_v2_trench15_pooled_bank_20260902`
(1,440 maps), seed 20260901, 4 x GH200, W&B offline. Run id
`terra-v2specpc-obs-ba08192-s20260901`; chain **A `4592317`** (running,
`nid005567`) -> **B `4592319`** (afterany A, `--resume-latest`) -> **C
`4592320`** (afterany B). Steady rate about 2.2 s/update between evals
(updates 40 -> 80 in 88 s), unchanged from the veto-era run. An interim
chain for run id `terra-v2specpc-7989a9b-s20260901` (Terra a4da127a,
without the observation flags; jobs `4592193` / `4592194` / `4592197`) was
cancelled before it reached update 500. The user's own chain
`terra-v2spec-obs-ad9ee96-s20260901` (jobs 4588229-4588231) is untouched.


## September 9, 2026: excavation reliability submission preparation

Status: local gates passed; Euler inputs staged; no job submitted. Terra
`ba9cc214` supplies strict occupied footprints, eligible-soil selection and
short tracked maneuvers. Baselines `9354b89` adds retained work-pose metrics and
a one-RTX4090, 24-hour full-bank 2x recipe. Four local native updates including
an ordinary process restart pass finite and transition-integrity checks; the
PPO executable cache is reused. The specialist regression screen is 175/224
versus 188/224. The mixed-bank generalist u5000 remains the initializer: it
completes 32/608 full-panel cases under the repaired environment, versus 0/608
for the easy-foundation 2x u15000 checkpoint. Both remain weak on foundations.
Independent review and Slurm test-only validation pass. Ready for one bounded
training allocation; see the [current readiness report](../../../../.artifacts/terra_excavation_reliability_20260909/SUBMISSION_READINESS.md)
for the selected checkpoint, exact source/runtime/bank identities and launch.
Canonical checkouts and unrelated running/queued jobs are preserved.

## September 9, 2026: scratch foundation comparison on CSCS

The user replaced the proposed Euler continuation with scratch initialization
on the four-GPU CSCS node. The prepared comparison is control versus 2x costs,
each at seeds 20260909 and 20260910, using the same repaired Terra `ba9cc214`
and easy-foundation bank. One 24-hour node is at most 96 GPU-hours. No generalist
or Euler duplicate is included. No job has been submitted: the CSCS certificate
expired at 15:10:46 CEST and SSH currently rejects authentication.

The scratch launcher passes 43 focused recipe tests and shellcheck. A local
512x32 two-update scratch smoke passes: next update 2, Adam step 128, finite
model/optimizer/loss and zero transition-integrity counters. Independent code
review found no actionable issues; actual Daint binding/runtime still needs
the in-allocation checks. See the [current plan and evidence](../../../../.artifacts/terra_foundation_scratch_cscs_20260909/PLAN.md).


## September 10, 2026: paired foundation and trench scratch comparison

The user approved foundation control/2x and trench control/2x as four independent
one-GPU policies on one 24-hour CSCS node (at most 96 GPU-hours). All use seed
20260909 and repaired Terra `ba9cc214`; no historical initializer is imported.
This replaces the two-seed foundation-only proposal above. Foundation uses the
existing 256-map easy bank, trench the existing 1,440-map/15-condition pooled
bank including junctions. Comparisons are within task, with one paired seed.
Job **4634548** was submitted September 10 at 00:29:56 CEST and is
**PENDING (Priority)** at 00:31:34. The provisional Slurm start estimate is
22:55 CEST September10 and can change. ReqTRES: 256 CPUs, one node, four GPUs;
normal/d130, 24 hours. Staged source is baselines `30f5f97` with Terra `ba9cc214`.
The 44 focused tests, independent review, both local family scratch smokes and
Slurm test-only passed. The new trench smoke completed u1/u2 with finite
model/optimizer/loss and zero integrity counters. Its local log retains a cuDNN
bf16 autotuning mismatch warning. Daint has not allocated a node or run its
numerical gates yet. See the [current plan and execution evidence](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/PLAN.md)
for current submission/runtime status. No generalist or Euler duplicate is
included. Evaluate matched checkpoints separately on the 64 easy-foundation
validation cases and 224 trench rows of the unchanged 608 development panel.


### September 10, 09:46 CEST: all four scratch gates passed

Job 4634548 is RUNNING on nid005895, started 09:32:30 with four distinct GH200s;
end time is September 11 09:32:30. Node-level conv backward/NCCL and each arm's
own conv backward/u1/u2 gates passed. All four checkpoints have finite model,
optimizer and loss, Adam 128 and zero transition-integrity counters. Each arm
restored its own new u2 optimizer state for production; first-update compilation
was still running at 09:46:18. No production checkpoint or matched behavioral
evaluation exists yet. This is a numerical-startup result only.

The fixed environment's random-transition and recorded-state probes support
mass/occupancy correctness and restored short moves. Frozen specialist replay
regressed 188→175/224; generalist trench replay improved 30→32/224 with foundations
still 0/384. Thus there is no demonstrated overall learned-policy improvement.
The scratch comparison holds the fixes constant and tests zero versus 2x costs
within each task, with one paired seed. See the [status and interpretation](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_MORNING.md)
and its per-arm live evidence. No jobs or training settings changed during the
status check.

At 09:48:15, both foundation arms had production receipts through u31 and steady
updates at roughly 7,150 transitions/s (2.3 s/update); both trench arms were still
compiling their first resumed production update. The foundation u31 batch had
512 timeouts and no successes in each arm; this is too early to rank costs or
claim saturation. The linked morning status retains the exact observations.


### September 10, 13:35 CEST: all arms past u5000; cost concern

CSCS 4634548 remains RUNNING on four GH200s, with recorded updates F0=5500,
F2=5921, T0=5381, T2=5861. Latest saved checkpoints are 5500/5500/5000/5500.
All four retrieved u5000 checkpoints have finite model/optimizer/loss, zero
transition integrity and Adam 320000; native continuation is working. No
execution/cuDNN/nonfinite error was found. Rates are around 6100-7200
transitions/s/GPU. No training settings or jobs changed.

The matched online update 4000-5000 window raises concern about early cost
suppression: foundation mean excavation 68.9% control vs 15.4% 2x; trench 48.3%
vs 0.32%. These are sampled training episodes, not a held-out efficiency result.
The authorized 24-hour screen continues. Four matched u5000 evaluations are
running locally; the completed foundation control is 0/64 success and 48.9%
excavated, with zero integrity failures. Other results are pending; comparison
is scheduled after all four finish. See the [afternoon evidence report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_AFTERNOON.md).


### September 10, 15:31 CEST: completed u5000 comparison

Job 4634548 remains healthy at about six hours: updates F0=8171, F2=8901,
T0=8041, T2=8891; no execution/cuDNN/nonfinite errors. All four u5000 fixed
results are complete. Foundation control/2x: 0/64 successes each and 48.91% versus
15.49% excavation. Trench control/2x: 3/224 versus 0/224 and 46.71% versus 0.178%
excavation. Trench 2x digs no fresh soil on 220/224 maps. There are no common
successes within either pair; reduced travel cannot be promoted as efficiency.
All 1344 evaluated episodes have zero integrity failures/unavailable counters.
Independent review checked matching identities, reset receipts, hashes and
settings; all result counts reproduce. The newer sampled online u7000-8000
window still shows large 2x progress suppression. These are early one-seed
cost-treatment results, not convergence or an environment-fix ablation.
Continue the authorized 24-hour screen; the next planned evaluation is u10000.
See the [completed comparison report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_1530.md).

At 15:41 CEST, a reviewed local helper started waiting for all four u10000
checkpoints and u10001 receipts (managed session 88723, PID 3052299). It has a
three-hour readiness deadline, checks the actual Adam count and local GPU
availability, then runs the unchanged matched evaluations and summary inline.
No u10000 result is available yet; no Slurm job or training setting changed.


### September 11, 00:30 CEST: component arms submitted; evaluation recovery

Original CSCS 4634548 remains RUNNING (last live check 00:18), about 14h46m into
its 24-hour allocation. Recorded updates are F0=20201, F2=22291, T0=19911, T2=22501.
The sampled online u18000-19000 window gives exact success 4.84%/0% foundations
and 25.69%/0% trenches (control/combined 2x). Excavation is 86.66%/5.48% and
70.36%/0.135%. No runtime failure or physical/per-step integrity violation was
found; one small informational accumulated reward-drift count is documented.
These online results are separate from the fixed held-out evaluation.

The old automatic u10000 evaluation had stopped before downloading checkpoints
on SSH exit 255. All four checkpoints were retrieved and passed finite checks
and actual Adam 640000 today; recovery evaluation began 00:08:59 locally.
Foundation results are complete: control 0/64 exact, 66.64% dug; combined 2x 0/64,
14.76% dug. Both integrity checks pass. The trench evaluations are still running.

Lorenzo authorized four additional overnight experiments. Submitted CSCS 4642631
at 00:18:13 for one additional 24-hour four-GPU node (at most 96 GPU-hours), with
foundation/trench lateral-only (0.5,0,0) and relocation-only (0,0.01,0.04), same
scratch seed 20260909, data, model, PPO and repaired environment. Original jobs
are unchanged. New immutable source: Terra ba9cc214 and baselines 275571b;
changes are launcher/verifier wiring and docs only. The 38 existing launch tests,
CLI comparison, shellcheck, known finite checkpoint verifier and independent
review passed. Each new arm still requires its actual per-GPU finite-u2 gate.

New job state at 00:18:41 was PENDING(Resources), with an estimated 16:35 start
that may change. The SSH certificate expired 00:19:13 after submission. Two
read-only shorter-segment queries failed authentication; no walltime change or
extra submission occurred. Root requested renewal if Lorenzo was still awake.
Submitted/running jobs do not require continued SSH access.

At 00:29:41 the reviewed serial evaluation driver started in local tmux
terra-components-eval-20260911 (worker 3821255), waiting for current u10000 before
original u15000/u25000 and new component u5000/u10000. Retrieval requires renewed
authentication; each helper has a 12-hour bounded readiness/download window.
No new generalist, recipe promotion, or saturation claim is made.
See the [current status](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260911_0025.md)
and [component plan](../../../../.artifacts/terra_excavation_cost_components_cscs_20260911/PLAN.md).


### September 11, 09:03 CEST: completed u10000 comparison; live status blocked

The four original u10000 evaluations completed at 00:46:14. Foundation
control/2x have 0/64 exact each and 66.64%/14.76% excavation. Trench control
reaches 14/224 exact and 52.05% excavation; trench 2x has 0/224 and no fresh
digging on any of the 224 cases. Both pairs still have no common successes, so
no comparative efficiency result is available. All 1344 episodes pass integrity
checks. Independent review confirmed hashes, update/Adam 640000, fixed settings,
per-map/reset identities and the result counts.

Against u5000, control foundation excavation improves 48.91%→66.64%; trench
success 3→14/224 comprises one retained, thirteen new and two lost successes.
All 14 current trench successes are straight-trench conditions; segmented,
network and T-junction conditions remain unsolved. This is an early checkpoint
trend, not the end-of-allocation or convergence result.

SSH still fails authentication at 09:03. The last confirmed scheduler states
remain original 4634548 RUNNING and additional 4642631 PENDING(Resources) at
00:18:41; do not report these as current. New-arm startup and later checkpoints
are unverified. The persistent local evaluation driver is alive, waiting at
original u15000 retrieval within a 12-hour deadline until about 12:46:56. No
u15000/u25000 or component u5000/u10000 results exist yet. Lorenzo was asked to
renew authentication. No jobs, training settings or runtime source changed.
See the [morning report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260911_MORNING.md).


### September 11, 09:33 CEST: historical foundation comparison

Recomputed all eight archived CSCS control/2x foundation panels at u7000,
u10000, u15000 and u25000. At 10,000 additional easy-bank updates, old u15000
solves 63/64 and 64/64 versus current scratch u10000 at 0/64 each. Evaluation
maps, resets and settings match. Initialization and environment source differ:
the old policies inherited 327.68M generalist transitions, for 491.52M total
versus current 163.84M. The actual original u1000 predecessor confirms scratch
initialization and the unchanged four-device pretraining batch shape. Older
scratch experiments do not provide a matched easy-bank learning-speed control.

Two bounded local frozen-policy replays under current Terra ba9cc214 completed.
Compared with their old local replays, control changes 63→62/64 exact and
99.67%→99.56% excavation; 2x changes 64→51/64 and 100%→95.27%. All 128 new
episodes pass integrity checks and retain the same checkpoint/reset identities.
The thirteen lost 2x successes are a material behavior regression under fixed
dynamics, distinct from the current scratch 2x policy's early work suppression.
No claim about the environment's causal effect on scratch learning is made.

The waiting local automatic evaluator was paused for the serial replays and
restarted at 09:32:28 in tmux terra-components-eval-20260911. It is alive and
waiting for original u15000 retrieval; CSCS SSH still fails authentication.
The previous retrieval log is preserved and the helper has a fresh bounded
12-hour readiness window. No Slurm jobs or training settings changed, and no
live scheduler update is available. See the [historical report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/historical_foundation_comparison_20260911/REPORT.md).

### September 11, 10:51 CEST: foundation movement bug confirmed and corrected

The frozen 2x u15000 checkpoint now solves **63/64** easy-foundation validation
maps with corrected movement, versus 51/64 under ba9cc214 and 64/64 under
original fa8d5d13. Twelve lost completions recover and no new ones are lost.
Mean excavation improves 95.274% -> 99.938%; no-effect actions 76.000 -> 4.016
and steps 134.625 -> 60.875. The same checkpoint, all map/reset identities,
reward/treatment receipt, greedy policy, seed 20260907, 450 horizon and 32-row
forward chunks match. All 64 episodes pass existing integrity checks. Slot 38
remains unsolved at 96.0317% dug.

Seven first divergences came from rounded intermediate chassis centers
stepping sideways into cells outside the actual straight sweep. Six others
first diverged at intended global soil/rotation protections; five of those
also recover after movement repair. The correction uses whole straight swept
paths plus valid endpoints and retains strict soil-free chassis rules.
It is committed locally as Terra 7fb30402 in the isolated
`terra_straight_sweep_20260911/terra` worktree. Eighteen focused CPU tests,
384 independent old-endpoint comparisons and thirteen exact GPU transitions
pass. The explicit thirteen-state probes preserve chassis clearance and mass;
ordinary full-episode integrity counters do not independently measure chassis
soil occupancy. Independent review confirms the fix and final paired counts.

No policy was retrained and no Slurm job, cost, observation or PPO setting was
changed. The original CSCS cohort and its automatic evaluator retain ba9cc214.
The evaluator is alive after its 10:25:31 resumption and still awaits original
u15000 retrieval; there is no refreshed scheduler result after the recorded
authentication failure. Use the correction for subsequent submissions and
re-evaluate scratch checkpoints before attributing old failures to reward
weights. See the [diagnosis and complete evidence](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/foundation_regression_diagnosis_20260911/REPORT.md).

### September 11, 11:48 CEST: corrected scratch comparison submitted

CSCS authentication was refreshed and verified as lterenzi. Original job
4634548 reached its 24-hour limit with numbered checkpoints F0=u33000,
F2=u36500, T0=u32500 and T2=u37000. All four are preserved on CSCS and copied
locally under the restart artifacts. Pending component job 4642631 had never
started; it was cancelled before submitting replacements. The old local
evaluation driver was stopped because it waited on that cancelled cohort.

Both replacement jobs use clean published Terra main **46738cde** (movement
fix 7fb30402) and terra-baselines main **2fb7863**, staged once under
`/capstor/scratch/cscs/lterenzi/terra-training/snapshots/excavation-straight-scratch-20260911`.

| Job | Suite | Independent policies | State at submission |
| --- | --- | --- | --- |
| 4645193 | paired | foundation/trench × control/combined 2x | PENDING(Priority) |
| 4645194 | components | foundation/trench × lateral-only/relocation-only | PENDING(Priority) |

Each requests one four-GH200 node for 24 hours, four processes with one GPU
each: at most 192 GPU-hours total including startup. Campaign directories are
`runs/excavation-straight-paired-20260911` and
`runs/excavation-straight-components-20260911` beneath the same CSCS root.
There is no Euler duplicate or full-bank generalist in this comparison.

Every policy starts at update zero with seed 20260909, fresh parameters,
Adam and exploration. Costs are control=(0,0,0), lateral=(0.5,0,0),
relocation=(0,0.01,0.04), combined=(0.5,0.01,0.04). PPO remains one device ×
512 environments × 32 steps, two epochs and 32 minibatches. Data, executable
fresh-dig observations and the corrected environment are shared within each
task family. Only each arm's own newly trained u2 checkpoint is continued.

Local foundation and trench smokes each completed two fresh updates with
finite model/optimizer/loss, Adam 64→128, 32,768 transitions and zero recorded
transition-integrity errors. CUDA convolution-backward passed. Independent
review resolved all eight actual launch mappings and both phases (16 argument
sets), checked fresh initialization and reviewed explicit EDF suite selection.
Shell syntax, shellcheck and both Slurm dry submissions passed. Local cuDNN
autotuning emitted algorithm-comparison warnings, including one already seen
in the preceding foundation smoke; both finite-update checks completed.

No CSCS allocation or learning result is established by this submission.
Each allocation must pass four-GPU convolution/NCCL, one-GH200 binding per arm,
and all four fresh u2 checks before production. Numbered checkpoints remain
every 500 updates and the absolute target remains 500,000; walltime bounds
this segment. W&B is offline per job/arm, with no new online history yet.

Evaluate equal updates u5000/u10000 before ranking behavior costs. Foundation
uses its 64 validation episodes (seed 20260907, forward chunk 32); trench uses
the original 608 development reset cohort (seed 20260724, chunk 120), reporting
224 trench rows separately. Both use greedy decoding and a 450-step horizon.
Completion and excavation come first; compare productive setups, workspace
area, retained-pose travel and adjacency on common successes. Keep the old
ba9cc214 training cohort separate. One seed and separate nodes limit variance
and factorial-interaction claims. The [restart plan and artifacts](../../../../.artifacts/terra_movement_restart_cscs_20260911/PLAN.md)
contain source revisions, smoke checks, launch review and submission records.

At 11:52 CEST, both jobs remain pending; Slurm estimates September 12 at 01:10
CEST on nid005875/nid005885. These are scheduler estimates, not allocations.
Local tmux `terra-straight-eval-20260911` runs the corrected-cohort retrieval
and evaluation driver with a 72-hour deadline. It retrieves u5000/u10000 only
after the following rollout receipt exists, saves the per-arm smoke results,
and evaluates serially when the local GPU is free. Scheduler snapshots and
driver output live in the restart artifact directory. This replaces the old
driver waiting on the cancelled component job; no new metrics exist yet.

### September 12: all eight u5000/u10000 comparisons completed

The bounded local evaluator finished successfully at 01:51:17 CEST. A prior
summary error (`KeyError: 'geometry'`) was caused by optional metadata missing
from trench rows. Required identities remain strict; optional geometry now
requires matching presence and values. Saved component rollouts were reused
and validated, then paired u5000 and both u10000 suites completed. All four
milestone/suite directories contain their summaries and completion markers.
The driver exited after its planned work; it is no longer a live monitor.

| Costs | Foundation exact at u10000 | Foundation dug | Trench exact at u10000 | Trench dug |
| --- | ---: | ---: | ---: | ---: |
| control | 1/64 | 71.03% | 20/224 | 47.26% |
| lateral only | 1/64 | 55.00% | 2/224 | 43.36% |
| travel/turn only | 0/64 | 12.47% | 0/224 | 0% |
| combined 2x | 0/64 | 8.28% | 0/224 | 0% |

Control foundation excavation improves 45.09%→71.03% from u5000→u10000;
trench completions improve 2→20/224. Lateral-only improves 48.49%→55.00%
foundation excavation and 0→2/224 trench completions. Both travel-cost
policies make zero fresh excavation in all 224 trench cases at both milestones.
No foundation successes are common between control and lateral-only; their
trench overlap is only one success. There is no supported comparative
efficiency benefit or policy promotion.

All 5,376 saved full-panel episode rows pass the recorded integrity and reset
checks. All checkpoint checks retain finite model/Adam/loss validation, exact
updates, actual Adam counts and hashes. Within-family comparisons verify the
same full reset identities and matching treatment/reward contracts after
removing run name and the three cost fields. One seed and separate nodes limit
variance and factorial-interaction claims. These results concern early scratch
learning, not the later trained policies or convergence.

At the September 12 morning check, SSH failed with `Permission denied
(publickey)`; certificate validity ended at 11:32:17 CEST. Last live manual
scheduler verification was 00:02:55, both RUNNING; successful later checkpoint
retrieval does not establish present scheduler state. The 24-hour allocations
were scheduled to end around 16:07 CEST today. Authentication renewal is needed
for a fresh status and later checkpoints. No jobs, costs, PPO settings or
allocation limits changed in this check.

The next-recipe hypothesis is to reduce or delay travel/turn costs until the
policy learns reliable completion. It has not been tested, and no new run is
submitted. Finish the existing bounded screen and examine later held-out
checkpoints before a convergence or saturation conclusion. See the
[complete comparison and raw evidence](../../../../.artifacts/terra_movement_restart_cscs_20260911/REPORT_20260912.md).

### September 12: delayed behavior-cost recipe approved and prepared

Continue the corrected zero-cost foundation and trench controls, preserving
their native model/Adam/update clocks, Terra 46738cde environment, existing banks,
observations and PPO configuration. No fresh initialization or generalist is
planned. Do not renew the cost arms that suppress early digging.

Each family must independently reach at least 90% exact completion on two
successive fixed-panel evaluations at least 2,500 updates apart: 58/64
foundations or 202/224 trenches. Keep greedy decoding, 450 steps and verified
resets; evaluate all 608 trench-panel rows. The saved u10000 controls have
1/64 and 20/224 successes, so both remain at zero added costs. Those native
parents have finite parameters/optimizer state and actual Adam count 640,000.

After eligibility, freeze the accepted zero-cost reference and retain a
zero-cost sibling. Add 25%, then 50%, then 100% of the previous combined 2x
costs, holding each stage for 5,000 additional updates. Evaluate at +2,500 and
+5,000; both must stay above 90% completion and within three percentage points
of the original reference before another increase. Preserve completion before
ranking work-pose efficiency and adjacency. The
[stage launcher](../scripts/foundation_reward_sweep/README.md) defaults to an
evidence check and prints the native training command; `--execute` belongs
inside a checked GPU allocation. Copy `penalty_stage.json` with each parent
checkpoint and provide it for later increases.

CSCS SSH still failed after certificate expiry at the 12:46 CEST check, so jobs
4645193/4645194 have no refreshed live state. Latest checkpoint retrieval,
evaluation, the next allocation's runtime check and a two-update native
continuation smoke remain pending. No new training, submission or cancellation
has occurred for this recipe. The implementation passed 72 focused CPU tests
and 22 subtests, shellcheck, syntax checks and independent review. Both real
u10000 native parents and their existing bank identities pass CPU inspection;
wrong update/Adam metadata, partial resets and nonfinite model data are rejected.
Both real completion gates stay closed. See
`.artifacts/terra_delayed_penalties_20260912/` at the workspace root for evidence.

## 2026-09-12 smooth behavior-cost ramp and GPU-layout validation

The approved recipe now ramps linearly for 2,500 updates and holds for 2,500
before the next offline promotion decision. Two canonical full-start greedy
evaluations, sustained 90% exact completion, and the frozen original-reference
loss guard remain required. Failed stages are rejected; no automatic rollback
or automatic promotion is implemented. The wrapper supports 1/2/4 GPUs while
retaining 512 global environments, 16,384 transitions and 64 Adam steps per
update. Device-local advantage normalization changes PPO numerics on a GPU
layout change; qualify both zero-cost reports on the new layout before a cost
fork, and keep that layout across subsequent stages.

Local CUDA convolution preflight and a 32-environment functional smoke passed.
The real zero-cost foundation u10000 checkpoint continued to u10002, saved
halfway through a four-update test ramp, and a new process resumed through
u10005. All seven periodic/FINAL artifacts have finite model/Adam/loss data,
zero transition-integrity counters, and Adam=64*next_update. Effective costs
reached the exact target at u10004 and remained fixed at u10005. There was one
PPO update signature per process; the resumed process hit the persistent
`pmap__update_step` cache. This is a small runtime check, not a throughput or
behavior comparison. Focused CPU validation passed 122 tests and 22 subtests;
independent source review has no remaining findings.

Euler diagnostic **13935300**, account **lterenzi**, requests four RTX4090s,
12 CPUs, 6 GB/CPU and at most 45 minutes. It measures 1-vs-4 GPU throughput
from the same u10000 parent/global batch, then verifies a native mid-ramp
resume at the full global batch. At 15:52 CEST it was PENDING/Priority, with
no runtime acceptance yet. Scheduler remapped the short request to gpuhe.4h.
The immutable snapshot is
`/cluster/scratch/lterenzi/codex_terra_edge_validation/terra_smooth_ramp_20260912_1345`;
run root is
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_smooth_ramp_20260912`.
The snapshot contains baselines 555ed53 plus the recorded ramp patch and
Terra 46738cde; archive SHA-256 is
`87c89c8eaf6800dc30ed0476503d4aaa20ca330b299483a2fda66515174a645f`.

At 15:53 CEST, corrected CSCS control checkpoints were foundation u32500
(SHA 04dcca21cdf7fac2e2053d46099709b75bb36dc358c7d3d8ae4160eb06463f43)
and trench u32000
(SHA e3cf178099ea09451527cd5df010405139a8de00712f4ef2fde71fe868d8a5b4).
Both were downloaded and verified finite with exact Adam clocks and zero
added costs. Final parent selection and fixed-panel evaluation follow the old
allocation's end; do not use the old local u10000 diagnostic parent for a
production restart. A CSCS diagnostic is being prepared while Euler queues.

The closed Weber account was removed from active SSH configuration and
installed workflow defaults by an independent agent. Commit c2df04a updates
28 Terra routing/helper/launcher files to lterenzi and rejects retired output
roots. Historical records and shared read-only input identities remain.

Artifacts: `/home/lorenzo/moleworks/.artifacts/terra_delayed_penalties_20260912/smooth_ramp/`
(`focused_tests.log`, `local_verification.json`, local GPU logs, launch files,
source archive/diff, and `latest_controls/native_validation.json`). Account
cleanup evidence is in `.artifacts/retire_euler_weber_20260912/STATUS.md`.

## 2026-09-12 latest control evaluations and conditional continuations

Both corrected scratch allocations, 4645193 and 4645194, ended after 24 hours
with `TIMEOUT`, exit 0:0. CSCS access is restored under lterenzi. Selected
zero-cost parents are foundation u33000 and trench u32000; native CPU checks
verify finite model/Adam/loss, exact optimizer clocks and zero costs. Their
hashes and paths are in
[parent_selection_local.json](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/parent_selection_local.json).

Local greedy evaluation uses the unchanged 450-step horizon, baselines 866e8e
and Terra 46738cde. Foundation has 64 validation episodes; the trench evaluation
retains all 608 development episodes and reports its 224 trench episodes.

| Control checkpoint | Exact completion | Excavated | Disposed |
| --- | ---: | ---: | ---: |
| Foundation u33000 | 8/64 | 93.2860% | 90.9650% |
| Trench u32000 | 36/224 | 65.9106% | 62.8468% |

The earlier matched u10000 controls achieved 1/64 and 71.033% excavation for
foundations, and 20/224 and 47.257% for trenches. All 672 new full-panel rows
have zero recorded integrity failures, nonfinite states, target/obstacle
mutations, termination disagreements, unavailable integrity and mass residual.
Progress improved, but neither family reaches 58/64 or 202/224. Keep added
costs at zero. See the [evaluation artifacts](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/evaluation/).

CSCS diagnostic **4652857** started at 16:35:10 CEST on nid005780. The 16:37
check confirms four GH200s, actual parent/bank checks, cuDNN backward and NCCL
preflight. By 16:44 its foundation four-GPU test advanced u33000 to u33016;
all 17 periodic/FINAL checkpoints pass finite model/Adam/loss and integrity
checks. Median throughput after the first two updates is 15,935.585 global
transitions/s; the process took 455 seconds including startup and compilation.
The matched one-GPU test is running, so scaling and trench native-resume
acceptance remain pending. This one-hour diagnostic compares 16 foundation updates on four versus one GPU
from the same parent, then checks
four-GPU trench continuation across a 2+14-update resume.

Its `--continue-trench` hook submits one 24-hour four-GPU trench continuation after all
diagnostic checks pass and foundation speedup reaches at least 1.5x. That is an
allocation guard, not evidence of better trench learning. A failed speed guard
records a reason and submits no job. The accepted parent would be the verified
diagnostic trench u32016 FINAL. Production repeats runtime checks and two
finite native updates, then continues at zero costs to absolute target 500000
with checkpoints every 500 updates, bounded by one 24-hour allocation. There
is no further allocation chain or automatic penalty promotion. No production
child is submitted at this check.

Euler replacement **13939497** is submitted under lterenzi for four RTX4090s
and at most 45 minutes, with `AUTO_CONTINUE_FOUNDATION=1`. At 16:40 it is
PENDING because nodes are down, drained or reserved, with no allocated GPU or
reliable start estimate. Old 13935300 was cancelled only while pending under
that user. After runtime acceptance, the
hook selects four GPUs at speedup >=1.5x, otherwise one, and submits one
24-hour zero-cost foundation continuation from the actual u33000 parent. It
checks two finite native updates before production; the diagnostic u10000
parent is used only for its runtime comparison. No replacement runtime result
is available yet. Both recipes retain 512 global environments and 64 Adam
steps per update. GPU-local advantage normalization changes with layout, so
collect both qualifying zero-cost reports and the reference on the selected
layout before any cost fork.

Training source published to baselines main is
b6d1597e96e63aeaef40373d0909e31a1d28ed0d, including
smooth ramp b6754540, lterenzi routing c2df04a and the portable hash fix; Terra
remains 46738cde. Validation includes 122 CPU tests plus 22 subtests, 65 tests
after the Python 3.10 compatibility fix, shell checks, independent review and
bounded submission-stub checks. The real local RTX4090/32-env ramp/resume test
produced seven finite checkpoints, retained Adam clocks and crossed the ramp
endpoint with a persistent executable cache hit. CSCS four-GPU foundation
native updates now pass; the remaining runtime and scaling checks are pending.

Euler source:
`/cluster/scratch/lterenzi/codex_terra_edge_validation/terra_smooth_ramp_20260912_b6d1597`.
The `source_b6d1597.tar.gz` SHA-256 is
`6d304c995a04ef002ff96ba850845baa567ab83016a76d65f54c613122b629b6`.
CSCS source:
`/capstor/scratch/cscs/lterenzi/terra-training/snapshots/terra-smooth-ramp-20260912`,
with clean b6754540 plus the portable hash patch matching b6d1597. Current
routing uses lterenzi; historical dataset and W&B names remain. The
[current artifact status](../../../../.artifacts/terra_delayed_penalties_20260912/STATUS.md)
tracks diagnostic results and conditional submission receipts.

## September 12, 23:03 CEST: trench continuation running; foundation queued

CSCS diagnostic **4652857** completed with exit 0:0 from 16:35:10 to 17:07:07
CEST (31m57s). All four diagnostic phases passed, including the trench native
resume. Median foundation throughput after the first two updates was
15,935.585 transitions/s on four GPUs versus 6,961.24 on one GPU: **2.2892x**.
The four-GPU trench resume reached 15,778.96 transitions/s. These short samples
establish runtime acceptance and throughput, not faster policy learning.
Its resumed PPO executable hit the persistent cache: the XLA compilation
stage took 4.153 seconds versus 65.402 seconds initially. Tracing/lowering
still occur when a training process restarts.

The accepted scaling result triggered the already-authorized zero-cost trench
continuation **4652918**. It started at 17:10:25 on **nid005935**, using four
GH200s. CPU parent checks, cuDNN/NCCL preflight and two finite native updates
passed from u32016 to u32018. At 23:03 the production run is RUNNING after
about 5h52m, near u48919: 16,901 updates beyond the production start. Recent
training logs report roughly 13.5–13.9k global transitions/s. The one 24-hour
allocation is scheduled to end September 13 at 17:10 CEST; no further
allocation is automatically submitted.

Training retains zero lateral/travel/turn costs, 512 global environments
(4 × 128), 32-step rollouts, 64 Adam steps per update and checkpoints every
500 updates. W&B is offline under
`terra-trench-control-4gpu-after-4652857-4652918`. The native run directory is
`/capstor/scratch/cscs/lterenzi/terra-training/runs/terra-smooth-ramp-20260912/continuations/4652918`.

Retained **u48500** was copied locally and checked: Adam **3,104,000**, finite
model/optimizer/loss, zero effective behavior costs and zero recorded
transition-integrity counters. Its SHA-256 is
`a0cba3c9fd4889a84de3d699e60880d5887899df64412e1a3b7d0eb0c73ef82f`.
The full 608-row greedy 450-step development evaluation completed at 23:40;
the 224 trench rows improve from **36/224 to 147/224 exact** (16.1% to 65.6%),
**65.9106% to 91.3255% dug**, and **62.8468% to 87.0979% disposed**. All 608
rows have zero recorded integrity failures, nonfinite states, map mutations,
termination disagreements and mass residual. Reset, bank, R2 and normalized
treatment checks match; only the intended training GPU layout and run name
differ. There are 115 newly solved episodes and four formerly solved episodes
that now fail. Foundation u33000 remains 8/64 exact.

| Trench layout | u32000 exact | u48500 exact |
| --- | ---: | ---: |
| Straight | 35/64 | 48/64 |
| T-junction | 1/32 | 25/32 |
| Multiple segments | 0/32 | 23/32 |
| Two-sided network | 0/64 | 51/64 |
| Road-constrained network | 0/32 | 0/32 |

On the 32 common successes, efficiency is almost unchanged: productive base
poses 7.406 to 7.375, unique area per productive setup 2.948 to 2.951 m2,
retained-work straight-line travel 28.578 to 28.398 m, and edge-adjacent
workspace transfers 100% in both. This supports continued completion learning;
it does not establish a workspace-efficiency gain. Keep costs at zero because
147/224 remains below the 202/224 threshold and a second qualifying checkpoint
would still be required.

At 23:41, CSCS4652918 is still running around u50831, with u50500 saved and
recent median throughput 13,681.645 transitions/s. Euler **13939497** remains
PENDING/Priority with no reliable start estimate; the earlier September 13
20:25 estimate is no longer available. No
foundation production continuation has started or been submitted by its hook.
Changing GPU count preserves global batch size but changes per-device
advantage normalization; qualifying evaluations and the accepted zero-cost
reference must use the selected training layout before a later penalty fork.
No jobs or training settings were changed during this status check. Local
checkpoint/evaluation evidence is in
[status_2302](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_2302/).

## September 13, 08:43 CEST: trench u77000; foundation continuation recovery

CSCS **4652918** is RUNNING after about 15h33m near u77361, with u77000 saved
and recent throughput 13,618.59 global transitions/s. Its existing allocation
ends today at 17:10 CEST. Downloaded u77000 passes native finite model/Adam/loss
checks at Adam **4,928,000**, with all added costs and integrity counters zero.
Its SHA-256 is
`b19836680aebbae7924c9efe96d020fcf0c2e1059d13b9f45f72b74e1b16d110`.
The same greedy 450-step, full 608-episode development evaluation completed
at 08:59 CEST with EVAL_DONE and its complete JSON report. The 224 trench rows
improve **147/224 → 161/224 exact (65.6% → 71.9%)**, **91.3255% → 94.4698%
dug**, and **87.0979% → 89.8239% disposed**. There are 24 gained and 10 lost
successes. All 608 rows have zero recorded integrity failures, nonfinite states,
map mutations, termination disagreements and mass residual. Reset, bank,
source, treatment and four-GPU layout identities match.

| Trench layout | u48500 exact | u77000 exact |
| --- | ---: | ---: |
| Straight | 48/64 | 52/64 |
| T-junction | 25/32 | 27/32 |
| Multiple segments | 23/32 | 22/32 |
| Two-sided network | 51/64 | 60/64 |
| Road-constrained network | 0/32 | 0/32 |

Among 137 common successes, productive poses change 9.358 → 9.314, unique
area per setup 2.712 → 2.724 m2, retained-work travel 40.754 → 40.705 m and
edge adjacency 89.58% → 89.45%. These are essentially unchanged. Raw Terra
travel falls 58.196 → 53.625 m and steps 67.52 → 63.88; these are secondary
to deployed workspace transfers. Completion is improving unevenly, without
an established workspace-efficiency gain. No penalty stage is active.

Euler diagnostic **13939497** ran 00:38:02–01:13:23 CEST (35m21s) and ended
`FAILED`, exit 124:0. Both zero-cost one/four-GPU controls completed all 16
updates and 17 checkpoints each. Recovered verification confirms all 34 are
finite, preserve native Adam clocks, retain 512 global environments and have
zero added costs/integrity counters. Matched median throughput was 13,503.175
versus 4,585.225 transitions/s: **2.9449x** four-GPU speedup. The separate ramp
phase hit its 650-second startup timeout before writing any checkpoint; this
is not evidence of numerical failure, and it does not qualify the full ramp.
The recovered receipt explicitly accepts only zero-cost continuation.

Requiring the unused ramp test before zero-cost production prevented the
foundation hook from submitting. To recover the already-authorized one-day
continuation, it is being moved to CSCS. At this check, resource estimates are
September 13 at 12:13 CEST on CSCS versus September 14 at 11:45 on Euler;
these are volatile estimates. No Euler production job was submitted, so this
move creates no duplicate or extra trial. Actual replacement **4655350** was
submitted at 09:00:10 and started at 09:00:11 CEST on nid005799, ahead of the
test-only estimate. Slurm confirms four GPUs, account lterenzi/d130, no
dependency, no requeue and a 24-hour limit ending September 14 at 09:00.
Parent/bank and cuDNN-backward/NCCL runtime checks passed. Native startup
updates are in progress; production is not yet verified.

The submitted four-GPU foundation job uses diagnostic 4652857's accepted
u33016 FINAL (Adam 2,113,024), SHA-256
`51bb2905a689e8202ba78ef1e7eb3e15fd1a1dc8849b6843b4f880aa2f930e24`.
It keeps the existing 256-map foundation bank, training source b6d1597 and
Terra 46738cde, 512 global environments (4 × 128) and 64 Adam steps per update.
The new allocation must repeat CPU parent/bank checks, CUDA/NCCL preflight and
two finite native updates u33016→u33018 before zero-cost production. It then
targets absolute u500000 with checkpoints every 500 updates, bounded by one
24-hour allocation, with a unique offline W&B history and no automatic
follow-on allocation. It does not depend on the failed Euler job. GPU-local
advantage normalization changes with layout, so later penalty qualification
still needs matching zero-cost reports on the selected layout. Evidence is in
[status_20260913_0843](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_0843/).

At 09:04 CEST, the trench worker has reached approximately u78310. An
independent review reproduced all u77000 evaluation counts, grouped results,
matched identities and common-success metrics from the raw reports, with no
remaining findings. Foundation runtime checks have passed; its first native
training updates are still compiling at this observation.

## September 13, 11:44 CEST: both continuations training; native checkpoints pass

CSCS foundation **4655350** is RUNNING on nid005799 after 2h44m, around
u41113 (8095 production updates beyond u33018). Its startup smoke passed
all three finite native checkpoints through u33018 in 425 seconds, and
production is now sustained at roughly 14.7–15.3k global transitions/s.
The saved u41000 checkpoint passes finite model, optimizer and loss checks
at Adam 2,624,000. Allocation ends September 14 at 09:00 CEST.

CSCS trench **4652918** is RUNNING on nid005935 after 18h33m, near u85992.
Recent log samples report 13.1–13.3k transitions/s. Saved u85500 passes the
same native checks at Adam 5,472,000. Allocation ends today at 17:10 CEST.
Both checkpoints preserve four devices × 128 environments, training-bank
identity, native parent/clock, zero added behavior costs and zero recorded
transition-integrity counters. No jobs or training settings changed.

No new held-out evaluation was run during this status check. The latest
complete results remain trench u77000 at 161/224 exact (71.9%), 94.4698% dug,
and foundation u33000 at 8/64. These are not evaluations of the newer
checkpoints. Workspace efficiency and road-network completion have no new
measurement; the completion gate remains unmet and penalties stay off.
Evidence: [status_20260913_1144](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_1144/native_validation.json).

## September 13 evening: CSCS auth expired; retained foundation u41000 regresses

At 22:22 CEST, live CSCS status could not be refreshed: lterenzi's SSH
certificate expired at 15:52:28 and retry returned `Permission denied
(publickey)`. Last verified scheduler/runtime observations remain those at
11:44. Trench 4652918 was scheduled to end September 13 17:10; foundation 4655350
was scheduled to end September 14 09:00. These scheduled times do not establish
current job state. No further jobs or reward changes were made.

The already-downloaded, native-validated foundation u41000 and trench u85500
checkpoints were evaluated locally in persistent tmux using the frozen source
and unchanged respective panels. These checkpoints predate training later in
the day. Foundation's complete 64-row greedy 450 evaluation gives **3/64 exact**
versus **8/64 at u33000**, **79.4091% versus 93.2860% dug**, and **77.1420%
versus 90.9650% disposed**. All 64 reset identities and integrity checks pass;
checkpoint hashes and effective zero-cost treatment match. Independent review
reproduced the counts from raw rows.

All 8 earlier successes are lost and 3 new successes appear. Square drops 8/21
to 1/21, rectangle rises 0/22 to 2/22, L remains 0/21. Average excavation decreases
in all three shape groups. On each checkpoint's failing rows, longest material
stall rises 387.18 to 408.87 steps; failure sets differ. No common-success
intersection exists, so no success-conditioned workspace-efficiency comparison
is available. Raw travel across all 64 rows rises 54.54 to 104.74 m while average
excavation falls. Fewer productive poses here cannot establish efficiency.

This establishes regression at the retained u41000 checkpoint, not a cause
or the latest remote model's behavior. Continued learning and layout migration
from 1x512 to 4x128 are confounded; per-device advantage normalization changes
while the global batch is preserved. The added penalties remain zero. Retrieve
and evaluate the latest native checkpoint after authentication renewal before
deciding how to continue or change the foundation recipe.


## September 13, late evening: historical trench replay and manual regression audit

Fresh matched trench u85500 evaluation gives **162/224 exact** versus 161/224
at u77000, with mean excavation 93.53% versus 94.47%. The 151 common successes
show essentially unchanged productive-pose, area-per-setup, retained-work travel
and workspace-adjacency metrics. Road-constrained completion remains 0/32.

Historical fixed reports confirm 196/224 at u51500 and 188/224 at u83500.
The latter completed 25/32 road maps; non-road completion is almost unchanged
at 163/192 historically versus 162/192 now. A fresh replay of old u83500 weights
under the current repaired environment completes **179/224**, including
**20/32 roads**, with matched resets and zero integrity failures. Thus the
current environment still permits successful road behavior. This comparison
retains each policy's trained observation interface and different training
histories; it does not isolate the cause of the new policy's deficit.

Manually inspected trajectories show the current policy repeating blocked
base movements, ineffective cabin/dig cycles and WAIT despite legal productive
alternatives. The old policy completes the two selected road cases in 95 and
231 actions under current physics. For foundations, old u33000 completes a
selected square in 63 actions while u41000 waits 448 actions after two setup
actions. Same-input inference confirms a changed learned action preference.
Sampled u41000 diagnostic completes 6/64 versus greedy 3/64; this is not a
qualified deployment improvement.

Native clocks, optimizer state and effective environment settings pass audit;
63 focused CPU environment tests pass. No hidden added penalties or positive
WAIT reward were found. Relevant differences still requiring controlled tests
are per-device PPO advantage normalization (512 to 128 local samples) and the
historical trench run's four-times-larger transition budget per update and
entropy-decay experience budget. Current loose-soil observations remain present.

Foundation greedy replay endpoints match all 64 rows. The current trench trace
replay differs on eight unselected episodes; all five selected clips match
checked endpoints. The full-panel parity failure is retained and not presented
as a pass. No production source, reward or job changes were made. CSCS state
remains unverified after certificate expiry; these are morning checkpoints.

See [completion regression audit](research/COMPLETION_REGRESSION_20260913.md)
and the [manual rollout viewer](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/qualitative/index.html).


## September 13, 23:49 CEST: two authorized overnight recovery screens submitted

The user explicitly authorized new overnight training. Two distinct 24-hour
Euler allocations were submitted as lterenzi, each requesting four RTX 4090 GPUs:

| Arm | Job | Native parent | Batch | Treatment | Fixed evaluation |
| --- | ---: | ---: | --- | --- | --- |
| Foundation screen |14055215|u33000|4×128env|Global minibatch advantage normalization|u41000,64-map greedy validation|
| Old trench recovery |14055463|u83500|4×512env|Original local normalization and legacy observation, current physics|u86000,224 trench rows of608-map greedy development|

Foundation was confirmed PENDING (Priority); neither job has passed its remote
startup gate at this submission observation. Trench's sbatch receipt is valid;
queue queries are intermittently timing out. Do not call submitted jobs healthy
training. Existing CSCS state is still unverified after certificate expiry.
No duplicate local-normalization foundation control or CSCS run was submitted.

Both local actual-parent GPU smokes passed two updates and three saved native
checkpoints at 1×32 environments. Model/Adam/loss are finite, actual clocks advance
128 Adam steps, and restored environment/bank/zero-cost contracts pass. This is
functional evidence only; the allocation requires CUDA convolution backward,
NCCL and two validated updates at the exact production shape before proceeding.
Startup failure ends the allocation without silently shrinking the batch.

The normalization change is opt-in, preserves the default arithmetic, records
its mode in checkpoints/W&B/evaluation fingerprints, and rejects native resumes
that would silently drop it. 26 CPU tests pass, including four-device versus
merged-minibatch equivalence through 64 real PPO/Adam steps. Independent review
found no remaining source, native-validator or watcher findings. Training source
is baselines 46adaec50ed4dd225117b2fb8b721ee85649efa7 paired with
Terra 46738cde28e455da7c466fc0a2cb64f677d86401; remote files pass SHA verification.

All added behavior penalties remain zero. Both runs target absolute u500000,
save every 500 updates and are bounded by 24 hours, with no automatic next job.
The old trench arm preserves seed 20260901, 65,536 transitions/update, 64 Adam steps,
entropy floor 0.02 and executable_dig_observation=False. Foundation preserves
seed 20260909, 16,384 transitions/update and 64 Adam steps, opting into global rather
than per-device normalization. Its existing CSCS control differs in hardware,
RNG and restart boundaries; this is a screen, not a causal A/B claim.

A 26-hour local tmux watcher, terra_recovery_overnight_20260913, monitors both
jobs, downloads only the selected milestones, verifies native checkpoint SHA/
mode and serializes fixed greedy evaluation on an idle local GPU. It reuses
completed reports after restart, kills its evaluation process group on timeout,
and neither submits jobs nor enables penalties. The evaluator stays frozen at
baselines 866e8e2/Terra 46738cde; a separate native SHA/mode receipt accounts for
its historical treatment fingerprint, which predates the normalization flag.

[Campaign plan and evidence](../../../../.artifacts/terra_regression_recovery_20260913/PLAN.md) ·
[Submission manifest](../../../../.artifacts/terra_regression_recovery_20260913/manifest.json) ·
[Overnight status](../../../../.artifacts/terra_regression_recovery_20260913/overnight_status.json).


## September 14, 12:54 CEST: recovery evaluations complete; both Euler jobs training

Both Euler jobs are RUNNING and have passed their full production-workload
startup gates: four RTX4090 GPUs, CUDA convolution backward, NCCL, two finite
native updates and checkpoint/environment/bank/integrity validation.

| Run | Approximate live update | Saved checkpoint | Recent global transitions/s | Scheduled end (CEST) |
| --- | ---: | ---: | ---: | --- |
| Foundation global normalization 14055215 |57,686|57,500|12,594|September 15 03:37|
| Old trench recovery 14055463 |87,591|87,500|16,923|September 15 08:05|

Latest saved checkpoints above are located, not evaluated or independently
loaded during this update. The completed milestone evaluations below use
validated earlier checkpoints, fixed greedy 450-step panels and zero added costs.

| Evaluation | Exact completion | Mean dug | Accepted disposal |
| --- | ---: | ---: | ---: |
| Foundation global u41000 |11/64 (17.2%)|91.09%|90.61%|
| Foundation prior local u41000 |3/64 (4.7%)|79.41%|77.14%|
| Foundation common parent u33000 |8/64 (12.5%)|93.29%|90.97%|
| Trench recovery u86000 |192/224 (85.7%)|96.78%|95.97%|
| Old trench parent u83500 under current physics |179/224 (79.9%)|94.71%|93.66%|

Trench road completion rises 20/32→24/32; non-road 159/192→168/192.
There are 22 gained and 9 lost successes. On 170 common successes, productive
poses 9.524→9.553, unique area/setup 2.808→2.803m², retained-work travel 44.03→
44.48m and workspace adjacency 88.97%→88.49% show no efficiency improvement.
The main gain is completing more maps and disposing more spoil correctly.

Foundation completes 8/21 squares,3/22 rectangles and 0/21 L maps. It improves over
the regressed local-normalization run but remains weak and does not exceed the
parent's average excavated fraction. Hardware/RNG/restart differences mean the
normalization comparison is a screen, not proof that normalization alone caused
the improvement.

Both reports finished with their EVAL_DONE markers. All 672 rows have zero
recorded integrity failures; reset identities and actual checkpoint hashes pass.
The bounded overnight watcher completed both planned evaluations and exited
normally at 11:27. It is no longer a live monitor. Training continues within the
existing 24-hour allocations; no new jobs or reward changes were made today.
Penalties remain off: neither 58/64 foundation nor 202/224 trench qualifying
completion threshold has been met. Let the existing allocations continue.

CSCS SSH access was checked again at 12:54 and still returns Permission denied(publickey);
its older jobs' final states and checkpoints remain unverified.

[Live runtime receipt](../../../../.artifacts/terra_regression_recovery_20260913/status_20260914_1253/live_runtime.json) ·
[Foundation evaluation](../../../../.artifacts/terra_regression_recovery_20260913/evaluation/foundation_global/fixed.json) ·
[Trench evaluation](../../../../.artifacts/terra_regression_recovery_20260913/evaluation/trench_recovery/fixed.json) ·
[Watcher completion](../../../../.artifacts/terra_regression_recovery_20260913/watcher_finished.json)


## September 14, 13:49 CEST: old foundation policies still solve 62–63/64 under current rules

Fresh complete frozen-weight replays under Terra 46738cde and evaluator 866e8e2
solve **62/64 for old zero-cost control u15000** and **63/64 for old 2× u15000**,
versus **11/64 for new global-normalization u41000**. The same old weights had
solved 63/64 and 64/64 in their original reports. The current comparison uses
identical 64-map resets, greedy inference, seed 20260907 and horizon 450. Both
replays have EVAL_DONE markers; independent review validates all 448 rows across
seven historical/current reports and the five actual retained checkpoint hashes.

Old control excavates 99.56% and correctly disposes 99.55%; old 2× achieves
99.94% for both. Each completes every map solved by new u41000, plus 51 and 52
additional maps. The old zero-cost control is the primary foundation reference.

The foundation bank has no trench metadata and receives no trench-yaw restriction.
All 70 native environment fields agree except live episode counters, and all
1,280 local training reset arrays match their manifest. Global soil/chassis and
movement repairs do affect foundations; the known intermediate-rounding collision
bug is fixed. Current Terra has the exact corrected 7fb30402 runtime tree.

The old runs adapted a pretrained generalist with native Adam/clock state; new
foundation learning started from random initialization. Old u15000 had 491.52M
total transitions versus 671.744M for new u41000. The new scratch u33000 already
scored 8/64 at the old 1×512 layout, before four-GPU normalization changed.
Neither fewer total samples nor later GPU scaling alone explains the large gap.
Initialization, optimizer state, experience distribution, entropy schedule history
and physical-rule training history remain confounded. Successful old-policy
execution does not prove unchanged difficulty of scratch learning.

Keep soil-free chassis protections and zero added costs. The old control is a
stronger recovery-parent candidate; no training jobs, source code or rewards were
changed in this audit. This entry adds historical comparison, not a new live
scheduler check or an evaluation of checkpoints newer than u41000.

[Foundation regression analysis](research/FOUNDATION_LEARNING_REGRESSION_20260914.md) ·
[Fresh report comparison](../../../../.artifacts/terra_foundation_lineage_audit_20260914/current_environment_comparison.json) ·
[Independent review](../../../../.artifacts/terra_foundation_lineage_audit_20260914/independent_review.md).


## September 14, 14:21 CEST: foundation teacher-KL recovery ready locally; CSCS auth blocks submission

The user requested a foundation-only CSCS run initialized from a strong policy
with KL guidance from a frozen teacher. Prepared one 24-hour, four-GH200 Daint
allocation using lterenzi. No new job has been submitted. CSCS SSH again rejects
the certificate with Permission denied(publickey); Euler jobs are unchanged by
this preparation.

Student and teacher use old zero-cost foundation control u15000, SHA
`d1a6c07d9d8a40b7b7b60bd0b54313aa46a9b50fb09c789ed4ebffddb2b488b3`,
which freshly solves 62/64 under current Terra 46738cde. Keep the same easy bank,
seed20260907, current soil/chassis physics, native Adam and entropy clock. The
4×128 layout keeps global512 environments and64 Adam steps/update; global
advantage moments match merged minibatch mathematics. Teacher KL is1.0 at
u15000, cosine-annealing to0 at u35000; value distillation and LR warmup are0.
All added behavior penalties and their ramp remain disabled. Evaluate u17500,
u20000 and u25000 on the same64-map greedy450-step panel, including workspace
and retained-work travel metrics. This is retention/adaptation, not a causal
KL A/B or proof of full-foundation generalization.

Teacher continuation needed fixes for its absolute coefficient origin and
constant-LR optimizer structure. The actual-parent GPU smoke also caught a
missing executable-dig observation selector in the teacher model stub; that
interface is now preserved and validated. Inference clears the teacher origin
when disabling distillation. Seven focused native-teacher and47 existing
training-utility tests pass; independent source/launch review has no findings.

The corrected local1×32 run completes two finite updates and a further native
resume: u15001/u15002/u15003, Adam960064/960128/960192. Teacher KL measures
0.0264/0.0444/0.0516 with a coefficient near1, immutable teacher SHA, finite
student/teacher/Adam/loss and unchanged environment/bank/R2/zero-cost contracts.
The original failed smoke is preserved. The second process still spent about
206 seconds before its first update; cache reuse is not claimed. Full4×128
CUDA/conv/NCCL plus two-update qualification remains an in-allocation gate.

Source and input staging, duplicate-safe submission, and the bounded milestone
watcher are prepared. They are not running remotely. Authentication renewal is
the outstanding submission dependency.

[Teacher-KL run design](research/FOUNDATION_TEACHER_KL_RECOVERY_20260914.md) ·
[Native resume smoke](../../../../.artifacts/terra_foundation_strong_recovery_20260914/local/resume_checks.json) ·
[Manifest](../../../../.artifacts/terra_foundation_strong_recovery_20260914/manifest.json).


## September 14, 14:54 CEST: matched scratch versus pretrained teacher-KL pair ready locally

The user reports better past results from random student initialization with
teacher KL and requested checking that against direct strong-policy initialization.
The unsubmitted single native-recovery proposal is superseded. Prepare one
24-hour CSCS d130/lterenzi node with four GH200s total: two independent2-GPU
arms, scratch_kl(primary) and pretrained_kl(full-model parameters only).
Both start with fresh Adam and clocks0, the same old zero-cost u15000 teacher,
seed20260907, easy foundation bank, current Terra46738cde physics and zero
added behavior penalties. Each arm keeps global512envs, 16,384transitions and
64Adamsteps/update. Constant LR3e-4/entropy0.02 and teacherKL1 cosine-to0 over
20,000newupdates are identical; value distillation and LR warmup are0.

The optional initialization receipt verifies actual first-rollout model, Adam,
reset and RNG states. All57CPUtests pass. Both local1x32 arms complete finite
u1/u2 plus nativeu3, with Adam64/128/192 and zero integrity failures. Initial
reset/RNG/history/teacher and fresh-optimizer hashes match; pretrained model
matches the teacher, scratch differs; both resumes preserve their own u2 state.
Independent review covers arm/parent binding, immutable source/launch/input
hashes and paired metrics. Actual weak-policy fixtures exposed legitimate
missing workspace metrics; aggregation now retains missingness and matched
per-metric sample counts. These are runtime gates, not learning results.

Fixed evaluations will compare both arms atu2500/5000/10000/20000 on the same
64-map greedy450-step panel. Report all-map coverage, common-success workspace
and retained-pose continuity metrics, and secondary raw Terra travel. No
additional allocation or penalty stage is automatic. The full2x256-per-arm
CUDA/conv/NCCL/startup and paired-state gates are mandatory in the allocation.
CSCS still rejects SSH authentication; no new job was submitted. Existing
Euler jobs are unchanged. The old native-only proposal is explicitly disabled.

[Current design](research/FOUNDATION_TEACHER_KL_RECOVERY_20260914.md) ·
[Campaign manifest](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json) ·
[Paired local qualification](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/local/paired_checks.json).


## September14,16:32CEST: CSCS foundation teacher-KL initialization pair submitted

Authentication now succeeds as lterenzi on daint-ln004. Submitted one reviewed
24-hour d130/normal job, **4665916**, at16:28:16CEST. It requests one node with
two tasks, each two GH200GPUs and32CPU cores; Slurm confirms totalfourGPUs and
per-task binding. Scratch+KL is the primary arm; pretrained full-model weights
with the same teacherKL are the comparison. Both retain fresh Adam and clocks0,
global512 environments each, identical schedules/maps and zero behavior costs.

Pinned training sources are baselinesa3a2119af014f9afe018ed69b023a2e9c1acc4dc
and Terra46738cde28e455da7c466fc0a2cb64f677d86401. Remote source, launch, bank,
teacher and container hashes match the reviewed staging receipts. There are
no other Terra jobs in the lterenzi CSCS queue at the authentication check;
unrelated running Newton job4663069 was left unchanged.

At16:31CEST job4665916 is PENDING/Priority. The provisional start estimate is
September15 at06:03CEST, not a reservation or proof of startup. There are no
new training updates/checkpoints or evaluations. Both full2x256 startup gates
must pass before production. The queue-aware runtime coordinator is running in tmux
terra_foundation_kl_init_runtime_20260914. Its live pending-state probe passed;
it starts the reviewed26-hour evaluation window when the allocation begins
and captures both full-size startup receipts. Pending time does not reduce
evaluation coverage. No second allocation or penalty phase is automatic.

[Submission receipt](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/submission_receipt.json) ·
[Scheduler receipt](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/scheduler_after_submission.txt) ·
[Live manifest](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json).


## September 15, 08:44 CEST: both arms recover strong easy-foundation completion

CSCS job **4665916** is RUNNING on nid005694, started at04:10:32CEST,
with walltime ending September16 at04:10:32CEST. Both full2x256 startup and
paired initialization gates passed. Live production logs have reached about
7,700 updates per arm; both have u7500 checkpoint files. The latest checkpoints
independently downloaded and native-validated for evaluation are u5000.
Training currently takes about2seconds/update per arm (about8,200 transitions/s).

| Arm | u2500 exact completion | u5000 exact completion | u5000 mean dug |
| --- | --- | --- | --- |
| Random student + teacher KL | 62/64 | 63/64 | 99.740% |
| Pretrained student + teacher KL | 62/64 | 63/64 | 99.933% |

All four fixed evaluations are complete:64 identical validation resets each,
greedy450steps, verified checkpoint hashes, Adam counters and pinned sources.
An independent agent checked all256 rows and transition integrity. On the62
common successful maps at u5000, scratch/pretrained productive poses are
6.839/6.823, unique area per productive setup6.136/6.132m², retained straight-line
travel26.574/26.579m, and fresh-workspace adjacency97.480% for both. There are
no exact-pose revisits or ABA returns in this cohort. Retained straight-line
travel is a geometric lower bound, not a Nav2 path measurement.

Scratch fails slot43/mapL00300 at83.33%dug; pretrained fails slot33/mapL00290
at95.72%dug. There is no clear completion or efficiency winner. Both match or
slightly exceed the frozen control's62/64 on this easy panel. This is one paired
seed with a shared teacher, not a no-KL ablation or full-dataset result.
Teacher KL is still active (coefficient about0.854 at u5000); the bounded watcher
is alive and will evaluate both arms at u10000 and u20000. Both numerical
completion gates now pass, but promotion is separate: all added behavior costs
remain zero and no new allocation or penalty phase has been started.

[Live evidence](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/live_status_20260915.json) · [Paired u5000 report](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/evaluation/paired/5000.json).
