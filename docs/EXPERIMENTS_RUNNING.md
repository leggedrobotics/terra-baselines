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
deadlock (C). Next: per-cell admission variant probe (in flight).

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
