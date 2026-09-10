# Experiments — completed log

## 2026-09-07 local pipeline correctness smoke

Validated the junction observation, native checkpoint replay, and JAX cache
fixes on starship's single RTX 4090 with W&B disabled. Paired Terra is
`46b140f8373e098ad832e4968d8136a5ba861bf6`; the trainer changes are on
`fix-training-continuation-cache`, based on `d410062b`. No Slurm job was launched.

The current v2 encoder (2,311,701 parameters) completed four updates at 8
environments × 4 steps, 2 epochs, and 2 minibatches. A separate process then
resumed checkpoint 2 with receipts 3/4 already present and completed absolute
update 5. All 431 optimizer leaves were restored exactly at step 8; final Adam
count and train-state step were 20. Model, optimizer, and stored losses were
finite. Replayed receipts 3/4 were archived unchanged and canonical receipts
1–5 remained available.

Both processes had one stable update signature and an explicit persistent-cache
hit for `pmap__update_step`. The original reset counter's weak-to-strong int32
change had caused a second full compile; its two reset initializers are now
explicit int32. Periodic cache eviction is removed. Process startup still traces
and lowers the graph; this small-batch smoke is not a production throughput,
multi-GPU, or learning-quality benchmark.

CPU validation passed junction/DO parity, reward and counter regressions, and
39 trainer/launcher tests. Four edited launchers passed shellcheck. Full local
commands, logs, checkpoints, and independent review are under
`/home/lorenzo/moleworks/.artifacts/terra_pipeline_fixes_20260907/`.

The first table is the historical spatial-encoder line. Metrics:
eval/positive_terminations /
eval/rewards (final), swhr = eval/success_within_horizon_rate, ep_len =
eval/avg_positive_episode_length. Full incident trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`.

| Date | Run | W&B | Config | Final | Verdict |
|---|---|---|---|---|---|
| 07-19/20 | spatial A/B | pqtmfmqy | resnet_spatial_8x8 base, 10k updates, mb32 | 2.810 / 0.131 | spatial beats atari +39% per-sample at 3.9× wall-clock; became the E3 teacher |
| 07-20 | atari control | nnsksyva | atari base, 10k, mb32 | 2.020 / 0.095 | control; 113k steps/s |
| 07-21 | E2 | nr032qs7 | se+bf16+critic512 FROM SCRATCH, 20k | 2.329 / 0.108, swhr 0.976 | below teacher — from-scratch drag of heavier bundles (bf16 confirmed +50% throughput) |
| 07-22 | E1 | 3buorfp3 | spatial_8x8 + algo fixes (no value clip, flat shuffle, 19k ent), 20k | 2.833 / 0.132, swhr 0.997, ep_len 59.2 | beats teacher on its own encoder → PPO fixes net-positive |
| 07-22 | E3 | j0bs2fkl | medium se, grown init + kickstart from pqtmfmqy, 20k | **3.054 / 0.142, swhr 0.997, ep_len 55.2** | **best policy to date (+8.7%); kickstart playbook validated — surpassed teacher at 30% budget** |
| 07-22 | E4 | k8vnwp5u | se_xattn FROM SCRATCH, 20k | 2.586 / 0.121, swhr 0.993, ep_len 63.0 | xattn beats SE from scratch +11% (2.59 vs 2.33) — real architecture win; still below warm-started runs |
| 07-22 | E5 | gud7cbwg | dumpzone transfer (E3 warm-start + teacher), CANCELLED @850/20k | 0.000 / −0.005 | naive cross-task kickstart does NOT bootstrap a new task family (no reward stream found); E5b = 2-stage curriculum + higher entropy when dumpzone becomes priority |
| 07-22 | E9 | — | 128×128 pilot, 5-stage medium SE, teacher_obs_downsample=2 | FAILED in smoke | teacher module was built with the student 128 env, causing 128-row position embeddings for a 64-row teacher checkpoint; fixed locally with regression + real-checkpoint CPU smoke before relaunch |
| 07-22 | E9b | — | fixed 128×128 pilot relaunch, same 512 env/GPU memory shape | FAILED in smoke | teacher-env fix worked, but 128×128 PPO update OOMed (`RESOURCE_EXHAUSTED`, temp 10.70 GiB plus 11.49 GB allocation attempt); next gate should try `num_minibatches=64` |
| 07-23 | E9c | 0ixsswn4 | fixed 128×128 pilot relaunch, same 512 env/GPU with `num_minibatches=64` | CANCELLED @~1.4k; all-NaN from smoke update 0 | memory-fit shape worked, but smoke gate missed NaN loss/params: smoke FINAL and production checkpoint both had all model params NaN; forward-only behavior was a symptom, not ordinary learning failure |
| 07-23 | E9d | — | E9c plus embedding clamp, local-map `IntMap`, local-map area scale 4, loaded/downsample fix, finite guard | SUBMITTED 8323457 | exact local full-shape 1-update gate passed finite before Slurm submit; job pending priority |

Cross-run findings:
- Warm-start (grow + kickstart) >> from-scratch for introducing architecture/capacity
  changes (E3 vs E2; E4' line continues this).
- bf16 encoder compute: +50% production throughput (43k vs 28.8k steps/s), numerics clean.
- Finite-loss/param checks are now mandatory for smoke gates: E9c completed update 0 and
  entered production despite all-NaN model params/loss scalars. The E9d script keeps
  `--fail_on_nonfinite` through smoke and production.
- Attention follow-up launched as E10 (pending at submission): `--attention_compute_dtype
  float32` isolates attention-softmax precision inside a bf16 trunk;
  `--token_mixer_residual_init_scale 0.001` wakes v5 mixer gradients without changing the
  default identity-at-init contract.
- pmap scaling ~95% (single-GPU probe 11.4k vs 10.8k/GPU ×4).
- Entropy-schedule stretch makes mid-run comparisons vs 9.5k-schedule runs invalid —
  compare finals or matched entropy phase only.
- Episode length: E3 55.2 vs E1 59.2 steps — bigger warm-started net is also faster per
  episode; 300-step horizon (E6) exerts no pressure since episodes are ~55 steps.

## Completed training awaiting fixed selection

The 2026-08-21 paired movement-feedback GRU pilot completed 50,000 updates in
both arms. Control job `11364188` and six-bit feedback job `11364189` exited
`0:0`; their final checkpoint SHA-256 values are respectively
`5459bd5347dbdf64431cd78df5f61f22b75ee56bc2b15662d9751fb2959a7f84`
and `8cde5ccd4fd4ef5b1ed716a9c5c3a4c4b43f69d44db66d29ed7db86f2ad7d7df`.

The final 1,000-update online aggregate has effectively tied success
(0.99019 control, 0.99037 feedback), while feedback reduces no-effect rate
from 0.03152 to 0.01450. This supports retaining the optional observation path
but is not a promotion verdict. The accepted development-720 evaluation is
still missing, so neither final policy is selected and feedback stays off by
default. See
[`research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md`](research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md)
for the frozen decision gate and full provenance.

## Accepted-bank campaigns

The metrics below are deterministic fixed-bank terminal absolute completion,
not the legacy online spatial-run summaries above. Promotion and development
remain separate. Exact is solved maps / evaluated maps.

| Date | Campaign / arm | Slurm | Selected update | Macro P / D | Exact P / D | Verdict |
|---|---|---:|---:|---:|---:|---|
| 08-02 | P5 six-arm accepted-bank screen | — | 2,000 | generalists 0.574--0.588 / 0.574--0.577 | at most 1/512 | all six completed and passed; `G-ADAPTIVE` selected only by the predeclared retention gate, not as a general scheduler claim |
| 08-02 | P5b `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | 2,000 | 0.652 / 0.625 | 1/512 / 2/512 | completed 2,000, `PASSED`; strongest selected constrained distance axis |
| 08-02 | P5b `G-DEEP-ADAPTIVE-WARM` | `9378175` | 1,000 | 0.653 / 0.628 | 2/512 / 2/512 | completed 2,000, `PASSED`; transient matched gain, not retained at 2,000 |
| 08-02 | P5b `G-MEDIUM-UNIFORM-WARM` | `9378176` | 1,000 | 0.647 / 0.664 | 2/512 / 6/512 | completed 2,000, `PASSED`; best selected development family floor, transient at 2,000 |
| 08-03 | all-free capability-floor evaluation | local fixed eval | selected P5/P5b checkpoints | 0.385--0.718 / 0.465--0.736 | generalists 0/32; trench specialist 1/32 promotion only | integrity-clean diagnostic; physically easier but target-mask OOD, excluded from constrained macro |
| 08-03 | P5c five-arm low-entropy screen | `9461489`, `9461500`, `9461504`, `9461507`, `9461512` | none | deep latest 0.624 / 0.586 | deep latest 168/512 / 143/512 | all fixed evaluations clean; no arm passed the long-run gate at two consecutive checkpoints |
| 08-10 | V8 Atari-base small-system control | `10128519` | 19,000 (descriptive promotion selection) | 0.457 / 0.428 | 16/752 / 20/752 | completed 20k; zero mastered conditions and depth 0/0; negative 480k-system result, not an encoder-only ablation |
| 08-14 | V8 v6.1 reward-v2 + stall age + final-v3 continuation | `10625259` | 40,000 | 0.956 / 0.959 | 657/720 / 663/720 | completed u14->u40; promotion exact 407->657 with 254 conversions and 4 regressions; u39->u40 was high-churn 38/32 for only +6 net, so this is a strong combined-treatment capability result but not stall-age or sampler attribution |
| 08-23 | trench-aligned 37-condition generalist, initial production attempt | `11529891` | 0 | — | — | exact update-1 smoke `11529665` passed, but production failed before update 1 on `eu-g6-065` with four-replica `CUDNN_STATUS_EXECUTION_FAILED`; no checkpoint and zero W&B training updates, so this is a runtime incident only |

P5b result root:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`.
Standard leaderboard:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`.
Capability-floor results:
`/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`.
The parameters-only current-protocol E8 compatibility replay scores
`0.013/0.027` macro and `0/32` exact; its historical near-1 online `swhr` is a
different evaluation contract and is not numerically comparable.

P5b deep used function-preserving growth (`2,441,223 -> 2,699,117`), a fresh
optimizer, and the frozen parent as KL/value teacher. E8 was not a larger E3:
both had `2,441,223` parameters. The likely recipe mismatch is P5b entropy
`0.15 -> 0.005 / 7,600`, still about `0.137` at the synchronized update-1,500
decline when KL reached zero. This is the P5c hypothesis, not a post-hoc claim
that entropy caused the decline.

## Completed fixed-evaluation screen: P5c

P5c freezes five 4,000-update arms with evaluation every 500 updates: medium
adaptive, medium uniform, deep uniform, foundation medium-uniform, and trench
medium-uniform. Allocated update-1 smoke jobs `9458568`, `9458581`, `9458585`,
`9458616`, and `9458619` all completed `0:0` and passed. Screen jobs `9461489`,
`9461500`, `9461504`, `9461507`, and `9461512` were then submitted from
revision `3478af87950d3d35059344b078209d00785c8481` and crossed finite update
1 with transition-integrity checks enabled. All five subsequently completed
4,000 updates in 6.2--8.0 hours. All share entropy
`0.02 -> 0.005 / 10,000` and the common P5 parent/teacher. The specialists are
family dose ceilings; they do not enter the causal sampler/depth comparison.
All 40 numbered checkpoints were evaluated on constrained
promotion/development and diagnostic all-free promotion/development: 160
integrity-clean evaluations and 43,520 episodes with no integrity failure.
Deep uniform at update 4,000 is the strongest descriptive endpoint:
promotion/development macro `0.624/0.586`, foundation `0.556/0.533`, trench
`0.711/0.654`, and exact `168/512` / `143/512`. Its checkpoint SHA-256 is
`4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc`.

This endpoint is not a formal selection. Foundation specialist had one clean
interval at update 3,000, deep uniform one at 4,000, medium adaptive one at
2,500, trench specialist one at 3,500, and medium uniform none. No arm passed
the predeclared improvement/retention gate at two consecutive checkpoints.
No 120-hour continuation or P6 training was launched. Online training success
was still rising, so the result is also not saturation evidence; it says the
current learning curve is unstable on fixed held-out panels.

The condition-balanced leaderboard is frozen at
`/home/lorenzo/moleworks/.artifacts/terra_p5c_leaderboard_20260803_3478af8/LEADERBOARD.md`
with input digest
`ac665b7088942b66159a52f7170c1484dc6e36175f2ec7decbd8c4383094c5ac`.
The complete read-only campaign archive is
`/cluster/work/rsl/lterenzi/terra_p5c_campaign_20260803_3478af87950d3d35059344b078209d00785c8481/`;
its payload-manifest SHA-256 is
`605922d0965206f82e7fe54a10fac202e028b548de24454febcd2691709ff42f`.
Future behavioral screens receive at least 24 healthy hours with an oversized
absolute update target; an admitted checkpoint continues with true resume
state on the 120-hour queue. See
[`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
section 12.


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
