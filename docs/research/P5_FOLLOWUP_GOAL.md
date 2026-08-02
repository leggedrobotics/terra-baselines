# P5 Follow-up: Condition-Level Diagnosis and Matched Improvements

- Status: active execution; P5b complete, P5c low-entropy screen in preparation
- Date: 2026-08-03
- Owner: Codex with Lorenzo review
- P5 implementation contract:
  [`P5_ACCEPTED_BANK_EXPERIMENTS.md`](P5_ACCEPTED_BANK_EXPERIMENTS.md)
- Map and training authority:
  [`D5_D7_IMPLEMENTATION_PLAN.md`](/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730/D5_D7_IMPLEMENTATION_PLAN.md)
- Curriculum taxonomy:
  [`CURRICULUM_TAXONOMY.md`](/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730/CURRICULUM_TAXONOMY.md)
- Research-code constraint:
  [`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)
- Frozen accepted bank:
  `/home/lorenzo/moleworks/.artifacts/terra_p5_accepted_bank_20260801_a6e6e5bc_prng_v102`
- Retrieved P5 evidence:
  `/home/lorenzo/moleworks/.artifacts/terra_p5_results_20260802_f8aac348`
- Retrieved P5b evidence:
  `/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`
- P5b standardized leaderboard:
  `/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`
- Capability-floor diagnostic bank:
  `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_controls_20260802_0306c3cd`
- Capability-floor evaluations:
  `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`

## 1. Objective

Turn the completed P5 screens into a verified per-panel, per-family, and
per-condition account of what the policies learned; use that evidence to
select and run the smallest matched follow-up campaign that can distinguish
continued optimization, curriculum choice, and additional model capacity; and
finish with either a demonstrably improved checkpoint or a documented
negative result that selects the next bottleneck.

The goal includes launching the selected Euler screens after their local and
allocated smoke gates pass. A long 120-hour run is admitted only after a
bounded matched screen shows improvement on both frozen public panels without
hiding a failed family or condition.

## 2. Frozen starting evidence

All six P5 jobs completed 2,000 PPO updates and 262,144,000 transitions with
the same accepted bank, environment protocol, medium E8 architecture, seed,
reward, horizon, and PPO shape. Promotion and development evaluations enumerate
their frozen panels exactly.

At update 2,000:

| Arm | Primary scope | Promotion macro | Development macro | Promotion exact | Development exact |
|---|---|---:|---:|---:|---:|
| `F-ANCHOR` | foundation | 0.361 | 0.371 | 0/288 | 0/288 |
| `F-SPECIALIST` | foundation | 0.566 | 0.559 | 0/288 | 0/288 |
| `T-ANCHOR` | trench | 0.277 | 0.295 | 1/224 | 0/224 |
| `T-SPECIALIST` | trench | 0.450 | 0.445 | 1/224 | 2/224 |
| `G-UNIFORM` | all | 0.574 | 0.577 | 0/512 | 0/512 |
| `G-ADAPTIVE` | all | 0.588 | 0.574 | 0/512 | 1/512 |

These aggregate values are orientation only. They do not choose a policy:
specialists received more per-condition training dose, uniform versus adaptive
is the only clean sampler comparison, and the condition distributions include
dead or regressing tails that pooled means can hide.

## 3. Standard leaderboard contract

One deterministic command must consume the immutable P5 and follow-up result
bundles and emit machine-readable JSON/CSV plus one concise Markdown report.
It must fail closed on inconsistent campaign, bank, protocol, panel,
checkpoint, or condition identities.

The report has four non-substitutable aggregate/detail tables for each
checkpoint and panel:

1. **Policy summary:** condition-macro completion, micro mean and p10, exact
   count/rate, worst condition, and integrity status.
2. **Family summary:** the same completion and exact accounting separately for
   foundations and trenches. A generalist comparison also reports the minimum
   family value.
3. **Factor-axis summary:** anchor, geometry, dump layout, capacity, distance,
   site, and composed slices, each using condition-macro rather than map-count
   weighting.
4. **Condition summary:** every canonical condition ID, its family and taxonomy
   fields, map count, exact count/rate, mean/median/p10 terminal completion,
   and rank within its family.

A fifth paired-delta table compares every declared treatment with its exact
control at the same added update after verifying identical ordered episode
IDs. All summaries include zero-completion rate, near-complete rate at `0.95`,
and total no-effect actions divided by total evaluated steps.

Promotion and development remain separate columns and rows. They are never
averaged to settle disagreement. Sealed evaluation remains untouched during
screening. Reward return is not a ranking metric. Specialist primary results
use only the trained family; their other-family values are labelled transfer
diagnostics. Leaderboard order is descriptive, while treatment claims require
a matched comparison.

## 4. Analysis questions

Before selecting follow-up jobs, answer from the fixed evaluations and training
receipts:

- Which individual foundation and trench conditions improved, plateaued,
  regressed, or remained near zero?
- Are failures aligned more strongly with geometry, dump constraint, site
  constraint, distance, or family?
- Did adaptive sampling materially change exposure and the condition tail, or
  is it tied with uniform within this single-seed screen?
- Does the anchor-to-specialist gain indicate learnability under concentrated
  dose, and where does the specialist still fail?
- Is a larger architecture a plausible capacity treatment, and can a P5
  checkpoint be grown without changing its initial function?
- Is continued medium-model optimization still improving at update 2,000?

## 5. Follow-up experiment contract

The final matrix is written here before submission and is constrained as
follows:

- test one hypothesis per matched pair;
- use the same immutable train bank and promotion/development panels;
- use the same parent checkpoint, sampler, reward, horizon, observations,
  actions, optimizer schedule, seed, update budget, and environment count
  within an architecture pair;
- compare each treatment to a shared parameters-only medium/adaptive warm-start
  control, changing either residual depth or sampler rule but never both;
- verify every update-zero initialization against the parent, archive the
  per-leaf growth report, and record that every arm resets optimizer state;
- retain numbered checkpoints and run fixed promotion and development
  evaluation at the same added-update cadence; and
- limit the first screen to the minimum arms needed by the condition-level
  diagnosis. Do not launch every plausible architecture or sampler variant.

The existing `G-UNIFORM` versus `G-ADAPTIVE` result remains a one-seed screen,
not a state-of-the-art curriculum claim. A scheduler confirmation requires
paired seeds after the treatment has a meaningful effect size.

### Frozen P5b matrix

The formal P5 selector chooses `G-ADAPTIVE` update 2,000 as the common parent:

- checkpoint SHA-256:
  `76b5189955735741b0cd4b3444fbda8ffdb8be4b29582509dafad85fa7cfb45a`;
- promotion macro `0.588`, development macro `0.574`; and
- parent exact `0/512` promotion, `1/512` development.

The original widened-large candidate was rejected before training. Its
update-zero promotion macro collapsed from the parent's `0.588` to `0.071`.
Recovering that transplant would spend the screen relearning the parent rather
than testing capacity.

The admitted capacity candidate keeps the medium widths and readout and adds
three identity-initialized residual blocks: stage channels remain
`(24, 48, 64, 96)`, while stage blocks change from `(1, 2, 2, 2)` to
`(2, 2, 3, 3)`. Parameters increase from `2,441,223` to `2,699,117` (`+10.6%`).
This is a depth treatment, not a general "large model" claim.
The admitted checkpoint SHA-256 is
`6bf014c7b9074564df9e1b36fd4e4106bfeb61f1dfa17b7fbd728314c958ba9b`;
its growth and same-runtime update-zero controls are under
`/home/lorenzo/moleworks/.artifacts/terra_p5b_parent_20260802/`.

The bottom-quartile retention panels are frozen from the common parent, not
reselected after training:

- promotion (`0.359` macro): `fnd-slab-apron-d16`,
  `fnd-slab-side1-obj`, `fnd-proc-side1-road`, `fnd-slab-apron-d12`,
  `fnd-slab-side1`, `trn-net4-side1-road`, `fnd-slab-split`, and
  `trn-straight-altsides`;
- development (`0.358` macro): `fnd-slab-apron-d16`,
  `fnd-proc-side1-road`, `fnd-slab-apron-d12`, `fnd-slab-side1-obj`,
  `trn-straight-side1-tight`, `trn-net3-side1-road`,
  `trn-straight-altsides`, and `fnd-slab-side1`.

The first follow-up screen has exactly three arms:

| Arm | Initialization | Architecture | Question |
|---|---|---|---|
| `G-MEDIUM-ADAPTIVE-WARM` | parent parameters | medium SE, adaptive sampler | shared re-optimization control |
| `G-DEEP-ADAPTIVE-WARM` | depth-grown parent | medium-width deeper SE, adaptive sampler | does added residual depth improve the fixed tail? |
| `G-MEDIUM-UNIFORM-WARM` | parent parameters | medium SE, uniform sampler | does balanced exposure recover the remote-map tail? |

All arms use parameters-only warm start, a fresh optimizer and sampler, and the
frozen medium parent as kickstart teacher. They share seed, maps, reward,
horizon, action/observation interface, PPO settings, entropy and teacher
schedules, `4 x 1024 x 32` rollout shape, and 2,000 added updates per arm
(262,144,000 transitions). `G-DEEP-ADAPTIVE-WARM` changes only residual depth;
`G-MEDIUM-UNIFORM-WARM` changes only the sampler rule. No reward, action-mask,
attention, map, or partial-reset change enters this matrix. A deep-uniform arm
is deferred unless both independent treatments win.

The parent checkpoint predates sampler-state checkpointing, so a true adaptive
continuation is unavailable. Calling `--resume_from` would either restart or
invent its curriculum state. The three warm-start arms make that reset explicit
and matched. New periodic/final checkpoints must contain exact versioned host
sampler state.

Fixed evaluation is run at added updates 0, 500, 1,000, 1,500, and 2,000 on
promotion and development. Update zero measures transplant damage. Every
comparison is aligned by added transitions. This is a P5b recipe screen on the
64-layout bank, not P6 and not evidence for a 20,000-update run.

### Admission gate for a bounded screen

Each job must pass:

1. clean-tree and immutable revision receipt;
2. accepted-bank and environment-protocol verification;
3. checkpoint-parent and architecture receipt verification;
4. CPU tests for the changed path;
5. local CUDA reset/inference and initial-output parity when growing a model;
6. allocated update-1 smoke with finite loss, checkpoint, and GPU activity;
7. no sealed-panel evaluation.

### Improvement gate

A treatment advances only when promotion selects it and development confirms
it. Relative to its matched control at the same added update, both panels must
improve condition-macro terminal completion by at least `+0.01`.

In addition:

- neither family macro may regress by more than `0.02`;
- the frozen parent's bottom-quartile condition macro on each panel may not
  regress by more than `0.02`;
- a treatment is rejected if the same condition regresses by at least `0.10`
  on both panels;
- integrity, panel enumeration, and condition coverage must be exact.

Exact completion, zero-completion rate, near-complete rate at `0.95`, micro
p10, and no-effect-action rate remain mandatory diagnostics. Exact success
alone cannot promote a treatment, and micro p10 is not a gate while it remains
zero for every arm. The uniform treatment must additionally improve the
two-cell `d12`/`d16` distance-tail macro by at least `+0.01` on both panels.

For noisy close calls, the result is inconclusive and receives paired seeds;
it is not promoted by aggregate rounding. A 120-hour continuation additionally
requires a positive trend across at least two consecutive fixed-evaluation
checkpoints and no worsening condition tail.

For the depth claim, `G-DEEP-ADAPTIVE-WARM` is compared only with
`G-MEDIUM-ADAPTIVE-WARM`. For the sampler claim,
`G-MEDIUM-UNIFORM-WARM` is compared only with
`G-MEDIUM-ADAPTIVE-WARM`. For selecting a recipe to extend, the winning arm
must also pass relative to the common frozen parent. If neither treatment
helps, capacity/curriculum escalation stops and action masking becomes the next
bounded diagnostic for the zero-completion/no-effect-action tail.

## 6. P5b terminal result and corrected architecture belief

The three P5b screen jobs completed their full 2,000-update contracts and
passed their final integrity receipts:

| Arm | Slurm | Selected update | Promotion macro | Development macro | Foundation P / D | Trench P / D | Exact P / D |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | 2,000 | 0.652 | 0.625 | 0.639 / 0.588 | 0.670 / 0.674 | 1/512 / 2/512 |
| `G-DEEP-ADAPTIVE-WARM` | `9378175` | 1,000 | 0.653 | 0.628 | 0.625 / 0.571 | 0.689 / 0.702 | 2/512 / 2/512 |
| `G-MEDIUM-UNIFORM-WARM` | `9378176` | 1,000 | 0.647 | 0.664 | 0.625 / 0.630 | 0.675 / 0.708 | 2/512 / 6/512 |

These selected checkpoints describe the best retained policies; they are not a
causal comparison across different update counts. At the matched update 1,000,
deep/adaptive improves promotion/development macro by `+0.023/+0.013` and
medium/uniform by `+0.017/+0.049` relative to medium/adaptive. Both treatments
then lose that advantage by update 2,000. All three arms show a synchronized
drop at update 1,500, when the frozen-teacher KL coefficient reaches zero,
followed by only partial recovery. Any P5b-to-P5c entropy claim is therefore
restricted to matched checkpoints through update 2,000.

The selected factor view is:

| Factor | Medium adaptive P / D | Deep adaptive P / D | Medium uniform P / D |
|---|---:|---:|---:|
| Anchor | 0.718 / 0.706 | 0.731 / 0.713 | **0.740 / 0.765** |
| Capacity | **0.766 / 0.673** | 0.738 / 0.644 | 0.722 / **0.711** |
| Composed | 0.428 / 0.467 | **0.510 / 0.499** | 0.485 / 0.471 |
| Distance | **0.361 / 0.435** | 0.264 / 0.310 | 0.330 / 0.364 |
| Dump layout | **0.697** / 0.516 | 0.635 / 0.567 | 0.626 / **0.634** |
| Geometry | 0.695 / 0.696 | **0.703** / 0.714 | 0.677 / **0.721** |
| Site | 0.653 / 0.603 | **0.661** / 0.599 | 0.647 / **0.656** |

Distance and composed maps remain the primary condition tails; foundations
remain weaker than trenches. Exact completion is still too rare for any P5b
checkpoint to count as reliable full-task competence.

The historical architecture analogy is also corrected. E8 did **not** enlarge
E3: E3 and E8 both have `2,441,223` parameters and E8 reused E3 as a
parameters-only parent plus frozen KL/value teacher. The earlier capacity jump
was the approximately `994,825`-parameter policy to E3's `2,441,223`-parameter
medium SE model. P5b deep did implement that grow-and-teach pattern correctly:
it used exact function-preserving initialization, a fresh optimizer, the frozen
P5 parent as KL/value teacher, KL `1.0 -> 0` over 1,500 updates, and value
distillation `0.5 -> 0` over 500 updates. The new model has `2,699,117`
parameters. The plausible recipe mismatch is instead P5b's high entropy
schedule, `0.15 -> 0.005` over 7,600 updates, which was still about `0.137` at
the shared update-1,500 collapse. This is a testable hypothesis, not a proven
cause.

## 7. Capability-floor diagnostics

The accepted 32-condition bank contains no genuinely unconstrained dump
control: dumpability means that accepted cells are physically usable, not that
all free ground is a valid dump target. A separate diagnostic bank therefore
adds two source- and geometry-paired counterfactuals:

- `fnd-slab-allfree`, paired with `fnd-slab-ring3x`; and
- `trn-straight-allfree`, paired with `trn-straight-side2`.

For each pair, source identity, split, dig mask, obstacles, reset protocol, and
full-reset episode stay fixed. Only the visible accepted-dump mask expands to
every legal non-dig cell. The controls have 64/16/16/32 train/promotion/
development/sealed maps per condition, but they are diagnostic-only and are
explicitly excluded from the constrained 32-condition macro.

Current generalist results are:

| Policy checkpoint | Macro P / D | Foundation P / D | Trench P / D | Exact P / D |
|---|---:|---:|---:|---:|
| Frozen P5 parent | 0.385 / 0.465 | 0.423 / 0.493 | 0.347 / 0.436 | 0/32 / 0/32 |
| P5b medium/adaptive @2,000 | 0.629 / 0.613 | 0.655 / 0.635 | 0.604 / 0.591 | 0/32 / 0/32 |
| P5b deep/adaptive @1,000 | **0.718 / 0.736** | **0.770 / 0.757** | **0.666 / 0.715** | 0/32 / 0/32 |
| P5b medium/uniform @1,000 | 0.484 / 0.540 | 0.452 / 0.459 | 0.517 / 0.622 | 0/32 / 0/32 |

For historical calibration, E8's model parameters were transplanted
shape-for-shape into a current evaluation skeleton because its old serialized
`EnvConfig` no longer unpickles. They score only `0.013/0.027` macro and
`0/32` exact. The old near-1 online `swhr` is therefore not comparable to the
current fixed full-reset/exact-completion protocol.

The existing P5 family specialists provide the matching per-family check:
the foundation specialist reaches `0.668/0.640` on the all-free foundation
condition (exact `0/16` on both panels), while the trench specialist reaches
`0.556/0.606` on the all-free trench condition (exact `1/16` promotion and
`0/16` development). Specialization therefore does not restore the old
near-1.0 completion floor either.

These maps are physically easier, but their almost-all-green target masks are
out of the frozen training distribution. Zero exact completion therefore
diagnoses both incomplete cleanup and missing target-mask support; it does not
prove that the mechanics are hard. Every P5c checkpoint will be evaluated on
these controls, but the controls are not silently added to P5c training.
Adding them to training would be a separate treatment and a versioned
34-condition successor bank whose original 32 conditions remain byte-identical.

## 8. Frozen P5c low-entropy matrix

P5c tests the smallest common recipe correction that follows from P5b: all
five arms use entropy `0.02 -> 0.005` over 10,000 updates, the same frozen P5
parent and frozen-parent KL/value teacher, the same accepted-bank release,
reward, horizon, PPO shape, full resets, seed, and 4,000-update budget. The
generalists use all 32 conditions; the specialists select the existing
18-foundation or 14-trench subset. Fixed evaluation runs every 500 updates on
constrained promotion/development and on both capability-floor panels.

| Arm | Support | Sampler | Architecture | Matched question |
|---|---|---|---|---|
| `G-MEDIUM-ADAPTIVE-WARM` | all 32 | adaptive | medium | low-entropy common control |
| `G-MEDIUM-UNIFORM-WARM` | all 32 | uniform | medium | sampler effect at medium capacity |
| `G-DEEP-UNIFORM-WARM` | all 32 | uniform | deep | depth effect under the selected uniform exposure |
| `F-MEDIUM-UNIFORM-WARM` | 18 foundations | uniform | medium | foundation-only dose ceiling |
| `T-MEDIUM-UNIFORM-WARM` | 14 trenches | uniform | medium | trench-only dose ceiling |

The generalist arms form two matched causal edges: medium/adaptive versus
medium/uniform isolates the sampler, and medium/uniform versus deep/uniform
isolates residual depth. The two specialists are family dose ceilings, not evidence that a
specialized architecture is intrinsically better and not participants in the
generalist promotion macro.

A long 20,000-update/120-hour continuation is admitted only after P5c shows a
positive trend at multiple fixed checkpoints, improvement on both public
panels, no hidden foundation/trench or bottom-tail regression, and improvement
or at least stable behavior on both capability-floor controls. A finite loss,
online reward, or one favorable checkpoint is insufficient.

## 9. Work ledger

- [x] Freeze and retrieve all six P5 promotion/development result bundles.
- [x] Verify P5 receipts and produce policy/family/factor/condition/delta leaderboards.
- [x] Write the condition-level failure and sampler analysis.
- [x] Reject widened-large growth at update zero and build the depth-only candidate.
- [x] Complete same-runtime parent/depth update-zero parity on both panels.
- [x] Freeze the minimal matched follow-up matrix in this document.
- [x] Implement and test only the support required by that matrix.
- [x] Pass CPU, local CUDA, and allocated update-1 admission gates.
- [x] Launch the bounded Euler screens.
- [x] Evaluate every numbered P5b checkpoint on promotion and development.
- [x] Update the standardized P5b leaderboard and record the bounded-screen
  decision for every treatment.
- [x] Add paired foundation/trench capability-floor controls without changing
  the constrained macro.
- [x] Evaluate the P5 parent and selected P5b checkpoints on both diagnostic
  panels.
- [ ] Pass CPU, local CUDA, and allocated update-1 gates for all five P5c arms.
- [ ] Launch and evaluate P5c every 500 updates on constrained and diagnostic
  panels.
- [ ] If admitted, queue and monitor the long run through fixed evaluation;
  otherwise record why no long run was justified.

Current admission evidence:

- full CPU suite: `277 passed` against Terra `a6e6e5bc`;
- local CUDA update 1: medium/adaptive and depth/adaptive passed the existing
  smoke checkpoint verifier, including finite transition state, params-only
  optimizer reset, and versioned pooled-sampler state;
- local smoke artifacts:
  `/home/lorenzo/moleworks/.artifacts/terra_p5b_local_smoke_20260802.TwbRgP/`;
- exact-shape allocated smokes passed for all three arms before screen
  submission.

Allocated execution receipt (2026-08-02):

- immutable terra-baselines run revision:
  `6c56610ea5af1d029b736b2c4c1a8c2be3f5bc36`;
- smoke jobs `9377609`, `9377610`, and `9377611`: all `COMPLETED 0:0`,
  explicit verifier `passed=true`, update 1, and `status=PASSED`;
- smoke throughput: `304.81`, `293.10`, and `307.11` steps/s for
  medium/adaptive, deep/adaptive, and medium/uniform respectively; and
- screen jobs `9378174`, `9378175`, and `9378176`: all `COMPLETED 0:0` with
  final `status=PASSED`, 2,000 updates, and complete fixed-panel evaluations.
- P5c has no Slurm job IDs yet; its state is `PREP`, not `RUNNING`.

## 10. Completion evidence

This goal is complete only when the repository contains:

- the deterministic leaderboard tool and its tests;
- an archived leaderboard and analysis tied to exact input hashes;
- the predeclared follow-up matrix and immutable run contracts;
- smoke and screen receipts for every launched P5c job;
- fixed constrained and capability-floor results at every declared P5c
  checkpoint; and
- a final decision naming the selected policy/checkpoint, or a falsifiable
  negative conclusion with the next bottleneck.

`RUNNING`, GPU utilization, finite PPO loss, online reward, or an unevaluated
checkpoint is execution evidence only and cannot complete the goal.
