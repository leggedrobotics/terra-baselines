# V8 failure-remediation execution

Date: 2026-08-14

Completed Codex goal: `01a000fe-f5b4-7380-8df1-4ea8f2e343f1`

Goal objective: complete the non-overlapping Terra V8 failure-remediation
workstream: repair canonical evaluation no-effect accounting, expose a clear
dig/terminal/off-zone/load progress vector, perform an evidence-backed
traversability-observation versus movement-physics incidence audit, and
evaluate the stall-age plus Continuous Banded v3 continuation when its final
checkpoint is available.

## Scope boundary

This work does not edit trench generation, trench excavation ordering, or the
partial-reset generator. It also does not mutate the allocated continuation,
change reward-v2, or add action masking. The separate fresh partial-reset arm
now consumes the paired Terra generator through its narrow sidecar API and
adds only the reset-context observation described below.

The paired partial-reset implementation is owned at
`/home/lorenzo/moleworks/.worktrees/terra_v8_relay_corridor_20260814`. This
baselines workstream does not edit, format, clean, commit, or otherwise mutate
that worktree. Training consumes only a Terra-validated sidecar bank whose
content digest is bound into checkpoints and evaluation receipts.

## Why these items come first

The Oracle staging review identifies missing transition-outcome information as
the strongest policy-side issue, but the existing evaluation has two upstream
measurement defects: its canonical no-effect count is not sourced from Terra's
authoritative transition flag, and its scalar completion metric hides progress
at loaded timeouts. Those defects must be repaired before another behavioral
treatment is interpreted.

Synthetic staged-soil states also make the existing discrepancy between the
observed traversability map and the real movement predicate more relevant.
The discrepancy will therefore be measured now, but environment or policy
semantics will change only if the incidence audit shows that it affects actual
decisions.

## Work packages

### 1. Canonical no-effect accounting

- Compute `no_effect_action_count` from
  `timestep.info["action_had_effect"]` for every active evaluation transition.
- Remove the raw-observation comparison from the canonical evaluator.
- Add one focused regression test covering a terrain height greater than one,
  where clipped preprocessing previously corrupted the comparison.
- Preserve the unmasked policy contract.

Acceptance: the aggregate count exactly equals the recorded authoritative
effect sequence on a bounded rollout.

### 2. Material progress vector

Expose episode-terminal normalized components with direct semantics:

- `dig_fraction`;
- `terminal_soil_fraction`;
- `off_zone_staged_soil_fraction`;
- `loaded_soil_fraction`.

The components must use one declared source-volume denominator and remain
interpretable at loaded timeouts. `absolute_completion` remains dashboard-only
and is not a promotion metric. New output must not describe legal off-zone
staging as illegal dumping.

Acceptance: a mass-conserving constructed episode partitions source soil across
terminal, off-zone, and loaded states, and a loaded near-complete timeout keeps
its material progress rather than collapsing to zero.

### 3. Traversability incidence audit

- Compare the policy-visible traversability value with the exact movement
  result for forward, backward, clockwise, and anticlockwise actions.
- Cover ordinary full starts and staged-soil states without changing either.
- Separate false-blocked and false-free observations by soil height/density and
  action type.
- Report incidence and representative counterexamples in an artifact.

Acceptance for this workstream: the receipt identifies the pinned Terra
revision, selected archived state source, transition count, and mismatch
counts.  Coverage of the parallel partial-reset curriculum is explicitly
deferred until that bank is frozen and immutable; this receipt must not claim
to cover it.  No observation or dynamics change is authorized by the audit
alone.

### 4. Stall-age plus final-v3 continuation readout

Slurm job `10625259` is the immutable capability run. It uses Terra
`c2d2a94a`, terra-baselines `dddc691`, eight RTX 4090 GPUs, material stall age,
and family-free Continuous Banded v3. It does not include later Terra commits
`88c0099e` or `30ad500f`.

Evaluate the final durable checkpoint on:

- the fixed 720-map promotion panel;
- the existing foundation/trench and per-condition decomposition;
- the frozen recurrence failure strata;
- stall-age mean and saturation; and
- the successful `d16` multileg relay as a negative control against disrupting
  legitimate empty relocation.

Acceptance: compare exact completion against the u14k `407/720` source with
the same evaluation contract and record failure-mechanism changes separately
from the headline score. This combined run cannot attribute a gain to stall age
or v3 individually.

## Already implemented for a future fresh runtime

- Terra `88c0099e`: remove the arbitrary minimum actionable-unit veto.
- Terra `30ad500f`: capacity-bounded, mass-conserving staged-soil relift.

These fixes are not retroactively injected into job `10625259`.

## Separate next treatments

Only after the work packages above:

1. test previous-transition outcome observations as a matched fresh arm;
2. test the five-way current-DO affordance separately if loaded/carry failures
   persist and the decodability audit supports it;
3. align a movement-observation channel with physics if the traversability
   mismatch has material incidence;
4. audit dynamic station/heading serviceability before changing reward-v2; and
5. use progress age, then an actor-only GRU, only if explicit observation
   repairs leave multi-step recurrent cycles.

Defer global pile-height preprocessing changes, accepted-first dump fallback,
time-to-go, action masking, and broad reward redesign until higher-priority
evidence requires them.

## Backplay-inspired synthetic partial-reset arm

The relay/cleanup exposure treatment is implemented as one fixed, absolute
update schedule. Update indices are zero-based and do not restart when a
24-hour segment resumes:

- updates 0--2,499: 25% partial lanes at 90% completion;
- updates 2,500--4,999: 25% partial lanes distributed across 75% and 90%;
- updates 5,000--7,499: 25% partial lanes distributed across 50%, 75%, and
  90%;
- updates 7,500--9,999: the same cumulative tier window while total partial
  share fades linearly from 25% to zero; and
- update 10,000 onward: full starts only.

The three tier shares and their aggregate target share are logged separately.
Partial lanes draw from the current Continuous Banded v3 probabilities
renormalized over the one condition subset supported by all three tiers. Full
lanes retain ordinary v3 draws. All assignment, reset, and transition exposure
counts remain truthful to the actual training mixture; only partial-start
episode outcomes are excluded from v3 competence and mastery updates.

The legacy `train/episode_success_rate` remains unchanged for dashboard
continuity. During the partial curriculum it mixes full- and partial-start
episodes and is therefore not a matched online comparison. The separate
`train/full_start_episode_success_rate` uses only completed tier-0 episodes and
is the comparable online diagnostic across control and treatment. It remains
secondary to the untouched fixed 720-map full-start evaluation, which is the
primary decision evidence.

The causal pair uses the same fresh v6.1 architecture and reward-v2 contract:

- matched control: full starts plus
  `--reward-v2-reset-context-observation`;
- treatment: the same flag plus `--partial-reset-root` and its validated bank
  digest.

The observation is the constant-per-episode pair
`[Q_reset, H_reset / V0]`, added through separate zero-initialized actor and
critic embeddings. Both control and treatment reject parameter-only warm
starts so they begin from fresh parameters. Native `--resume_from` remains
supported for segmented runs; the treatment additionally requires the same
bank digest and continues the schedule from the absolute checkpoint update.
Action masking and stall age are excluded from this causal arm. Online and
fixed-bank evaluation force tier 0 and therefore remain full-start tests.

This is **Backplay-inspired**, not an implementation of Backplay. It moves a
synthetic start-state distribution backward through completion windows, but
does not reset along successful policy trajectories, replay demonstrations, or
provide per-state success witnesses. The implementation is untrained and
there is no claim yet that it improves relay learning, solves the generated
states, or improves the untouched 720-map full-start panel.

## Issue-resolution ledger

The immutable u40 continuation tests stall age plus final v3 only.  It does
not contain the later Terra mechanics commits and must not be credited for
their effects.

| Issue | Status class | Current evidence | Bounded next action |
| --- | --- | --- | --- |
| Stall age plus final v3 | **Trained/evaluated** | The combined u14-to-u40 continuation improves exact completion from 407/720 to 657/720, but five selected recurrent failures remain and late checkpoint churn is high. | Continue the identical treatment for one more 24-hour segment and evaluate several retained checkpoints.  Do not attribute the gain to either component alone. |
| Evaluator no-effect accounting and material-progress vector | **Diagnostic fixed** | Canonical no-effect counts now use Terra's transition flag, and fixed evaluation measures terminal dig/accepted/off-zone/carried fractions.  These repairs postdate the source that generated u40. | Keep these diagnostics in every subsequent fixed-panel readout; they are measurements, not policy treatments. |
| Missing action-outcome feedback and greedy attractors | **Unresolved** | Exact no-effect fixed points and effective short cycles remain.  Prior physical effect plus material/load change is a supported fresh-arm hypothesis, not a guaranteed planning fix. | Run a matched fresh observation arm.  Promote only on reduced recurrence without full-panel regression; otherwise proceed to the recurrent-policy rung. |
| Minimum actionable-unit veto | **Implemented/untrained** | Terra `88c0099e` removes the arbitrary singleton veto.  It is absent from u40 and its direct continuation. | Carry it into the next fresh environment and retain exact-completion and mass-conservation tests.  Its measured direct ceiling is small. |
| Atomic rejection of an over-capacity positive relift | **Implemented/untrained** | Terra `30ad500f` implements capacity-bounded, mass-conserving `load what fits`.  It is absent from u40 and its direct continuation. | Train and evaluate the new transition before attributing policy benefit. |
| Relay and cleanup underexposure | **Treatment implemented/untrained** | Terra `67c72d09` adds natural relay-corridor partial resets and `794d4759` preserves trench access. The baseline side now implements the fixed 10k schedule, reset provenance, common-support v3 sampling, full-start-only mastery updates, reset-context matched control, full-start evaluation, and bank-bound native resume. | Run the fresh matched control/treatment pair and judge only on the untouched full-start panel plus recurrence/relay strata. Do not credit this arm with Backplay demonstrations or merge it into the direct u40 continuation. |
| Accepted-first dump without off-zone fallback | **Unresolved, low observed incidence** | The source-level risk remains, but the selected lifecycle audit observed zero accepted-invalid/off-zone-valid states. | Keep the reason-code diagnostic and add two-pass fallback only after a real trajectory reaches that branch. |
| Traversability observation versus physics | **Audited/unfixed** | Sparse height-one soil was rarely false-blocked.  The causal selected case was under-base holes hidden by the agent overlay. | Prevent dig/relift under the exact base footprint, define collision-reducing escape for pre-existing overlap, and preserve underlying blockers in the visible channel.  Do not add an action mask. |
| Hidden `last_dig_mask` and ambiguous `DO` outcome | **Deferred pending decodability** | Hidden state changed counterfactual DO eligibility in 38 visited states but produced zero measured exact full-input aliases. | Do not expose raw history now.  Use a large decodability audit to decide whether a compact five-way DO affordance should subsume it. |
| Point-geodesic potential ignores serviceability | **Audited/unfixed** | One local counterexample improved H by 7.72 but had no same-base relift heading; this does not prove global inaccessibility. | Keep reward-v2 and measure station, heading, and path serviceability over a larger panel before changing the potential. |
| Clipped global pile heights | **Deferred** | They did not cause the measured target recurrences: no repeated policy-input hash had a different raw action-map hash. | Revisit only if a separate capacity or long-range planning audit finds height-dependent aliases. |
| Progress age and actor GRU | **Deferred rung** | Stall age delays some traps but saturates, and material-changing ping-pong resets it.  No matched recurrent treatment has run. | Try progress-aligned observations first; if explicit repairs leave multi-step cycles, test actor-only GRU-64 with contiguous sequence PPO and a sequence-batched feed-forward control. |

## Execution status

### Completed measurement repairs

- The canonical evaluator now counts ineffective actions exclusively from
  `timestep.info["action_had_effect"]`.  Action-map clipping now copies the
  observation mapping instead of mutating the raw simulator observation.
- Training receipts, W&B, and fixed-panel rows expose the four-component
  material vector.  Fixed evaluation measures terminal carried soil from the
  preserved simulator state; training explicitly infers it from the enforced
  mass ledger because terminal states auto-reset.  Both use required negative
  target volume as the denominator.
- The fixed evaluator validates shapes, finiteness, ranges, measured load, soil
  partition conservation, and stall-age count/fraction consistency before it
  writes JSON.  Malformed or nonfinite scientific receipts fail closed.
- The historical absolute-completion comparison screen is now labeled
  `advisory_diagnostics_only`; its `passed` field is not a promotion decision.
  This workstream promotes only on exact-success gain plus failure-mechanism
  review.

Fifty-nine focused aggregate, evaluator, fixed-bank, W&B, and sampler tests
pass against paired Terra `c2d2a94a`; Python compilation and diff checks also
pass.  The matched d16 action trace described below also passes its full-panel
parity and mechanism checks.

### Traversability result

The targeted audit is complete on pinned Terra `c2d2a94a`.  It exactly replayed
5,400 archived action effects and enumerated 13,024 unloaded base-action
counterfactuals.  Within these deliberately selected 12 rollouts, not as a
720-panel prevalence estimate, the old concern that sparse height-one soil was
routinely shown as blocked is not supported: only 17 counterfactuals (0.13%)
were visible-false-blocked, and the archived policy selected one of them,
which succeeded.  The still-parallel synthetic partial-reset bank has not
been audited by this result.

The audit instead found 802 visible-false-free actions, all in promotion slot
300.  The policy selected 399 of those no-ops; every selected base action from
steps 53--449 was visibly free but physically blocked.  An effective `DO` had
dug three holes beneath the current base footprint.  The wrapper subsequently
hid those holes with its agent overlay, while movement physics continued to
treat them as blockers.  Slot 338 had no such mismatch and remains a separate
planning/progress failure.

This supports a future fresh-environment fix in the following order: prevent
dig/relift under the exact base footprint, permit only collision-reducing
escape actions from a pre-existing overlap, then align the visible terrain
mask with the physical predicate without erasing underlying blockers.  It does
not support action masking or broadly weakening collision checks.

Artifacts:

- `.artifacts/terra_v8_v61_failure_audit_20260813/traversability_audit_v1/receipt.json`
  (`244c4d24ab8a511cb7b1fdc1a0f6014884f632438a2fcea8222d3bb74c5a7891`);
- `.artifacts/terra_v8_v61_failure_audit_20260813/traversability_audit_v1/INTERPRETATION.md`;
- `.artifacts/terra_v8_v61_failure_audit_20260813/traversability_audit_v1/selected_policy_mismatches.json`;
- `.artifacts/terra_v8_v61_failure_audit_20260813/traversability_audit_v1/mismatches.jsonl`.

### Continuation

Job `10625259` completed with Slurm exit code zero after `22:48:04`.  The
immutable final checkpoint is
`v8_v61_stall_age_v3_u40000_FINAL_17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8.pkl`
(SHA-256
`17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8`).
The local durable copy, run contract, and Slurm log are under
`.artifacts/terra_v61_stall_age_continuation_20260814/final_u40000/`.

The exact u40 reproduction boundary is immutable and intentionally excludes
all later diagnostics, mechanics, and partial-reset work:

- terra-baselines source:
  `dddc691c93ee21488cd7eeb8e01b067bf1f9733c`;
- Terra runtime source:
  `c2d2a94a124759e9f21c2b37930f717e299f0c46`;
- final checkpoint SHA-256:
  `17cbd702f8b7558fb91538debcefac6f15f1554ea8ac800b2f213612004fb6d8`;
- checkpoint clocks: `next_update=40000` and optimizer step `2560000`;
- W&B run:
  `v8_v61_stall_age_v3_dddc691c93_phase2_10625259`;
- Slurm job: `10625259`;
- run-contract SHA-256:
  `4954a456a9c6ef5be40146e1993025cbbdb4d879531aa7957d1c9240ea042502`;
  and
- Slurm-log SHA-256:
  `dd4381faa93de53c71f0976b86fb65b0e8ecd686c516c2e7257ebe10fbc0ef04`.

The paired annotated Git tag `v8-v61-stall-age-u40-20260814` points to the
respective source commit in each repository.  The direct one-day continuation
must resume this checkpoint with these exact sources.  It must not use Terra
`88c0099e`, `30ad500f`, `67c72d09`, or `794d4759`; those form a separate fresh
environment/curriculum line.

The unchanged extra-day segment is implemented by launcher commit
`bbaebc04c2ddc7c3ae667e434e223e1d01b95f84` and was submitted as Slurm job
`10752100`.  It resumes the same W&B run with `resume=must`; the checkpoint at
u40 is ahead of the last logged `train/update=39991`.  Its absolute target is
u70,000, deliberately beyond the approximately u67k expected to fit in the
23:45 allocation, and it checkpoints every 500 updates.  At the recorded
snapshot the exact 8xRTX-4090 request is `PENDING (Priority)` with no allocated
node and a scheduler-estimated start of 2026-08-15 07:15 CEST, so it has
produced no new training evidence yet.

The final sampler state has 46/47 mastered conditions.  Its only open
condition is `fnd-slab-side1-obj`, which receives the 0.15 cap; the remaining
0.85 is distributed across mastered conditions.  The resulting family masses
are 0.593478 foundation and 0.406522 trench, confirming that final v3 did not
retain the superseded 50/50 family allocation.  Sampler ESS is 26.174.

The prepared u14 checkpoint initialized both 704-dimensional stall-age
embeddings to exact zero.  At u40 their L2 norms are 11.263 for the actor and
12.964 for the critic.  This proves that optimization used the scalar path; it
does not establish that the learned dependence is beneficial.

The independent targeted u40 recurrence probe passed receipt and NPZ
recomputation (`receipt.json` SHA-256
`94c29f5a3384426b3e24ede605540973fdc4393b6698698ee67d48dbe4d6537b`;
arrays SHA-256
`f75cdd7ab980aba949ca9d2e834b720f69cb42ef6493feb9d3f599f91504d877`).
It solves 7/12 selected resets versus 2/12 in the historical u14 targeted
rerun: both controls, all four formerly sampling-rescuable clean/high-dig
failures, and carry slot 100.  Slot 100 is a real relay rather than a direct
shortcut: it stages 28% of the soil off-zone, makes 30 empty/material-neutral
decisions, relifts at stall age 0.9375, and finishes in 127 steps.

All five remaining targeted failures revisit an exact full policy input.  Four
saturate stall age and then enter one fixed point or short cycles; obstacle
recurrence starts at decision 33, immediately after the 32-step cap.  Slot 17
instead alternates dump and relift in a six-action cycle.  Those material
changes continually reset stall age even though terminal progress is flat.
Thus the combined continuation supports useful relay behavior and converts
the easier attractor class, while stall age itself is neither an unbounded
repetition counter nor a progress measure.  This is combined stall-age, v3,
and 26k-update evidence, not component attribution or population-level
evidence.

The authoritative promotion-panel readout is complete under one matched
Supercluster RTX-3060 inference contract: forward chunk 120 and
`--xla_gpu_enable_cudnn_frontend=false`.  This is a new matched u14/u39/u40
contract rather than a bit-identical replay of the old backend, but the u14
readout independently reproduces the historical exact headline of `407/720`.
Reset, identity, mutation, nonfinite, exact-completion, material-partition,
and stall-age integrity gates pass for all three 720-map records.

| Checkpoint | Exact | Foundation | Trench | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| u14 | 407/720 | 115/384 | 292/336 | 313 |
| u39 | 651/720 | 322/384 | 329/336 | 69 |
| u40 | 657/720 | 324/384 | 333/336 | 63 |

From u14 to u40, 254 maps convert to exact success and four regress, for a net
gain of 250.  The regressions are promotion slots 84
(`fnd-slab-apron-c3x`), 220 (`fnd-slab-ring3x-road`), 232
(`fnd-slab-side1`), and 637 (`v7-trn-dogleg-adjacent`).  The descriptive
diagnostics of the differently composed failure sets move in the same
direction: authoritative no-effect actions fall from 72,245 to 8,831; mean
terminal-soil fraction rises from 0.400 to 0.529; mean loaded soil falls from
0.0112 to 0.0051; and mean off-zone staged soil falls from 0.0420 to 0.0289.

The final thousand updates are not monotone at map level.  From u39 to u40,
38 maps convert and 32 regress, for only +6 net.  In particular, selected
carry slots 17 and 234 solve at u39 but regress at u40.  Therefore u40 is the
best aggregate checkpoint, while u39 must be retained as a scientifically
useful comparator; the +6 headline must not conceal the high late-training
churn or be called a stable per-map improvement.

The u39 checkpoint is preserved locally at
`.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/checkpoints/v8_v61_stall_age_v3_u39000_103e4c8903fa7e16f55e4e4c9df6e925b83212f0206b903c54664850019ae249.pkl`
(SHA-256
`103e4c8903fa7e16f55e4e4c9df6e925b83212f0206b903c54664850019ae249`),
so the comparator is reproducible rather than only a retained JSON row.

At u40, 72/720 episodes reach the stall-age cap for 15,691 decisions.  Among
the 63 failures, 53 saturate the scalar.  The remaining failures still include
all five exact-input recurrent cases found by the targeted probe, so the
aggregate result agrees with the mechanism readout: the continuation removed
most old failures but did not eliminate recurrent policy attractors.

The residual set still supports the parallel relay/cleanup curriculum,
without making it a universal explanation: using a `1e-6` positive-mass
tolerance, 23/63 failures terminate with off-zone or loaded soil, while 10/63
have already excavated at least 95% of the target.
The hardest condition remains `fnd-slab-side1-obj` at 5/16 exact, followed by
`fnd-proc-side1-road` at 8/16.  By contrast, trenches are 333/336 exact.  The
partial-reset arm should therefore remain foundation- and relay/cleanup-focused
and must still be judged on untouched full starts.

The 720-map development-panel comparison also passes every integrity gate:
u14 solves 377/720 and u40 solves 663/720.  Its manifest slot 119
(`fnd-slab-apron-d16`, `curriculum-diverse-320-9150`) is a conversion rather
than a retention control.  At u14 it times out after 450 steps with 47.97%
excavated, 14.19% in the terminal zone, 33.78% staged off-zone, 404 no-effect
actions, and 373 stall-saturated decisions.  At u40 the same reset reaches
exact completion in 132 steps, with eight no-effect actions and no stall
saturation.  The source checkpoint therefore did not already solve this
current development reset.

The core map geometry is byte-identical to the historical successful d16
relay, but the canonical distance sidecar and scenario identifier changed;
the historical 163-step, 17-dump, 9-rehandle trace also came from a different
compact checkpoint.  Fixed JSON establishes the u40 conversion but cannot
show which actions produced it, so a matched full-720, chunk-120 action trace
was run under the same development-panel contract.

That mechanism trace passes.  On u40, the policy makes five off-zone staging
dumps totaling 115 units, five off-zone relifts totaling 115 units, and seven
terminal deposits totaling all 148 source units before exact completion at
step 132.  Its first explicit relay stages 21 units at step 11, aligns the
cabin at steps 12 and 14, makes effectful unloaded base movements at steps 13
and 15, relifts the 21 off-zone units at step 16, and deposits them in the
terminal zone at step 19.  Cabin turns do not count as base relocation.  The
same trace reaches a peak off-zone inventory of 94 units and finishes with no
off-zone or carried soil.

The u14 policy already executes one short stage-relocate-relift primitive at
steps 17--22, so this is not evidence that the continuation invented staging
from nothing.  It stages 71 units but relifts only 21, then times out with 50
units off-zone.  The bounded result is therefore that the combined
continuation learned to close this multistage relay on one canonical d16
development reset; it is not component attribution to stall age or v3, a
retention result, or a population-level relay-success estimate.  The event
classifier follows aggregate off-zone mass through time and does not claim
particle-level soil lineage.

The durable trace artifact is
`.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/d16_trace/`.
Its `comparison.json`, `receipt.json`, `trace_rows.jsonl`, and
`full_panel_action_arrays.npz` hashes are respectively
`6da467c9d891ff8e979e7525fc09dff85dd4502f6c54018d2e1be60676922cc0`,
`ea1dfbde180bca4965ab5357bc96561432ff3d0ccd9d3a8d165ac6f3441cd98c`,
`fd55ede383d1f91e685d549c40ef2b829d8eb37fe276ab3905f4725e9f79f226`,
and `610307430d96dd1e535cf2d5fa4721ca718ae15b0d6d7929d6e5ca85387f09e1`.
The trace has exact full-panel parity with both fixed development receipts;
cross-process action identity to those earlier fixed runs is not claimed.

The fail-closed combined analysis is
`.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/analysis/readout_analysis.json`
(SHA-256
`c3ae10f2df65e2f8a8b075047414363ecfce19ccca01c5e285e993e68d5a9953`).
The development u14 and u40 JSON hashes are respectively
`6189432fea05adb20a208ebcde0e3b3572fe9d184e096605646cf2d32acfa6b5`
and
`ddb006b31ae36d01bffc3e162ea82915909c85266bac9bb84734a500621fdbf2`.

The paired fixed-panel artifact is
`.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/fixed_panel/u39000_u40000_fixed_panel_chunk120_legacy_cudnn.json`
(SHA-256
`0b1c8cca42566259c36369d1fef6c78f0f05b7db2b0064f60cd4af5dab047f3a`).
The matched u14 artifact is
`.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/fixed_panel/u14000_fixed_panel_chunk120_legacy_cudnn.json`
(SHA-256
`cdd9bacb29de7032664acfde77171100bee6011fa921c9fa31a86d1fee313be6`).
The targeted recurrence probe uses its separately declared 8+4 forward split
and remains diagnostic only.  No preliminary training curve is treated as
evaluation evidence.

## Evidence sources

- `ORACLE_TERRA_STAGING_REVIEW_20260814.md`
- `V8_V61_FAILURE_AUDIT_20260813.md`
- `V8_REWARD_TERMINATION_AUDIT.md`
- `V8_IMPROVEMENT_SET_20260810.md`
- `../EXPERIMENTS_RUNNING.md`
