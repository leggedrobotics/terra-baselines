# V8 failure-remediation execution

Date: 2026-08-14

Active Codex goal: `01a000fe-f5b4-7380-8df1-4ea8f2e343f1`

Goal objective: complete the non-overlapping Terra V8 failure-remediation
workstream: repair canonical evaluation no-effect accounting, expose a clear
dig/terminal/off-zone/load progress vector, perform an evidence-backed
traversability-observation versus movement-physics incidence audit, and
evaluate the stall-age plus Continuous Banded v3 continuation when its final
checkpoint is available.

## Scope boundary

This work does not edit trench generation, trench excavation ordering, or the
partial-reset generator. Those are being developed in parallel. It also does
not mutate the allocated continuation, change reward-v2, add action masking,
or bundle a new policy observation into the partial-reset experiment.

The parallel partial-reset implementation is owned at
`/home/lorenzo/moleworks/.worktrees/terra_v8_relay_corridor_20260814`. This
workstream must not edit, format, clean, commit, or otherwise mutate that
worktree. A later audit may consume a completed immutable dataset from it, but
must not import uncommitted source as evaluation authority.

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

Acceptance: the receipt identifies the pinned Terra revision, evaluated state
source, transition count, mismatch counts, and whether the mismatch reaches
states used by the partial-reset curriculum. No observation or dynamics change
is authorized by the audit alone.

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

## Conditional next treatments

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

Focused aggregate, evaluator, fixed-bank, W&B, and sampler tests pass.  The
final combined validation and commit remain pending until the continuation
readout is complete.

### Traversability result

The targeted audit is complete on pinned Terra `c2d2a94a`.  It exactly replayed
5,400 archived action effects and enumerated 13,024 unloaded base-action
counterfactuals.  The old concern that sparse height-one soil was routinely
shown as blocked is not supported: only 17 counterfactuals (0.13%) were
visible-false-blocked, and the archived policy selected one of them, which
succeeded.

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

Job `10625259` remains the immutable stall-age plus final-v3 capability run.
Its final `u40000` checkpoint and fixed-panel/recurrence readout are still
pending; no preliminary training curve is treated as evaluation evidence.

## Evidence sources

- `ORACLE_TERRA_STAGING_REVIEW_20260814.md`
- `V8_V61_FAILURE_AUDIT_20260813.md`
- `V8_REWARD_TERMINATION_AUDIT.md`
- `V8_IMPROVEMENT_SET_20260810.md`
- `../EXPERIMENTS_RUNNING.md`
