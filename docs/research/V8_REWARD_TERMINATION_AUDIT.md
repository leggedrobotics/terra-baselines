# V8 reward and termination audit

Status: working design and experiment note; no reward or termination change is
approved by this document alone.

Snapshot: 2026-08-10 CEST

Runtime evidence:

- Terra runtime: `eb3835c1d17af81e970b973ed5abf687ca6f3a26`
- training source: `dcc4f955347182e57e6f16e9df81a3f170564d97`
- compact job: `10128518`
- selected checkpoint: update 20,000, SHA-256
  `0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834`
- accepted bank: 47 conditions x 96 training maps, release
  `terra_v8_v6_constraints_v7_adjacent_train96_v5`

Job `10128518` completed `0:0` after 20,000 updates and all four fixed-panel
sweeps. The pinned result files are:

- main development:
  `/cluster/work/rsl/lterenzi/terra_v8_architecture_control_v1/dcc4f955347182e57e6f16e9df81a3f170564d97/screen/all47/s20260807/compact_xattn/eval/development.json`,
  SHA-256 `dd8c3b381e57889827462222c81f29003a8b19f6285abd87247db5e60a2fea26`;
- capability development: the adjacent `capability_development.json`, SHA-256
  `8b9733b6d542f851d141803ec8bdaef7e4ef2db939a0e4bcb3fcad663e6df6be`.
- distance-field metadata:
  `/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2/train/006__fnd-slab-apron-d12/dataset.json`,
  SHA-256 `2e6fbb2a18a46b11f56130abb8664a2ae2179ba9da781181e905c29e72700e15`;
  it pins `8_connected_cardinal_1_diagonal_sqrt2` and
  `per_map_max_to_1` for the 96-slot `v6_train96` lineage.

The per-identity counts below were derived read-only from the main file. D0
must materialize a machine-readable analysis receipt before they are reused in
a paper table.

This is the shared note for reward, completion, termination, and closely
related failure diagnostics. Agents should add evidence before changing the
decision table. Reward, horizon, action masking, map sampling, and environment
mechanics are separate causal variables.

## Executive decision

The strict V8 success rule should remain strict. The compact policy's remaining
failures do not justify accepting soil outside the visible dump mask, leaving a
bucket loaded, or relaxing complete excavation.

The main semantic problem is instead that `absolute_completion` serves two
incompatible roles:

1. a strict success predicate; and
2. a continuous progress score used by dashboards and the dense timeout bonus.

Because `unloaded_completion` is binary and participates in a minimum, a
nearly completed episode ending with any load reports zero completion. Exact
failure is correct; zero graded progress is not.

The minimum safe direction is therefore:

1. preserve exact boolean success and the 450-step benchmark;
2. report material progress separately from success;
3. diagnose horizon, obstacle-action, and dump-capacity failures before PPO;
4. run at most one matched dense-versus-annealed reward screen from the same
   compact checkpoint; and
5. prototype a simpler normalized material-potential reward only if the
   transition traces support it.

Sparse reward is an ablation, not the presumed repair. It removes the existing
relocation signal that specifically supports distant dumping.

## Current evidence

The update-20,000 compact checkpoint was selected on promotion. On the
source-disjoint main development panel it scores:

| Slice | Exact | Macro graded completion |
|---|---:|---:|
| all 45 non-anchor conditions | 546/720 | 0.861 |
| foundations | 221/384 | 0.756 |
| trenches | 325/336 | 0.981 |

All 174 main-development failures are 450-step timeouts. Of these, 163 are
foundations. The all-free anchors score 31/32 on both promotion and
development; the latter is foundation 15/16 and trench 16/16 with macro
completion 0.977. Basic excavation and unconstrained dumping are therefore not
the dominant limitation.

The observed failures separate into different mechanisms:

| Failure class | Evidence | Most plausible first lever |
|---|---|---|
| distant dumping | `d12` 3/16 and `d16` 1/16, despite ordinary 124--242-cell excavation volume | longer-horizon inference and relocation traces |
| obstacle conjunctions | seven extreme no-effect stalls, concentrated in side-one-plus-object and ring-plus-object maps | action feasibility and workspace geometry |
| large foundations | every one of the 18 maps above 500 required cells fails; many continue productive work | work budget, horizon, and volume efficiency |
| endgame cleanup | 29 failures retain illegal soil and 30 end loaded, with overlap | decomposed material progress and D5 capacity traces |
| partial clean timeout | 107 failures are clean and unloaded but only partially excavated | planning, workload, and generalization |

No-effect actions are not the general explanation: 153 of 174 failures have
zero no-effect actions. Similarly, a longer horizon may help large and
near-finished maps but does not by itself explain low-completion distant maps.

All four fixed sweeps are complete and integrity-clean.

## Effective V8 contract

The current run uses one solo tracked excavator, horizon 450, full resets,
dense reward, no trench-specific shaping, and no foundation-edge alignment
requirement. The continuous map sampler changes exposure but not reward,
horizon, action, observation, or dynamics.

At the pinned revisions, the primary Terra source anchors are
`terra/state.py:1969` (accepted mask), `terra/state.py:2332` (dump transition),
`terra/state.py:3125` (reward assembly), and `terra/state.py:3373` (completion
components). `terra/env.py:339` and `terra/env.py:462` expose the auto-reset and
no-reset step paths. These anchors are navigation aids, not substitutes for
the frozen source hashes above.

### Exact dump and success

The accepted dump mask is the visible `target_map > 0` region excluding
obstacles. It is the acceptance/accounting mask, not the complete physical
action mask: reach, dumpability, traversability, and workspace exclusions also
constrain a dump. A solo excavator can use off-zone staging when no accepted
cells are reachable, but every off-zone pile must be removed before success.

Current exact success is the minimum of:

- a nonempty valid task;
- total required excavation completion;
- optional edge completion, disabled in V8;
- dump purity;
- accepted dump volume divided by required dig volume;
- every active carrier unloaded; and
- static dump-mask integrity.

`task_done` is true at `absolute_completion >= 1 - 1e-6`. An episode ends on
`task_done` or at 450 steps. Transition mass conservation and fixed-evaluation
map/reset integrity are separately checked.

This exact contract is coherent and should not be loosened.

### Dense reward

For solo V8, action, terminal, and existence terms are summed and divided by
70. Representative nonterminal values are:

| Transition | Normalized reward |
|---|---:|
| any productive fresh dig | +0.01071 |
| no-effect dig | -0.00529 |
| ordinary move | -0.00500 |
| colliding move | -0.00786 |
| base turn | -0.00429 |
| cabin turn | -0.00386 |
| explicit no-op | -0.00357 |
| failed dump | -0.01786 |

A productive dig receives a hard-coded `+1` whether it moves two units or a
full workspace. The configured `Rewards.dig_correct` value is not read.

A successful dump receives signed relocation progress:

```text
carry credit
+ off-zone distance potential before
- off-zone distance potential after
```

multiplied by `1.5` and a clipped target-size scale. This permits useful
staging: moving a pile closer is positive, moving it farther is negative, and a
closed lift/dump shuttle does not create net undiscounted relocation progress.
The frozen V8 bank uses 8-connected obstacle-aware geodesic distance normalized
by each map's finite maximum. For a 52-unit median-source load, the approximate
full source-to-zone normalized reward is about `+0.35` at `d12` and `+0.42` at
`d16`; mean-source estimates are about `+0.49` and `+0.52`. Exact credit is
cell-dependent and may be realized across several staging dumps; each dump
earns only its realized distance reduction. Distant maps therefore do not
simply lack positive hauling reward; the credit is delayed until a useful dump
is realized. These descriptive estimates use all 96 training arrays per cell
and `52 * 1.5 / 70` times the per-map median or mean target-cell distance; D4
must record actual transition credit before the numbers are used as a result.

Exact success gives a solo dense terminal component of approximately `6.857`.
A timeout instead receives approximately
`0.571 * absolute_completion^2`. A loaded ending has
`absolute_completion == 0`, so it loses all partial timeout credit regardless
of its material progress.

### Terminal objective and annealing

The supported terminal objective is not the legacy `Rewards.sparse()` preset.
It gives:

- `0` on nonterminal transitions;
- `-1` on every timeout; and
- approximately `5.714` to `6.857` on exact success, using soft
  productive-workspace and step-efficiency bonuses.

The annealed mode linearly mixes the complete dense and terminal objectives.
At its terminal endpoint it removes relocation shaping, invalid-action costs,
and every other nonterminal signal. It also assigns the same `-1` to a
zero-progress failure and a 99%-complete loaded failure.

## Findings

### Keep

1. One accepted visible dump mask is the shared acceptance/accounting mask for
   classifying action outcomes, reward accounting, completion, termination,
   and evaluation. Physical action feasibility remains stricter.
2. Strict cleanup, unloading, and mass conservation.
3. Foundation/trench-neutral material relocation shaping. V8 did not use
   trench alignment shaping.
4. Source-disjoint fixed evaluation for checkpoint selection; online
   sampler-weighted success is not the benchmark.

### Fix or separate

1. **Strict success is not graded progress.** A binary loaded flag collapses
   the scalar to zero. Keep the boolean predicate and expose a separate progress
   vector.
2. **Fresh-dig reward is action-count based.** A small productive scoop earns
   the same `+1` as a large one. This can reward fragmented work on large
   foundations.
3. **Two reward APIs conflict.** Per-level `DENSE/SPARSE` and global
   `dense_skill/annealed_objective/terminal_objective` coexist. Legacy
   `Rewards.sparse()` still pays the hard-coded productive-dig reward and must
   not be called sparse in V8.
4. **Static validity is mixed into progress.** Dump-mask integrity belongs to
   fail-loud bank/reset admission. Edge and inner completion are useful
   diagnostics but are redundant in generic exact termination when total dig
   completion is already one; edge enforcement is disabled here.
5. **Reachable dump capacity has a confirmed control-flow limitation.** If any
   legal dump cell is reachable but that legal subset cannot contain the
   complete load, the dump no-ops instead of attempting allowed off-zone
   staging. Global map capacity validation does not prove capacity of every
   later workspace cone. Whether this occurs in a real failure, and whether
   fallback is preferable to repositioning, remain unverified until D5. Any
   future change must preserve accepted-first, mass-conserving two-pass
   semantics rather than blindly spilling whenever a legal deposit fails.
6. **Obstacle stalls are mechanics/action issues.** Any obstacle intersecting
   the whole dig cone vetoes the entire dig. The runtime computes an action mask
   helper but emits an all-zero informational mask to the policy. Do not hide
   this with obstacle-specific reward bonuses.
7. **Success and timeout share one PPO `done`.** GAE bootstraps neither. Keep
   the present within-450 objective for the frozen benchmark, but represent
   `task_done` and time-limit truncation explicitly before reusing the trainer
   for a different horizon interpretation.
8. **Distance shaping is obstacle-aware but still delayed.** The frozen V8 bank
   pins 8-connected geodesic distance with per-map finite-max normalization.
   This improves the destination potential around obstacles, but it does not
   expose valid immediate actions or pay intermediate loaded movement.

## Proposed semantic separation

### Exact success

Keep the present exact result while testing a clearer equivalent form for
mass-conserving, full-reset V8 tasks:

```text
valid admitted task
AND remaining required excavation == 0
AND accepted dump volume >= required volume
AND illegal positive soil == 0
AND every active carrier is empty
```

Do not switch implementations until constructed edge cases and the full fixed
bank prove bit-identical success decisions. Relocation-only and partial-reset
tasks require their own explicit contract rather than hidden branches in V8.

### Continuous progress

Expose, at minimum:

```text
dig_fraction
accepted_volume_fraction
illegal_volume_fraction
loaded_volume_fraction
```

A simple dashboard scalar may be
`work_progress = 0.5 * dig_fraction + 0.5 * accepted_volume_fraction`, but it
must never be named success or used alone for checkpoint promotion. Illegal and
loaded fractions remain first-class columns. First recompute this vector on the
174 failures. Keep the scalar dashboard-only initially: substituting it into
timeout credit without explicit loaded/illegal treatment could reward a dirty
or loaded timeout.

### Candidate simplified material reward

The current dense reward should not gain more topology-specific knobs. If the
diagnostics support a redesign, the candidate is one normalized material
potential:

```text
U(s) = remaining required excavation / required excavation

H(s) = (
    sum_over_cells(remaining required depth * dump distance)
  + sum_over_cells(off-zone positive soil * dump distance)
  + sum_over_active_carriers(relocation credit)
) / max(reset-state haul work, epsilon)

r_t = (1 - sparse_mix) * [U(s_t) - U(s_t+1) + H(s_t) - H(s_t+1)]
      - step_cost
      + success_bonus * exact_success_transition
```

Reject a degenerate zero-work task at admission. Exact constants are treatment
parameters and must be frozen only after D4, before PPO.

This form has the desired qualitative properties:

- excavation credit scales with volume rather than action count;
- lifting transfers haul work into carrier credit instead of creating reward;
- closer placement is positive and farther placement is negative;
- relift/redump cycles telescope in the undiscounted sum rather than paying
  repeatedly;
- map work and distance are normalized;
- exact cleanup and unloading remain mandatory; and
- increasing `sparse_mix` removes progress shaping while retaining an exact
  success objective and a small step-efficiency pressure.

This is a candidate, not an implemented or accepted replacement. Initially do
not add workspace, edge, family, obstacle, or per-condition bonuses. Because
PPO uses `gamma < 1`, either use gamma-consistent potential shaping or add a
discounted-cycle regression test; undiscounted telescoping alone is not enough.

## Minimum experiment plan

Diagnostics use the selected compact update-20,000 checkpoint and the frozen
development identities. They do not change training.

| ID | Treatment | Decision it supports |
|---|---|---|
| D0 | reproduce 546/720 exact and materialize a hashed per-identity analysis receipt | integrity baseline; fixed evaluator complete, derived receipt pending |
| D1 | rerun the 174 failed identities with horizon 900; require an identical deterministic 450-action prefix | distinguish budget-limited from policy-limited failures |
| D2 | categorical action sampling at temperature 1 with eight frozen action seeds on the same failures | distinguish greedy attractors from missing capability |
| D3 | oracle immediate-effect masking on the seven severe obstacle-loop cases and matched successful obstacle cases | upper-bound action-feasibility benefit and guard regressions; not a deployable mask result |
| D4 | log per-action fresh volume, relocation progress, remaining potential, no-effect streak, last-progress step, load, and illegal soil | test the flat-dig and delayed-haul hypotheses |
| D5 | constructed saturated-legal-cone dump test, followed by inference on any matching failures | verify the control-flow risk and decide between repositioning and accepted-first two-pass fallback |

After D0 parity and D4 trace integrity, run at most one initial PPO reward
screen. D1--D3 may run in parallel for interpretation. D5 is an independent
mechanics follow-up and does not block the reward screen.

| Arm | Common parent and budget | Reward |
|---|---|---|
| R1-control | exact same compact checkpoint, optimizer, sampler state, bank, horizon, seed, and 6,000-update budget | continue current dense reward |
| R1-treatment | same | linearly fade current dense to the implemented terminal objective for 5,000 updates, then 1,000 terminal-only updates |

R1 answers only whether removing dense shaping after the current competence
level improves fixed exact completion and successful-episode efficiency. It is
not expected to repair distant hauling, and one seed is only a screen. The two
children are a matched statistical fork, not a trajectory-identical
counterfactual: environment state, action history, execution RNG, and GPU
numerics restart or diverge. Propose replacing the old online-depth trigger
with an explicit fixed-evaluated parent receipt; the authoritative trigger
remains unchanged until that amendment is explicitly accepted and
implemented. The map curriculum has continuous bands, not external stages.

Use fixed promotion and development exact counts, macro completion, family and
condition tails, and all-free retention. Compare steps and productive
workspace cycles only on identities solved by both policies. Never compare raw
return between reward schemes. A material screen effect requires at least
three paired resume seeds, with promotion used for checkpoint selection and
development only for confirmation, before a paper claim.

Only if D4 confirms fragmented low-volume digging should a later R2 compare
the candidate normalized material reward against the same dense control. Do
not combine R2 with a horizon, sampler, action-mask, map-bank, or dynamics
change.

## Decision table

| Item | Decision | Rationale |
|---|---|---|
| exact visible dump mask | frozen | matches the accepted task contract |
| cleanup and unloaded final state | frozen | current failures should remain failures |
| 450-step fixed benchmark | frozen pending D1 | changing it now breaks comparability |
| topology- or condition-specific reward | rejected | hides the mechanism and scales poorly |
| legacy `Rewards.sparse()` for V8 | rejected | it is not terminal-only and has misleading controls |
| continuous material-progress vector | accepted for diagnostics | fixes interpretation without changing success |
| dump reachable-capacity fallback | confirmed code-path risk; behavioral decision pending D5 | may create localized endgame no-ops |
| dense-to-terminal R1 | proposed exploratory matched screen after D0/D4 | directly answers the requested sparse-reward question |
| normalized material reward R2 | conditional | run only if traces support the mechanism |
| obstacle action masking | separate treatment | action feasibility is not reward design |
| workload-aware horizon | separate treatment | requires D1 and benchmark-policy discussion |

## Claim and implementation boundaries

- This audit changed no environment, trainer, bank, or checkpoint, and launched
  or modified no job.
- A successful unit test or update-1 smoke is implementation evidence, not a
  reward result.
- A one-seed R1 result is a screen, not a paper-level causal estimate.
- Promotion selects checkpoints; development confirms. Sealed evaluation is
  used only after treatment selection.
- Reward return is not comparable across dense and terminal objectives.
- Efficiency comparisons are restricted to jointly successful identities.
- Fixing dump fallback, action masking, horizon, or reward in one arm would make
  attribution impossible.

## Collaboration log

The 2026-08-10 audit used three independent passes:

- reward semantics and scale audit;
- exact completion, termination, mass, and reset audit; and
- failure-to-experiment synthesis against the compact fixed evaluation.

Append future changes here with date, revision, evidence artifact, and whether
the entry is an observation, hypothesis, accepted decision, or superseded
decision.

| Date | Kind | Entry |
|---|---|---|
| 2026-08-10 | accepted | preserve exact termination and separate continuous material progress |
| 2026-08-10 | observation | compact u20 failures split across remote planning, obstacle stalls, high workload, and endgame state; no single reward cause is supported |
| 2026-08-10 | defect | legacy sparse productive-dig semantics are inconsistent with the sparse name |
| 2026-08-10 | risk | reachable-legal-cone dump fallback is a confirmed control-flow limitation; behavioral relevance awaits D5 |
| 2026-08-10 | experiment | complete D0/D4 before the single matched R1 reward screen; run D1--D3 and D5 as separate diagnostics |
