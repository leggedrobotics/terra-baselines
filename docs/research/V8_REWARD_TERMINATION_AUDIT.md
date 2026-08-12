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

The focused relocation replay below used the selected checkpoint and the exact
720-environment Python-loop evaluation graph. All nine selected traces matched
the frozen development result on steps, exact success, material completion,
load, illegal soil, and no-effect counts. Smaller batches and a fused
`lax.scan` changed some greedy bfloat16 actions, so evaluator batch/graph shape
is part of the deterministic replay protocol. The replay was read-only and did
not produce a durable receipt; D4a must materialize one before paper use.

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
3. expose time-to-go for a genuinely finite-horizon policy, as a separate
   observation treatment;
4. replace action-count and per-map-scaled dense rewards with one normalized
   material potential, first as a matched screen with the same carry-state
   observation repair in both arms;
5. diagnose obstacle-action and dump-capacity failures separately; and
6. fade only the selected dense shaping after it reliably supports exact
   completion.

Sparse reward is an ablation, not the presumed repair. It removes the existing
relocation signal that specifically supports distant dumping.

## Objective hierarchy

The long-term project objective is a global policy that can navigate,
excavate, relocate soil, finish precisely, and dump legally across arbitrary
feasible foundation and trench scenes within a declared admitted protocol
envelope. Reward is only a learning aid for that objective. Keep the following
hierarchy explicit:

1. **Admission:** the task is nonempty, physically representable, has sufficient
   dump capacity, and satisfies static map integrity.
2. **Exact task correctness:** all required excavation is complete, all
   displaced task soil is in the accepted mask, and every carrier is empty.
3. **Fixed-budget evaluation:** exact completion within 450 steps is primary;
   family/cell tails and all-free retention prevent pooled scores from hiding a
   failed capability.
4. **Continuous diagnostics:** dig, accepted, illegal, loaded, and remaining
   haul fractions explain failures but never redefine success.
5. **Dense credit assignment:** normalized excavation and relocation progress
   help PPO learn long action sequences without geometry-specific bonuses.
6. **Efficiency:** steps, workspace cycles, and rehandling are compared only
   among exact successes; they must not compensate for a failed task.
7. **Reward curriculum:** once the dense recipe is selected, fade only its
   physical shaping toward the exact objective.

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

### Focused relocation replay

The current carrier-credit ledger is algebraically sound. On all nine replayed
traces, every lift conserved the remaining-haul quantity `H` within
`7.6e-6`, and cumulative dump relocation matched `H_reset - H_terminal`
within the same tolerance.

| Trace | Result | Relocation evidence |
|---|---:|---|
| nearby foundation | success in 82 steps | 11 direct dumps; `H` 33.08 -> 0; relocation return 0.709 |
| `d12` foundation | success in 172 steps | 11 direct dumps; `H` 86.52 -> 0; relocation return 1.854 |
| `d16` foundation | success in 163 steps | 17 dumps, 9 rehandles, 5 off-zone dumps; `H` 87.13 -> 0; return 1.867 |
| `d12` illegal-soil failure | timeout | `H` 59.73 -> 62.97; undiscounted relocation -0.0695 but PPO-discounted relocation +0.0079 |
| `d16` near-finished loop | timeout | 45 rehandles, 39 zero-progress dumps; the last 100 steps repeatedly move four units while `H/H_reset` stays 0.0689 |

The successful `d16` trace contains one negative and two zero-progress dumps.
Therefore a blanket rehandling, negative-dump, or zero-dump penalty would reject
valid plans. Conversely, the failed `d16` loop has frozen
`no_effect_action_count == 0` because every lift and redump changes physical
state even when material work is unchanged. Add material-neutral-cycle and
longest-`H`-stagnation diagnostics; no-effect alone is not a loop detector.

Two other failures show the boundary of reward repair: one `d12` episode ends
fully dug, clean, and 96.4% dumped while carrying five units; one `d16` episode
moves for 450 steps without touching soil. More relocation magnitude alone
cannot solve either policy-selection failure.

### Relocation scale and observation defects

Across all 4,512 V8 training maps, the projected total normalized relocation
return under the current scale ranges from 0.0527 to 15.34, a 291x spread.
Foundation median is 0.959, trench median is 0.108, and 34 maps have more
available relocation return than the approximately 6.857 exact-success bonus.
This is a reward-budget mismatch, not meaningful task difficulty.

A read-only overlap query reproduced all of those aggregates and classified
the 34 maps. Contrary to the motivating hypothesis, none is `d12`, `d16`, or
any apron map. All are depth-1 `adjacent_generous` foundations:

| Condition | Above 6.857 | Budget range | Required-volume range |
|---|---:|---:|---:|
| `v7-fnd-slab-adjacent` | 19/96 | 6.926--15.336 | 511--792 |
| `v7-fnd-bearing-walls-adjacent` | 9/96 | 6.957--9.421 | 395--518 |
| `v7-fnd-irregular-adjacent` | 3/96 | 7.067--8.298 | 530--595 |
| `v7-fnd-courtyard-pads-adjacent` | 2/96 | 7.064--7.728 | 404--441 |
| `v7-fnd-courtyard-adjacent` | 1/96 | 6.957 | 397 |

`d12` has median/p95/max projected budgets `1.541/2.976/4.388`; `d16`
has `1.585/2.936/3.816`. The defect therefore connects most directly to the
large-foundation workload class: current shaping can rival finishing because
it multiplies excavation volume by a per-map-normalized distance field. It
does not explain the remote failures by excessive reward magnitude. Trenches
receive about nine times less median relocation budget yet have much higher
fixed-policy macro graded completion (`0.981` versus foundation `0.756`); this
is confounded by task difficulty but rejects any simple "more shaping mass
means more success" narrative.

The provisional read-only query used
`H_reset * 1.5 * clip(170 / max(negative_target_cells, 1), 2, 5) / 2 / 70`,
where `H_reset` is the current per-map-normalized initial haul ledger,
and the threshold `200 * 1.2 * 2 / 70`. Under that ad-hoc serialization, the
sorted 34-map ID set has SHA-256
`658a5abb657f5dd73b1bedbae6026b004eadd3c014e48f251764082995a781a2`;
this is not yet a durable receipt. D4b must materialize the full per-map rows,
serialization, and input provenance before using this as a paper figure or
freezing constants.

The per-map finite-maximum distance normalization also destroys physical
comparability. A roughly 1.09-tile source distance is stored near 0.774 for an
all-free straight trench but near 0.034 for an adjacent straight trench, about
23x different credit for the same physical transport. Irrelevant far free
space can rescale every relocation reward on a map.

Finally, `carry_relocation_credit` changes the next dump reward but is absent
from the policy observation. Two states with byte-identical observations, the
same eight-unit load, and the same accepted dump produced normalized rewards
0.425 and 0.211 solely because hidden carry credit differed. Retaining the
ledger therefore requires exposing normalized carry work, or explicitly
accepting a partially observed reward.

## Effective V8 contract

The current run uses one solo tracked excavator, horizon 450, full resets,
dense reward, no trench-specific shaping, and no foundation-edge alignment
requirement. The continuous map sampler changes exposure but not reward,
horizon, action, observation, or dynamics.

The 450-step cutoff is part of the benchmark objective, not merely a rollout
collector limit. GAE therefore correctly zero-bootstraps both exact success and
horizon failure, while ordinary 32-step PPO rollout boundaries bootstrap from
`last_val`. The corresponding state representation is incomplete, however:
neither `env_steps` nor normalized time-to-go reaches the model. Identical
physical states near step 1 and step 449 can require different decisions but
are aliased in the observation.

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
The focused replay verifies that accounting, including necessary staging and
rehandling. The defects are instead the clipped target-cell-count scale,
per-map distance normalization, hidden carry credit, and gamma-inconsistent
timing. The `d12` regression trace demonstrates that undiscounted telescoping
alone is insufficient: early improvement followed by later regression can
still yield positive discounted relocation return.

Valid actions also have unequal costs on top of the common existence cost.
No-op is cheaper than cabin rotation, base rotation, and navigation. When the
policy believes completion is unlikely, the reward therefore makes waiting or
turning locally cheaper than exploring a distant workspace. One uniform step
cost is a simpler expression of efficiency.

Digging positive soil from an accepted dump cell incurs an additional
`-1.2 * lifted_volume` penalty. Arbitrary constrained sites can legitimately
require in-zone rearrangement; signed material potential plus elapsed steps
already makes unnecessary rehandling costly.

Exact success gives a solo dense terminal component of approximately `6.857`.
A timeout instead receives approximately
`0.571 * absolute_completion^2`. A loaded ending has
`absolute_completion == 0`, so it loses all partial timeout credit regardless
of its material progress.

### Terminal objective and annealing

The supported terminal objective is not the legacy `Rewards.sparse()` preset.
It gives:

- `0` on nonterminal transitions;
- `-1` on horizon termination without exact success; and
- approximately `5.714` to `6.857` on exact success, using soft
  productive-workspace and step-efficiency bonuses.

Exact success takes precedence when action 450 simultaneously sets
`task_done` and the horizon flag.

The annealed mode linearly mixes the complete dense and terminal objectives.
At its terminal endpoint it removes relocation shaping, invalid-action costs,
and every other nonterminal signal. It also assigns the same `-1` to a
zero-progress failure and a 99%-complete loaded failure.

The dense and terminal endpoints also disagree about workspace efficiency:
the dense `+1` fresh-dig event favors splitting a fixed volume across more
productive scoops, while the terminal workspace bonus favors fewer loading
cycles. Its lower bound `ceil(required_volume / 52)` also ignores geometry and
necessary rehandling. Step count and workspace cycles are better retained as
secondary evaluation metrics until exact completion is reliable.

Finally, `_calculate_terminal_reward` contains a threshold and quadratic
completion curve, but it is called only after `done_task` is already exact.
The partial-completion branch is therefore dead complexity for success; the
dense success component is effectively a fixed bonus.

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
3. **Action costs are anti-navigation.** A no-op is cheaper than every valid
   motion or rotation. Replace the six action-specific costs with one uniform
   step cost in reward-v2; do not add a generic invalid-action knob until the
   uniform treatment is measured. This removes the reward-only collision
   surcharge, not collision mechanics; keep collision counts as diagnostics
   and deployment safety in action feasibility/mechanics.
4. **Accepted-zone rehandling is over-penalized.** The explicit volume penalty
   can oppose necessary rearrangement in constrained zones. Let signed material
   potential and time cost judge the complete sequence.
5. **Relocation scale is not a task invariant.** The target-cell-count heuristic
   and per-map distance maximum create a 291x reward-budget spread and physically
   incomparable maps. Use required volume and one protocol-wide physical
   distance scale.
6. **Reward-bearing state is hidden.** Carry credit changes dump reward but is
   absent from the observation. The distance sidecar must also be a canonical,
   validator-recomputed function of visible target/obstacle maps rather than an
   independently trusted reward label.
7. **The finite horizon is hidden.** Zero-bootstrap is correct for the frozen
   within-450 objective, but normalized time-to-go must be observed for a Markov
   finite-horizon policy. Treat this as a separate observation ablation.
8. **Two reward APIs conflict.** Per-level `DENSE/SPARSE` and global
   `dense_skill/annealed_objective/terminal_objective` coexist. Legacy
   `Rewards.sparse()` still pays the hard-coded productive-dig reward and must
   not be called sparse in V8.
9. **Static validity is mixed into progress.** Dump-mask integrity belongs to
   fail-loud bank/reset admission. Edge and inner completion are useful
   diagnostics but are redundant in generic exact termination when total dig
   completion is already one; edge enforcement is disabled here.
10. **Reachable dump capacity has a confirmed control-flow limitation.** If any
   legal dump cell is reachable but that legal subset cannot contain the
   complete load, the dump no-ops instead of attempting allowed off-zone
   staging. Global map capacity validation does not prove capacity of every
   later workspace cone. Whether this occurs in a real failure, and whether
   fallback is preferable to repositioning, remain unverified until D5. Any
   future change must preserve accepted-first, mass-conserving two-pass
   semantics rather than blindly spilling whenever a legal deposit fails.
11. **Obstacle stalls are mechanics/action issues.** Any obstacle intersecting
   the whole dig cone vetoes the entire dig. The runtime computes an action mask
   helper but emits an all-zero informational mask to the policy. Do not hide
   this with obstacle-specific reward bonuses.
12. **Timeout progress and exact success are conflated.** Remove the squared
   `absolute_completion` timeout bonus once normalized material potential is
   available. Keep `task_done` and horizon failure distinct in logging, and do
   not reuse zero-bootstrap semantics if a future horizon is only a training
   truncation.
13. **Distance potential cannot solve pose planning.** It values material
   destination progress but does not expose valid immediate actions or guide a
   policy across a flat-`H` navigation plateau. Pose/access shaping or action
   masking remains a separate treatment.

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

The current dense reward should not gain more topology-specific knobs. The
focused replay now supports testing one family-neutral reward-v2 bundle.

Let `V0` be original required excavation volume. Define one canonical physical
distance field:

```text
D(x) = obstacle-aware geodesic_metres(x, accepted_dump_mask) / D_ref
```

`D_ref` and an admitted dimensionless `D_bound` are frozen benchmark-wide,
never per-map. Do not clip: clipping makes farther placement indistinguishable
after saturation. The validator rejects non-finite fields and maps outside the
release's distance bound; a broader site requires a new protocol version with
a larger bound. Generator, validator, and runtime must use the same
deterministic distance routine.

For full resets, define:

```text
Q(s) = completed required excavation volume / V0

H(s) =
    sum_over_cells(remaining required depth * D)
  + sum_over_cells(off-zone positive soil * D)
  + sum_over_active_carriers(carry relocation credit)

P(s) = (H_reset - H(s)) / V0
Phi(s) = alpha * Q(s) + beta * (P(s) + D_bound)

r_t = B * exact_success_transition
      - F * horizon_failure_transition
      - c_step / 450
      + shaping_weight * [potential_gamma * Phi(s_t+1) - Phi(s_t)]
```

`B` is one flat, map-independent exact-success payment. Reward-v2 does not
reintroduce productive-workspace, step-efficiency, family, or geometry bonuses
inside `B`; those quantities remain evaluation diagnostics. Earlier exact
success is preferred through discounting and the uniform step cost, not by
changing `B`.

`H_reset` is an additive baseline only; it is never the denominator. Dividing
by reset haul work would amplify legal regressions on nearby tasks by as much
as 34.6x and would give trivial and remote jobs the same relocation budget.
Do not clip signed `P`: farther placement must remain negative. Because
`potential_gamma < 1`, reset-centering is not a mathematically neutral constant;
it deliberately makes dwelling after accumulated progress costly.

Under mass conservation and `D in [0, D_bound]`, require
`P in [-D_bound, D_bound]` and `Q in [0, 1]` as fail-loud invariants. The global
`D_bound` shift makes `Phi` nonnegative without clipping regressions; it adds a
uniform dwell cost to the shaping surrogate. `horizon_failure_transition`
explicitly excludes exact success on action 450.

Freeze `potential_gamma` in the reward protocol and assert that it equals PPO
`gamma` in the run contract, checkpoint, smoke, and resume verifier. The
current comparison value is `0.9984`. Do not call this scalar `lambda`, because
`gae_lambda` already names a different quantity.

Apply the potential term on every transition, not only dumps. This fixes the
observed discounted progress-then-regress defect. With nonnegative `Phi`, a
flat material loop cannot earn positive shaping solely from discount timing.
Still require every constructed and replayed closed cycle to have nonpositive
discounted shaping plus step cost.

The first dense reward-v2 screen should retain the physical terminal `Phi` at
both exact success and horizon failure so useful partial work remains a training
signal. This adds map-dependent shaping to successful returns, so offline
trace rescoring alone cannot establish objective dominance. Before PPO, derive
an analytic bound over every admitted `Q`, `P`, terminal type, success step,
and horizon step using `D_bound`, `potential_gamma`, `B`, `F`, and `c_step`.
The minimum possible discounted exact-success return must exceed the maximum
possible discounted horizon-failure return; enumerate the frozen 1--450-step
range rather than relying on observed traces. Trace rescoring remains a sanity
check on that proof. The treatment is intentionally a dense surrogate; do not
claim formal policy invariance. A strict episodic potential-based variant would
set the absorbing terminal potential to zero and should be treated as a
separate experiment.

Anneal only `shaping_weight` from one to zero. Keep `B`, `F`, and the uniform
step cost fixed, so the sparse endpoint is exact completion versus horizon
failure with shorter successful plans preferred. Latch one `shaping_weight`
for each complete episode and advance the schedule only for newly reset
episodes; checkpoint the schedule cursor and each live episode's latched value.
The potential accounting and closed-cycle claims apply only within a fixed
episode weight. R3 is deliberately a small nonstationary reward curriculum
across episodes. Initially add no generic invalid-action penalty: every wasted
action already consumes the same time budget. Workspace cycles, rehandled
volume, and no-effect counts remain diagnostics rather than reward knobs.

Exact constants are treatment parameters and must be frozen after offline
trace rescoring and before PPO. A reasonable screen starts near the existing
scale (`B` about 6, `F` about 1, `alpha=1`, `beta=1.5`, and at most one
cumulative unit of explicit step cost over 450 actions), but this document does
not approve those numbers. `D_bound` and the nonnegative shift must be included
when rescoring these coefficients.

The shifted potential also creates an implicit state-dependent dwell cost,
including a constant-shift component, of
`shaping_weight * (1 - potential_gamma) * Phi` on a flat transition. This
may be the dominant time pressure and must be chosen deliberately. For example,
with the illustrative but unapproved values `potential_gamma=0.9984`,
`alpha=1`, `beta=1.5`, `D_bound=2`, and `shaping_weight=1`, the per-step cost is:

| `P` | `Q=0` | `Q=0.5` | `Q=1` |
|---:|---:|---:|---:|
| `-D_bound` | 0.0000 | 0.0008 | 0.0016 |
| `0` | 0.0048 | 0.0056 | 0.0064 |
| `+D_bound` | 0.0096 | 0.0104 | 0.0112 |

The illustrative explicit step cost `1/450 = 0.00222` is smaller over much of
that grid. The offline receipt must reproduce the grid using the proposed
constants, tabulate the admitted extrema, and include both implicit and
explicit time pressure before any coefficient is frozen.

This form has the desired qualitative properties:

- excavation credit scales with volume rather than action count;
- lifting transfers haul work into carrier credit instead of creating reward;
- closer placement is positive and farther placement is negative;
- at fixed shaping weight, discount timing cannot make a nonnegative-potential
  flat loop profitable; constructed closed cycles remain an acceptance gate;
- map work and physical distance use globally comparable units;
- exact cleanup and unloading remain mandatory; and
- decreasing `shaping_weight` removes physical progress shaping and its
  shifted-potential dwell cost while retaining a fixed exact-success objective
  and explicit uniform step pressure.

This is a candidate, not an implemented or accepted replacement. Initially do
not add workspace, edge, family, obstacle, invalid-action, or per-condition
bonuses.

The first reward-v2 treatment remains full-reset only. For future partial
resets, retain original full-task `V0` but center both terms on the episode
reset state: `q = Q - Q_reset` and `p = (H_reset - H) / V0`. These reset
baselines are history-bearing state and must be observed or otherwise made
Markov before mixing partial resets into this treatment.

R2 uses carry credit inside `Phi` on every transition, so leaving it hidden
would worsen partial observability relative to the control. Both R2 arms must
therefore receive the same added scalar channel with output-preserving
initialization, but its value is the exact reward-bearing carry work for that
arm: the frozen legacy ledger in the control and the globally normalized ledger
in reward-v2. R2 is consequently a reward-plus-reward-state bundle, not a pure
reward-only ablation. The target, dump, obstacle, initial-pose, and slot
identities stay fixed, while each arm's distance protocol and sidecar are
treatment-specific, validated, and hashed. Time-to-go remains the separate D6
treatment. Also prove that each distance field is uniquely derived from its
visible map inputs and frozen protocol.

## Minimum experiment plan

Diagnostics use the selected compact update-20,000 checkpoint and the frozen
development identities. They do not change training.

| ID | Treatment | Decision it supports |
|---|---|---|
| D0 | reproduce 546/720 exact and materialize a hashed per-identity analysis receipt | integrity baseline; fixed evaluator complete, derived receipt pending |
| D1 | rerun the 174 failed identities with horizon 900; require identical actions and physical states through action 450 | distinguish budget-limited from policy-limited failures; terminal reward/flags/reset observation necessarily differ at the boundary |
| D2 | categorical action sampling at temperature 1 with eight frozen action seeds on the same failures | distinguish greedy attractors from missing capability |
| D3 | oracle immediate-effect masking on the seven severe obstacle-loop cases and matched successful obstacle cases | upper-bound action-feasibility benefit and guard regressions; not a deployable mask result |
| D4a | exact full-graph replay of nine targeted traces | complete read-only: ledger correct; scale, hidden-state, discount, and flat-potential defects confirmed; durable receipt pending |
| D4b | log per-action fresh volume, normalized potential, material-neutral cycles, longest stagnation, load, and illegal soil on the full failure set; materialize the 34-map query and the dwell-cost/bound grid | quantify flat-dig and remaining mechanisms; confirm the high-budget overlap with large nearby foundations and freeze shaping constants knowingly |
| D5 | constructed saturated-legal-cone dump test, followed by inference on any matching failures | verify the control-flow risk and decide between repositioning and accepted-first two-pass fallback |
| D6 | append normalized time-to-go with output-preserving initialization | separate finite-horizon observation ablation; do not bundle with reward-v2 |

R2 admission requires D0 parity, a durable D4a replay receipt, a materialized
D4b scale/overlap and dwell-cost receipt, offline reward-v2 rescoring, and the
analytic terminal-dominance proof. D1--D3 and D5--D6 are nonblocking
diagnostics or independent treatments and do not belong in the reward arm.
After those R2 gates pass, run at most one initial PPO reward-repair screen.

R2 pins `continuous_banded_v2` from terra-baselines `60e7510` in both arms.
The selected compact checkpoint contains v1 sampler state. Before creating the
two children, one prepared-fork path must validate its saved probability vector
under v1; preserve competence, mastery, current/closed windows, exposure
counters, refresh state, and NumPy RNG; recompute only probabilities under v2;
and materialize one common migrated sampler state. Both arms restore that exact
state. The run contract records source rule, target rule, migration receipt,
post-migration state hash, and explicitly asserts `settings.rule` rather than
accepting the shared v1 schema name as proof of the active rule. The
reward-plus-ledger bundle is the only arm-level treatment after this common
migration. This is necessary because v1 would leave the depth-2 foundation
targets of the relocation repair at preview-level exposure during a short
screen.

| Arm | Common parent and budget | Reward |
|---|---|---|
| R2-control | same output-preserving carry-input expansion of compact u20, migrated `continuous_banded_v2` state, fresh optimizer, entropy 0.02, map identities, horizon, seed, LR warmup, and 4,000--6,000-update budget | continue current dense reward and its frozen carry ledger |
| R2-reward-v2 | same | normalized `Phi` bundle above, with `shaping_weight=1` |

R2 is deliberately one repair bundle: volume-normalized digging, globally
normalized relocation, gamma-consistent timing, fixed success, and one step
cost. It can determine whether the simpler reward is better, but cannot assign
credit to one constituent change. Add normalized carry work to both policies,
zero-initialize its contribution so the parent outputs are unchanged, and use
the same fresh-optimizer treatment and short frozen LR warmup in both arms.
Each receipt labels the carry channel with that arm's exact ledger and distance
protocol. Run, checkpoint, and evaluation receipts store the protocol ID and
distance-sidecar hash and fail closed on mismatch; observation arrays from
different arms must never be concatenated as if the channel had identical
numerical semantics. The two children are a matched statistical fork, not a
trajectory-identical counterfactual:
environment state, action history, execution RNG, and GPU numerics restart or
diverge.

Only if reward-v2 wins R2 should R3 fork the same selected reward-v2 checkpoint,
optimizer, sampler state, source seed, and map identities into two statistical
resume children. The sampler's NumPy RNG is restored, but JAX rollout RNG,
live environments, and action history restart and then diverge. One child keeps
`shaping_weight=1`; the other follows the episode-latched fade. Only the frozen
weight schedule may differ. This is the clean
sparse-reward question. Resolve D6 before declaring a final fixed-horizon
recipe or running the later scratch confirmation that must show the recipe can
teach the full V8 distribution rather than only preserve or repair an already
competent checkpoint. The previously implemented whole-objective anneal,
formerly called R1, is retired from the mainline and will not be launched: it
simultaneously changes action costs, timeout reward, success scale, and
efficiency bonuses. It may be reconsidered only as nonblocking paper color
after R2/R3, under a separately recorded decision. Use an explicit
fixed-evaluated parent receipt; online sampler depth does not authorize the
transition. The map curriculum has continuous bands, not external stages.

Use fixed promotion and development exact counts, macro completion, family and
condition tails, and all-free retention. Compare steps and productive
workspace cycles only on identities solved by both policies. Never compare raw
return between reward schemes. A material screen effect requires at least
three paired resume seeds, with promotion used for checkpoint selection and
development only for confirmation, before a paper claim.

No sampler change beyond the pinned common R2 migration is allowed, and neither
matched fork may change its sampler. The migration is a pre-fork prerequisite,
not the R2 treatment. Do not combine R2 or R3 with a horizon, action-mask,
time-observation, map-bank, or dynamics change.

Implementation follows `$simple-research-code`: add one named v2 preset and one
named prepared-fork initializer that expands the model, retains absolute PPO
update 20,000 and migrated sampler history, creates a fresh optimizer at local
step zero, freezes entropy at the parent's `0.02` endpoint, and keys one
treatment-local LR warmup from that optimizer-local step before emitting the
two matched children. Existing warm-start drops sampler history and existing
true-resume keeps the old optimizer, so neither is silently repurposed. Add one
common
carry channel, one global distance routine, one potential formula, and one
episode-latched fade. Use only claim-driving tests for the distance/ledger
math, discounted cycles and terminal dominance, v1-to-v2 sampler migration,
output-preserving input expansion, and prepared-fork/latch state. Do not build
a generic reward framework or compatibility matrix. Keep the work as a
reversible experiment commit and revert it if the matched screen loses.

## Decision table

| Item | Decision | Rationale |
|---|---|---|
| exact visible dump mask | frozen | matches the accepted task contract |
| cleanup and unloaded final state | frozen | current failures should remain failures |
| 450-step fixed benchmark | frozen pending D1 | changing it now breaks comparability |
| topology- or condition-specific reward | rejected | hides the mechanism and scales poorly |
| legacy `Rewards.sparse()` for V8 | rejected | it is not terminal-only and has misleading controls |
| continuous material-progress vector | accepted for diagnostics | fixes interpretation without changing success |
| current carrier-credit accounting | retain | exact replay confirms lift/dump/handoff conservation and path accounting |
| reset-haul denominator | rejected | amplifies nearby regressions and erases meaningful distance differences |
| per-map maximum distance normalization | rejected for reward-v2 | physically identical hauling receives incomparable credit |
| globally normalized physical distance | proposed for R2 | one family-neutral unit and bounded reward budget |
| flat productive-dig event reward | replace in R2 | rewards fragmented excavation rather than moved volume |
| action-specific valid-motion costs | replace in R2 | no-op is currently cheaper than navigation; one step cost is simpler and also removes the reward-only collision surcharge while physical collision mechanics remain unchanged |
| accepted-zone relift penalty | remove in R2 | can oppose necessary rearrangement; signed potential and time already price it |
| positive graded timeout bonus | remove in R2 | `absolute_completion` is a strict gate-min and collapses while loaded |
| workspace/step terminal bonuses | diagnostics for now | structural rehandling and gamma already confound these soft bonuses |
| carry-work observation | common channel and output-preserving parent for both R2 arms | values follow each arm's exact hashed ledger, so R2 is explicitly a reward-plus-state bundle |
| terminal objective dominance | analytic admission gate for R2 | bounded exact-success return must exceed every admitted horizon-failure return; trace rescoring alone is insufficient |
| reward fade schedule | latch per episode | prevents mid-episode coefficient changes from invalidating the fixed-weight potential accounting |
| R2 sampler | pin `continuous_banded_v2` in both arms | per-condition graduation prevents v1 from starving the depth-2 maps targeted by reward-v2 |
| optimizer restart | fresh optimizer plus identical short LR warmup in both R2 arms | matched treatment for input expansion and critic-target refitting |
| normalized time-to-go | separate D6 treatment | required for a fully observed fixed-horizon policy, but not a reward ablation |
| dump reachable-capacity fallback | confirmed code-path risk; behavioral decision pending D5 | may create localized endgame no-ops |
| former R1 whole-objective anneal | retired; do not launch on the mainline | changes several causal variables simultaneously and answers a weaker question than R2/R3 |
| normalized material reward R2 | proposed after durable D4a/offline rescore | traces now support testing the mechanism as one repair bundle |
| shaping-only fade R3 | proposed only if reward-v2 wins R2 | clean dense-to-sparse question; keep success and step objective fixed |
| obstacle action masking | separate treatment | action feasibility is not reward design |
| workload-aware horizon | separate treatment | requires D1 and benchmark-policy discussion |

## Claim and implementation boundaries

- This audit changed no environment, trainer, bank, or checkpoint, and launched
  or modified no job.
- A successful unit test or update-1 smoke is implementation evidence, not a
  reward result.
- A one-seed R2 or R3 result is a screen, not a paper-level causal estimate.
- Promotion selects checkpoints; development confirms. Sealed evaluation is
  used only after treatment selection.
- Reward return is not comparable across dense and terminal objectives.
- Efficiency comparisons are restricted to jointly successful identities.
- Fixing dump fallback, action masking, horizon, or reward in one arm would make
  attribution impossible.

## Research basis

- Ng, Harada, and Russell, [Policy Invariance Under Reward Transformations:
  Theory and Application to Reward
  Shaping](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html),
  ICML 1999: the gamma-consistent potential form.
- Grzes, [Reward Shaping in Episodic Reinforcement
  Learning](https://kar.kent.ac.uk/60614/), AAMAS 2017: terminal-state handling
  matters in episodic potential shaping.
- Pardo et al., [Time Limits in Reinforcement
  Learning](https://proceedings.mlr.press/v80/pardo18a.html), ICML 2018: a true
  fixed-period task should expose remaining time; a collector truncation should
  instead bootstrap.

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
| 2026-08-10 | observation | exact nine-trace replay confirms carrier-ledger accounting and necessary zero/negative relocation steps |
| 2026-08-10 | defect | current relocation budget spans 291x; per-map distance scaling and hidden carry credit break comparability/observability |
| 2026-08-10 | defect | fixed 450-step objective is partially observed because time-to-go is absent; current zero-bootstrap remains correct |
| 2026-08-10 | rejected | normalize relocation by reset haul work; use original required volume and one global physical-distance reference |
| 2026-08-10 | proposed | R2 tests one normalized material-potential bundle; R3 later fades only selected shaping |
| 2026-08-10 | experiment | complete D0 and durable D4a receipt before R2; run D1--D3, D5, and D6 as separate diagnostics/treatments |
| 2026-08-10 | accepted | both R2 arms use one prepared `continuous_banded_v2` sampler migration; v1 starvation cannot confound the reward screen |
| 2026-08-10 | observation | all 34 train maps whose projected relocation budget exceeds the current success bonus are large nearby V7 foundations, not remote/apron maps; durable D4b receipt pending |
| 2026-08-10 | accepted | reward-v2 uses flat `B`, explicit dwell-cost and dominance receipts, arm-labelled carry semantics, and matched optimizer-local LR warmup |
| 2026-08-10 | rejected | former R1 whole-objective anneal as a mainline run; retain implementation history but spend no compute before R2/R3 |
