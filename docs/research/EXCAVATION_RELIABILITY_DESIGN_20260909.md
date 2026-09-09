# Excavation reliability: soil occupancy, positioning and recovery

Status: first implementation and local validation completed on September 9,
2026. Environment invariants, shorter legal maneuvers and evaluation
diagnostics are implemented in isolated paired branches. Decoder recovery and
behavioral training follow measured evaluation; they are not part of this
increment. A two-update training smoke is complete.

## Problem and evidence

The u83500 trench specialist fails 36/224 development episodes. Replaying all
36 finds deterministic cycles of 2–16 actions, consuming 83.4% of their action
budgets. Exact saved-state probes demonstrate several mechanisms:

- Soil underneath the chassis can activate loose-only digging even though that
  soil is excluded from pickup. Removing only this ordering error admits seven
  fresh cells in one case.
- Four unloaded poses have no valid standard base action but admit a shorter
  translation with unchanged terrain and footprint.
- Productive actions are sometimes ignored: immediate DO can excavate 12 cells;
  another case needs only two chassis turns and DO to excavate eight cells.
- Four failures repeatedly rehandle soil. Loading-cycle and material-stall
  counters can report activity without useful task progress.

One matched sampled evaluation reaches 199/224 versus 188/224 greedy, with 15
rescues and four regressions. This is decoder sensitivity, not a reliable
recovery solution. Three failures complete excavation but leave unaccepted
spoil. A stubborn one-cell remainder has a verified digging pose, but no
verified approach route from the failed pose.

Evidence: [failure diagnosis and 36-case atlas](../../../../../.artifacts/terra_training_restart_20260907/failure_analysis_20260909/REPORT.md).
Both older checkpoint comparisons already use the corrected per-cell junction
rule. This design does not attribute every failure to junction admission or
infer that further PPO training alone will resolve it.

## Intended behavior

The excavator works from valid, productive base poses with adjacent workspaces
where possible. Intermediate Terra navigation actions are discarded at
deployment: the navigation stack connects the retained work poses. Extra Terra
maneuvers are acceptable when they improve workspace placement or completion.
We measure the continuity and productivity of those retained poses, alongside
raw travel and action counts. Overlapping dig workspaces are acceptable; fresh
area per setup matters more than geometric overlap alone.

The user clarified that loose soil should never be underneath the chassis.
This is an environment invariant, not a condition to reward the policy for
avoiding. Native terrain at height zero is valid ground. Positive loose soil,
negative holes, static obstacles and other occupied chassis footprints block
base placement. Dumping is not subject to the lateral fresh-dig penalty, and
loose-soil pickup does not require fresh-trench alignment.

## First increment

### 1. Make occupied footprints consistent

Use exact rasterized chassis footprints in reset placement, motion, excavation,
ground dumping and soil relaxation. Treat every positive soil cell as blocked
for chassis placement; remove the current shallow/scattered-soil exception.
Use the union of all active chassis footprints to exclude ground soil changes.
Loading directly into a truck remains a transfer between machine loads.

Do not delete, flatten, teleport or otherwise sanitize soil to satisfy the
invariant. From a valid initial state, a transition must preserve zero loose
soil under every chassis and conserve terrain-plus-carried material. If a
historical saved state already violates the invariant, retain it as diagnostic
evidence; it must not be silently repaired. A legal move may leave that state
for a valid footprint. Host-side reset validation must reject invalid requested
poses where available; random placement must use the same occupancy rule.

Resolve loose-versus-fresh selection after candidate geometry, occupied-base,
last-work and depth eligibility. An ineligible positive cell cannot suppress
fresh digging. Genuinely eligible loose soil retains the current priority.
The final fresh mask retains per-cell admission through any aligned owning
trench section. There is no wider fresh-first or radial soil-priority redesign.

### 2. Allow shorter tracked-excavator maneuvers

Keep the eight-action policy interface. Forward/backward for an unloaded
tracked excavator move through the longest valid prefix up to the configured
nominal distance. For each integer distance from one to `move_tiles`, compute
the rounded position from the original pose and heading, then test its full
footprint. Stop at the first blocked candidate. Never select a later free
endpoint past an invalid one.

On open ground, the endpoint stays equal to the current nominal move. Near a
blockage, the action may stop sooner. Every intermediate discrete candidate is
checked even if the nominal endpoint is clear. Do not repeatedly add a rounded
one-tile vector: that would change travel direction at 30-degree headings.
Keep loaded-excavator movement rejection, angular bins, map resolution and
machine dimensions. Other embodiments retain their existing movement model.

This is discrete Terra path checking. It is not a continuous swept-volume
certificate, a chassis-turn sweep, a Nav2 route, or physical traversability
validation. Check exact movement-feasibility observations against the same
handler. Use a bounded JAX loop with fixed-shape state; do not run a Python
search inside the compiled transition.

### 3. Measure useful progress and failure tails

Retain existing fresh area per productive setup, unique productive poses,
travel, heading changes and lateral-work metrics. Add clearly named raw soil
unit metrics for new excavation, positive-soil relifting, refill/redig and
accepted-disposal stock. Do not label cell-depth units as cubic metres.

Credit excavation only beyond the best previous depth of each target cell.
Report net accepted stock and increases beyond its previous maximum
separately, so repeatedly lifting and replacing accepted soil cannot inflate
progress. Record longest intervals without new excavation or accepted-disposal
progress, separately from intervals without any material/load change. Both are
diagnostics: intermediate relocation can be necessary without increasing either
completion component.

Track repeated action patterns up to the observed 16-action period with a
bounded history, explicitly distinguished from exact physical-state loops.
All metrics include the final transition of each initial episode and stop at
termination; auto-reset jumps must not count as travel or work. Do not silently
change the meaning of legacy loading-cycle or stall-age observations.

## Following increments

1. Reuse the existing executable fresh-dig observation for the new policy. Test
   exact base feasibility and separate fresh/relift availability as explicit
   observation treatments; any changed model input shape needs an intentional
   initialization or transfer, not a purported identical native continuation.
2. Add an optional evaluation-only recovery decoder, initially for the greedy
   feed-forward policy. Trigger on a repeated
   physical/task state including load, work exclusion and relocation state,
   plus policy action history and enabled feedback observations, while ignoring
   the monotonic episode clock and cumulative loading counter. Require no new
   unique excavation or accepted-stock maximum through that cycle. Recurrent
   policies need hidden-state handling before support is added. Retain a
   bounded history and per-episode override budget. First try a legal productive DO or
   cabin turn to productive work; otherwise explore untried effective actions
   using policy probabilities. Record every intervention and feed the actual
   executed action back into policy history. Multi-action recovery must fit
   the remaining horizon. Apply the same exact
   digging/dumping/movement handlers; never declare a heuristic mask to be a
   proof of a valid dump. Default greedy evaluation stays available separately.
3. Mine analogous late-stage states only from the training bank. Keep these
   diagnosed development cases for evaluation, and leave the sealed panel
   untouched during tuning. Train on recoverable residual and disposal states
   before increasing cost weights.

Do not enable PPO action masking without verifying rollout and optimization
use the same support for log probabilities, entropy and probability ratios.
No new time observation, encoder change, entropy sweep or reward coefficients
are introduced in the first increment. Keep 2× as the generalist candidate.
Blindly increasing navigation penalties could discourage needed repositioning.

## Validation and comparison

Use focused CPU contracts for occupied-base exclusion and mass conservation,
eligible-loose selection, safe-prefix movement, and honest progress accounting.
Include an isolated shallow soil cell, another active chassis, dumping followed
by relaxation, and a clear endpoint separated by a blocked intermediate pose.
Verify unchanged long endpoints in open space and loaded movement rejection.

Replay the diagnostic saved states as transition witnesses, labelling existing
invalid initial occupancy. Old witnesses that depended on driving over soil
must be rechecked; there is no guarantee that all four shorter moves remain
legal under strict occupancy. Then run one bounded vectorized environment smoke
and a finite PPO first-update smoke before any proposed long training run.
Use the existing CUDA convolution-backward preflight. Report compile and steady
step time separately; stricter movement checks add work per movement action.

When screening policy quality, compare both checkpoint and environment changes
explicitly. The same checkpoint on the repaired environment is a transition
treatment, not a retraining improvement. First compare the unchanged greedy
decoder; sample/recovery results are separate arms. Report exact success and
failure counts by foundation/trench family, excavation, accepted disposal,
productive setup area/count, work-pose continuity and the unproductive tail.
Efficiency comparisons must include common-success episodes to avoid rewarding
policies that stop early or fail to excavate.

The map-bank identity stays fixed; environment source revisions identify the
changed transition semantics. Historical frozen sources and checkpoint bytes
remain intact. Do not rewrite old benchmark receipts to make a new environment
look like the original experiment.

## Implementation locations and status

The paired worktrees are
`/home/lorenzo/moleworks/.worktrees/terra_excavation_reliability_20260909/terra`
and its `terra-baselines` sibling, both on branch `excavation-reliability`.
Starting revisions are Terra `1ea555d7` and baselines `c797ea3`. They include
the foundation behavior observations/costs and the inert-default benchmark
compatibility repair. The much older, dirty canonical checkouts are preserved.

- Soil occupancy and digging eligibility: implemented in `terra/state.py`.
- Short tracked maneuvers: implemented in the same state transition handler.
- Progress and repetition metrics: implemented in `utils/behavior_metrics.py`
  and exposed through normal fixed-bank evaluation, including failure summaries.
- Recovery decoder, observation expansion and full training experiment:
  designed, not started.

The occupancy tests confirm the first historical slot379 ingress: BACKWARD
action53 covered an existing one-unit soil cell without changing terrain. The
new move rejects that occupied endpoint. The seven-cell eligibility witness
also passes from its explicitly invalid historical state without removing the
existing under-base unit.

All four original shorter-movement cases retain an effective translation under
strict occupancy. Slot422 initially contains one under-base soil unit; its new
forward move leaves that invalid state for a clear footprint. Slots429/438/454
start clear and remain clear. All tested movement transitions preserve terrain
and total material. This proves local movement availability, not route-to-goal
or episode completion.

Validation completed:

- Six movement tests, four occupied-chassis contracts, a 15-case existing
  dump/trench/relocation subset, and the 44-case foundation/occupancy/step/protocol
  subset. The latter initially failed only an old assertion that excluded loose
  soil must suppress fresh work. Its corrected expectation passes separately;
  the initial failing log is retained.
- 99 baseline tests covering metrics, fixed-bank results and foundation behavior
  loading/training recipes. The 34 metrics/fixed-bank cases pass again after
  limiting relift calculations to rows whose buckets actually load.
- CUDA convolution-backward preflight, then 512 environments × 64 random-action
  steps from 53 recorded initial states. Zero material residual, zero positive
  soil under active bases, finite rewards. Compilation/first step: 165.2 s;
  subsequent environment-only throughput: approximately 25,782 transitions/s.
- Two native PPO updates, u5000→u5002, with 512×32 rollout shape, two epochs,
  32 minibatches and the 2× costs. Both saved checkpoints have finite model,
  Adam and loss values and zero existing transition-integrity counters. Adam
  advances 320,000→320,128; the parameter tree stays at 2,311,701 parameters.
  This uses the easy-foundation bank for runtime validation, not the proposed
  full generalist bank. Total training process time was 270.2 s, dominated by
  compilation; the second update reported approximately 4,114 transitions/s.

New diagnostics add host evaluation cost: in a bounded 608-map stationary-state
probe, accumulation took approximately 53 ms/step versus 29 ms previously.
This excludes device transfers, inference and environment stepping. These
counters are outside the PPO hot path. The measurements do not establish a
throughput improvement over the old environment.

The [implementation report and test logs](../../../../../.artifacts/terra_excavation_reliability_20260909/REPORT.md)
retain exact commands, state witnesses and smoke checkpoints. No cluster jobs
were submitted or modified. Full fixed-panel policy evaluation, recovery and
long training remain next-stage work. Local correctness and finite updates do
not establish policy improvement, Nav2 feasibility or physical acceptance.


## Submission preparation follow-up

The full-bank native 2x smoke now passes through u5002, and a fresh-process
ordinary continuation passes through u5004. Adam advances from 320,000 to
320,256, all model/optimizer/loss checks are finite, and transition-integrity
counters are zero. The resumed PPO executable hits the persistent cache;
loading and Python tracing still incur startup cost. Continuous training keeps
that cost outside subsequent updates.

The normal fixed-bank evaluator now also measures retained effective DO
setups, including dump/relift poses, transfer-distance lower bounds, pose and
A-B-A returns, and fresh-workspace edge/corner adjacency. It preserves raw
navigation metrics separately. Forty focused tests and 272 comparisons against
16 earlier replay episodes pass. Geometric-cone overlap and Nav2 paths remain
outside these counters.

The unchanged specialist solves 175/224 trenches in the repaired environment,
versus 188/224 previously, with seven rescues and twenty regressions. Its
remaining failures still have long repeated-action tails. This is not a
checkpoint promotion. The original generalist u5000 is also weak on the full
panel: 0/384 foundations under both environments, and 30/224 trenches before
versus 32/224 after repair. Its 29 common successes have identical retained-work
efficiency. The reviewed easy-foundation 2x u15000 checkpoint completes 0/608
cases under the repaired environment; its mean foundation excavation is 9.44%,
versus 59.85% for the generalist. Reset and integrity checks pass for both.
Retain the mixed-bank generalist as the initializer. This compares different
weights, training data and behavior features; it does not isolate reward effects.
Completion must improve before foundation workspace efficiency can be judged.

An independent Codex review found no actionable issues in the launcher, native
continuation or retained-work metrics. The Euler runtime gate verifies every
installed package against the complete 81-package lock. The staged one-GPU,
24-hour resource request passes Slurm test-only validation without creating a job.

The [submission-readiness report](../../../../../.artifacts/terra_excavation_reliability_20260909/SUBMISSION_READINESS.md)
records the final comparison, selected parent, runtime gates and exact staged
source. The [one-GPU training and evaluation recipe](../../scripts/excavation_reliability/README.md)
keeps a 24-hour allocation, periodic native checkpoints and separate per-job
receipts. Recovery decoding, new observations and stronger rewards remain
separate treatments. No new cluster job has been submitted during preparation.
