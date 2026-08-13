# V8 v6.1 policy failure audit (2026-08-13)

Living research note for the selected `v6_1_rv2` update-14,000 policy.  This
note separates measurements from hypotheses and records simulator probes as
they finish.  It is not a new experiment launcher or a component attribution
claim.

## Frozen subject and evidence

- Checkpoint: `v8_v6_yolo_rv2_v61_9abf88eb60df_s20260807_update_014000.pkl`
- Checkpoint SHA-256:
  `79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609`
- Training source: `9abf88eb60dfc0eb2395a5cc799b933928b6952c`
- Terra runtime: `3051054bc4c713d95905d3f954e6eabf55d6a85a`
- Fixed promotion result: 407/720 exact, macro completion 0.7265.
- Code-matched compact control: 281/720 exact, macro completion 0.6157.
- Pairing: 170 maps are solved only by v6.1, 44 only by the compact policy,
  237 by both, and 269 by neither.
- The gain is broad rather than trench-only: foundations improve from 51 to
  117 exact (+66), and trenches from 230 to 290 (+60).  The dominant residual
  wall is therefore a subset of foundation geometry, not a failed architecture
  replacement overall; 46 trench failures remain too.
- Promotion JSON SHA-256:
  `8973c9af9b95530c4d69d9e507bc239ab7b42ed62506dd02781bda0efa700cae`.
  All 720 rows pass the recorded reset/state-integrity checks.

The fixed-panel comparison uses the chunk-120 evaluator described in
`V8_V61_ABLATION_RESULT_20260813.md`; its measured near-tie variation is much
smaller than the +126-map v6.1 margin.  Claims below concern the bundle, not an
individual mixer, flatten projection, or latent-query change.

## Where v6.1 still fails at update 14,000

All 313 fixed-panel failures are 450-step timeouts.  They are concentrated in
foundations: v6.1 solves 117/384 foundation episodes and 290/336 trench
episodes.  The six zero-exact cells are:

| condition | exact | mean absolute completion |
|---|---:|---:|
| `v7-fnd-bearing-walls-adjacent` | 0/16 | 0.187 |
| `fnd-slab-side1-obj` | 0/16 | 0.193 |
| `v7-fnd-slab-adjacent` | 0/16 | 0.267 |
| `fnd-slab-ring3x-obj` | 0/16 | 0.289 |
| `v7-fnd-courtyard-pads-adjacent` | 0/16 | 0.327 |
| `fnd-slab-ring3x-obj1` | 0/16 | 0.494 |

The failures are not one mechanism:

- 26/313 never dig; 134/313 dig at least half the target; only 27/313 dig at
  least 90%.
- 92/313 retain illegal soil, 28/313 end loaded, and 120/313 have dump purity
  below `1 - 1e-6`.  These categories overlap.
- 46/313 reach at least 75% exact completion and 20/313 reach at least 90%, so
  a real near-finish/horizon subset exists, but it is not the majority.
- Failed maps have mean required volume 253.9 versus 115.4 for successes;
  16 failures exceed 500 volume units.  Large adjacent foundations are a
  workload/coverage cluster.
- Remote d12/d16 and obstacle conjunctions fail at ordinary volumes too, so
  volume alone is not an explanation.

Examples make the separation concrete.  `v7-fnd-slab-adjacent` failures have
mean required volume 619 and 12.1 productive work cycles, consistent with a
large-workload problem.  `fnd-slab-side1-obj` and
`fnd-slab-ring3x-obj` have ordinary mean volume 188; the targeted true-effect
traces below confirm an action-selection stall on one reset from each cell.
d12/d16 failures frequently retain illegal soil.  Conversely,
`v7-fnd-pads-adjacent` reaches 14/16 and every trench condition is at least
10/16, which rules out a global inability to excavate, relocate, or finish.

### No-effect metric warning

The stored success/completion/integrity fields are sound, but the historical
fixed-eval `no_effect_action_count` is not authoritative for this audit.
`clip_action_map_in_obs` mutates the observation used by the evaluator; once a
cell magnitude exceeds one, comparing the next raw map with that clipped prior
map can make a true no-op look effective.  New traces must use Terra's
`timestep.info["action_had_effect"]` directly.

## Sampler state at the selected checkpoint

Online sampler telemetry is not the fixed panel, but it exposes a continuation
issue:

- all 22 trench conditions are mastered (mean EMA 0.966);
- only 7/25 foundation conditions are mastered (mean EMA 0.453);
- v2 still assigns exactly 50% of population mass to each family, so 51.4% of
  total assignment mass is on already-mastered conditions while every open
  condition is a foundation;
- maximum cell mass is only 0.04113 and ESS is 38.62/47.

A cap-only draft was inactive because no condition exceeded 0.15. It prevented
a single-cell monopoly but did not release the half-batch quota held by the
mastered trench family, so that draft was rejected. Final Continuous Banded v3
instead removes family quotas and uses the global 80% open / 20% mastered
replay mixture documented below. The sampler-state diagnosis motivates that
change; it is not itself held-out evidence of a policy improvement.

## Structural hypotheses and decisive probes

Run on `rsl@supercluster` with one batched 12-environment rollout per RTX
3060: horizon 450 on GPU 0 and horizon 900 on GPU 1.  The primary workstation
GPU is not used.  The selected set contains four loaded failures, four
high-dig clean failures, two obstacle stalls, and two success controls.

The diagnostic evaluator imports Terra `46b5a1ddcd3b0e3a0d9e637af2e4ea94af51b4c8`
because the current baselines evaluator expects its reward-timing selector;
phase-1 Terra `3051054b` lacks that interface.  The source delta is confined to
reward protocol/timing selection and tests, not physical transitions, and the
900-step counterfactual forces the dense reward because reward is not a policy
input.  This is an explicit diagnostic compatibility layer, not a claim of
byte-identical runtime provenance.

1. **Horizon/workload:** replay the frozen panels with the same greedy policy
   and physical dynamics but a diagnostic 900-step horizon.  Because time and
   reward are absent from the policy input, the action/physical prefix through
   step 450 should match apart from the termination transition.  Rescue of
   high-progress failures supports a workload/horizon problem; no rescue
   refutes horizon as the dominant cause.
2. **Action feasibility:** on the stratified 12-map batch, log the
   true action-effect flag and all eight one-step counterfactual effects at
   each timeout's pre-final decision state.  This is not an action-mask
   training experiment.  Effective alternatives indicate selection
   inefficiency; no effective alternative points to dynamics/geometry or an
   irreversible state.
3. **Deterministic traps:** compare greedy with four sampled-policy seeds on
   the same resets and record outcomes, true action effects, and last progress
   step.  Sampled-only success proves that the categorical policy can escape a
   greedy trajectory attractor; logit-margin and observation-alias probes are
   deferred until the outcome screen says where they are useful.
4. **Observation limits:** the policy has no time-to-go, only five previous
   actions, no enabled reachability channel, and a globally clipped action
   map.  Probe obstacle and carry-channel sensitivity only after the behavior
   tests identify which bottleneck dominates; do not add attention tooling
   pre-emptively.

There is also a concrete observation/physics mismatch: the wrapper's global
`traversability_mask` marks every nonzero soil cell blocked, while movement
physics allows isolated height-one positive soil and blocks only holes, piles
above one, and sufficiently dense patches.  The optional reachability channel
that could disambiguate actual access is disabled in this run.  This can make
legal navigation look blocked to the policy, but it is not an environment
infeasibility proof.  Action-logit masking is deliberately disabled: prior
training experiments did not show useful improvement, so this audit does not
propose another masking arm.  Its all-action enumeration is diagnostic only --
it distinguishes a bad policy choice from a state with no effective local
action and never changes the trained policy distribution.

### Source-proven mechanics that now have prevalence tests

These are real runtime semantics, but they require measured prevalence before
they explain aggregate failure.  The Supercluster trace and frozen-panel
reduction record that incidence before any environment change is proposed.

- A dig footprint containing at most one eligible tile is discarded.  A
  terminal state with exactly one globally remaining target cell is therefore
  directly unfinishable from that pose/terrain unless a multi-step sequence
  first creates another eligible target tile.  This is a concrete endgame
  obstruction, not by itself a proof that the map or state is globally
  unrecoverable.
- A loaded solo excavator cannot translate or rotate its base.  The terminal
  audit enumerates all 12 cabin headings and classifies complete legal and
  off-zone unloads.  A loaded state with no complete unload heading is an
  exhaustive local deadlock, not merely a greedy-policy stall.
- Positive-soil relifts use the same minimum-two-tile gate and reject a whole
  selected pile if its volume exceeds int8 capacity.  This creates possible
  cleanup traps, but it should be changed only if an actual policy terminal
  state satisfies a fail-loud predicate.
- The reward distance is point-geodesic and assigns accepted-zone soil zero
  remaining haul cost.  It does not model the 7x11 footprint, five-tile motion,
  headings, or future serviceability.  This can make a locally positive dump
  dynamically harmful; incidence requires trajectory evidence.

### Targeted horizon-450 simulator result

The independent RTX-3060/chunk-8 rerun completed and passed exact replay parity
for its recorded actions, Terra action-effect flags, and continuous completion
traces.  It reproduced the frozen success vector and episode lengths for all
12 selected slots (two successes, ten timeouts).  Its terminal graded values
do differ from the frozen chunk-120 trace, so these mechanism rows belong only
to this independent rerun and are not merged into the 407/720 paper result.

The selected sample is deliberately mechanism-enriched and cannot estimate
population prevalence.  Within it:

- `fnd-slab-apron-near` slot 142 ends clean, unloaded, 99.36% complete, with
  exactly one unit/cell left.  The minimum-two-tile rule makes that state
  directly unfinishable: effective repositioning actions still exist, but no
  one-step action improves the exact objective and the unchanged greedy policy
  does not escape by step 900.  This is a confirmed one-step mechanics
  obstruction in one targeted trajectory, not a multi-step impossibility
  proof.  All four sampled-policy trajectories avoid this endpoint and solve the
  same reset, confirming that the map itself is feasible.
- `fnd-slab-side1-obj` slot 247 and `fnd-slab-ring3x-obj` slot 177 take 450/450
  true no-effect actions and never change physical/task state.  Both finish by
  choosing `backward`, while rotations (and, for slot 177, `forward`) are
  physically effective.  This proves deterministic policy-choice traps on
  those resets, not global environment deadlock.  It does not motivate action
  masking; a multi-step choice/observation problem remains.
- The action sequences make the attractors explicit.  Slots 247 and 177 choose
  `backward` all 450 times.  Loaded slot 234 chooses cabin-clockwise 426 times
  consecutively while `DO` is an improving unload.  Loaded slot 17 chooses a
  no-effect `backward` 346 times consecutively.  Near-finish slots 68 and 300
  settle into two-action backward/anticlock cycles; slot 338 keeps moving and
  rotating but makes no task-state progress after step 83.  Thus no-effect,
  cabin-spin, and motion-cycle failures all coexist.
- Three rerun endpoints are loaded.  Exhaustive classification over all 12
  cabin headings finds zero loaded deadlocks: d12 and side1 have 11 off-zone
  unload headings, and `proc-side1-road` has two legal unload headings.  In
  each case `DO` immediately improves exact completion, but the greedy policy
  chooses cabin rotation or a no-effect backward action instead.
- Three other clean near-finish cases have only 2--4 units left, but no single
  action at the pre-final decision improves the exact objective.  They are
  local planning/pose stalls, not proven mechanics impossibilities.
- All ten failures have at least one physically effective action at the
  pre-final decision state.  Seven have no immediately
  exact-objective-improving action.  The
  distinction matters: effective rotation is an escape opportunity, not task
  progress by itself.

Focused runtime unit tests independently pass for rejecting one-tile dig masks
and preserving multi-tile masks.  The later 900-step matched trace passed its
horizon gate by matching actions, effects, and every completion component
through step 450.

The frozen 720-panel result also bounds the prevalence of the singleton defect.
Because every admitted target cell has unit depth, multiplying each terminal
dig fraction by its frozen target volume reconstructs an integer count (maximum
rounding residual `2.17e-5`).  Exactly 2/313 failures end with one undug target
cell: slot 142 (`fnd-slab-apron-near`) and slot 185
(`fnd-slab-ring3x-obj`).  Eleven failures end with at most four undug cells.
Thus undug-target singleton endpoints are rare (0.64% of failures) and cannot
explain the main performance gap; the reduction measures endpoint incidence,
not global state impossibility.  The frozen JSON does not expose spatial
positive-soil cleanup singletons, so their prevalence remains unquantified.  The durable
reconstruction receipt is
`/home/lorenzo/moleworks/.artifacts/terra_v8_v61_failure_audit_20260813/singleton_prevalence_v1.json`
(SHA-256
`41ea8df9ef2d3f7db38e6d05cb826057c8371e975a9658eb8608e1cf35846a22`);
the analysis script SHA-256 is
`2e467722d31141785cdd1d75a8d6922079d4fb968b2fcd49857f3bc5dbc39beb`.

A static geometry gallery for the 12 frozen scenarios is stored at
`/home/lorenzo/moleworks/.artifacts/terra_v8_v61_failure_audit_20260813/v61_targeted_maps_v2.png`
(SHA-256
`9142738d58dee2286ed38eb29465f9001a7054bdbbad2be01a0a6b6bbfcc1ac6`).
Blue is required excavation, yellow is accepted dump area, black is obstacle,
and the panel-border color marks the diagnostic role.  It visualizes task
geometry only, not the terminal terrain or policy trajectory.

### Matched horizon-900 result

The gate passed: the 450 and 900 executions have exactly identical action IDs,
Terra action-effect flags, and all seven continuous completion-component traces
through step 450 (maximum component error 0).  The only intended execution
difference is the episode horizon; the 900 diagnostic uses dense reward solely
to bypass reward-v2's hard 450 validation, and reward is neither a policy input
nor a physical-dynamics branch.

Extending the same unmasked greedy policy to 900 actions converts **0/10**
selected failures.  Exact remains 2/12 (the two controls).  None of the four
near-finish failures converts.  More strongly, every failure's last task-state
change stays at the same early step seen in the 450 run: 111 or earlier.  The
singleton remains at one cell; the 2--4-cell clean failures remain at 2--4;
the loaded failures retain the same load/illegal soil; and the two obstacle
resets remain at zero progress after 900 no-effect actions.

This targeted result refutes “450 is simply too short” for these selected
failure mechanisms.  It does not estimate the effect of horizon over the full
313-failure population.  The next high-value probe is policy-side: stochastic
or short intervention branches on the exact stalled states, not a global
horizon increase and not another action-mask experiment.

### Sampled-policy result

Four independent sampled-policy batches (seeds 20260808--20260811) ran on the
same 12 resets in two paired waves, one 12-environment batch per Supercluster
GPU.  They sample the checkpoint's existing categorical policy without
masking, retraining, or changing reward or dynamics.  The greedy run solves
2/12; the four sampled runs solve 6/12, 5/12, 6/12, and 6/12.

The slot-level agreement is more informative than the totals:

- All four seeds rescue exactly the same four greedy failures: slot 142
  (`fnd-slab-apron-near`), slot 68 (`fnd-slab-apron-c2x`), slot 300
  (`trn-net3-side1-road`), and slot 338 (`trn-net4-side1-road`).  They finish in
  98--217, 55--84, 93--178, and 94--119 steps respectively.  All four were
  clean/high-dig failures under greedy execution.  This is strong evidence
  that their 450-step failures were deterministic trajectory attractors rather
  than inadequate horizon or structurally impossible maps.
- None of the four seeds rescues any loaded/high-carry diagnostic reset or
  either obstacle reset.  Fifteen of the sixteen sampled carry trajectories
  end unloaded; slot 17 under seed 20260810 retains two units.  All still
  finish with illegal soil and/or undug target.  On the two obstacle resets sampling
  reduces the extreme 450/450 greedy no-effect count to 144--324 and all eight
  sampled trajectories change task state, yet none finishes.  These eight
  sampled trajectories therefore do not establish stochastic
  exploration as sufficient for the harder obstacle/cleanup and
  relocation-planning mechanisms.
- All four sampled seeds retain the `v7-fnd-pads-adjacent` success control.
  Three retain `fnd-slab-ring3x-road`, but seed 20260809 loses it and stalls
  loaded after its last task change at step 59.  Stochastic inference is
  consequently a diagnostic, not a proposed deployment policy or a
  population-level improvement.

The result splits the next work cleanly.  Near-finish failures call for better
training-time choice stability, endgame state representation, or short action
history—not more horizon.  Obstacle and carry failures require a separate
planning/observation investigation because all 24 sampled trajectories across
those six resets fail to convert them.  No action mask is introduced or
recommended by this result.

### Greedy policy-input recurrence and logits

A final replay evaluates the unmasked policy on the exact validated greedy
action trace.  It hashes the actual post-preprocessing 22-leaf policy input for
each slot and decision, including the five previous actions, and separately
hashes the unclipped action map.  It uses the same chunk-8 forward split (8+4)
as the source trace.  Its fail-loud gates pass: zero active argmax mismatches,
zero action-effect mismatches, zero error in all seven completion traces, zero
nonfinite logits, and zero cases where an identical policy input produces
different logits.

The failures are recurrent policy dynamics, but not all have the same
confidence profile:

- The two obstacle stalls are exact observed fixed points.  Each sees only six
  distinct full policy inputs; from decision 6 onward one identical input
  repeats for 445 decisions, including an identical raw action map and saturated
  five-action history.  Greedy chooses `backward` every time.  On the modal
  input, its chosen probability is only 0.224 for slot 247 and 0.272 for slot
  177, with top-two logit margins 0.198 and 0.480 and entropy 1.982 and 1.936
  nats (`ln(8)=2.079`).  This is not a highly confident policy.  Greedy argmax
  converts a broad categorical distribution into a permanent no-effect fixed
  point; all eight sampled obstacle trajectories escape the fixed point and
  make task progress, but none sequences a complete solution.
- Carry/cleanup slots 250 and 17 also enter exact one-step fixed points: their
  modal input repeats 325 and 341 times, with chosen probabilities 0.166 and
  0.154, margins 0.084 and 0.023, and high entropy 2.031 and 2.055.  Slots 100
  and 234 instead enter exact 14-step and 12-step input cycles.  Their modal
  choices are more separated (post-stall median probabilities 0.342/0.389 and
  margins 0.869/1.047), showing that carry failure is not uniformly a near-tie
  problem.
- All four clean/high-dig greedy failures enter short recurrent cycles before
  timing out: lags 7, 2, 2, and 18 for slots 142, 68, 300, and 338.  Slot 142
  is frequently near-tied, but slots 68/300 have post-stall median margins
  0.653/1.267, and slot 338 has 101 post-stall decisions with chosen
  probability at least 0.8.  The fact that every one of 16 sampled trials
  solves these maps therefore does not mean every greedy failure is caused by
  a near-tied top two; sampling can break moderately or strongly preferred
  short cycles too.
- In contrast, the two greedy success controls have a different full policy
  input at every active decision (79/79 and 81/81 unique).  Across the complete
  probe, 3,788/4,660 active decisions revisit an exact prior input and 1,552
  repeat the immediately prior input.  There are zero repeated policy-input
  hashes paired with a different raw-action-map hash, so action-map clipping is
  not the observed source of these particular aliases.

The policy input omits time-to-go and retains only five action IDs, with no
record of whether they changed physics or exact task state.  Once a full input
fixed point or short cycle is reached, deterministic feed-forward inference has
no internal state with which to count repetitions and change strategy.  This
is a source-grounded limitation, not proof that any particular new observation
will improve training.  The smallest causal follow-up is a bounded branch from
the first repeated input: perturb one action, then return to greedy execution.
That distinguishes one-action basin escape from failures requiring sustained
exploratory sequencing.  If a learned treatment follows, test one compact
stall/time feature or recurrent state—not an action mask—and keep reward,
sampler, and architecture otherwise frozen.

### Recurrent-policy design review

The recurrence evidence does not, by itself, prove that Terra requires a
recurrent policy.  An observation-identical deterministic controller must emit
the same logits, but a different stationary controller could still choose the
escape action.  The first-repeat action branch therefore remains the causal
gate before interpreting memory as the remedy.

If a recurrent treatment is run, the narrow implementation is one actor-only
GRU after the existing v6.1 spatial and state fusion:

```text
v6.1 fused feature -> Dense(160) + ReLU -> GRU(64)
                   -> Dense(48) + ReLU -> 8 action logits
```

The critic stays feed-forward, as do the SE-ResNet, token mixer, latent-query
readout, agent encoder, carry feature, and five-action history.  The expensive
v6.1 encoder should still run once on a flattened batch of observations; only
its fused actor features are reshaped to environment-by-time and scanned
through the recurrent head.  Scanning the full spatial encoder one time step at
a time would be an avoidable throughput regression.  Relative to the selected
2,303,421-parameter v6.1 policy, Flax 0.8.2's `GRUCell(64)` should add 38,656
parameters, for 2,342,077 total; an implementation receipt must recompute rather
than trust that projection.  Run the recurrent cell in float32 even though the
spatial encoder uses bfloat16.

Hidden state is zeroed on reset/done and otherwise carried across steps.  PPO
must store the pre-rollout actor state, shuffle it with the same environment
permutation as each intact trajectory, and use 32-step truncated BPTT with done
masks.  Flat transition shuffling is incompatible and must fail closed.  The
current 32-step rollout covers every observed 2--18-step cycle.  The final value
bootstrap uses the feed-forward critic-only path and must not advance actor
memory.  Evaluation must carry and selectively reset hidden state and chunk the
observation and state together; MCTS is outside this pilot because each search
node would need its own recurrent state.

The local `rsl_rl` checkout already implements the relevant GRU/LSTM rollout
contract in PyTorch: record hidden state before action, reset terminated
environments, store recurrent state with transitions, and replay contiguous
sequences during PPO.  Terra is JAX/Flax, so this is a design reference rather
than reusable code.  The smallest acceptance suite is: stepwise-versus-scan
forward parity including selective done resets; hidden-state reset parity;
rejection of flat minibatch shuffling; and one finite update/checkpoint/greedy
evaluation smoke.  A matched feed-forward control must use the same
sequence-preserving minibatches, otherwise the batching change is confounded
with recurrence.  The first experiment is from scratch and changes only the
temporal policy mechanism; reward-v2, curriculum, and v6.1 spatial readout stay
fixed.  Unsupported visualization/inference entry points must reject the
recurrent checkpoint rather than silently zeroing memory at every step.

Lux provides precedent for gated temporal state but not for a GRU specifically.
The Lux AI Season 3 winner used a ResNet followed by ConvLSTM and spatial
Transformer, and the tenth-place end-to-end JAX PPO solution fused entity,
spatial, and scalar features before a 384-unit LSTM.  Conversely, the Season 1
winner used a feed-forward 24-block SE-ResNet with explicit day/night and game
phase features, and a Season 3 gold solution used a ten-frame stack rather than
an RNN.  These results justify a bounded temporal-state test, not copying a
large spatial ConvLSTM or claiming that Lux established GRU superiority.
Primary implementation sources are the
[Season 3 winning model](https://github.com/tonykozlovsky/lux-ai3-pub/blob/8001d70d939c78725d198a70586e8ba77efa2a24/final_versions/07_03_tune_against_mask/lux_ai/nns/models.py#L1253-L1354),
its [recurrent-state handling](https://github.com/tonykozlovsky/lux-ai3-pub/blob/8001d70d939c78725d198a70586e8ba77efa2a24/final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/create_buffers.py#L42-L104),
the [tenth-place JAX write-up](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/boey-10th-place-solution-boey-end-to-end-jax-rl),
the [frame-stack solution](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/6f91e4eddddcbd0473b6948d23a384fda6b55768/write-up.md#L101-L146),
and the [Season 1 winning architecture](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021#neural-network-architecture).

The cheaper learned alternative remains one normalized stall-age scalar.  Let
the exact material signature contain the raw unclipped action map plus every
active agent's loaded amount and carry-relocation credit.  Reset a counter to
zero whenever that signature changes; otherwise increment it, capped at 32,
and expose `stall_age = counter / 32` after the existing fusion and before both
heads.  Pose and cabin motion therefore do not erase the evidence that material
work is stalled.  The 32-step bound is a pilot choice covering every observed
2--18-step recurrent cycle; shadow traces must reject it if successful normal
navigation commonly saturates the counter.  It directly breaks the measured
input equality without recurrent-PPO plumbing, but it cannot represent
arbitrary history.  A checkpoint-preserving continuation adds separate
zero-initialized 704-wide actor and critic embeddings, so the old head matrix
shapes and update-14,000 outputs remain unchanged at `stall_age=0`; matching
zero Adam moments are added for those two leaves.  This is a continuation
screen against the already measured v6.1 curve, not a newly trained matched
control.

Normalized time-to-go is a separate finite-horizon correctness treatment, not
something to bundle with stall age or the GRU.  The branch probe decides
whether a one-action basin escape makes stall age the first learned screen;
failure of all one-action branches, or evidence that the same current input
needs sustained history-conditioned action sequences, is the stronger case for
the GRU.  An independent GPT-5.6 Pro Oracle review of this note and the pinned
model/training/environment sources reached the same ordering and explicitly
rejected treating recurrence alone as proof of a POMDP or GRU requirement.
This separation follows the time-limit distinction of
[Pardo et al. (2018)](https://proceedings.mlr.press/v80/pardo18a.html): a
true fixed-period task should expose remaining time, whereas recurrent models
address a broader missing-history problem.  Recurrent model-free RL is a
credible later treatment, but its performance depends on real sequence
training rather than merely inserting a cell into a feed-forward learner
([Ni et al., 2022](https://proceedings.mlr.press/v162/ni22a.html)).

## Supercluster execution receipt

- Dedicated account-scoped keys were created for `euler-alesweber` and
  `euler-lterenzi`; existing identities were not modified.
- Isolated worktrees:
  `/home/rsl/moleworks/.worktrees/terra-baselines-v61-failure-audit-20260813`
  at baselines `288130df`,
  `/home/rsl/moleworks/.worktrees/terra-v61-failure-audit-20260813` at phase-1
  Terra `3051054b`, and
  `/home/rsl/moleworks/.worktrees/terra-v61-horizon-runtime-20260813` at the
  diagnostic trace runtime `46b5a1dd`.
- GPU environment: Python 3.12.3, JAX 0.4.26, Flax 0.8.2, Optax 0.2.1.
- A real CUDA convolution backward pass succeeded on the otherwise-idle RTX
  3060 GPU 1.  The matched stochastic probes used both GPUs concurrently, one
  12-environment policy batch per GPU; maps were not evaluated one at a time.
  Both processes exited cleanly and released their allocations.
- Exact checkpoint, bank, and code-matched promotion JSONs were copied and
  SHA-verified under
  `/home/rsl/moleworks/terra_runs/v61_failure_audit_20260813`.
- Exact executed one-off analysis and rollout sources are archived locally under
  `/home/lorenzo/moleworks/.artifacts/terra_v8_v61_failure_audit_20260813/supercluster/executed_scripts`;
  the source hashes below match those archived files.
- The static 720-map decomposition is preserved as
  `static_panel_audit_u14000_v2.txt`, SHA-256
  `02f415310fb3cb3663a99bc41d8ca8546b77dd449c09dd061605e36dc7654b6b`;
  its one-off analysis script SHA-256 is
  `25045103d841059f1b98bffc686f8be89250d33753b20f9701a8395326ad9941`.
- Targeted horizon-450 receipt SHA-256:
  `6e5aa750c3e9cb59056cb1ee8c9b4454d375ef883439b688c604442412f448e4`;
  trace-array SHA-256:
  `51efce34fe8343c33c3b4123160118e4e219fc40b57d6da8f67cf66589dc8d42`;
  executed diagnostic script SHA-256:
  `9152799a060881de22ce8012a03a199c8bc44309bfbf29b6fe0bb5bc2c2a9822`.
- Targeted horizon-900 receipt SHA-256:
  `e509664eb8dc2bc73563dd704dbaff9600662a9725e9e134b8c137da35d39eb5`;
  trace-array SHA-256:
  `c9bcf34b79d43cec3f02cd8f11c092929b79156ebe9190224b669c84f87090e4`.
- Matched horizon-comparison receipt SHA-256:
  `628db35abd1e44a18df997b262fdcaf38c02422ee9ac4d2899219b35abe8d96b`;
  comparison script SHA-256:
  `c4194150a9823048c17ecfaaac03136af8e60b858efb63aec012470feee73fa1`.
- Sampled-policy seed-20260808 receipt/array SHA-256:
  `bc082f99bef8133307547b212f984f80d3bab553217d59ceb2f9112eb42af449`
  / `5b682839e5d2d6ba5417cfb9ab17386b6eefcdeb40f3a3608e9263fe263992ea`;
  execution-log SHA-256:
  `29fef046487784d6eb1c95d56b1fe9e1d26f80c8d63917eda36b414ef05aec17`.
- Sampled-policy seed-20260809 receipt/array SHA-256:
  `3717dfbc27ed7dd0e9d4e7751cab7b7444132122efbd09cd510ce83c1abc3cfb`
  / `18000662ad0a17463ffe954778dd8a20bba20f268ab3462fc1c2f9c38e8b1a42`;
  execution-log SHA-256:
  `43befa37872391baa855470b1b86060f6d7d5264b6dfff1be3563a262df665e9`.
- Sampled-policy seed-20260810 receipt/array SHA-256:
  `be4f982e2afc709e1b2c32ebff4501aa1f2d7aa778a687a3b7af6a4b730d5de6`
  / `7e33a972c8594a52b65575f00d83695c86d936f42fa659d95ffa90a8ade1da57`;
  execution-log SHA-256:
  `47bdf2372a809d92e53ff730ff4853e5562116312ff68918dec8b070f6cf4438`.
- Sampled-policy seed-20260811 receipt/array SHA-256:
  `e9151fb8d02692478a0d5906b43f426feb299f67a5e7c1d3886fd3f8d8883922`
  / `69d78a60326f4df3d1d427e586c1c5d1cc25315716ce7a42a58f11262b64d6d5`;
  execution-log SHA-256:
  `fc66b126bf47fd94bfe042c1480f3d796376769464163653d4aa2002af25a056`.
- Executed sampled-policy diagnostic script SHA-256:
  `8fe17d487ee0384f725fb20bb77fb901b12c17820b3c8f04b2c760249cfe04b8`.
- Sampled-policy comparison receipt SHA-256:
  `f90370043a30a39b2cbfa011c23834789824e352bbb6cd417d12df673e6de0c2`;
  comparison script SHA-256:
  `ddda04e51282300b636b3a9b6d868c0a4bb7e590bcd3b54091794a891b258dff`.
- Greedy logit/input-alias receipt/array SHA-256:
  `df4b8d5376277f5d28f94ebcbfc049c9eebab7af94beca980579fc8e9907be63`
  / `1cc2aa57deea4153358596585705524d9593efa0924f89bb46e57a0959366d5d`;
  execution-log SHA-256:
  `b94d9747c37febe849b5a3699bc89a1c73f07453b181ccf56e49f4a523375898`;
  executed script SHA-256:
  `9646dae6706c24eded89b0de07117dcc3351bfb1a4da360ae2d94a6ada2aa85a`.
- Independent GPT-5.6 Pro Oracle review:
  `/home/lorenzo/moleworks/.codex_tmp/oracle_v61_policy_attractor_answer.md`,
  SHA-256
  `c84efe77897177fedd7a54b4cb1c0f3c713e83f66eca3b958e2e4438d78665c8`;
  its substantive response ends with
  `ORACLE_V61_FIX_REVIEW_COMPLETE`.

## Stall-age continuation contract

Job `10616190`, submitted from terra-baselines `6ad2eb1`, was cancelled at
2026-08-13 18:20 CEST while pending. Slurm records zero runtime and no
allocation, so it produced no checkpoint, W&B run, or training evidence. Its
v2-sampler contract was superseded rather than resumed.

The old 8-GPU continuation path is retired.  Job `10569391` allocated eight
RTX 3090 GPUs but failed before update 14,001 in the v6.1 flatten-reduce
convolution with `CUDNN_STATUS_EXECUTION_FAILED`; it produced no checkpoint or
training evidence.  Its replacement `10572344` was cancelled while still
pending and never allocated.

The replacement has one supported phase-2 recipe. It requests eight RTX 4090
GPUs for 23:45 and reshapes phase 1 from 4×512 to 8×256 environments. It
keeps 2,048 total environments, 65,536 transitions per update, 32 minibatches,
two epochs, and the absolute optimizer/entropy clocks. The prepared checkpoint
adds two zero stall-age embeddings and migrates v2 to the final family-free
`continuous_banded_v3` rule. It preserves mastery, competence, the closed
window, refresh grid, and sampler RNG, but clears the source's 50-update partial
window before resume. Runtime therefore performs a native v3 restore, not a
mid-run migration.

Final v3 assigns 80% to a global depth-weighted open pool, 20% to uniform
mastered replay, and caps any one condition at 15%; foundation/trench labels
are diagnostics only. At u14k, 29 conditions are mastered and 18 foundations
remain open, so the migrated distribution is exactly 80% open and 20%
mastered replay, with maximum condition mass 6.96%. Reward-v2 and its timing,
action masking, horizon, bank, learning rate, and v6.1 encoder remain fixed.
Time-to-go is absent.

The target remains absolute update 40,000, but the 24-hour segment may end
earlier.  A finite rolling checkpoint every 500 updates is the continuation
unit.  This remains a statistical continuation because environments, rollout
RNG, and action history restart at the segment boundary; it is not bit-exact.
The source-to-treatment transformation is pinned by source SHA
`79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609`
and prepared SHA
`68aea1a0f5dc3c05d11319fdf640ade05495125225533bc99ad92592475fcb75`
(`v8_v61_stall_age_v3_u14000_prepared.pkl`, 27,741,529 bytes). Independent and
canonical materializations were byte-identical. The implementation uses Terra
`c2d2a94a`; the exact
terra-baselines revision and replacement job id must be recorded at launch.

If the combined recipe improves held-out exact completion and reduces
repeated-input failures, retain it as the practical best path while evaluating
the stall-age and curriculum telemetry separately. If fixed points remain, the
next bounded treatment is the actor-only GRU-64 design above. That experiment
is separate: it holds reward, sampler, and spatial encoder fixed, trains real
contiguous 32-step PPO sequences, carries and resets hidden state correctly,
and compares against a sequence-batched feed-forward control. A larger
ConvLSTM, action mask, time-to-go feature, or reward change is not part of this
ladder.

There is deliberately no matched control from u14k. The new curve answers the
practical question "does continuing v6.1 with stall age and final v3 improve
the policy?" It cannot separately attribute gains to the observation or
curriculum. Any paper-level component claim requires a later matched run or
direct mechanism evidence.

## Probe status

- Static fixed-panel and checkpoint decomposition: **complete**.
- Matched targeted horizon-450 batch: **complete; receipt passed**.
- Matched targeted horizon-900 batch: **complete; receipt passed**.  It
  reproduces the exact action, action-effect, and continuous completion prefix
  through step 450, then converts 0/10 selected failures by step 900.
- Four sampled-policy h450 batches: **complete; receipts passed**, seeds
  20260808--20260811.  All four rescue the same four clean/high-dig greedy
  failures; none rescues the six loaded/obstacle failures.  No action mask or
  environment change was used.
- Greedy logit/policy-input recurrence replay: **complete; receipt passed**.
  It exactly reproduces the source actions, true effects, and completion traces
  and confirms exact full-input fixed points or short cycles in every targeted
  greedy failure.
- First-repeat all-action branch, first attempt: **rejected by its parity
  gate**.  Expanding the source evaluator's heterogeneous 12-slot `8+4`
  forward context into ten homogeneous chunks of eight changed a low-margin
  bfloat16 argmax for all eight replicas of slot 142 before its branch.  It is
  not policy evidence.  The failed log and executed script are preserved under
  `.artifacts/terra_v8_v61_failure_audit_20260813/supercluster/one_action_first_repeat_failed_v1`.
- First-repeat all-action branch, second attempt: **partially accepted, then
  stopped by its parity gate**.  All four carry/cleanup slots passed their own
  original-action full-suffix shams.  Across the 28 non-greedy one-action arms,
  none reached exact success.  Forced `do` left the exact recurrent basin and
  later changed material state on all four, but greedy continuation still
  failed; this supports a multi-step selection/planning limitation rather than
  a one-action carry remedy.  The next clean slot changed a pre-branch
  bfloat16 argmax under the mapped execution shape, so no clean/obstacle result
  from this process was accepted.
- A third attempt executed each arm in the original 12-slot `8+4` context, but
  **also stopped before intervention** when slot 142 changed its greedy action
  at decision 69 in a fresh process.  This shows that matching the visible
  batch/chunk shape is insufficient for cross-process bitwise greedy parity on
  this bfloat16 path.  GPU 1 was released and no result was claimed.  The
  failed receipt is preserved under
  `.artifacts/terra_v8_v61_failure_audit_20260813/supercluster/one_action_standalone_failed_v3`
  (log SHA-256
  `998093b833d2149d84c5eb5786fde164f7d5dcd2f61deb106f3c9af40df908e6`).
  A future branch probe must generate its greedy control and interventions in
  the same executable realization and judge paired outcomes there, rather than
  requiring suffix identity to an archived process.  Clean/obstacle
  first-repeat branches therefore remain pending.
- The superseded 32-anchor capability sweep and 720-map trace were stopped;
  their partial logs are preserved.  They compiled the same expensive graph
  while answering a less decisive question than the matched targeted batch.
- Both completed horizon runs record Terra's true action-effect flag, terminal
  singleton and loaded-deadlock predicates, and all eight one-step alternatives
  at each timeout's pre-final decision.  Enumerating alternatives is a
  recoverability diagnostic only; no action mask is applied to the policy or
  proposed for training.
