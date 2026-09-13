# Completion regression investigation — September 13, 2026

The retained foundation checkpoint regresses on an unchanged held-out panel.
Manual replay and executed legal-action witnesses identify avoidable waiting
and loss of task focus. The historical trench deficit is concentrated in
road-constrained spoil rehandling, not a broad failure to dig every trench.
Added behavior penalties remain zero; stronger travel costs are not the first
response supported by these failures.

## Matched outcome evidence

| Policy comparison | Exact completion | Mean excavated | Mean accepted disposal |
| --- | ---: | ---: | ---: |
| Foundation u33000 → u41000 | 8/64 → 3/64 | 93.29% → 79.41% | 90.97% → 77.14% |
| Current trench u77000 → u85500 | 161/224 → 162/224 | 94.47% → 93.53% | 89.82% → 89.02% |
| Historical trench u83500 → current u85500 | 188/224 → 162/224 | Different effective environment | Different training history |
| Old u83500 weights in current environment | 179/224 | 94.71% | 93.66% |

The first two comparisons match map/reset identity, evaluator source, greedy
policy, 450-step horizon, reward/action contract and checkpoint hashes. All
672 current evaluation rows pass recorded integrity checks. Foundation changes
training layout from 1 × 512 to 4 × 128; current trench stays 4 × 128. The
historical comparison also matches all 608 map/reset identities and exact
completion definitions, but uses a different effective environment and training
history. It is not an isolated treatment comparison.

Older trench performance was genuinely stronger: u51500 solved 196/224 and
u83500 solved 188/224. Road networks explain 25 of the 26 net missing successes
against u83500: 25/32 → 0/32. Non-road trenches are almost tied at 163/192 →
162/192. Current road rows average 92.44% excavation, 62.63% accepted disposal,
and 29.81% off-zone soil, with only nine relifts across 32 episodes. All are
unloaded at the endpoint. This identifies recovery of staged soil as an
important bottleneck.

## Historical weights under the current environment

A completed 608-row replay of old u83500 weights under current Terra
`46738cde` and evaluator `866e8e2` solves **179/224 trenches**, including
**20/32 road networks** and **159/192 non-road trenches**. It preserves the
old trained observation interface (`executable_dig_observation=False`), the
same 450-step greedy panel, checkpoint SHA and reset identities. All recorded
integrity counters are zero. The newer u85500 policy solves 162/224, including
0/32 road networks, under the same current physical rules.

This separates two observations: changing the old policy's environment gives
188 → 179 exact completions and 25 → 20 road completions; changing from old
to new policy under current physical rules gives 179 → 162 overall and
20 → 0 on roads. This is not an additive causal decomposition: training
experience, observation interface and optimization differ, and policies can
respond differently to physical changes. It nevertheless rules out the claim
that current physics makes every road map impossible.

Manual inspection of the saved current-environment trajectories confirms that
the old policy completes road slots 290 and 345 in 95 and 231 decisions.
The newer policy fails both: slot 290 escapes toward a boundary, while slot
345 leaves staged spoil and cycles cabin/dig actions. The old policy places
all soil inside the accepted zone in both cases. All five selected old-policy
clips match their actual replay endpoints and complete. Slot numbers in this
report and viewer are one-based `slot_index`, not `slot_index_zero_based`.

## Direct behavior observations

The foundation u41000 instrumented replay reproduces every endpoint field on
all 64 rows. The selected examples are descriptive, not an unbiased prevalence
sample.

- Square slot 7 moves backward once, turns the cabin once, then waits for 448
  actions without any excavation. At its early waiting state, DO immediately
  excavates two units; one cabin turn followed by DO excavates 28 units.
- Slots 2 and 33 also settle into WAIT with unchanged model inputs. Executed
  early-state probes find useful movement and short turn-and-dig sequences,
  with 447–449 decisions remaining. These are not geometric dead ends.
- Square slot 9 excavates and disposes 82.07%, then leaves the remaining work,
  travels to the map boundary, and repeatedly requests a blocked forward move.
- Rectangle slot 19 completes in 44 actions. Its six fresh digs use a cabin
  yaw 60 degrees from the chassis axis (lateral score 0.75). Completion does
  not imply the desired digging posture while added lateral cost is zero.

The failed WAIT states have probabilities approximately 0.56, 0.77 and 0.59
for WAIT, with identical model-input hashes over their repeating suffixes.
Their WAIT reward is negative, about -0.008222 per step and -1.008222 on the
final timeout. No positive-reward WAIT exploit was found. The value function
was trained for stochastic policy futures, so comparing it directly with an
indefinite greedy WAIT return does not by itself prove critic corruption.

The same-input actor comparison strengthens this diagnosis. At saved slots
2/7/33 with identical observed state and action history, u33000 selects useful
turns while u41000 selects WAIT. At slot 9, the old actor chooses backward
instead of the newer actor's blocked forward command. CPU inference matches
recorded input hashes and all current argmaxes; it is not bit-exact with GPU
inference (maximum probability difference 0.001582), so only robust action
preferences are interpreted.

A one-seed sampled-decoder diagnostic on the same u41000 weights and 64 resets
solves 6/64 versus greedy 3/64, with excavation 88.66% versus 79.41%. It gains
four successes and loses one. Selected WAIT cases 2 and 7 complete; slot 33
reaches 98.4% but does not complete. Sampling demonstrates an escape mechanism,
not a reliable deployment solution. Its output is explicitly a diagnostic
schema and cannot qualify the greedy completion gate.

Manual road views show slot 290 finishing most work by step 71, then moving
away to the upper-left boundary and repeating an invalid base turn. Slot 345
finishes excavation by step 113 but leaves 43.4% of soil outside the accepted
zone, then repeats an 18-action cabin/DO sequence. CPU witnesses find legal
movement and local loose-soil pickup alternatives with 337–379 decisions left
after the last material change. These witnesses do not prove complete disposal
plans, but they rule out a claim that every available action is blocked.

The current trench observer replay has a preserved full-panel parity failure:
23 endpoint fields differ across eight unselected episodes. All five selected
visual cases match every checked endpoint, and the total trench exact count
remains 162. Use the scoped receipt for these clips; do not call this full-panel
trace parity. The cause of the other trajectory differences is unestablished.
Both foundation greedy replays match all 64 endpoints.

## What the code and checkpoint audit establishes

Native model, Adam state and loss are finite. The actual Adam clock advances
from 2,112,000 to 2,624,000, with learning rate 3e-4 and absolute entropy floor
0.02 unchanged. No ramp, warm start, bank transfer, optimizer reset or schedule
restart occurred. All 70 saved environment leaves match except a live
curriculum failure counter. Actual resume/overlay code introduces no hidden
environment setting change. Sixty-three focused CPU environment tests pass,
including short movement, chassis-soil protection, junction approaches and
executable-DO observation parity.

Both policies omit stall age, previous-action outcome and movement-feasibility
observations. They have five previous actions and no recurrent memory. The
environment records stall age, but the policy does not receive it. Once a
no-effect action repeats long enough to fill action history, the greedy actor
can receive an identical input indefinitely. The existing optional action
mask deliberately keeps WAIT valid, so enabling that mask alone would not
eliminate these explicit WAIT failures.

One-to-four-GPU scaling also changes PPO semantics: the local normalized
minibatch shrinks from 16 trajectories × 32 steps to 4 × 32 steps. Advantages
are centered and scaled locally before gradients are averaged across devices.
The global 512 samples per optimizer step and 64 Adam steps per update are
preserved, but the normalized objective is not identical. This is a causal
hypothesis requiring a controlled comparison, not a demonstrated cause of the
foundation regression.

Historical trench training used 2048 global environments versus 512 now:
65,536 versus 16,384 transitions per update. Its 20,000-update entropy decay
therefore spanned 1.311 billion transitions versus 0.328 billion now. The
current u85500 has about 1.401 billion transitions, versus 3.375 billion at
historical u51500. Compare experience budgets as well as update counts.

Cross-era physical rules also changed: positive soil blocks chassis travel,
tracked movement checks the swept footprint, and digging/deposition exclude
chassis support cells. The first repair briefly introduced spurious rounded
intermediate-path collisions; current continuous-sweep code fixes those.
Old weights still solved 19/32 road maps under the first repaired environment,
including selected slots 290 and 345. This rules out a blanket claim that
all road maps became impossible. Preserve the soil-free chassis requirement.

The newer executable-dig observation does not hide loose soil. Both old and
new admissible-dig channels describe fresh excavation; positive-soil local
observations remain available. At failed road slot 345, correct cabin headings
show 2–6 units of recoverable soil, whereas the policy attempts DO at empty
headings. Soil-height clipping existed before this change. A separate
executable-relift channel would be a new feature, not restoration of deleted
information.

## Next decisions

Retrieve and evaluate the newest remote checkpoints after CSCS authentication
renewal. The local results above concern checkpoints saved this morning, not
unverified evening training. Keep added behavior penalties off and retain the
older checkpoints for matched comparison.

Retain old u83500 as a recovery benchmark: its current-environment replay
shows substantially stronger road-soil handling. It is a candidate parent for
a future controlled recovery experiment, not automatically a promoted policy.
The completed greedy-versus-sampled diagnostic identifies foundation WAIT
traps but does not qualify a greedy deployment policy.

For the next controlled training comparison, preserve a common parent, bank,
global batch, optimizer clock and transition budget, and isolate global versus
per-device advantage normalization. Qualify policies with the actual deployment
decoder. Consider explicit outcome/feasibility feedback or a bounded recovery
decoder only as a declared treatment with its own completion and workspace
checks. Do not use stronger base-travel penalties to try to cure WAIT collapse.

## Evidence

[Evening comparisons](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/evaluation_comparison.json),
[historical comparison](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/historical_trench_audit.md),
[environment audit](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/environment_contract_audit.md),
[optimizer audit](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/optimizer_contract_review.md),
[same-input and decoder review](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/stalled_policy_decoder_review.md),
[native resume receipt](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/foundation_resume_audit.json),
[rollout viewer](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/qualitative/index.html),
[early legal-action witnesses](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/qualitative/foundation_41000/early_legal_action_probes.json).

[Old weights in current environment report](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/qualitative/old_trench_83500_current_env/fixed.json).

Independent final review verifies all 608 reset identities across old/native,
old/current and new/current reports; all 1,824 rows pass integrity checks.
Both checkpoint SHA values match their actual files. See the
[final replay review](../../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/old_trench_current_environment_review.md).
