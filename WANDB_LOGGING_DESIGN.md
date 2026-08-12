# Terra W&B logging design

Status: design validated against the completed P5c foundation and trench
specialists; implementation complete and API-validated; rendered human review
pending

## 1. Goal

The default W&B view must let a researcher answer, in this order:

1. Is the policy completing tasks on fixed unseen scenarios?
2. What map population is it training on?
3. Is useful excavation behavior improving?
4. Is PPO and kickstart optimization healthy?

The present P5c foundation specialist exposes 344 user metric keys. Ninety-four
come from the legacy `curriculum_levels` tree, 184 come from sampler telemetry,
and 30 are numerical diagnostics. Many are constant, duplicated, or one scalar
per condition. W&B therefore creates hundreds of single-metric panels, while
the fixed-bank success results that decide whether a policy generalizes are
only stored in Euler JSON files.

The replacement is one small scalar schema, two detail tables, and one curated
workspace. Full receipts remain files; they are not time-series charts.

## 2. The completion numbers

The first dashboard section must use these exact definitions.

| Metric | Definition | Use |
|---|---|---|
| `eval/<split>/exact_success_rate` | Fixed scenarios with `absolute_completion >= 1 - 1e-6` divided by all scenarios in the split | Primary answer to “what fraction completed correctly?” |
| `eval/<split>/macro_completion` | Mean of per-condition mean terminal absolute completion | Graded progress when exact success is sparse |
| `eval/<split>/worst_condition_completion` | Minimum per-condition mean terminal absolute completion | Exposes a dead condition |
| `train/episode_success_rate` | Exact task completions divided by all ended training episodes in the logging window | Online training diagnostic |
| `train/episode_timeout_rate` | Timeouts divided by all ended training episodes in the logging window | Online failure diagnostic |

`train/episode_success_rate` is undefined when no episode ended and must be
logged as NaN, never zero. Evaluation rates enumerate each fixed initial
scenario once and are bounded in `[0, 1]`.

Do not show these legacy/proxy quantities in the default workspace:

- `progress/episode_completion_rate`: final-rollout-step terminations, including
  timeouts;
- `eval/positive_terminations`: successful auto-reset episodes per initial
  environment, which can exceed one; or
- reward return as a substitute for task completion.

## 3. Default workspace

The saved workspace is named **Terra RL - Human v1**. It has at most sixteen
visible panels. Related series share a panel instead of generating one panel
per scalar.

### A. Task outcome

1. **Fixed exact success**: promotion and development exact-success rates.
2. **Fixed graded completion**: promotion/development macro completion and
   worst-condition completion.
3. **Family exact success**: foundation and trench rates for promotion and
   development. Always show both families when the fixed evaluation contains
   both; the run config identifies the trained family. This makes specialist
   transfer or regression visible instead of hiding it.
4. **Online outcomes**: training success and timeout rates plus inline-evaluation
   success and termination within the horizon when inline evaluation is enabled.

Fixed evaluation uses `eval/update` as its own step metric. Evaluation may run
after training; it must still plot against the checkpoint update rather than
W&B's append order.

### B. Curriculum population

5. **Current family population**: foundation and trench fractions of active
   environments.
6. **Current depth population**: Anchor, One-axis, and Composed fractions.
7. **Sampler concentration**: target-distribution effective sample size and
   normalized entropy.
8. **Condition snapshot**: one table, updated only at sampler refresh or
   checkpoint, with:

   `condition`, `family`, `depth`, `target_probability`,
   `active_population_fraction`, `reset_exposure_fraction`,
   `ended_episode_fraction`, `train_success_rate`, and
   `mean_absolute_completion`.

The table replaces all per-condition `sampler_q/*`,
`sampler_*_current/*`, and `sampler_*_closed/*` scalar series. Detailed JSON
receipts remain authoritative. Family/depth scalars are retained because they
are small and useful as curves.

### C. Behavior and reward

9. **Task progress**: mean absolute, dig, and legal-dump-volume completion over
   ended training episodes.
10. **Dump quality**: dump purity and no-effect action rate.
11. **Action distribution**: the eight action fractions in one panel, retained
    to expose policy collapse without creating eight standalone panels.
12. **Reward**: mean episode return plus dense-agent, terminal, trench, and
    existence components per ended episode in one panel. Disabled components
    may be absent; they do not get standalone zero-valued panels.

Mean episode length and productive workspace cycles remain available in the
collapsed details section rather than consuming an overview panel.

Reward curves diagnose learning incentives. They never promote a checkpoint.

### D. Optimization

13. **PPO losses**: total, policy, and value loss.
14. **Policy distribution**: policy entropy, entropy coefficient, approximate
    KL, and clip fraction.
15. **Fit and gradients**: explained variance and global gradient norm.
16. **Kickstart**: teacher-policy KL, teacher-value MSE, KL coefficient, and
    value coefficient. These remain visible because the current runs are
    warm-started and the kickstart schedules materially change the objective.

Throughput is available in a collapsed **System** section with steps per
second and cumulative environment steps. It is not part of the scientific
overview.

## 4. Scalar schema

Future runs use `logging_schema=terra_wandb_human_v1` in W&B config. The
intended scalar history is bounded; condition count must not create new scalar
keys.

```text
train/update
train/episode_success_rate
train/episode_timeout_rate
train/ended_episodes

behavior/absolute_completion
behavior/dig_completion
behavior/dump_volume_completion
behavior/dump_purity
behavior/no_effect_action_rate
behavior/mean_episode_length
behavior/productive_workspace_cycles_per_episode
behavior/action_fraction/forward
behavior/action_fraction/backward
behavior/action_fraction/base_clockwise
behavior/action_fraction/base_anticlockwise
behavior/action_fraction/cabin_clockwise
behavior/action_fraction/cabin_anticlockwise
behavior/action_fraction/do
behavior/action_fraction/no_op

reward/episode_return
reward/agent
reward/terminal
reward/trench                 # only when enabled/nonzero
reward/existence              # only when enabled/nonzero

curriculum/population/foundation
curriculum/population/trench
curriculum/population/Anchor
curriculum/population/One-axis
curriculum/population/Composed
curriculum/target_ess
curriculum/target_entropy_normalized
curriculum/refreshes

ppo/total_loss
ppo/policy_loss
ppo/value_loss
ppo/entropy
ppo/entropy_coef
ppo/approx_kl
ppo/clip_fraction
ppo/explained_variance
ppo/grad_norm

kickstart/kl
kickstart/value_mse
kickstart/kl_coef
kickstart/value_coef

system/steps_per_second
system/environment_steps
```

Fixed evaluation adds only this bounded family of keys:

```text
eval/update
eval/<split>/exact_success_rate
eval/<split>/macro_completion
eval/<split>/worst_condition_completion
eval/<split>/zero_completion_rate
eval/<split>/<family>_exact_success_rate
eval/<split>/<family>_macro_completion
```

`<split>` is `promotion` or `development`; `<family>` is `foundation` or
`trench`. Per-condition evaluation belongs in one table, not scalar keys.

Optional inline evaluation uses the same `train/update` x-axis and only three
scalars: completed-episode success, success within the horizon, and termination
within the horizon.

## 5. What leaves scalar history

- All `integrity/*`, finite-fraction, mutation, mass-residual, and reward-
  residual series. They remain hard failures and machine-readable receipts.
- `curriculum_levels.*`, including duplicated agent-type population trees.
- Per-condition sampler scalars and the `current`/`closed` Cartesian product.
- Constant horizon, learning rate, GPU count, condition count, sampler mode,
  intended uniform masses, code revisions, and hashes. These belong in W&B
  config or final summary.
- Absolute-max numerical probes other than global gradient norm. They remain in
  failure receipts when a finite check trips.
- Raw action counts and per-termination/per-condition internals. These remain
  in the episode-aggregate JSON and detail tables.

Existing P5c histories are immutable and remain verbose. A curated report can
make them readable, but they are not rewritten or presented as v1-schema runs.

## 6. Logging cadence

- Training, behavior, PPO, kickstart, and aggregate curriculum scalars: the
  existing training log interval.
- Condition snapshot table: sampler refresh and checkpoint only.
- Fixed evaluation scalars/table: once per evaluated checkpoint, using
  `eval/update` as the x-axis.
- Config: once at run initialization.
- Final receipt/status: once in run summary and in the existing artifact.

No scalar is logged merely to prove that an invariant stayed true.

## 7. Implementation boundary

Keep one direct path:

1. Replace `_episode_aggregate_wandb_metrics` with the v1 human metrics.
2. Replace `get_curriculum_levels(...)` and the unbounded sampler telemetry in
   `train_mixed.py` with family/depth aggregates plus one condition table.
3. Rename the small PPO scalar set and add approximate KL/clip fraction at the
   existing loss-reduction point.
4. Add a small post-evaluation logger that consumes the existing fixed-bank
   JSON and resumes the same W&B run using `eval/update`.
5. Materialize one saved workspace/report from an explicit panel list.

Do not add a logging framework, compatibility aliases, a second evaluator, or
a database. Old runs keep old keys; new runs use the new schema directly.

## 8. Validation gates

The design is accepted only when all of the following pass:

1. A projection from the completed P5c foundation and trench specialists
   produces the outcome, curriculum, behavior, PPO, and kickstart panels from
   real data.
2. Fixed evaluation JSON produces bounded exact rates and condition-macro
   completion at all eight checkpoints, with exact agreement against the
   leaderboard inputs.
3. An offline two-step W&B smoke contains no `integrity/*`,
   `curriculum_levels*`, `sampler_q/*`, or per-condition scalar keys.
4. The scalar-key budget is at most forty-eight training keys regardless of whether
   the run has 14, 18, or 32 conditions.
5. Four compact contract tests cover: episode formulas and zero-episode NaN;
   curriculum population plus the condition table; fixed-evaluation
   reconstruction; and the bounded schema plus manual workspace layout.
6. A rendered desktop view puts fixed exact success in the first row, contains
   no single-panel constant flags, and keeps condition details in tables.
7. Existing episode receipts, checkpoint contents, environment transitions,
   PPO settings, and fixed-evaluation semantics remain unchanged.

This is research instrumentation, not a production observability platform. Do
not add a test per helper or metric. During iteration run the four focused
contracts; run the existing shared PPO/aggregate tests once before handoff, and
use one two-step offline W&B smoke as the only new integration check.

## 9. Validation receipt

The design was checked against the completed P5c runs rather than invented
from desired metric names.

### 9.1 Current-history audit

| Run | User scalar keys | Constant scalar keys | Legacy curriculum keys | Sampler keys |
|---|---:|---:|---:|---:|
| Foundation specialist | 344 | 61 | 94 | 184 |
| Trench specialist | 296 | 57 | 94 | 136 |
| Medium uniform generalist | 519 | 76 | 94 | 359 |

The proposed overview metrics can be projected from both specialist histories:
outcomes, task progress, dump quality, return, PPO losses, entropy, fit,
gradients, kickstart losses and coefficients, and throughput are all present.
The existing histories lack only the intentionally new aggregates (mean episode
length, clean reward-component means, clean population summaries, approximate
KL, and clip fraction) and fixed-evaluation curves.

The generalist's active population also demonstrates that family and depth are
the useful curriculum summaries. From first to last logged update, foundation
population was 0.559 to 0.562 and trench was 0.441 to 0.438; Anchor was 0.279 to
0.281, One-axis 0.600 to 0.593, and Composed 0.121 to 0.125. The hundreds of
individual condition-assignment flags add no overview value.

### 9.2 Fixed-evaluation reconstruction

For every one of the eight evaluated checkpoints in both specialists and both
splits, exact success, macro completion, and worst-condition completion were
recomputed directly from `per_map` rows. All reconstructed values agree with
the saved fixed-evaluation summaries within `1e-12`.

| Specialist / split | Exact, first to last | Macro, first to last | Worst condition, first to last |
|---|---:|---:|---:|
| Foundation / promotion | 0.002 to 0.188 | 0.591 to 0.558 | 0.270 to 0.248 |
| Foundation / development | 0.004 to 0.201 | 0.566 to 0.557 | 0.163 to 0.323 |
| Trench / promotion | 0.000 to 0.227 | 0.598 to 0.401 | 0.173 to 0.054 |
| Trench / development | 0.002 to 0.209 | 0.588 to 0.374 | 0.183 to 0.053 |

This is the key design validation: exact success alone gives a misleading view
of the trench specialist. Its exact completion improved while average graded
completion, the worst condition, and the zero-progress rate deteriorated. The
first row therefore needs exact, macro, and worst-condition curves together.

### 9.3 Design decision

The bounded scalar schema and sixteen-panel layout are accepted. Implementation
may proceed, but the remaining gates in section 8 still apply to the code and
rendered workspace. No training result is reinterpreted by this decision.

### 9.4 Accepted review correction

The first implementation draft removed the legacy per-action evaluation keys
along with the noisy diagnostics. Review established that action distribution
is useful for spotting collapse. The eight fractions are therefore retained as
one grouped training panel. This raises the bounded schema from 39 to 47
possible training scalars, including the explicit `train/update` x-axis; it
does not reintroduce condition-dependent keys.

### 9.5 Implementation receipt

- The implemented training schema contains 47 possible scalar keys, independent
  of condition count. Runtime rejects unknown keys or more than 48 scalars.
- Four compact contract tests cover the load-bearing formulas, population and
  condition accounting, bounded metric selection and workspace layout, and
  fixed-evaluation reconstruction. They run with the existing sampler tests in
  0.11 seconds; the shared aggregate/PPO tests passed once before handoff.
- One two-update offline W&B smoke accepted the custom `train/update` step,
  grouped action fractions, and bounded metric names without any banned noisy
  prefix.
- Real P5b promotion/development files validate at updates 500, 1000, 1500, and
  2000 and produce 17 fixed-evaluation scalars plus condition tables.
- The saved workspace is
  <https://wandb.ai/aless-weber-eth/mixed-agents?nw=84ulh56rci5>. An API
  round-trip confirms four open four-panel overview sections, one collapsed
  four-panel details section, disabled automatic panel generation, fixed exact
  success first, and grouped action distribution present.

The remaining validation gate is Lorenzo's rendered browser review of panel
readability. Existing P5c runs cannot retroactively populate metrics that were
not logged; the v1 schema applies to future runs.
