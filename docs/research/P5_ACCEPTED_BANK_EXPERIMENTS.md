# P5 Accepted-Bank Experiment Implementation

- Status: P5, P5b, and P5c fixed evaluations complete; no long continuation
  admitted; checkpointed-duration continuation protocol accepted
- Date: 2026-08-03
- Canonical authority:
  [`D5_D7_IMPLEMENTATION_PLAN.md`](/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730/D5_D7_IMPLEMENTATION_PLAN.md)
  §P5a
- Research-code constraints:
  [`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)
- Terra branch: `experiment/simple-mapbank-reward-v3`
- terra-baselines branch: `experiment/simple-mapbank-reward-v3`

This document is the executable terra-baselines side of the canonical P5a
plan. The canonical plan should link here rather than duplicate these commands
or implementation details.

Final P5 evidence, the standardized condition leaderboard, and the frozen
three-arm warm-start depth/sampler screen are recorded in
[`P5_RESULTS_ANALYSIS.md`](P5_RESULTS_ANALYSIS.md) and
[`P5_FOLLOWUP_GOAL.md`](P5_FOLLOWUP_GOAL.md). Those documents do not change the
P5 treatment after the fact.

## 1. Question and minimal matrix

The initial screen has six scratch-trained arms:

| Arm | Training conditions | Condition sampler | Question |
|---|---|---|---|
| `F-ANCHOR` | accepted foundation `Anchor` conditions | uniform | are the new foundation anchors learnable? |
| `F-SPECIALIST` | every accepted foundation condition | uniform | can a foundation-only policy learn the full foundation distribution? |
| `T-ANCHOR` | accepted trench `Anchor` conditions | uniform | are the new trench anchors learnable? |
| `T-SPECIALIST` | every accepted trench condition | uniform | can a trench-only policy learn the full trench distribution? |
| `G-UNIFORM` | every accepted condition | uniform | can one generalist learn the target distribution directly? |
| `G-ADAPTIVE` | the identical all-condition bank | adaptive progressive | does progressive exposure improve the generalist? |

The anchor controls and family specialists are feasibility diagnostics. They
do not enter the causal curriculum ablation or promotion decision, which
compares only `G-UNIFORM` and `G-ADAPTIVE`.

They are not clean negative-transfer A/B comparisons with the generalist.
With the frozen 18-foundation/14-trench/32-total condition set, uniform sampling
gives each `F-SPECIALIST` condition `1/18` of assignments and each
`T-SPECIALIST` condition `1/14`, versus `1/32` in `G-UNIFORM`: respectively
`1.78x` and `2.29x` more dose per condition. The specialists answer whether a
family is learnable under concentrated family-only compute. A specialist is
invalid when its accepted family contains only anchors, because that would
duplicate the anchor control; the loader rejects this before JAX starts.

No strict stage unlock, per-environment demotion, partial reset, reward
curriculum, teacher, or old 12-map bank is part of this comparison.

All six arms freeze:

- agent-neutral `relocation_progress_mult = 1.5`;
- horizon 450, `DENSE`, trench absolute shaping off;
- tracked single-excavator actions and observations;
- scratch initialization;
- medium E8 MLP with `resnet_spatial_8x8_se`, bf16 encoder and
  `512,256` critic;
- PPO and entropy settings supplied by the one shared entrypoint; and
- one accepted-bank protocol hash.

## 2. Accepted-bank input contract

`--accepted-bank-root` is required. Its `dataset.json` must use
`terra_curriculum_loader_bank_v1` and contain:

- `environment_protocol = "environment_protocol.json"` plus its canonical
  SHA-256;
- `reset_prng = {jax_default_prng_impl: threefry2x32,
  jax_threefry_partitionable: true}` inside that hashed protocol;
- `scenario_identity_contract = "terra_reset_arrays_sha256_v1"` at the bank
  root, as emitted and enforced by the paired Terra commit;
- `source_registry = "source_registry.jsonl"` plus its file SHA-256;
- `train[]` entries with `condition_id`, `family`, `branch_depth`,
  `maps_path`, and exactly `map_count = 64`;
- exact MapsBuffer datasets under every training `maps_path`; and
- `promotion`, `development`, and `sealed` evaluation panels.

The caller must pass the exact immutable Terra revision recorded in the source
archive manifest. Before JAX initialization, the loader:

1. verifies the protocol receipt hash and equality with the protocol derived
   from the imported Terra code plus that explicit revision, without consulting
   `.git`;
2. rejects a missing or different reset-PRNG contract before constructing an
   environment;
3. verifies the source-registry hash;
4. verifies every staged training path declares exactly 64 maps, has one
   condition with contiguous manifest slots `1..64`, requires local
   `slot_count = 64`, and contains exactly `img_1.npy..img_64.npy` in each of
   the five reset-array directories;
5. verifies every evaluation panel path, count, condition count and contiguous
   manifest; and
6. recomputes each evaluation `episode_id` from `scenario_id`, `reset_seed`,
   and the frozen protocol hash.

This makes review-only banks, the legacy 12-map bank, stale Terra revisions,
and inherited `DATASET_PATH` values invalid inputs.

## 3. Adaptive sampler

There is one small host-side sampler. Every map condition owns one Terra level;
the trainer writes an explicit categorical level assignment for every
environment.

The uniform arm keeps `q = Uniform(conditions)`.

Every 150 PPO updates, the adaptive arm computes a completion EMA for each
condition with enough completed episodes, then:

```text
q = 0.20 * Uniform + 0.80 * Frontier
```

`Frontier` is a temperature-softmax over completion of conditions below the
0.75 mastery threshold. A condition stuck at zero therefore cannot monopolize
training; the most advanced unsolved conditions receive more exposure. A
mastered condition remains in the uniform floor and re-enters automatically if
its completion EMA falls below the threshold. Per-condition mass is capped at
0.15.

The receipt keeps three quantities separate: sampled level assignments, maps
actually instantiated by reset, and completed episodes used to estimate
competence. It reports both the accumulating current window and the last
closed window; completed-episode mass is never labelled as realized exposure.

## 4. Fixed evaluation and comparison gate

Use `eval_fixed_bank.py --accepted-panel` for the frozen `promotion`,
`development`, or `sealed` panel. It consumes each row's frozen `reset_seed`
and verifies that it selects the declared exact slot. Seed-to-slot mapping is
part of the executable protocol: the evaluator asserts partitionable
`threefry2x32` before deriving reset keys, so an import-order or JAX-config
change fails rather than silently shuffling or duplicating panel slots.

Every checkpoint reports:

- exact successes and rate;
- condition-macro terminal absolute completion;
- micro completion mean, median, p10 and p25;
- worst condition and its completion; and
- per-family and per-condition results.

For each checkpoint after the first, the evaluator writes a comparison to the
previous checkpoint. The gate passes only when integrity is clean and:

```text
progress =
    at least one additional exact-success map
    OR condition-macro completion gain >= 0.01

guards =
    micro p10 delta >= -0.05
    AND worst-condition delta >= -0.05
```

The exact-rate quantum is recorded as `1 / panel_size`. There are no
`26/32`, `6/8`, or other panel-size-specific counts.

The two specialist policies still evaluate the complete frozen panel so their
cross-family behavior remains visible, but their primary diagnostic fields are
the trained family's values:

- exact successes in `summary.by_family[family]`;
- macro completion, micro p10 and worst-condition completion in
  `summary.graded.by_family[family]`; and
- clean global `summary.integrity`.

The untrained family's scores are transfer diagnostics. The evaluator's global
`comparison_to_previous` gate is not a specialist feasibility gate: its macro
and worst-condition terms include the untrained family. The bounded specialist
screen is interpreted as progress when trained-family exact success increases
by at least one map or trained-family macro completion increases by at least
`0.01`, while that family's p10 and worst-condition completion each regress by
no more than `0.05`. These diagnostics never select the generalist promotion.

## 5. Entrypoints

The six YAML config names are the arm names themselves. A single non-Slurm
entrypoint keeps architecture and PPO arguments shared:

```bash
export TERRA_ROOT=/path/to/source-archive/terra
export TERRA_REVISION="$(cat /path/to/source-archive/terra/REVISION)"
export RUN_ROOT=/path/to/run-artifacts
export PYTHON_BIN=/path/to/python
export SEED=0

scripts/run_accepted_bank_screen.sh \
  G-UNIFORM /path/to/accepted-bank p5-g-uniform-s0 2000
```

`NUM_DEVICES`, `NUM_ENVS_PER_DEVICE`, and `NUM_STEPS` are operational
parameters; their defaults are `1`, `1024`, and `32`. The script computes
`total_timesteps` from those values and the required update count. `SEED` is
required and must match between paired `G-UNIFORM` and `G-ADAPTIVE` runs.
The production PPO shape is frozen at 2 update epochs, 32 minibatches and
learning rate `3e-4`; inline evaluation is disabled, and numbered checkpoints
are retained every 500 updates. `FINITE_CHECK_INTERVAL` defaults to 10 for
2k/20k runs; set it to 1 for an update-1 smoke. The script neither submits a
job nor chooses an Euler partition. It accepts no trailing training arguments,
so callers cannot override the frozen treatment.

Example fixed evaluation:

```bash
python eval_fixed_bank.py \
  --checkpoint /path/to/checkpoint.pkl \
  --bank-root /path/to/accepted-bank \
  --accepted-panel development \
  --terra-revision "$TERRA_REVISION" \
  --expect-completion-contract exact_visible_dump_v1 \
  --output /path/to/development_eval.json
```

The immutable Euler preparation, smoke, screen, reset-parity and fail-closed
future-20k boundary are documented in
[`scripts/euler_accepted_bank_v1/README.md`](../../scripts/euler_accepted_bank_v1/README.md).
`SUBMIT=0` is local-only and performs no remote mutation.

## 6. Implementation checklist

- [x] reject legacy/review-only/stale bank roots before JAX;
- [x] select foundation anchors, all foundations, trench anchors, all trenches,
  or all accepted conditions from descriptor fields rather than directory-name
  conventions;
- [x] port only the global uniform/adaptive sampler, not the old M3 bank or
  strict staged scheduler;
- [x] freeze reward and horizon across condition levels;
- [x] disable the per-environment Terra ratchet;
- [x] log intended mass, sampled assignments, reset exposure and completed
  episodes without conflating them;
- [x] add the six parameterized configs and one shared local/allocation
  entrypoint;
- [x] freeze and test the production PPO shape, explicit seed, checkpoint
  cadence, and finite-check cadence in that entrypoint;
- [x] add condition-macro, micro-p10 and worst-condition evaluation;
- [x] replace fixed panel counts with a continuous, map-count-aware comparison
  gate;
- [x] use frozen evaluation reset seeds and episode identities;
- [x] bind the JAX reset-PRNG mode into the protocol and reject runtime/config
  disagreement before fixed-panel reset;
- [x] validate archive runs from an explicit frozen Terra revision without
  requiring `.git`;
- [x] require reset-array scenario identity before JAX;
- [x] separate assignments, reset exposure and completed-episode competence;
- [x] reject mixed-treatment or duplicate-update fixed-evaluation lists;
- [x] implement content-addressed Euler smoke and screen launch;
- [x] fail closed on P6 until the separate 256-train-layouts/condition bank
  exists; retain the generalist selection receipt as the future P6 gate;
- [x] pass the full terra-baselines CPU suite against the paired Terra commit
  (`249 passed`);
- [x] complete one local or allocated-GPU first-update smoke per arm;
- [x] submit the bounded screens only after the accepted bank is frozen.

The unchecked items are execution gates, not evidence supplied by this
implementation commit alone.

## 7. 2026-08-01 allocated-gate finding

The first content-addressed six-arm smoke completed update 1 and passed its
checkpoint receipts. The following screen attempt stopped before W&B or PPO in
the CPU/GPU reset-parity gate: the loader had generated scalar reset seeds with
legacy `jax_threefry_partitionable=false`, while `train.py` and
`train_mixed.py` explicitly run with `true`. Under the real runtime the
512-slot promotion panel selected only 319 unique slots.

The accepted correction is to preserve all reviewed maps and source-disjoint
splits, pin partitionable Threefry in the hashed environment protocol,
regenerate only evaluation reset seeds and episode IDs, and rerun the six
smokes and screens. Forcing the evaluator back to the legacy mode is rejected
because it would create a different stochastic environment contract from the
trainer.

## 8. Final P5 execution result

The PRNG-corrected six-arm campaign
`f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa`
completed on 2026-08-02. Every job produced a passing receipt, 2,000 updates,
262,144,000 transitions, and exact deterministic promotion/development
evaluation. The formal selector chose `G-ADAPTIVE` because it alone passed the
update-1,000 to update-2,000 promotion-retention gate. This is one paired seed,
not a general scheduler-superiority claim. P6 remains fail-closed until the
separate 256-training-layouts-per-condition bank exists.

## 9. P5b execution result

The matched P5b jobs completed on 2026-08-02:

| Arm | Slurm | Final state | Selected fixed checkpoint |
|---|---:|---|---:|
| `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | `COMPLETED 0:0`, `PASSED` | 2,000 |
| `G-DEEP-ADAPTIVE-WARM` | `9378175` | `COMPLETED 0:0`, `PASSED` | 1,000 |
| `G-MEDIUM-UNIFORM-WARM` | `9378176` | `COMPLETED 0:0`, `PASSED` | 1,000 |

The selected policies score promotion/development macro
`0.652/0.625`, `0.653/0.628`, and `0.647/0.664`, respectively. These
different-update selections are descriptive. At the matched update 1,000,
deep/adaptive improves over medium/adaptive by `+0.023/+0.013` and
medium/uniform by `+0.017/+0.049`; neither advantage survives to update 2,000.
The full condition and factor tables are frozen in
`/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`.

P5b deep used exact function-preserving growth from `2,441,223` to
`2,699,117` parameters, a fresh optimizer, and the common frozen parent as KL
and value teacher. That is the intended grow-and-teach implementation. E8 is
not the historical bigger-network analogue: E8 and E3 both have `2,441,223`
parameters; the historical growth step was the approximately `994,825`-
parameter model to E3.

All P5b arms show a synchronized fixed-evaluation decline at update 1,500 as
KL reaches zero. Because P5b entropy remains about `0.137` at that point under
its `0.15 -> 0.005` over 7,600-update schedule, the next matrix tests the common
lower historical entropy schedule. This diagnosis is correlational. P5b/P5c
entropy comparisons are interpreted only at matched checkpoints through
update 2,000.

## 10. Diagnostic capability-floor contract

The two capability-floor conditions are:

- `fnd-slab-allfree`, paired to `fnd-slab-ring3x`; and
- `trn-straight-allfree`, paired to `trn-straight-side2`.

The pairing freezes source group, pair slot, split, excavation mask, obstacle
mask, reset seed, horizon, dynamics, and evaluation protocol. Only the target
mask changes: every legal non-dig cell is an accepted visible dump cell. Each
condition has 64 training, 16 promotion, 16 development, and 32 sealed maps.

This bank uses an explicit diagnostic-control contract and is rejected by the
ordinary accepted-bank path unless the evaluator opts into diagnostic mode.
Its two conditions:

- are excluded from all constrained 32-condition macro, family, factor, and
  promotion computations;
- are evaluated as their own promotion and development panels;
- preserve frozen reset seeds and episode identities; and
- cannot be used as P5c training support.

The current bank and result roots are:

- `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_controls_20260802_0306c3cd`;
- `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`.

The P5 parent, P5b medium/adaptive @2,000, deep/adaptive @1,000, and
medium/uniform @1,000 score promotion/development macro
`0.385/0.465`, `0.629/0.613`, `0.718/0.736`, and `0.484/0.540` on these
controls. Every evaluation is integrity-clean and exact is `0/32` on both
panels. These maps are physically permissive but target-mask OOD. A future
training-support treatment must create a versioned 34-condition successor
whose original 32 conditions remain byte-identical; it must not mutate P5c.

## 11. P5c low-entropy implementation contract

P5c consists of five arms, each run for 4,000 added PPO updates and evaluated
every 500 updates:

| Arm | Accepted support | Sampler | Architecture |
|---|---|---|---|
| `G-MEDIUM-ADAPTIVE-WARM` | all 32 conditions | adaptive | medium SE |
| `G-MEDIUM-UNIFORM-WARM` | all 32 conditions | uniform | medium SE |
| `G-DEEP-UNIFORM-WARM` | all 32 conditions | uniform | depth-grown SE |
| `F-MEDIUM-UNIFORM-WARM` | 18 foundation conditions | uniform | medium SE |
| `T-MEDIUM-UNIFORM-WARM` | 14 trench conditions | uniform | medium SE |

All arms share the P5 parent parameters, frozen P5 teacher, fresh optimizer,
entropy `0.02 -> 0.005` over 10,000 updates, KL/value schedules, accepted-bank
release, reward, horizon, observations, actions, PPO shape, full-reset
distribution, and seed. Generalists select all 32 conditions; specialists
select the existing family subset.
All also freeze `enforce_foundation_border_alignment=false`; edge alignment is
not part of the P5c completion or reward treatment. Any later straight-edge
experiment is a separate per-map ablation and cannot silently change these
results.
The causal comparisons are:

1. medium/adaptive versus medium/uniform for sampler choice; and
2. medium/uniform versus deep/uniform for residual depth.

The foundation and trench specialists are dose ceilings. Because they receive
more assignments per condition, they do not select the generalist and cannot
be used as negative-transfer evidence.

Every numbered P5c checkpoint must produce four integrity-clean fixed
evaluations: constrained promotion, constrained development, diagnostic
all-free promotion, and diagnostic all-free development. The long-run gate
requires improvement across multiple checkpoints, confirmation on both public
panels, no material family or bottom-tail regression, and stable or improving
capability-floor results. P5c allocated update-1 smoke jobs `9458568`,
`9458581`, `9458585`, `9458616`, and `9458619` all completed `0:0` and passed
their explicit contracts. Screen jobs `9461489`, `9461500`, `9461504`,
`9461507`, and `9461512` consequently completed 4,000 updates from the exact
tested revision. They are early learning-curve screens, not convergence runs.

## 12. Checkpointed-duration continuation contract

The completed P5c W&B histories establish that the prior compute budget was
too short for a saturation claim:

| Arm | W&B runtime | Final logged online `episodes/task_done_rate` |
|---|---:|---:|
| `G-MEDIUM-ADAPTIVE-WARM` | 6.27 h | 0.422 |
| `G-MEDIUM-UNIFORM-WARM` | 6.22 h | 0.394 |
| `G-DEEP-UNIFORM-WARM` | 7.98 h | 0.611 |
| `F-MEDIUM-UNIFORM-WARM` | 6.32 h | 0.522 |
| `T-MEDIUM-UNIFORM-WARM` | 6.21 h | 0.796 |

The online rates follow the sampled training distribution and do not prove
held-out generalization. They were nevertheless still rising late in training,
so the 4,000-update runs cannot be called converged or saturated. Their proper
interpretation is promising but undertrained, with a possible train/test or
condition-retention gap to measure using fixed evaluation.

That fixed evaluation is now complete. No arm passed the predeclared
improvement/retention gate at two consecutive checkpoints, so no P5c
checkpoint is formally selected and no 120-hour continuation was launched.
Deep uniform at update 4,000 is the strongest descriptive endpoint
(`0.624/0.586` constrained promotion/development macro and `168/512` /
`143/512` exact), but its preceding interval failed the family/tail retention
requirements. This is an unstable-learning-curve result, not a saturation
claim and not authorization to bypass the separate P6 bank gate.

Future behavioral training uses this duration protocol:

1. update-1/minute-scale jobs remain runtime smokes and provide no behavioral
   conclusion;
2. a research screen receives at least one healthy 24-hour allocation unless
   it fails numerically or is explicitly declared a shorter diagnostic;
3. configure an absolute update target above what the allocation can finish,
   retain a validated rolling checkpoint every 100--500 updates, and treat a
   wall-time exit with a valid checkpoint as `CONTINUABLE`;
4. continue promising policies with true `--resume_from` state on the 120-hour
   queue rather than restarting parameters or optimizer state;
5. preserve architecture, bank, reward, horizon, PPO shape, schedule, global
   update, and adaptive sampler state across segments; and
6. stop only after fixed promotion/development exact, macro, and tail metrics
   plateau across multiple checkpoints. Report matched updates/transitions and
   GPU-hours as separate comparison axes.

`total_timesteps` is the absolute final target, not the number of additional
steps in the next segment. Terra restores model, optimizer, update/schedule,
and pooled-sampler state, while RNG, live environment state, and action history
restart at the boundary; continuation is statistically continuous, not
bit-exact. If a hard timeout prevents the launcher from running evaluation,
evaluate the latest complete checkpoint in a separate job before interpreting
the segment.
