# Precision action-interface comparison, 2026-10-13

Compare scratch PPO with `primitive8` and `structured` actions for one 24-hour
GH200 allocation per arm, seed 20261013. Both arms use the corrected Terra
environment, observations, reward, campaign encoder and GRU. The question is
whether selecting a DO heading and variable motion amounts improves precision
learning relative to the old eight primitive commands. This does not compare
the full old campaign against the new campaign: the environment fixes are shared.

## Data and split

The input pool is the 95 precision slots in
`.artifacts/terra_pull_scratch_teacher_20261007/inputs/training_slots.json`.
The local dataset is
`.artifacts/terra_gru_bigbank_20260923/bank_balanced/train_v3_generalist_512`;
CSCS uses the existing `gru_gen512_20260923/bank` dataset root.
All 95 slots belong to `v7-fnd-slab-adjacent`. They passed static per-cell
reachability screening. Whole native episodes have not established finishability
for this pool. That limitation applies equally to both arms and does not block
this diagnostic training comparison.

Each slot has a distinct `source_id` and `dig_sha256`, but there are only 58
excavation shapes after cropping empty surroundings and identifying rotations
and reflections. Three repeated `source_scenario_id` values further connect
different shapes. The split groups the connected components of all these
identities, giving 55 groups. This conservatively keeps source variations and
identical geometry out of the opposite split even when placement or dump layout
differs.

`prepare_split.py` fixes 12 held-out groups using seed 20261013:

- Train: 43 groups, 74 slots, required volume 440–792 grid units.
- Held out: 12 groups, 21 slots, required volume 456–800 grid units.
- Main evaluation: one representative slot per held-out group, four reset seeds
  per representative, 48 identical native starts for both arms. Representatives
  are the smallest slot in each group; this choice is fixed before training.

The checked-in `train_slots.json`, `eval_slots.json` and `split_manifest.json`
record this split. Training samples uniformly over its 74 slots; shapes with
multiple slots therefore receive more exposure. Main evaluation weights each
held-out group equally. The dataset's historical `split: train` field describes
the old corpus partition; this experiment's held-out partition is in the new
manifest. No bulk training lanes or teacher are used.

## Matched settings

`scripts/precision_action_ab/run.sh PHASE INTERFACE` supports `smoke`, `probe`,
`production` and `eval`; `INTERFACE` is `primitive8` or `structured`.

- Four GH200 devices, 512 lanes per device, 32 decisions per rollout, two PPO
  epochs, 32 minibatches, learning rate 3e-4, clip 0.2, value coefficient 2,
  no value clipping. Both arms reuse `scripts/structured/campaign_model.json`.
- Precision fraction 1; safety, current edge observations and both certified
  dead-end checks enabled; edge share 0.5; per-mode advantage normalization.
- The episode ends at the same modeled-time deadline:
  `max(3600, 7960 + 3 * dig_units * tile_size^3 * 120)` seconds.
  Timing is exactly `scripts/structured/timing_simple.json`, including setup
  per dig, driving, chassis and cabin turns.
- A common decision guard `max(2000, ceil(8 * dig_units))` stops zero-time
  loops. On this pool the guard is 3520–6400 decisions. It is a guard, not the
  primary budget; report how often it ends episodes.
- Success-time bonus 2, success-decision bonus 0. Gamma 1, duration-scaled GAE
  lambda 0.95 per 300 modeled seconds. Existing material and success rewards
  are shared. Entropy coefficient 0.02 applies to each active conditional head;
  this is the joint action entropy with the same coefficient in both arms.
- Save the initial checkpoint, rolling checkpoints every 100 updates, retained
  checkpoints every 500. No warm start and no teacher.

Required launcher environment variables are `TERRA_RUN_DIR` (separate per arm),
`TERRA_AB_INPUTS` (shared split and evaluation files), `TERRA_STRUCTURED_INPUTS`
(contains `initial_states_game.pkl`), `TERRA_EXPERIMENT_INPUTS` (contains
`geometry_dependencies`) and `DATASET_PATH`. The caller sets the 24-hour Slurm
limit; this wrapper does not submit jobs. The launcher requires GH200 devices
and runs the existing CUDA convolution-backward and NCCL preflight.

Prepare the shared inputs once in the staged runtime, with the same `PYTHONPATH`
and `DATASET_SIZE=20480` as the launcher:

```bash
python scripts/precision_action_ab/make_eval_bank.py \
  --template "$TERRA_STRUCTURED_INPUTS/initial_states_game.pkl" \
  --split-manifest "$TERRA_AB_INPUTS/split_manifest.json" \
  --output "$TERRA_AB_INPUTS/eval_states.pkl"
```

The panel uses native `_reset`, precision alignment and corrected safety rules,
then checks the saved bank with `load_initial_bank`. `eval_states.json` records
slot, source, group, reset seed, position and headings for every start. Each
representative must have four distinct poses, obtained within 32 fixed reset
seeds; the builder fails if that bounded search cannot provide them.

## Qualification and evaluation

For each arm the smoke uses eight lanes per device, four-step rollouts and a
four-decision cap to exercise reset behavior. It completes two updates, saves
the rolling checkpoint, resumes to update three, and runs a required four-start
evaluation. These reduced limits apply only to integration
checks. `verify_smoke.py` checks finite updates, resume state and saved action
interface. Compare initial shared backbone weights between arms. A production
probe measures actual rollout/update throughput; successful compilation alone
is not qualification. Fail runtime, serialization, protocol, split or finite
checks before production. Poor early success alone is not a reason to stop.

Evaluate the initial, intermediate and last available production checkpoint on
the same 48 starts with greedy decoding and one fixed sampled seed (20261013).
Use an intermediate retained checkpoint nearest the midpoint of elapsed training
wall time; also compare the largest common completed update for sample exposure.
`TERRA_EVAL_CHECKPOINT` chooses the checkpoint and `TERRA_EVAL_OUTPUT` selects a
separate output directory; `run.sh eval INTERFACE` writes `greedy.json` and
`sampled.json`, with masked policy distributions and selected arguments every
32 decisions. Evaluation never chooses a different checkpoint by its score.
If the bounded evaluation allocation cannot cover all panels, prioritize final,
then intermediate, then common update, then initial. A bulk panel is optional secondary evidence;
it is not part of this precision-only conclusion.

The primary outcome is held-out precision success by source group. Also report
edge/interior completion, accepted soil, loaded-at-end rate, modeled completion
time, relifts, immobility, certified dead ends and deadline/decision-guard exits.
Use both wall time and cumulative native decisions/update count on learning
curves. Variable actions deliberately change decisions per completed operation;
equal PPO updates and equal wall time measure different efficiencies. All
primitive transitions must use the same modeled clock; command count must not
stand in for elapsed time.

Continue both valid arms for the authorized 24 hours. Stop for a reproducible
runtime or semantic bug that invalidates the comparison, not a flat early curve.
If decision guards materially bind (more than 1% of held-out episodes), report
the limit explicitly and do not attribute that difference solely to action
quality; any revised-budget experiment must rerun both arms. A winner should
improve held-out success in both decoders, retain edge progress, and show no
termination or timing artifact. If both arms fail, inspect paired traces before
claiming the action interface is irrelevant. With one training seed and 12
held-out groups, small differences are inconclusive; add a seed only if that
uncertainty changes the next decision.

## Local qualification, October 13

Twenty-seven focused CPU tests passed, including primitive masks and likelihood,
unused argument gradients, multi-device diagnostic reduction, checkpoint
interface handling, and existing episode/dead-end cases. One native parity case
checked cabin wrap followed by fresh digging against direct aimed DO: equal
terrain, reward and modeled time, with two versus one policy decisions. This
is not exhaustive native-action equivalence. Independent code reviews found no
remaining blocker.

All 48 evaluation starts passed native footprint, initial mobility, unloaded
bucket, step-zero and unfinished-task checks. The local CUDA primitive smoke
completed one finite update and saved its checkpoint; structured CUDA and CSCS
update/resume/evaluation/full-batch qualification are still pending at this
source snapshot. No production job has been submitted at this point.

Artifacts: `.artifacts/terra_precision_action_ab_20261013/`. Local reduced
GPU smokes use a historical replay bank that includes a held-out slot; their
weights are discarded. Production starts fresh from the new training split.
