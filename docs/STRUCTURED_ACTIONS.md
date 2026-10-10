# Structured Terra actions (opt-in)

`train_structured.py` is the training entrypoint for the solo tracked
excavator's `terra_structured_v1` action space. `train_mixed.py` remains the
legacy eight-way training path. Episodes start either from a saved native
initial-state bank (`--initial-states`: `initial` list, optionally `episodes`
metadata) or from a training map level under `DATASET_PATH` (`--maps-path`).
In map mode, `--env-template` is a saved bank whose native rules every lane
uses (for example the manual game's bank), the level is loaded through
`TerraEnvBatch` with the `--distance-protocol-id` sidecar and pull boundary
geometry, and `--precision-episode-fraction` of the lanes on every device run
precision episodes on the `precision_slots` of `--training-slots`. Other lanes
sample the whole level. Rollout (`lax.scan`), GAE and PPO run in one `pmap`
over `--num-devices`; gradients and minibatch advantage statistics are averaged
across devices. `--wandb-project` enables W&B logging.

The runner preserves the bank's pull-length and other native environment
settings. Native digging has no bucket-width constraint: trenches are widened
to bucket size during postprocessing. The temporary rectangular strip gate was
removed; old saved values of `dig_working_strip_width_m` are ignored by Terra.
See the paired checkout's `docs/PULL_DIRECTION_ALIGNMENT.md`. The manual game
uses a 1 m minimum pull; full-bank finishability remains a separate evaluation.

## Action and model contract

The existing type IDs are retained: forward/backward (0/1), clockwise and
counterclockwise base turns (2/3), DO (6), wait (7). Cabin-only types 4/5 are
masked out in the policy; the manual game still supports them. Type-conditioned
argument heads output move distances 1–5 cells, turn amounts 1–6 native
30-degree steps, and DO headings 0–11 (absolute cabin index relative to base).
DO performs one dig **or** one unload according to load; they are separate
policy decisions. The environment checks intermediate base orientations and
native movement clearance.

On the integer pose grid, one-cell moves are available only at cardinal base
headings. Oblique headings use amounts 2–5; requests that would clip to a
one-cell displacement are blocked. This prevents repeated rounded one-cell
moves from drifting along a different direction from the base heading.

The policy masks come from actual native outcomes, including relifting staged
soil and full-load unload checks. Wait is only the all-invalid fallback unless
`--allow-wait` enables it. Infeasible argument rows receive an internal dummy
categorical to keep unused-row gradients finite, but their type is masked out.

`utils/models.py` adds opt-in argument heads to the existing encoder/GRU.
Zero-initialized context projections feed both actor and critic remaining
machine time, remaining decision budget, timing visit state, and history of
amounts/headings/executed durations. Existing action-type history is retained.
Recurrent PPO replays complete trajectories with resets after terminal steps.

Rollouts store type, arguments, native masks, joint log probability and executed
duration. PPO clips the ratio of joint selected-action probabilities. Entropy
is the exact type entropy plus conditional entropy weighted by the probability
of each type. Four coefficients can be tuned independently. Unused argument
heads never contribute to a selected action's log probability.

## Time and return objective

The provisional defaults are 14,400 modeled seconds and a 450-decision guard.
`--time-budget-factor F` instead gives every episode `--time-budget-offset-s`
plus F times its map's dig-only time (dig units x tile^3 x `dig_s_per_m3`), at
least `--time-budget-s`.
A fixed budget cannot serve the generalist bank: under the default timing,
digging alone takes 0.9 h for a median trench, 2.4 h for a median foundation
and 7-9 h for every qualified precision slot. The time reward and remaining-time
observation are relative to the episode's own budget. Likewise
`--decisions-per-dig-unit R` sets the decision limit to R per dig unit, at
least `--decision-limit`; the October 8 oracle needs about 0.5.
`--no-time-limit` ends episodes only on success or the decision limit; the time
budget is then only a per-map time reference. `--success-time-bonus B` and
`--success-decision-bonus C` add `B x (1 - time/reference) + C x (1 -
decisions/limit)`, each clipped at 0, on success only; with `time_cost_total`
0 in the timing JSON there is no per-step time cost, so time never makes
progress worse than idling. `StructuredTimeConfig.setup_per_dig` charges the
setup on every dig instead of once per visit. The CSCS launcher uses all of
these (`scripts/structured/run.sh`, `scripts/structured/timing_simple.json`).
The observation contains
both remaining budgets because zero-duration actions still consume the guard. `StructuredClock` charges setup once per actual visit;
movement away and back opens another visit. `--timing-json` accepts fields from
`terra.structured_actions.StructuredTimeConfig`; timing estimates need physical
calibration. Changing these values changes the training problem.

The native `material_time_v1` reward is material-potential difference plus the
configured behavior costs, minus `3.6 * executed_seconds / time_budget_seconds`.
Exact completion within both budgets earns +6 once; timeout earns -1 once. A
completed action that overruns time is a timeout. Separate native dig/unload
events preserve material and handling accounting. This path does not call the
legacy 450-step reward-v2 formula.

`gamma=1` is deliberate: potential differences telescope over a finite episode,
and modeled time is penalized explicitly. An explicit `--gamma` below one uses
`gamma ** (executed_seconds / discount_reference_seconds)` and is a different
objective: the undiscounted shaping deltas no longer telescope. GAE uses
`lambda ** (executed_seconds / discount_reference_seconds)`; default lambda=.95
and reference=30 seconds. Lambda, reference duration, episode budget and entropy
are initial settings, not behaviorally tuned conclusions. Terminal episodes,
including timeouts and the decision guard, never bootstrap into a reset state.

## Bounded smoke

From this `terra-baselines` worktree (saved-bank mode):

```bash
export PYTHONPATH="../terra:.:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
/home/lorenzo/moleworks/.venv-terra-uv/bin/python train_structured.py \
  --initial-states /home/lorenzo/moleworks/.artifacts/terra_manual_edge_20261009/initial_states_cone30_perp.pkl \
  --bank-indices 0,1 \
  --output /home/lorenzo/moleworks/.artifacts/terra_structured_20261009/ppo_cpu_smoke \
  --num-envs 2 --num-steps 2 --epochs 1 --updates 1
```

The native masks/observation and transition kernels take minutes to compile on
CPU. `metrics.jsonl` records every update (losses, per-head entropy, episodes and
successes by mode, completion, modeled hours, budget, decisions, action mix and
throughput); `config.json` records the resolved protocol. `checkpoint.pkl` is
written atomically every `--checkpoint-interval` updates, with live
environments; `--keep-checkpoint-every` also keeps parameter-only
`checkpoint_update_NNNNNN.pkl` files.

`scripts/structured/run.sh {smoke,probe,production}` is the CSCS launcher for
the map-mode scratch run (campaign encoder and observations from
`scripts/structured/campaign_model.json`, 4 x 512 lanes, 32 decisions per lane
per update). `smoke` checks two updates plus a resume, `probe` measures
production-size throughput, and `production` resumes from its own rolling
checkpoint when one exists.

For CUDA, use `.venv-terra-gpu-uv`, export its NVIDIA library directories, disable
JAX preallocation on a shared GPU, and run the Terra RL `check_jax_runtime.py`
preflight before the same bounded command. A model update or a native first
update proves implementation startup, not policy quality or map finishability.

## Checkpoints and evaluation

`eval_structured.py --checkpoint CKPT --initial-states BANK --output OUT.json`
runs every start of a saved bank (greedy starts only when the bank records
decoders) to termination under the checkpoint's own rules, time budgets and
decision limits, greedily or with `--sampled`, and reports success,
completion, modeled hours and decisions per start. On the manual game's
bank it is directly comparable with the October 8 oracle panel.

`--warm-start-from LEGACY.pkl` reuses a shape-compatible encoder, recurrent core
and type head. It initializes new argument/context parameters, resets the critic
and optimizer, and starts fresh environments. Move arguments initially favor five
cells; base turns favor one step. Heading-conditioned DO and removed cabin types
mean the complete action distribution cannot exactly preserve legacy behavior.
Architecture mismatches fail clearly. Optional `--model-config FILE.json`
overrides architecture/observation settings for scratch; warm starts inherit the
source model configuration. Legacy config classes are registered before unpickling.

`--resume-from STRUCTURED.pkl --updates N` resumes the native optimizer, update
index, RNG, live terrain, recurrent carry, action history and time/visit clocks.
Repeat the original protocol arguments; `N` is the absolute final update target.
Changes to recorded timing, bank identity metadata, model, batch or PPO settings
reject native resume. Source code must retain the same dynamics and action
protocol; the loader does not detect arbitrary code edits. Output location and
checkpoint cadence may change.

`--teacher-checkpoint` rejects explicitly: there is no approved probability
mapping from an eight-way teacher's cabin actions to heading-conditioned DO.
Teacher guidance should be added only with an explicit mapped target and test.

Before selecting a training recipe, compare fixed initial maps on completion,
accepted disposal, modeled time and decisions. The small saved game bank is a
runtime diagnostic, not a source-disjoint generalization benchmark.
