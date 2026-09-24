# Multi-agent joint policy — plan (2026-09-23)

Branch `multiagent-joint-policy-20260923` in both `terra` and `terra-baselines`,
based on `oracle-time-work-costs-20260916` (terra 2122b2df) and
`oracle-followup-20260916` (terra-baselines a7e0da2).

## Why change the current multi-agent setup

The current setup serializes agents: one env step applies one agent's action,
`State._swap` advances a round-robin pointer, the observation is rolled so the
acting agent is slot 0, and one shared actor-critic picks that agent's action.
Review findings (2026-09-23, reproduced on CPU):

- Collision / turn / wheel-turn penalties read `new_state._get_next_agent_state()`.
  After the swap that is the actor only for N=2. For N=3 (`excavators_truck`) a
  blocked move is never penalized (−0.10 instead of −0.30).
- `_backfill_terminal_rewards` copies the terminal reward onto the N−1 preceding
  turns: terminal ×N mid-rollout, ×1 when the episode ends in the first N−1
  steps of a 32-step rollout. GAE already propagates that credit.
- `_get_action_mask_*` compares a handler result against `_get_prev_agent_state()`,
  a different agent for N ≥ 2 (blocked moves reported as allowed).
- γ, the existence cost and `max_steps` all count turns, so hyperparameters
  mean different things for different N; the global-map encoder runs N times
  per round of real progress; `prev_actions` is 5·N interleaved actions.
- reward_v2 (the current production reward) is single-agent only: its validity
  guard requires `num_agents == 1` and returns NaN otherwise.

No strong method in the literature spends a real env step per agent. Methods
that decide sequentially (MAT, ACE, SeqComm) do it inside one env step.

## Method (decision)

**Sequential joint-action PPO in the Multi-Agent Transformer family (MAT,
Wen et al. 2022), on a shared spatial trunk with per-agent readouts (Lux AI S1
winner pattern), with MAPPO's implementation details.**

- One env step = one joint action for all A agents (A = `len(agent_types)`,
  static). A round is one step: γ, step cost and horizon are per round.
- Policy factorization π(a|s) = Π_k π(a_{o_k} | s, a_{o_<k}), where o is a
  uniformly random permutation per step. Agent k's head sees the actions already
  chosen by agents earlier in o (MAT decoder). The env applies the actions in
  the same order, so every agent conditions on exactly the actions that will
  have been executed before its own. Random order removes first-mover bias.
- Parameter sharing: one network applied to each agent's view (agent-centric
  agent slots, own local maps, own action history, own retained-work context,
  own trench alignment, and per-view traversability and workspace maps; see E1
  below for why the maps must be per view). The map trunk runs once per view.
- Critic: the existing value head applied to each view (MAPPO agent-specific
  global state); team value = mean over agents; every view regresses the team
  return.
- Loss: per-agent conditional ratios clipped against the shared team advantage
  (MAT objective); entropy averaged over agents; value loss averaged over agents.
- Why MAT rather than factorized MAPPO: the sequential decode carries the
  multi-agent advantage-decomposition guarantee (monotonic improvement) and beat
  MAPPO/HAPPO on SMAC, MA-MuJoCo and Google Research Football. A ≤ 4 here, so the
  decode costs A small head passes per step. The decoder output is zero-init, so
  training starts from independent per-agent heads (MAPPO) and learns
  coordination on top.
- Warm start: every agent's head starts as the u110000 single-agent generalist
  (`resnet_spatial_8x8_se_sa_xattn`, medium, reward_v2, 2.95M params). The joint
  model has the same parameter tree plus the intent decoder, so the migration is
  function-preserving: the intent output projection is zero and the first-layer
  weights that read teammate agent slots 1..3 (never active in single-agent
  training, still at random init) are zeroed.

## Environment contract (terra)

1. Remove turn-taking: delete `_swap`, the `turn` flag and the
   `_get_next/prev_agent_state` accessors. Reward code reads the acting agent
   from the post-action state directly (no pointer moves during a sub-action).
2. Joint action: `Action.action` has shape `(A,)`; the execution order is a
   second `(A,)` array. For A=1 this is today's action.
3. `State._step_joint(actions, order)`: for k in 0..A−1, set `current_agent =
   order[k]`, apply that agent's action with the existing handlers, collect its
   action-dependent reward terms (legacy dense agent reward and trench reward;
   reward_v2 lateral/travel/turn/retained-work costs), then refresh the
   traversability mask (dump exclusion reads it). After all sub-actions, once:
   `env_steps += 1`, stall age, task completion, terminal reward, existence/step
   cost, reward_v2 potential shaping between the pre- and post-round states.
   Collisions resolve by execution order (move validity already checks other
   agents' current footprints).
4. reward_v2 validity accepts any number of tracked excavators (horizon 450
   rounds unchanged). Legacy dense: per-agent terms summed; the team terminal
   reward is paid once and no longer depends on agent count.
5. Observation: A=1 unchanged. A>1: global keys once; per-agent keys stacked on
   a leading agent axis (`agent_states`/`agent_active` rolled per view, 10 local
   maps, `retained_work_context`, the three trench-alignment scalars,
   `movement_feasibility` when enabled). Previous-outcome feedback is the
   team's transition outcome. The trainer rejects the reachability observation
   and partial-reset banks for A>1 (u110000 uses neither).
6. Diagnostics: `action_had_effect` = any agent changed something;
   `productive_workspace_cycle` summed over agents.
7. `_get_action_mask_*` compares the same agent (used by analysis scripts).
8. Known limitation kept for now: `world.last_dig_mask` is shared between agents.

## Model and trainer (terra-baselines)

- `SimplifiedCoupledCategoricalNet`: static `num_agents`. `_fused_features`
  flattens per-agent inputs to B·A and encodes global maps once;
  `Spatial8x8MapResNet` accepts an agent embedding `[B, A, D]` and broadcasts the
  trunk outputs to the per-agent readout. Parameter tree unchanged.
- `AgentIntentDecoder` (MAT decoder, small): tokens = (agent self-embedding,
  one-hot action) of already-decided agents; query = deciding agent's actor
  features; 4 heads × 64; zero-init output added to the actor features.
- Methods: `joint_policy(obs, actions, order)` (teacher forcing for PPO),
  `joint_act(obs, order, rng, greedy)` (sequential sampling), `joint_value(obs)`.
- `train_mixed.py`: joint path when A>1 at action selection, action wrapping,
  per-agent action history `[B, A, 5]`, Transition (`order`, `[B, A]` actions and
  log-probs, team value), GAE (team), PPO update (team branch of
  `ppo_update_networks`), inline eval, action histograms.
  `_backfill_terminal_rewards` is deleted.
- Warm-start migration `utils/team_migration.py`: single-agent checkpoint →
  team params; shared parameters keep the parent's Adam moments and count, the
  intent decoder starts with zero moments.

## Validation (CPU, minutes)

- A=1 parity: step, reward, reward components and observation identical to the
  pre-change code for fixed action sequences (includes reward_v2 bitwise tests
  already in the suite).
- Joint step: order respected, sequential collision resolution, per-agent terms
  summed, shaping/terminal/step cost once per round, `env_steps` +1 per round.
- Model: A=1 outputs identical to the old network; joint forward with zero intent
  equals the single-agent network on each view; teacher-forced log-probs equal
  sampling log-probs; migration of a real u110000 checkpoint is
  function-preserving on every view.

## Experiments

- E0 (local RTX 4090): update-1 smoke, 2 excavators, `train_v2_pooled_generalist`,
  warm start u110000; finite loss/params/Adam; throughput vs single agent.
- E1 zero-shot baseline: migrated u110000, no training, 2 excavators, fixed bank,
  greedy and sampled. Measures what independent copies of the generalist do.
- E2 joint MAT fine-tune from u110000, 2 excavators, 1 GPU (512 envs × 32 steps),
  checkpoint every 500 updates, ≥ 24 h before any negative claim.
- Metrics: success within 450 rounds, rounds to success, team speedup = paired
  single-agent steps / joint rounds on the same maps, no-effect action fraction
  per agent, agent–agent blocking events, dump purity.
- E3: heterogeneous excavator + truck and 3 agents (truck head learns from
  scratch; reward_v2 truck semantics to specify); optional teacher KL to the
  single-agent generalist per view as a retention guard.

## Risks

- Warm-start distribution shift: the trunk only ever saw one footprint and one
  interaction cone. Measured by E1; fine-tuning should absorb it.
- Shared team advantage with A ≤ 3 is well within MAPPO/MAT evidence; per-agent
  counterfactual baselines only if E2 stalls.
- 64×64 maps with 7×11-tile excavators: 2 agents fit; 3–4 are crowded.
- Shared `last_dig_mask` couples the dig/dump exclusion of different agents.

## Status (2026-09-23)

Implemented on the branch (uncommitted in the worktree
`.worktrees/terra_multiagent_joint_20260923`):

- Terra: turn-taking removed (`_swap`, prev/next accessors, `turn` flag);
  `State._step_joint` / `_apply_action` / `_agent_reward_terms`; team reward
  terms paid once per round; reward-v2 valid for tracked-excavator teams;
  per-agent observation views (`AGENT_VIEW_OBS_KEYS`); traversability refreshed
  between agents from one State helper that the wrapper also uses; team-summed
  productive cycles; start actor fixed at slot 0 (reset RNG split kept).
- Baselines: shared-trunk team forward (`Spatial8x8MapResNet` per-agent
  readout), `AgentIntentDecoder`, `joint_policy` / `joint_act` / `joint_value`,
  `joint_obs_to_model_input`, `select_joint_action`, team PPO branch in
  `ppo_update_networks`, rollout / inline-eval / episode-aggregate support,
  per-agent action history, terminal backfill deleted, warm-start migration
  with Adam-moment transplant (`utils/team_migration.py`), `scripts/team/`.

Validation:

- Single-agent parity: 8 envs × 470 steps on the real bank with random fixed
  actions (includes timeouts and resets). Every observation key, reward, done,
  info field and reward component is bitwise identical to the base commit,
  except the logging-only per-agent reward split: one value differs by 1 ulp.
- Terra tests: new `test_joint_step.py` (execution order decides contested
  space; one step and one existence cost per round; team reward = sum of agent
  terms; agent-centric views) plus the existing env/state suites.
- Baselines tests: new `test_team_policy.py` (migrated team equals the
  single-agent policy on every view; sequential sampling equals teacher
  forcing; causal decoding; team PPO update trains the decoder with zero
  initial KL; Adam transplant; per-agent histories) plus existing model,
  training, aggregate and evaluator suites.
- u110000 migrated to a two-excavator team matches the single-agent policy on
  every view on real bank observations; reward-v2 is finite for the team.
- RTX 4090 smoke: two team PPO updates from the u110000 warm start with finite
  checks, checkpoint save and final save.

Runs (CSCS Daint, account d130, 4× GH200; code snapshot
`/ritom/scratch/cscs/lterenzi/terra-training/snapshots/terra-team-mat-20260923`,
source receipt `runs/terra-team-mat-20260923/launch/source_revisions.txt`):

- Smoke 4746675 (debug, `runs/terra-team-mat-20260923-smoke`): 2 updates at
  4×256×32, warm start u110000
  (`runs/terra-efficiency-anchor-20260922/training/checkpoints/generalist-retained-p10-anchor_update_110000.pkl`),
  finite checks on, save/final save. First update 320 s (compile); second
  update 15,454 rounds/s (end-to-end 12,704). The single-agent anchor run on
  the same node type and batch logs about 15,800 transitions/s, so a round of
  two excavators costs about the same as one single-agent step.
- E2 screen 4746787 (`runs/terra-team-mat-20260923`): two excavators, one
  24-hour allocation toward u20000, checkpoints every 250 updates, W&B offline.

### E1 zero-shot (CSCS, 512 identical reset maps, seed 0)

Two copies of u110000 (migrated, independent heads) on the maps u110000
solves alone, first version (shared traversability with every chassis at -1,
workspace = union of both agents' cones):

| Setup | Success | Median steps/rounds | Agent 0 effective actions |
|---|---:|---:|---:|
| single agent, sampled | 99.4% | 64 | 98% |
| single agent, greedy | 99.6% | 63 | - |
| team of 2, sampled | 60.5% | 113 | 74% |
| team of 2, greedy | 52.9% | 100 | - |
| team, teammate idle, sampled | 28.7% | 82 | 53% |
| team, teammate idle, greedy | 23.0% | 73 | 38% |

A parked second excavator makes agent 0 waste about half its actions and fail
most maps, so the loss is mostly observational: the solo policy reads the
single -1 chassis blob as itself and the workspace channel as its own cone.
Fix (terra `_traversability_map(observer)`, `TerraEnv._observation`): each
view marks only its own chassis -1, teammates as blocked cells, and its own
workspace. The shared-trunk readout was dropped with it; the trunk now runs
per view. Screen 4746787 (first version) was cancelled at u2750; its u2750
checkpoint is evaluated for comparison.

Second version: training screen 4749240
(`runs/terra-team-mat-v2-20260923`, snapshot
`snapshots/terra-team-mat-v2-20260923`), zero-shot panel 4749241, first-version
u2750 panel 4749242.

First-version run 4746787 at u2750 (about 2 h of 4× GH200, 90M rounds), same
512 maps:

| Setup | Success | Median rounds | Paired speedup vs single |
|---|---:|---:|---:|
| zero-shot team, sampled | 60.5% | 113 | 0.54 |
| u2750 team, sampled | 98.8% | 74 | 0.93 |
| u2750 team, greedy | 95.1% | 70 | 0.96 |

Team PPO removes the interference quickly, but the team is not yet faster than
one excavator. Speedup (rounds of a single agent / rounds of the team on maps
both solve) is the target metric from here; perfect work splitting would give
up to about 2.

Second version (per-view maps), same 512 maps:

| Setup | Success | Median rounds | Paired speedup | Moves effective |
|---|---:|---:|---:|---:|
| zero-shot team, sampled | 46.9% | 188 | 0.42 | 15–19% |
| zero-shot team, teammate idle | 57.2% | 66 | 0.98 | 20% |
| u1000 team, sampled | 98.8% | 72 | 0.95 | - |
| u1000 team, greedy | 94.1% | 66 | 1.00 | - |

Zero-shot, the two identical solo policies drive to the same work area and park
side by side (render `runs/terra-team-mat-v2-20260923/render/u110000/`). With
an idle teammate, success is 79% when no chassis starts on target cells and 33%
when one starts on dig cells (a parked excavator makes the map unsolvable).
Team PPO recovers single-agent reliability within 1,000 updates (v1 needed
about 2,750). The team is still no faster than one excavator; the
milestone panels (u2750–u20000) track whether work splitting emerges. The
reward-v2 speed incentive is modest (finishing in 40 instead of 64 rounds is
worth about +0.4 return, ~5%); the v21 timing variant is the next lever if the
speedup stays near 1.

### Executed-plan time (the robot's real earthworks time)

The robot keeps only the base poses of effective DO actions; Nav2 drives
between them. `scripts/team/evaluate.py` logs, per machine, straight-line travel
between successive work poses and every scooped unit (digs and relifts);
`scripts/team/compare.py` converts them with 0.5 m/s travel and 30 s per
0.3 m³ scoop. A unit is 0.571 m × 0.571 m × unit depth; with no declared
vertical scale the default is a cubic cell (0.187 m³). The team time is its
slower machine's; waiting is not modeled.

u5000 vs single u110000, 512 maps (507 solved by both):

| | Single | Team u5000 |
|---|---:|---:|
| median executed time | 47.9 min | 33.0 min |
| throughput | 34.7 m³/h | 48.8 m³/h |
| median speedup (sampled / greedy) | - | 1.43 / 1.45 |
| maps where the team is faster | - | 86% / 90% |
| round speedup (simulator decisions) | - | 1.19 / 1.23 |

The speedup is insensitive to the vertical scale (1.42–1.44 for 0.2–1.0 m per
unit): scooping dominates (about 25 min per machine against 46 s of travel).
The team scoops exactly as much as the single agent (ratio 1.00), so double
handling is gone. The limit is load balance: the busier machine scoops a median
64% (over 80% on 30% of maps). Jobs under 100 units are effectively
single-machine (share 0.88, speedup 1.01); 200–400 units reach 1.64. The
training reward counts rounds, in which a move and a full-workspace DO cost the
same, so it does not target the makespan the robot experiences.

### Team milestones u5000–u10000 (2026-09-24 04:15)

Run 4749240 (CSCS, 4 GH200, 20,000-update target), paired with the single
agent u110000 on the same 512 reset maps (seed 0). The maps are drawn from the
training bank `train_v2_pooled_generalist`, the bank both the parent and the
team trained on: an in-distribution comparison, not a held-out test. Speedups
are medians over maps both solve (507 sampled, 501–502 greedy). Executed-plan
time: 0.5 m/s travel between work poses, 30 s per 0.3 m³ scoop, a unit =
0.571 m cube, team time = slower machine (waiting not modeled).

| (sampled / greedy) | u5000 | u7500 | u10000 |
|---|---:|---:|---:|
| success (single 99.4 / 99.6%) | 99.4 / 98.0% | 99.4 / 98.2% | 99.4 / 98.2% |
| maps gained / lost vs single, greedy | 1 / 9 | 1 / 8 | 1 / 8 |
| round speedup | 1.19 / 1.23 | 1.27 / 1.34 | 1.31 / 1.31 |
| executed-plan speedup | 1.43 / 1.45 | 1.47 / 1.53 | 1.46 / 1.54 |
| team faster on | 86 / 90% | 88 / 87% | 90 / 90% |
| throughput, m³/h (single 34.7) | 48.8 / 49.1 | 49.0 / 51.2 | 49.8 / 50.8 |
| busier machine's scooping share, median | 0.64 / 0.65 | 0.64 / 0.62 | 0.63 / 0.63 |
| maps where one machine scoops > 80% | 30 / 27% | 23 / 23% | 24 / 22% |

Executed-plan speedup by job size (sampled; units of required excavation):

| units | u5000 | u7500 | u10000 |
|---|---:|---:|---:|
| < 100 (n≈155) | 1.03 | 1.06 | 1.15 |
| 100–200 (n≈200) | 1.45 | 1.48 | 1.49 |
| 200–400 (n≈130) | 1.65 | 1.65 | 1.65 |
| ≥ 400 (n≈23) | 1.65 | 1.69 | 1.78 |

Gains since u5000 come from splitting work better (small jobs and fewer
lopsided splits); the bound for two machines is 2.0. Scooped volume equals the
single agent's (ratio 1.00): no double handling.

### Team run complete: u15000 and u20000 (2026-09-24 14:15)

Run 4749240 completed 20,000 updates in 17 h (ended 12:17). Same pairing as
above (512 training-bank maps, seed 0, single agent u110000):

| (sampled / greedy) | u15000 | u20000 |
|---|---:|---:|
| success | 99.2 / 98.4% | 99.6 / 98.4% |
| maps gained / lost vs single | 2/3 · 1/7 | 2/1 · 1/7 |
| round speedup | 1.32 / 1.37 | 1.36 / 1.39 |
| executed-plan speedup | 1.52 / 1.55 | 1.51 / 1.53 |
| team faster on | 88 / 90% | 89 / 90% |
| throughput, m³/h (single 34.7) | 50.9 / 51.6 | 50.9 / 51.3 |
| busier machine's share, median | 0.62 / 0.62 | 0.62 / 0.62 |
| maps where one machine loads > 80% | 20 / 20% | 19 / 19% |

Speedup by job size (sampled), u15000 / u20000: < 100 units 1.14 / 1.17,
100–200 1.54 / 1.54, 200–400 1.75 / 1.69, ≥ 400 1.76 / 1.73.

Training curve (online, train bank): success 97.6% (u1000) → 99.5% (u20000);
rounds per episode 102 → 62.5; entropy 1.02 → 0.59 (parent 0.17); KL 0.004–0.009.

The executed-plan speedup stopped improving after u15000 while the round
speedup kept rising: the reward counts rounds. Next lever: a reward aligned
with the executed-plan time (makespan), fine-tuned from u20000.

### Makespan cost: reward tied to each machine's soil and workspaces (2026-09-24)

Lorenzo: uneven work splitting is about the soil each machine moves and its
number of workspaces; tie the reward to that. The round reward cannot see it
(a move and a full-workspace dig cost one round each), and the executed-plan
speedup plateaued at ~1.5 after u15000 while the round speedup kept rising.

Design (terra `State.machine_work_s`, `_get_reward_v2`; docs/ENVIRONMENT.md):

- Every machine accumulates executed-plan seconds on its effective DO events:
  loaded units × 18.7 s (0.187 m³ unit, 30 s per 0.3 m³), + travel from its
  previous work pose at 0.5 m/s, + `makespan_setup_s` per new work pose
  (workspace).
- `F = max_i W_i / (V × 18.7 s)`: the busiest machine's time over the
  single-machine loading time of the job. Reward per round
  `-makespan_cost × (F' − F)`: only growth of the busiest machine is charged,
  so work by the other machine is free until it becomes the busiest. Summed:
  `-makespan_cost × F_final` (≈ −κ for one machine doing everything, ≈ −κ/2
  for an even split).
- Observation: `agent_states[..., 9] = W_i / (V × 18.7 s)` for every machine,
  own first (model flag `machine_work_observation`).
- Off by default; the reward is bitwise unchanged at 0 (tested), frozen
  benchmark hash unchanged (inert defaults excluded).
- Fine-tune from team u20000: `utils/machine_work.py` appends a zero input row
  (policy and value identical, tested) and zero Adam slots; `run.py` detects a
  resume without the observation (migrate once) and a change of the makespan
  settings (declared reward fine-tune), later segments resume ordinarily.
- Evaluation: `compare.py --setup-s` adds the same per-workspace overhead.

Smoke (CSCS debug 4759878, from u20000, 2 updates): migration, reward
fine-tune and settings in effect (config and env κ = 2, setup 30 s, R2
receipt records both), finite parameters, KL 0.014, clip fraction 0.08,
explained variance 0.78 (value adapting to the new term). The first smoke
(4759733) failed the episode-aggregate integrity check: the makespan term was
in the reward but not in the logged per-agent split; it is now shared equally
among the agents (test: components reconstruct the step reward).

Proposed run (needs CSCS approval): κ = 2, setup 30 s, from u20000 to u30000
on one 4-GH200 node (~9 h), panels at u22500/u25000/u27500/u30000 against
the single agent with `--setup-s 0` and `30`. The old recipe plateaued between
u15000 and u20000, so a rise above ~1.52 is attributable to the new term.

## Skid steer, solo (2026-09-24)

Step toward excavator + skid-steer teams: train the skid steer alone on
relocation maps first. A review of its MDP found it untrainable under
reward-v2, and fixed as follows (terra `state.py`, `env.py`, `maps_buffer.py`):

- Reward-v2 was NaN for every skid-steer or relocation map: tracked excavators
  only, and `P` divided by the dig volume, which is 0 on haul-only maps. R2 now
  admits tracked skid steers and normalizes haul-only maps by the soil to haul
  at reset (`State.material_v_reset`); maps with a dig target are unchanged
  bit for bit. The carry-work and reset-context observations use the same
  normalizer (they were 0 and ~3e7 on relocation maps).
- Pickup was all-or-nothing (a 54-cell bucket over 52 units loaded nothing)
  and relaxed soil out of the accepted zone. It now fills to capacity
  proportionally, never lifts accepted soil, and contains relaxation to the
  zone.
- Reverse with a lowered, loaded shovel dumped implicitly (or was blocked).
  `DO` is now the only dump.
- A wheeled skid steer can never load; the trainer rejects it, and capacities
  above 127 (int8 load).
- The R2 loader admits loose soil on maps without a dig target.

Behavior costs (lateral dig, travel, turn, retained work) are excavator-gated,
so the skid-steer reward is the R2 core: material potential, step cost,
+6 / −1 terminals.

Bank: `terra/tools/build_skid_relocation_bank.py`, strict hashed contract, R2
geodesic distances. Train 2048 maps (seed 20260924): 512 single-load (16–48
units, no obstacles), 768 `relocations_harder` style (3 piles, 40–50 units,
1–2 obstacles), 768 multi-trip (2–4 piles, 60–150 units, heights 1–2).
Eval 256 maps (seed 20260925, same mix). A map is kept only if every soil
cell lies in the bucket of a collision-free pose (Terra's own skid-steer
footprint and bucket, 12 headings). Local copy
`.artifacts/terra_skid_relocation_bank_20260924/`, CSCS
`runs/terra-skid-solo-20260924/inputs/bank/`.

Smoke (CSCS debug 4750244, both arms, 2 updates): finite parameters, gradients
and rollouts, so reward-v2 was valid on every transition; skid-steer behavior
costs are 0 as expected. The warm arm moves fast at first (KL 0.042, clip
fraction 0.18 at update 2, fresh Adam); its value transfers (explained
variance 0.93).

Zero-shot (u110000 driving a skid steer, eval bank, 512 envs): 0% success,
sampled and greedy. The excavator policy spends most actions rotating the
cabin (sampled 54%, greedy 75%), a no-op for the skid steer, scoops a median
11 units and almost never dumps (88 effective DOs in 512 episodes). The warm
arm starts by unlearning this; a per-type mask on the cabin actions is the
follow-up if that is slow.

Runs on Euler (compute rule: single-GPU work goes to Euler), one RTX 4090 per
arm, 512 envs × 32 steps (half the parent's per-update batch), the parent's
recipe otherwise; six chained 4 h segments per arm (`scripts/team/run_euler.sbatch`,
each resumes the newest checkpoint), target 40,000 updates. Warm 15008870–79,
scratch 15008881–90; project storage
`/cluster/project/rsl/lterenzi/terra_experiments/terra_skid_solo_20260924/`.

- `warm`: from u110000. The excavator's relift-and-dump habit maps onto the
  skid steer's scoop, lift, drive, dump. Entropy 0.05 → 0.02 over 2000 updates.
- `scratch`: same recipe, fresh initialization. Entropy 0.15 → 0.02 over 5000.

Evaluation: `scripts/team/evaluate.py --types 2` on the eval bank. Work events
now include skid-steer pickups (any own-load change), so the executed-plan
model covers both machines.

Early training (online episodes on the train bank, sampled policy; W&B
offline runs `igzj8p6i` warm, `e0tf7sz4` scratch), 2026-09-24 ~04:10:

| update | warm success | warm episode length | scratch success | scratch episode length |
|---:|---:|---:|---:|---:|
| 50 | 9% | 428 | 0% | 450 |
| 100 | 78% | 235 | 1% | 448 |
| 150 | 93% | 148 | 11%* | 431 |
| 300 | 98% | 84 | 2% | 447 |
| ~370 | 98% | 76 | 2% | 448 |
| 650 | 98% | 61 | – | – |

(*few ended episodes.) The warm start transfers within about 100 updates
(entropy rises from 0.46 to 1.9 while the cabin habit is dropped, then falls
to 0.6). Scratch stays near uniform (entropy 2.0 of 2.08 at coefficient 0.15).
Held-out eval-bank panels follow at u1000, u2500, u5000, ...

Held-out u1000 (256 eval maps, 512 first episodes, seed 0; Euler eval job
15019271):

| u1000 | sampled success | median steps | greedy success | median steps |
|---|---:|---:|---:|---:|
| warm | 98.6% | 42 | 93.0% | 41 |
| scratch | 9.0% | 313 | 3.3% | 381 |

Held-out u2500 (Euler eval job 15021113): warm 98.8% sampled (median 36
steps) / 95.9% greedy (35); scratch 13.5% sampled (327) / 1.8% greedy.

Training by 2026-09-24 13:50 (online, train bank, episode-weighted over the
last 100 updates):

| update | warm success | warm length | scratch success | scratch length | scratch entropy coef |
|---:|---:|---:|---:|---:|---:|
| 1000 | 98.4% | 55 | 9% | 438 | 0.138 |
| 2000 | 98.4% | 48 | 16% | 424 | 0.105 |
| 3000 | 96.8% | 55 | 24% | 401 | 0.065 |
| 4000 | 98.4% | 46 | – | – | – |
| 5000 | 98.6% | 44 | – | – | – |
| 7000 | 98.5% | 44 | 33% (u3811) | 373 | – |

The warm arm plateaus in success from u1000 and keeps shortening episodes;
scratch improves steadily as its entropy bonus anneals. Segments queue on
Euler priority between 4 h runs (warm 2 of 6 done, scratch 1 of 6).
