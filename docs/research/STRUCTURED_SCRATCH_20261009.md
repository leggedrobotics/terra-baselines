# Structured-action scratch run under the manual game's rules

Branches `pull-cone-viewer` (Terra) and `pull-cone-trap-dumpobs` (baselines) in
`.worktrees/terra_pull_cone_20261008/`. Action space and time model: Terra
`docs/STRUCTURED_ACTIONS.md`; trainer: `docs/STRUCTURED_ACTIONS.md` here.

## Question

Can a scratch policy learn the structured action space (move 1-5 cells, turn
1-6 steps, DO with a cabin heading) under the rules the manual game uses: 1 m
pull length, +-30 degree pull cone, perpendicular pulls at precision edges,
turn-keeping moves, native dump observation, no bucket-width gate?

## Protocol

- Rules: every lane uses the native `EnvConfig` of the manual game's bank
  (`terra_structured_actions_20261009/no_bucket_width/initial_states.pkl`):
  dig 4.0-6.5 m, dump up to 6.0 m, 0.57 m dug clearance, centred chassis.
- Maps: the 20,480-map generalist bank (`train_v3_generalist_512`, physical
  geodesic distance sidecar). Half the lanes on every device run precision
  episodes on the 95 qualified precision slots; the others sample the bank.
- Policy: campaign encoder and observations (`scripts/structured/campaign_model.json`,
  2.4 M parameters) with type-conditioned argument heads. Scratch; no teacher.
- Reward `material_time_v1` (potential difference, time cost
  -3.6 x duration / budget, +6 success, -1 budget exhaustion).
- Per-map limits: time budget = F x the map's dig-only modeled time (at least
  1 h); decision limit = 0.75 per dig unit (at least 450).
- PPO: 4 GH200 x 512 lanes x 32 decisions per update, 2 epochs x 32
  whole-trajectory minibatches, lr 3e-4, clip 0.2, vf 2, no value clip,
  entropy 0.02 per head, gamma 1, GAE lambda 0.95 per 300 modeled seconds.
  Seed 20261009.

## Why per-map limits

Under the structured time model (226 s/m^3 loading, 415 s per workspace
visit), digging alone takes 42 s per dig unit. Across the bank, dig-only time
is 0.9 h for a median trench, 2.4 h for a median foundation (13% above 4 h)
and 7-9 h for every precision slot. With the game's fixed 14,400 s budget,
every precision episode and many foundations would be unwinnable.

The October 8 greedy oracle, rerun under the game's rules, gives the scale of a
competent plan (`.artifacts/terra_structured_scale_20261009/oracle_panel.json`;
modeled time from a native replay of its action tape):

| Start (map, mode, seed) | Dig units | Finished | Modeled h | Dig-only h | Visits | Decisions |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| 17411 slab, bulk, 1 | 480 | yes | 9.04 | 5.62 | 27 | 171 |
| 17411 slab, precision, 1 | 480 | yes | 9.15 | 5.62 | 27 | 255 |
| 17413 slab, bulk, 1 | 570 | yes | 10.92 | 6.68 | 33 | 282 |
| 0 ring, bulk, 1 | 274 | yes | 6.21 | 3.21 | 24 | 140 |
| 10752 trench net4, bulk, 1 | 139 | yes | 3.69 | 1.63 | 16 | 143 |
| 13824 trench straight, bulk, 1 | 90 | yes | 2.85 | 1.05 | 14 | 121 |
| 15360 trench tee, bulk, 1 / 2 | 103 | yes | 3.20 / 3.47 | 1.21 | 16 / 18 | 98 / 116 |

The greedy oracle finishes 8 of the 20 unique panel starts (4/12 foundation,
4/8 trench); replay of every tape matches the planner exactly. The other 12
stop with stranded cells after an unlucky cut order (`no_reachable_cut`) or
because none of the 12 best-ranked cuts had a legal dump. Modeled time of the
finishes follows 1.58 h + 1.37 x dig-only time. The training budget,
2 h + 2.15 x dig-only, gives every finish 1.32-1.56x slack; the decision limit
(0.75 per dig unit, at least 450) gives at least 1.6x. Only the trench
finishes fit the game's 14,400 s; every foundation finish needs 6-11 h.

A finish is a constructive proof; an oracle failure is not an impossibility
proof. The greedy planner ignores modeled time, so its plans are not
time-optimal.

## Throughput

Kernel costs for 256 lanes on a shared RTX 4090, per decision: 12-heading DO
mask 630 ms, observation 267 ms, transition 204 ms. The DO mask dominates.
The turn mask now evaluates the first turn step only (identical masks, Terra
`252e5e54`).

Production-size throughput comes from the CSCS probe (3 updates at 4 x 512
lanes) in the smoke job.

## Launch

CSCS smoke 5014326 (2 updates, resume to 3, production-size probe) and
production 5014327 (`afterok`, 24 h, W&B project `mixed-agents`, run name
`structured_scratch_s20261009`). Snapshot Terra `252e5e54`, baselines
`898e01b`; run directory
`/ritom/scratch/cscs/lterenzi/terra-training/runs/structured-scratch-s20261009/`;
template `inputs/structured_20261009/initial_states_game.pkl` (sha256
`5f6a142e...`). Launch files:
`.artifacts/terra_structured_scale_20261009/cscs/`.
