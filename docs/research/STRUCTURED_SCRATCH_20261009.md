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
- Episode cap: per map, 300 + 1.2 macro decisions per dig cell (trenches 300,
  precision slabs 528-960 for their 440-800 cells), about 2x the oracle.
  Episodes end on success or at the cap; modeled time never ends an episode.
- Reward: material progress (about 1.15 over a whole map: completion 1 plus a
  small haul-distance term), +6 on success, -1 at the cap, and on success only
  2 x (1 - modeled time / T_ref) + 1 x (1 - decisions / cap). No per-step time
  cost.
- Modeled time (`scripts/structured/timing_simple.json`): 30 s per 0.25 m^3
  bucket (120 s/m^3), 278 s setup per dig, 0.5 m/s driving, 5 s/rad chassis
  turns, 0.28 rad/s cabin. T_ref = 2.2 h + 3 x the map's dig-only time.
- PPO: 4 GH200 x 512 lanes x 32 decisions per update, 2 epochs x 32
  whole-trajectory minibatches, lr 3e-4, clip 0.2, vf 2, no value clip,
  entropy 0.02 per head, gamma 1, GAE lambda 0.95 per 300 modeled seconds.
  Seed 20261009.

## Why a macro cap and a success-only time bonus

Under the field constants (226 s/m^3 loading, 415 s per workspace visit),
digging alone takes 42 s per dig unit: 0.9 h for a median trench, 2.4 h for a
median foundation and 5.2-9.4 h for the precision slots. A fixed time budget
such as the game's 14,400 s would make every precision episode and many
foundations unwinnable, so episodes are capped by macro decisions.

Those constants come from one fit over 12 field workspaces (R^2 0.26), and
interruptions inside sessions inflated it; without them it gives 278 s +
272 s/m^3. Time therefore enters only through a bounded bonus on success:
scaling errors cancel in t/T_ref, and an unfinished episode pays the same
whatever its pace. A per-step time cost was rejected: material progress totals
only about 1.15 per map, so any cost large enough to matter made unfinished
progress score worse than idling until the cap. Setup is charged per dig (the
workspace penalty); charged per visit, one stop could dig all its sectors for
a single setup.

The October 8 greedy oracle, rerun under the game's rules, gives the scale of a
competent plan (`.artifacts/terra_structured_scale_20261009/oracle_panel.json`;
modeled time from a native replay of its action tape, field constants):

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
finishes follows 1.58 h + 1.37 x dig-only time. The largest finish takes 282
decisions. Only the trench finishes fit the game's 14,400 s; every foundation
finish needs 6-11 h.

Retimed with the training model (`retime_oracle.py`, `retime_simple_model.json`),
the nine finishes take 5.4-5.5 h (17411), 6.55 h (17413), 3.85 h (ring) and
1.7-2.5 h (trenches, 60-70% of it setups), following 1.11 h + 1.50 x dig-only
time. T_ref is twice that line: every oracle finish uses 44-53% of its T_ref
and would earn about half of the time bonus.

Rerunning the five dump-limited starts with 60 instead of 12 checked cuts
finishes the straight trench 13824 (seed 2: 2.69 h, 84 decisions), so 9/20
starts now have a constructive finish. The road-side foundation 513 (both
seeds) and the 17413 precision rectangle (seed 2) still stop without a legal
dump: dumps must land on accepted cells 4.0-6.0 m from the base, a machine
rule shared with the legacy campaign.

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

CSCS smoke 5021368 (2 updates, resume to 3, production-size probe) and
production 5021369 (`afterok`, 24 h, W&B project `mixed-agents`, run name
`structured_scratch_s20261010`). Snapshot Terra `fddedf3e`, baselines
`90caf23`; run directory
`/ritom/scratch/cscs/lterenzi/terra-training/runs/structured-s20261010/`;
template `inputs/structured_20261009/initial_states_game.pkl` (sha256
`5f6a142e...`). Launch files: `.artifacts/terra_structured_scale_20261009/cscs/`.
The October 9 submission (5014326/5014327: time-limited episodes, per-step
time cost) was cancelled before it started.
