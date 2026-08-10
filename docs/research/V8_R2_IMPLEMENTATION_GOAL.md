# V8 R2 reward-v2 implementation and experiment goal

Status: active

Authority:

- [V8 reward and termination audit](V8_REWARD_TERMINATION_AUDIT.md)
- [V8 improvement set](V8_IMPROVEMENT_SET_20260810.md)
- [V8 scale-up record](V8_10M_SCALEUP.md), current-successor amendment

## Outcome

Implement the smallest credible R2 path, launch a matched compact-policy
comparison on Euler, complete its frozen fixed evaluations, and record whether
the normalized material-potential reward improves held-out V8 performance over
the current dense reward.

R2 has exactly two arms:

1. control: current dense reward and frozen legacy relocation ledger;
2. treatment: reward-v2 from the audit, with flat exact-success payment,
   globally normalized excavation/relocation potential, fixed horizon-failure
   and step terms, and fixed `shaping_weight=1`.

This goal does not implement or launch R3 reward fading. R3 is eligible only if
reward-v2 wins R2.

## Baseline

- Selected compact checkpoint: update 20,000, SHA-256
  `0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834`.
- Main development: 546/720 exact, macro graded completion 0.861.
- Capability development: 31/32 exact, macro graded completion 0.977.
- Parent sampler: `continuous_banded_v1`.
- Required child sampler: `continuous_banded_v2` from terra-baselines
  `60e7510`.
- Horizon: 450; full resets; 47 conditions x 96 train layouts; compact
  deep+xattn model; seed `20260807`.
- No reward-v2 implementation or valid R2 launch exists at activation.

## Frozen causal contract

Both arms must share:

- one output-preserving carry-input expansion of the compact parent;
- one prepared v1-to-v2 sampler migration and the exact same migrated sampler
  state;
- absolute PPO update 20,000, fresh optimizer-local step zero, identical short
  LR warmup, and entropy fixed at the parent endpoint `0.02`;
- physical map/source identities, target/dump/obstacle/action/pose arrays,
  graph, horizon, reset mode, architecture, PPO settings, source seed,
  transition budget, checkpoint cadence, and fixed evaluation panels;
- source-disjoint promotion/development plus all-free capability evaluations.

Only the reward-plus-ledger bundle may differ. The treatment uses a derived
bank whose physical/reset arrays are byte-identical to the control but whose
distance sidecar, scenario hashes, and protocol metadata are intentionally
different. Each arm records and validates its carry-channel protocol ID and
distance-sidecar hash. No arm-specific sampler, action mask, horizon, physical
map distribution, architecture, or dynamics change is allowed; both arms get
the same output-preserving carry-observation expansion.

Both arms execute Terra revision
`3051054bc4c713d95905d3f954e6eabf55d6a85a`. The reward-v2 tuple is frozen as
`D_ref=16 m`, `D_bound=2.5`, `gamma=0.9984`, success `B=6`, horizon failure
`F=1`, `alpha=1`, `beta=1.5`, total step cost `1`, shaping weight `1`, and
horizon `450`, with protocol IDs checked from that exact runtime revision.

## Simple implementation boundary

Follow [`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md):

- one named `continuous_banded_v2` preset;
- one named prepared-fork initializer;
- one canonical global distance routine;
- one carry-work scalar channel;
- one reward-v2 potential formula;
- no generic reward framework, compatibility matrix, or fallback modes;
- one reversible implementation commit per repository;
- one to four new claim-driving contract tests per repository unless a silent
  reward/termination error requires more.

The 6,000-update comparison has no resume path. If either arm fails or is
interrupted, that attempted pair is not evidence: retain its failure receipt,
discard both training continuations, and restart both arms from the exact same
prepared update-20,000 fork. Never resume one arm, continue from an intermediate
checkpoint, or splice jobs from different paired attempts. A retry uses a new
committed baselines revision and therefore a new immutable run namespace; prior
run directories are retained and never reused.

The old R1 whole-objective anneal remains historical code and receives no
compute. If R2 loses, retain its receipts and revert or abandon the experiment
commits rather than hardening an unsuccessful design.

## Admission gates

R2 cannot launch until all blocking gates pass:

- **D0:** reproduce the selected checkpoint's frozen evaluation and emit the
  per-identity analysis receipt.
- **D4a:** materialize the exact targeted relocation replay receipt, including
  evaluator graph/batch shape and ledger parity.
- **D4b:** materialize the 4,512-map scale/overlap rows, the 34-map identity
  set, the proposed `(Q,P)` dwell-cost grid, and admitted potential extrema.
- **Dominance:** analytically prove, over every admitted potential and success
  step 1--450, that the minimum discounted exact-success return exceeds the
  maximum horizon-failure return.
- **Implementation:** dense endpoint parity, signed-cycle accounting,
  output-preserving carry expansion, prepared-fork state, v1-to-v2 migration,
  LR warmup, paired restart contract, and finite-value tests pass.
- **Runtime:** each arm independently completes a W&B-disabled update-1 smoke
  after CUDA convolution-backward and NCCL all-reduce preflight on an approved
  RTX 3090/4090 allocation.

D1--D3 and D5--D6 remain nonblocking diagnostics or independent treatments.
They must not be bundled into R2.

## Experiment

- Compact R2 screen: two matched arms, 6,000 additional PPO updates from the
  prepared update-20,000 parent state.
- Checkpoints: at least every 500 updates.
- Fixed evaluations: retained 1,000-update checkpoints on promotion,
  development, and both capability panels; sealed only after treatment
  selection.
- Run allocation: verified 4x RTX 3090/4090 with runtime GPU guard, artifacts
  on Euler scratch/work rather than home, and W&B in
  `aless-weber-eth/mixed-agents`.
- A one-seed difference is a screen. A material effect requires at least three
  paired seeds before a paper-level causal claim.

## Decision rule

Checkpoint selection uses promotion only. Development confirms the selected
checkpoint. Compare:

1. all-47 exact success and condition-balanced macro graded completion;
2. foundation/trench and depth slices;
3. p10 and worst condition;
4. all-free retention;
5. `d12`, `d16`, large adjacent foundations, and obstacle conjunctions;
6. steps and productive workspace cycles only on identities solved by both.

Never compare raw reward between arms. Reward-v2 wins the screen only if it
shows a material fixed-panel improvement without family, tail, or all-free
regression at the selected checkpoint. Ambiguous or checkpoint-unstable results
trigger paired-seed replication, not post-hoc gate changes.

## Iteration loop

1. Inspect the next unchecked gate and its strongest failure evidence.
2. Make one narrow change within the frozen contract.
3. Run the smallest deterministic verifier that can falsify the claim.
4. Record command, revision, artifact, and verdict in this file and the
   experiment ledger.
5. Revert or revise on failure; advance only after the gate passes.
6. After submission, reconcile Slurm, logs, W&B, checkpoints, and fixed
   evaluations rather than trusting one source.

## Anti-cheating and safety

- Do not loosen exact excavation, accepted dump mask, cleanup, unloaded final
  state, mass conservation, or fixed evaluation.
- Do not change the physical map distribution, sampler between arms, horizon,
  architecture, seed, PPO shape, or action/observation contract beyond the
  common carry expansion. Only the treatment's canonical reward-distance
  sidecar and identities derived from that reset array may differ.
- Do not select on development, sealed results, online success, or reward.
- Do not call a queued job, running job, finite checkpoint, or update-1 smoke a
  learning result.
- Do not update expected hashes to bless evidence generated by older source.
- Do not write checkpoints, W&B files, or large logs under Euler home.

## Completion proof

The goal is complete only when all of the following exist:

- clean committed Terra and terra-baselines revisions for the direct R2 path;
- checked D0, D4a, D4b, dominance, CPU, and GPU-smoke receipts;
- exact parent, prepared-fork, dataset, graph, sampler, reward-protocol, source,
  and sidecar identities in both run contracts;
- one jointly released pair of Euler job IDs that each advanced beyond update 1
  and completed the declared 6,000-update screen from the shared prepared fork;
- fixed promotion/development/capability results for both arms with integrity
  checks;
- a result table and causal interpretation in the experiment ledger;
- a recorded decision: reject reward-v2, replicate it, or make it eligible for
  R3.

## Status checklist

- [x] G0 goal activated; clean implementation worktrees created
- [x] G1 D0 receipt complete
- [x] G2 D4a durable replay receipt complete
- [x] G3 D4b scale/overlap/dwell receipt and dominance proof complete
- [x] G4 Terra reward-v2 and carry-observation path implemented and committed
- [x] G5 baselines prepared fork, v2 preset, warmup, receipts, and launcher implemented and committed
- [x] G6 focused CPU tests and independent code review pass
- [ ] G7 both Euler update-1 smokes pass
- [ ] G8 matched 6,000-update R2 jobs jointly released and verified beyond update 1; failed pairs restart from the prepared fork
- [ ] G9 fixed evaluations complete and R2 decision recorded

## Worklog

- 2026-08-10: goal drafted from audit commit `9f34f6d`; no R2 code or job yet.
- 2026-08-10: activated goal; created clean Terra and baselines R2 worktrees
  from `eb3835c1` and `129a56d` respectively.
- 2026-08-10: copied and rehashed the selected parent locally at
  `.artifacts/terra_v8_r2_parent_20260810/`; verified update `20000`, optimizer
  step `1280000`, `2856685` parameters, seed `20260807`, all 47 sampler
  conditions, and saved `continuous_banded_v1` state. The exact parent SHA is
  unchanged at `0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834`.
- 2026-08-10: authoritative static admission receipts passed under
  `.artifacts/terra_v8_r2_admission_20260810/static_v2/`; this supersedes the
  provisional `static/` tree and additionally pins episode/reset-pose
  identity. D0 reproduced main
  development `546/720` (`0.860913` macro), main promotion `549/720`
  (`0.853249`), and both capability panels at `31/32`. D4b reproduced the
  legacy relocation budget range `0.052710--15.335504` (`290.938x`) and the
  exact 34-map high-budget overlap. The global-distance scan covered all
  `7520` admitted scenarios with `D_ref=16 m`, `D_bound=2.5`, and no
  disconnected traversable cells. Analytic terminal dominance passed:
  minimum success `0.771014`, maximum horizon failure `-0.815476`, margin
  `1.586490`. Receipt-manifest file SHA-256 is
  `9b16c391dbe0c108f4b79833f1940c5fc0ba31903a1e7edbfec1797aa53740d9`;
  its canonical receipt-tree SHA-256 is
  `5d5f31d5dc73a850e23a023e33a11efb7f296f6ddff80ddbfda3d0af29c7f291`,
  the sidecar-root SHA-256 is
  `f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980`,
  and the protocol SHA-256 is
  `ea7bf132f4d4f11265c30c443754619f1fb3ed0c6a07db229a72eb29c4b12ca3`.
  D4a remains pending.
- 2026-08-10: Terra reward-v2 committed as
  `3051054bc4c713d95905d3f954e6eabf55d6a85a`. The final focused/broader CPU
  gate passed `73` tests plus `4` subtests; exact dense reward goldens remain
  unchanged. The runtime now has one frozen reward-v2 formula, one canonical
  distance protocol, one ninth carry-work observation, and fail-loud full-reset
  validation. Dense control skips the unused reward-v2 computation.
- 2026-08-10: the first D4a attempt exposed an over-strict full-panel parity
  check: the local full-graph replay produced `549` successes while D0's frozen
  Euler panel records `546`. D0 remains the authoritative 720-map result under
  its declared Terra `a6e6e5bc` source. D4a instead uses the documented
  `dcc4f955`/`eb3835c1` targeted-replay source and gates exact ledger/terminal
  parity only for its nine preselected rows; it reports full-panel hardware
  drift without treating it as evidence. The invalid/contended attempt emitted
  no receipt and was stopped without touching the unrelated GPU job.
- 2026-08-10: the direct baselines path committed as
  `47e39f193d74003ceb27fc090c939a33d4a0bf4b`. Its focused integration suite
  passed `80` tests; Python compilation, Black, shell syntax, diff checks, and
  dry-rendered control/treatment commands passed. The shared prepared fork has
  SHA-256
  `8e01ebd3dfd99b36cea90a251dfe4a4e305228abeb2f5ecba633a9fc6805b1d0`
  and receipt SHA-256
  `d119f443613d4959d5f63918971c50c5ad204e4b6c1d65ec985c3fc31b005185`.
  Its value/logit deltas are exactly zero, and its saved config, accepted-bank
  profile, and sampler state all consistently select `continuous_banded_v2`.
  An independent baselines review and D4a remain pending; no PPO job has been
  launched.
- 2026-08-10: independent review found two launch blockers in the prepared
  training path. Commit `1bb4fedc1f358a3f6a8a2b1f86bcba4cebb07d8a`
  now overlays the arm-selected reward stage after loading the parent
  environment and leaves Terra's complete reset reward-component tree intact;
  the focused suite passed `82` tests. No PPO job was launched.
- 2026-08-10: D4a job `10285183` reproduced `546/720`, but its audit-only
  absolute lift gate rejected a float32 residual of `2^-15`. Commit
  `82b7de4a429761d895cb2d538247ad57ded30daf` records every lift's magnitude,
  location, and float32 spacing before gating and uses the explicit
  four-ULP bound. This changes the verifier only; a clean identical D4a rerun
  remains required before smoke submission.
- 2026-08-10: final launch hardening committed as
  `a94780d2099db57d500901f9b07c879fa51f2e74`. It pins Terra `3051054b`, the
  complete reward-v2 constant tuple and horizon, removes the unsupported resume
  promise, and submits the two arms held before releasing them together. The
  final focused suite passed `83` tests. Independent review additionally passed
  `147` relevant baselines tests and `12` Terra tests plus `4` subtests, with no
  remaining code or launcher blocker.
- 2026-08-10: final-authority D4a job `10289611` completed on one RTX 4090 in
  `12:19` with W&B disabled and no PPO. It exactly reproduced the frozen
  `546/720` panel and all nine targeted traces. Across `7179` lifts, zero failed
  the four-ULP gate; the largest absolute residual was exactly one ULP at slot
  `517`, step `9`, the maximum ULP residual was `2`, inert error was `0`, and
  telescope error was `4.1961669921875e-05 < 1e-4`. The durable receipt SHA-256
  is `6905300337310456a28ec6177a8c7d74f73892ebe052d11d29e9e0fa5bec7362`,
  manifest SHA-256 is
  `cc969a69810b5ed0d14b85d58a0932ae26659a34686c4eadb760ae24b7cc87a4`,
  and the successful execution receipt SHA-256 is
  `b472620f299bee3c174a9691064ae04a7208bbebf586ee0c8c25bd3a671c7ed3`.
  Failed job `10285183` remains archived separately. The final local
  `SUBMIT=0` smoke gate passed against this receipt; no PPO job had yet been
  launched at this checkpoint.
