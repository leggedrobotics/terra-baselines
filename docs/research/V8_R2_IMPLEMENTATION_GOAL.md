# V8 reward-v2 system-improvement goal

Status: **IMPLEMENTATION IN PROGRESS; DO NOT SUBMIT WITHOUT LORENZO'S APPROVAL**

Authority:

- [V8 reward and termination audit](V8_REWARD_TERMINATION_AUDIT.md)
- [V8 improvement set](V8_IMPROVEMENT_SET_20260810.md)
- [V8 scale-up record](V8_10M_SCALEUP.md)
- [`simple-research-code`](/home/lorenzo/.codex/skills/simple-research-code/SKILL.md)

## Outcome

Build and evaluate one from-scratch compact reward-v2 system on the full V8
distribution. This is a practical system-improvement experiment, not a new
two-arm reward study. Yesterday's completed compact dense run is a descriptive
reference only; no second dense job is launched.

The active implementation has one path:

- compact deep SE encoder plus cross-attention;
- random initialization with seed `20260807`, no teacher, warm start, prepared
  fork, or resume;
- common nine-feature agent input including normalized carry work;
- all 47 V8 conditions from update 0 under `continuous_banded_v2`;
- reward-v2 with the canonical global physical-distance ledger;
- 40,000 PPO updates and checkpoints every 500 updates;
- promotion main and capability panels evaluate every 1,000-update checkpoint
  from updates 1,000--40,000. They select by combined exact success, then
  47-condition macro completion, then worst condition, then earliest update;
- development main and capability panels evaluate only that promotion-selected
  checkpoint. Sealed evaluation remains unused.

## Historical descriptive reference

The completed dense compact system is job `10128518`, baselines revision
`dcc4f955347182e57e6f16e9df81a3f170564d97`, and selected update-20,000
checkpoint SHA-256
`0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834`.
It scored:

- development main: `546/720` exact, macro completion `0.861`;
- development capability: `31/32` exact, macro completion `0.977`;
- promotion main plus capability: `580/752` exact, macro completion `0.859`.

Those numbers provide practical context, not a required equal-budget gate. The
new run naturally emits update 20,000 because every 1,000-update checkpoint is
evaluated, but the headline result is the promotion-selected checkpoint over
the complete 40,000-update run.

## Bundled system changes

The new system differs from the historical dense reference in six connected
ways. Report them together and do not assign an observed gain to one item:

1. `dense_skill` becomes `material_potential_v2`.
2. The legacy per-map relocation distance and ledger become
   `obstacle_geodesic_8_physical_global_v1` with `D_ref=16 m` and
   `D_bound=2.5`.
3. The policy receives normalized carry work as agent-state feature 9. This
   changes the compact parameter count from `2,856,685` to `2,856,701`.
4. `continuous_banded_v1` becomes `continuous_banded_v2` from update 0.
5. Runtime Terra moves from the historical `eb3835c1` source to reward-v2
   revision `3051054b`.
6. The training target increases from 20,000 to 40,000 updates.

The treatment bank preserves all physical target, action, occupancy,
dumpability, pose, map-ID, and source-ID data. Only the canonical distance
sidecar and identities derived from it change. Therefore fixed-panel results
refer to the same physical V8 tasks even though scenario hashes differ.

Fresh v2 sampling retains foundation/trench family mass `0.5/0.5`. Its initial
aggregate depth mass is:

| Depth | Initial mass |
|---|---:|
| d0 | `0.11346390374331551` |
| d1 | `0.3836076203208556` |
| d2 | `0.5029284759358292` |

For context, the historical v1 start was approximately `75.4% / 17.8% / 6.8%`
over d0/d1/d2. This much broader hard-map exposure is an intentional part of
the improved system.

## Frozen run contract

- Runtime Terra:
  `3051054bc4c713d95905d3f954e6eabf55d6a85a`.
- Bank: 47 conditions x 96 train layouts; full 450-step resets; exact visible
  dump completion.
- PPO: 4 devices x 512 environments, 32 rollout steps, 32 minibatches, 2
  epochs, learning rate `3e-4`, gamma `0.9984`, no value clipping, flat
  minibatch shuffle.
- Entropy: `0.15 -> 0.02` over the first 20,000 updates, then remains `0.02`
  through update 40,000.
- Reward-v2: success `6`, horizon failure `1`, alpha `1`, beta `1.5`, total
  step cost `1`, shaping weight `1`, distance reference `16 m`, distance bound
  `2.5`.
- Compute: one 4xRTX4090 `gpuhe.120h` job after one independent 4xRTX3090
  update-1 smoke.
- Storage: live logs, W&B, and checkpoints under Euler scratch; immutable
  source and receipt hashes in the run contract.

There is no resume path. If the 40,000-update job fails or is interrupted, keep
its failure evidence and restart a reviewed revision from update 0. Do not
splice or promote an incomplete continuation as the declared system result.

## Admission and interpretation

Required before the long run:

1. committed clean Terra and baselines revisions;
2. D0, D4a, D4b, materialization, and analytic terminal-dominance receipts;
3. focused CPU tests and shell/static launcher checks;
4. one finite update-1 CUDA/NCCL smoke with zero transition-integrity errors,
   reward-v2 protocol receipt, v2 sampler receipt, 9-feature model, and no
   prepared-fork receipt;
5. explicit approval to submit.

Rank retained checkpoints using promotion exact success and condition-balanced
macro completion, then inspect family, depth, p10, worst-condition, and
all-free retention. Development is confirmation only. Efficiency comparisons
use jointly solved identities. Raw reward is never compared with the dense
reference.

An update-1 smoke proves only executable integration. A queued or running job
is not learning evidence. The 40,000-update result is one seed and should be
presented as a system screen unless later replicated.

## Status checklist

- [x] A0 reward-v2 runtime, distance, carry observation, and v2 sampler implemented
- [x] A1 D0/D4a/D4b/materialization/dominance receipts passed
- [x] A2 obsolete prepared-fork attempt archived; no repair planned
- [ ] A3 scratch-only launcher, checkpoint receipt, verifier, and docs committed
- [ ] A4 scratch update-1 smoke passes
- [ ] A5 40,000-update run explicitly approved, submitted, and verified beyond update 1
- [ ] A6 fixed evaluations complete and promotion-selected result recorded
- [ ] A7 promotion-selection receipt, selected and final checkpoints, run contract, and fixed-evaluation JSONs copied from Euler scratch to durable work/local artifacts with hashes verified

## Worklog

- 2026-08-10: Terra reward-v2 committed as
  `3051054bc4c713d95905d3f954e6eabf55d6a85a`; its focused gate passed 73
  tests plus 4 subtests and retained the exact dense goldens.
- 2026-08-10: static admission and bank materialization covered all 7,520
  admitted scenarios. The treatment-bank dataset SHA is
  `5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851`,
  tree SHA is
  `225e13aacd9047e7f241facd3397fd66794e3094a883cc6dc26304decc24d388`,
  and canonical-sidecar dataset SHA is
  `f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980`.
- 2026-08-10: final D4a job `10289611` reproduced the frozen `546/720` dense
  panel and all nine selected ledger traces. Across 7,179 lifts, no event
  failed the four-ULP gate; the durable receipt SHA is
  `6905300337310456a28ec6177a8c7d74f73892ebe052d11d29e9e0fa5bec7362`.
- 2026-08-10: the superseded prepared-fork implementation was retained in git
  history (`47e39f1`, `1bb4fed`, `82b7de4`, `a94780d`, `15641d5`). Its update-1
  jobs `10292301` and `10292324` both failed before producing a checkpoint on
  the same Optax `FrozenDict`/plain-dict tree mismatch. They consumed 11:03
  and 12:26 respectively, emitted no smoke validation, and are implementation
  failure evidence only. The direct scratch design deliberately does not fix
  or generalize that abandoned path.
- 2026-08-10: Lorenzo selected one practical from-scratch reward-v2 system
  instead of another dense control. The target was extended to 40,000 updates
  because the prior compact dense curve was still improving at update 20,000;
  its established entropy decay remains frozen to the first 20,000 updates.
