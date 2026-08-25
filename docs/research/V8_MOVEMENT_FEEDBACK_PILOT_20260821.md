# V8 movement-feedback pilot

Date: 2026-08-21

## Final training status

Training source is frozen at terra-baselines
`5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14` and Terra
`c8ab920504e09173760c8beba71589102d54ed21`. Control job `11364188`
completed update 50,000 on 2026-08-23 at 20:11 CEST; feedback job `11364189`
completed update 50,000 on 2026-08-23 at 22:37 CEST. Both exited `0:0`, wrote
rolling and `FINAL` checkpoints, and have finished W&B runs. There is no live
job left to cancel.

The immutable final checkpoints are:

- control: `v8_movefb_control_5d7284f6ca6d_s20260821_FINAL.pkl`, SHA-256
  `5459bd5347dbdf64431cd78df5f61f22b75ee56bc2b15662d9751fb2959a7f84`;
- feedback: `v8_movefb_feedback_5d7284f6ca6d_s20260821_FINAL.pkl`, SHA-256
  `8cde5ccd4fd4ef5b1ed716a9c5c3a4c4b43f69d44db66d29ed7db86f2ad7d7df`.

Their source run root remains
`/cluster/scratch/alesweber/codex_terra_edge_runs/terra_v8_movement_feedback_v1/runs/5d7284f6ca6d3c7a53a3ba2dea669c66d3c0ca14/c8ab920504e09173760c8beba71589102d54ed21/s20260821`.
The u40 and final checkpoints from both arms are also preserved outside
scratch under
`/cluster/project/rsl/alesweber/terra_runtime/archives/v8_movement_feedback_20260821/checkpoints`.
The u40 SHA-256 values are
`63a3d55d9b28b07b1acb9c11dfe0db9c22b3824cbacfd0905e00c0231f5ba524`
for control and
`e07fb0de0f941368501f801b18645a6fa4fe3aaabc8615a24bc9cd090c10324d`
for feedback.

## Question

Does compact movement feedback reduce the blocked-movement attractors that
dominate the current GRU failures without reducing global excavation ability?

This is a deliberately small two-arm capability experiment. It does not try
to attribute the result between current feasibility and previous outcome, and
it does not estimate the effect of the mechanics repairs shared by both arms.

## Common runtime

Both arms start from scratch with the same seed, source, banks, PPO recipe, and
Terra runtime. The common Terra runtime:

- removes the duplicate local-soil-relaxation call, retaining the single
  post-update relaxation in `_handle_dig`;
- excludes the exact current base footprint from dig and relift;
- keeps real blockers visible beneath the agent overlay;
- retains reward-v2, capacity-bounded partial relift, the existing relay
  partial-reset schedule, Continuous Banded v3, and unmasked actions.

The accepted training bank is release
`terra_v8_v6_constraints_v7_adjacent_train96_v5`, archive SHA-256
`b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725`.
The existing partial-reset bank archive is
`eb200b151f6b47d9f2ea5f53f6b13cdb45b595a54029fd5d866ec732fea1c8b8`
with semantic digest
`fb73b1d12dfad98c9aa79680d4d3ac178bf84b537e1be1e822535c65473a23f5`.

## Arms

| Arm | Current movement feasibility | Previous transition outcome |
| --- | --- | --- |
| control | absent | absent |
| feedback | four bits: forward, backward, base clockwise, base anticlockwise | two bits: any physical effect, material-or-load change |

The bits are observations only. They never mask logits. Previous outcome is
`00` on reset; thereafter the only valid codes are `00` (no effect), `10`
(agent/configuration effect only), and `11` (terrain or carrier changed).

Each optional vector has separate zero-initialized actor and critic embeddings.
With the same initialization key, all 225 shared parameter leaves and the
initial value, logits, and GRU hidden output must be bit-identical between
arms. The treatment adds exactly 8,448 zero parameters.

## Training contract

- paired seed: `20260821`;
- fresh scratch initialization, never the u44 checkpoint;
- actor: GRU64 concat-skip; critic remains feed-forward;
- four RTX 4090 GPUs per arm, 512 environments per GPU;
- 32 rollout steps, 32 recurrent minibatches, two PPO epochs;
- 65,536 transitions per update;
- primary checkpoint: absolute update 50,000, or 3,276,800,000 transitions;
- rolling checkpoints every 500 updates; and
- two independent Slurm jobs with identical resources and no dependency.

Every allocation first runs the full CUDA-library, convolution-backward, and
NCCL preflight, then one exact-shape W&B-disabled PPO update. Production starts
fresh only after that checkpoint has finite losses, model parameters, and
optimizer state and matches its arm contract. A queued or merely running job
is not evidence; each arm becomes healthy only after a real production update.

## Evaluation and decision

Online curves are diagnostic. The primary u50 pilot gate is the accepted
720-map `evaluation/main/development` panel. The previously mined promotion
panel is continuity evidence only. Report exact success, paired conversions
and regressions, hard-condition exact success, base no-effect rate and maximum
streak, cycle incidence/duration, material progress, and successful episode
length.

Practical pilot decision on the development panel:

The fixed hard-64 subset is 16 maps from each of
`fnd-slab-side1-obj`, `fnd-proc-side1-road`, `fnd-slab-ring3x-obj`,
and `fnd-slab-ring3x-obj1`.

- **GO:** feedback is at least control minus 2/720 globally, gains at least
  6/64 on the four hard conditions, reduces their base no-effect count by at
  least 20%, and loses at most 1/16 on d16;
- **NO-GO:** feedback is control minus at least 8/720 globally, or has no hard
  gain and less than 10% hard no-effect reduction; or
- **AMBIGUOUS:** all other outcomes. Continue both arms—not only feedback—to
  u70 only if the matched u40-to-u50 fixed readout is still rising.

These thresholds are engineering pilot gates, not statistical claims. A paper
claim requires paired seeds `20260822` and `20260823`, checkpoint selection on
source-disjoint development data, then one locked read on the separately
frozen 1,504-map sealed panel. The sealed main and capability-floor manifests
are `5a67ea5e948a014005e196eed6bf616308fbe55fc00efc9b469886d227c2ee2b`
and `b87c02b95254fe9d594c8f038bc48e53a04b75738389d93b5defe103d76c0015`.
They have been archived and statically audited, but no policy evaluation was
found in the audited records; call them policy-unseen, not access-unseen. Any
contrary prior result voids that designation. Promotion, pose, stochastic, and diagnostic
failure traces remain permanently ineligible for training or curriculum use.

## Completed online readout and promotion boundary

The final 1,000-update aggregate is a health and mechanism diagnostic, not the
preregistered selection result:

| Arm | Episodes | Full-start success | Terminal soil | No-effect rate | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 75,442 | 0.99019 | 0.99548 | 0.03152 | 88.93 |
| feedback | 75,602 | 0.99037 | 0.99591 | 0.01450 | 88.10 |

Online success is tied: feedback differs by only +0.018 percentage points.
The treatment nevertheless has a stable mechanism signal, cutting the online
no-effect rate by about 54% and shortening episodes by 0.83 steps. Feedback
took 66 h 22 min versus 64 h 17 min for control, about 3.2% more wall time.
The same pattern was present in each checked late window from u40 to u50; the
last single W&B point is therefore not evidence of a collapse.

No completed movement-feedback result was found for the accepted 720-map
development panel. Consequently:

- the optional observation implementation and repaired common runtime may be
  promoted to the main code line with feedback disabled by default;
- both u40 and u50 policies remain checkpointed candidates, not a selected
  policy; and
- feedback must not become a training default, nor be called a capability
  improvement, until the paired development-panel gate above is evaluated.

The next scientific action is the paired u40/u50 development-720 evaluation,
including hard-64, d16, conversions/regressions, no-effect streaks, and cycle
statistics. The promotion panel remains secondary continuity evidence.

## Issue checklist

| Issue | Current state |
| --- | --- |
| duplicate soil relaxation | fixed and trained in both arms; held-out effect not isolated |
| under-base excavation and hidden blockers | fixed and trained in both arms; held-out effect not isolated |
| blocked movement/no-op attractors | online no-effect rate fell about 54% with feedback; fixed recurrence panel pending |
| previous action outcome | implemented and trained as two exhaustive bits in the feedback arm; default remains off |
| current movement feasibility | implemented and trained as four exact unmasked bits in the feedback arm; default remains off |
| five-way `DO` affordance | deferred; real actions have compositional material outcomes |
| over-capacity partial relift | already fixed; zero incidence in current GRU-u44 failures |
| accepted-first dump fallback | deferred; observed incidence remains negligible |
| reward/serviceability redesign | deferred pending broader station/heading evidence |
| relay/cleanup exposure | held identical through the existing partial-reset bank |

## Entry points

- `scripts/run_v8_movement_feedback_v1.sh`
- `scripts/euler_v8_movement_feedback_v1/submit.sh`
- `scripts/euler_v8_movement_feedback_v1/run.sbatch`
