# V8 movement-feedback pilot

Date: 2026-08-21

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

## Issue checklist

| Issue | State entering this pilot |
| --- | --- |
| duplicate soil relaxation | fixed in the common runtime; untrained |
| under-base excavation and hidden blockers | fixed in the common runtime; frozen-policy transfer was non-monotone, so no benefit is claimed |
| blocked movement/no-op attractors | confirmed dominant behavior; six-bit feedback is the highest-coverage interface treatment |
| previous action outcome | implemented as two exhaustive bits in the feedback arm |
| current movement feasibility | implemented as four exact unmasked bits in the feedback arm |
| five-way `DO` affordance | deferred; real actions have compositional material outcomes |
| over-capacity partial relift | already fixed; zero incidence in current GRU-u44 failures |
| accepted-first dump fallback | deferred; observed incidence remains negligible |
| reward/serviceability redesign | deferred pending broader station/heading evidence |
| relay/cleanup exposure | held identical through the existing partial-reset bank |

## Entry points

- `scripts/run_v8_movement_feedback_v1.sh`
- `scripts/euler_v8_movement_feedback_v1/submit.sh`
- `scripts/euler_v8_movement_feedback_v1/run.sbatch`
