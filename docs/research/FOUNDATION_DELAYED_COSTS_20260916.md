# Foundation efficiency after completion: delayed costs, September 16, 2026

## Decision and experiment scope

One bounded comparison tests whether modest behavior costs improve an already
successful foundation policy while preserving completion. CSCS job **4675576**
was submitted on September 16 at 00:35 CEST. Its initial five-hour request was
reduced to four hours while held before its first start, after the scheduler
estimated a start too late to fit five hours before maintenance. The scientific
budget and recipe are unchanged. The
[campaign manifest](../../../../../.artifacts/terra_foundation_delayed_costs_20260916/manifest.json)
and experiment ledgers own subsequent submission and runtime status.

Both arms resume the same `scratch_kl` checkpoint at update 20,000 from the
[teacher-KL initialization comparison](FOUNDATION_TEACHER_KL_RECOVERY_20260914.md).
One keeps added costs at zero. The other introduces the first, 25% stage of the
previous combined 2x behavior recipe. Using one shared parent makes the added
costs the experimental difference; this is not another student-initialization
comparison.

The prior CSCS comparison, job **4665916**, was canceled with authorization on
September 15 at **23:55:43 CEST**. Both policies' update 20,000 and 30,000
checkpoints were retained and their SHA-256 hashes verified before cancellation.
Both policies completed 64/64 easy validation maps at update 20,000 and 63/64
at update 30,000. Efficiency gains on the common successful maps were modest,
so another unchanged allocation would not answer the next behavior question.
See the [preservation receipt](../../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/stop_20260915/preservation.json)
and [verified stop state](../../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/stop_20260915/verification_after.json).

## Common parent and fixed training contract

The selected parent passed the existing completion gate with **64/64** exact
successes at both updates **10,000 and 20,000** on the same 64-map panel.
The accepted zero-cost reference also completes 64/64. The resulting minimum
for this stage is **63/64**: an integer threshold that stays within three
percentage points of that reference and exceeds 90% completion. The
[qualification receipt](../../../../../.artifacts/terra_foundation_delayed_costs_20260916/qualification.json)
pins the reports, parent and proposed stage.

| Setting | Fixed value |
| --- | --- |
| Allocation | One CSCS Daint node, account `lterenzi`, project `d130`, at most four hours |
| GPU layout | Two concurrent processes, two GH200 GPUs per arm, four GPUs total |
| Batch per arm | 2 x 256 environments = 512 environments |
| Training bank | Same 256 easy foundation maps, full episode resets, one `train/all` curriculum level |
| Seed | 20260907 in both arms |
| PPO | 32 rollout steps, two epochs, 32 minibatches |
| Work per update | 16,384 transitions and 64 Adam steps per arm |
| Optimizer | Native Adam from the common parent; step 1,280,000 at update 20,000 |
| Learning rate and entropy | Constant learning rate 3e-4; constant entropy coefficient 0.02 |
| Advantage normalization | Global minibatch moments across each arm's two GPUs |
| Checkpoints | Numbered checkpoints at least every 500 updates |
| Final target | Absolute update 25,000: 5,000 new updates, 81.92M new transitions per arm |
| Evaluation milestones | Updates 22,500 and 25,000, plus a fresh frozen-parent benchmark |

The native parent SHA-256 is
`64e888f894d57cda973aded53c1402c966b09feb6bc1423c8e5f23cec0510e14`.
The source snapshot is unchanged from the successful initialization campaign:
Terra `46738cde28e455da7c466fc0a2cb64f677d86401` and terra-baselines
`a3a2119af014f9afe018ed69b023a2e9c1acc4dc`. The training-bank distance-sidecar
SHA-256 is
`6b2675998403ed2d6125d955fca446404fbdf260e0a0c2cf7b9864cbdd1fb2bf`.
Source and input manifests bind the complete files; these revision labels alone
are not a substitute for byte verification.

Both arms retain the same actor, critic, encoder, observations, map geometry,
movement, dumping and chassis rules. There is no adaptive sampler, mixed trench
curriculum, partial reset, task-bank transfer or architecture change. Native
resume preserves model parameters, Adam state and absolute clocks. As in the
existing trainer, it resets live environments, RNG and action history; this is
not bit-exact continuation of the old process. Paired initialization receipts
must verify that both new arms receive the same initial state.

## Treatment and exact ramp timing

| Arm | Lateral fresh-dig cost | Base-travel cost | Base-turn cost | Schedule |
| --- | ---: | ---: | ---: | --- |
| `control` | 0 | 0 | 0 | Remains zero |
| `penalty_p25` | 0.125 | 0.0025 | 0.01 | Linear increase over 2,500 updates, then hold for 2,500 updates |

The lateral term applies to newly excavated target soil relative to the chassis
axis. It does not penalize cabin orientation during dumping or loose-soil
handling. Travel and turn terms charge executed chassis motion. The lateral
score is an orientation preference, not a physical tipping-stability model.
The existing [behavior reward definitions](FOUNDATION_BEHAVIOR_20260907.md)
remain unchanged.

For completed-update index `u`, the penalty fraction is
`clip((u - 20000) / 2500, 0, 1)`. The initialization reset at update 20,000
therefore uses zero added costs. The first new rollout, producing checkpoint
update 20,001, uses **1/2500 of the targets**. At update 20,002, effective costs
are **2/2500 of the targets**. The full targets apply to the rollout producing
update 22,500 and remain fixed through update 25,000.

The startup gate resumes the common parent through update 20,002 in both arms.
Only the penalty startup passes `--finetune_foundation_behavior` and
`--behavior_cost_ramp_updates 2500`. Production then resumes each arm's own
accepted startup checkpoint with no new fine-tune request and no new ramp
override. The saved ramp keeps its original start at 20,000 and duration 2,500;
it must not restart at 20,002. TrainConfig and the R2 receipt declare target
costs, while the saved environment records the effective costs used by the last
rollout. Acceptance checks both quantities and the executed reward components.

The frozen teacher remains configured, with SHA-256
`d1a6c07d9d8a40b7b7b60bd0b54313aa46a9b50fb09c789ed4ebffddb2b488b3`.
Its original policy-KL schedule stays anchored at update zero, with coefficient
1 decaying by cosine to zero at update 20,000. Value distillation remains
disabled. Thus teacher regularization is **zero throughout all new training**;
neither its metadata nor its clock is reset. A logged teacher KL of zero in
this stage is the disabled-path placeholder, not measured policy agreement.

## Runtime acceptance and maintenance window

The local CUDA startup and native-resume gates passed for both arms. Their
initial model, Adam, reset state, RNG, history and teacher hashes match exactly;
the penalty ramp preserves its original clock through the third update. A
complete local replay with the current evaluator reproduces **64/64** frozen
parent completions. Independent review and remote payload verification passed.
Full production-shaped startup remains an in-allocation gate.

Before submission, validate the parent and teacher bytes, source and bank
identities, finite native model/optimizer/loss tensors, zero material-integrity
failures, and exact Adam clocks. Small 1 x 32 local diagnostics must be labeled
as such. In the allocation, each arm must pass CUDA convolution backward,
two-GPU collective communication and finite updates at the full **2 x 256**
shape. Both arms must pass their startup and paired-state checks before either
enters production. A startup failure ends the allocation.

The planned allocation has `--time=04:00:00` and
`--deadline=2026-09-16T06:55:00`, ahead of maintenance at **07:00 CEST on
September 16**. This is a completion deadline. With the full four-hour request,
the latest eligible start is **02:55 CEST**; submitting later does not make the
requested allocation fit. Do not carry a pending experiment across maintenance
or silently shorten one arm's workload. Record any incomplete stage as
incomplete and preserve the latest finite checkpoints.

This is one explicitly bounded 5,000-update behavior stage. It permits no
automatic second allocation, continuation beyond update 25,000, or escalation
to stronger costs. A timeout or missed evaluation is not evidence of saturation
or of a failed behavioral hypothesis.

## Evaluation and decision criteria

Re-evaluate the immutable update 20,000 parent with the same pinned runtime as
the new arms. Keep that report beside the historical qualifying report and
require its 64/64 completion to reproduce before applying the stage gate.
Evaluate both arms at updates 22,500 and 25,000 using the same **64 validation
maps, reset identities, seed 20260907, greedy actions and 450-step horizon**.
Preserve source-map and material hashes, checkpoint hashes, native validation
receipts, complete per-map rows and material-integrity results. Online return
and startup completion are not policy-quality evidence.

Always report exact completion, dug volume and accepted material over all 64
maps. The penalty stage must achieve **at least 63/64 exact completions at both
milestones**, which are 2,500 updates apart at the full target costs. Report the
matched control at both milestones as well, so ordinary continuation drift is
visible. Passing this coverage gate permits consideration of the behavior
results; it does not establish that the new behavior is better or authorize a
larger penalty.

Compare behavior first on maps completed by both arms, and also on the common
success cohort including the frozen parent. Retain cohort membership and each
metric's valid paired sample count. Keep missing workspace measurements
unavailable rather than replacing them with zero. Report failures and lost
parent successes separately; a shrinking successful cohort cannot count as an
efficiency gain.

The primary behavior measures are:

- **Workspace yield:** unique newly excavated area per productive base setup,
  its lower tail, and the number of productive setups. Overlapping digs can
  improve setup yield without increasing each individual dig's fresh area.
- **Retained work-pose travel:** distance between the productive poses retained
  for execution, along with revisits and backtracking. Straight-line distances
  are geometric lower bounds, not measured Nav2 paths.
- **Workspace continuity:** adjacency between consecutive fresh workspaces and
  unnecessary returns to earlier work areas.
- **Lateral fresh excavation:** volume-weighted lateral orientation score and
  the fraction of new excavation performed sideways relative to the chassis.

Deployment keeps the base poses and target workspaces, then uses the external
navigation stack to reach them. Raw Terra navigation-action counts and raw
travel remain secondary diagnostics. Extra simulator navigation can be useful
if it produces better productive setups, but adjacent workspaces and a coherent
work sequence remain objectives. Review representative successes, newly failed
maps and large metric changes qualitatively before preferring either policy.

This screen has one paired training seed, one easy-bank panel and one aggregate
cost stage. It tests the combined lateral/travel/turn intervention; it cannot
attribute changes to an individual coefficient. It does not establish an upper
penalty bound, saturation, general performance on the full dataset, or
asymptotic plasticity. Any next stage requires a separate decision based on
complete matched reports and behavior inspection.
