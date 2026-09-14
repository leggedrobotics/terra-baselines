# Foundation teacher-KL initialization comparison — September 14, 2026

The user reports that a randomly initialized student with teacher KL has worked
better than initializing directly from a strong policy. Test that observation
with a matched pair on the same easy foundation bank. This supersedes the
unsubmitted single native-continuation proposal prepared earlier today. The
primary arm is scratch plus KL; the comparison initializes the full model from
the old control and uses the same KL teacher, fresh Adam and clocks starting at
zero. CSCS job **4665916** was submitted on September14 at16:28CEST after
authentication was renewed. It is queued; the full production-shape runtime
gates have not run yet.

## September14 submission status

Job4665916 requests one24-hour d130 normal allocation as lterenzi, with two tasks,
two GH200 GPUs and32CPU cores per task. Slurm confirms totalfourGPUs and
`gres/gpu:per_task:2` binding. At16:31CEST it is PENDING/Priority, with a provisional
start estimate of September15 around06:03CEST; scheduler estimates can change.
Training has no completed updates yet. Full2x256-per-arm CUDA, convolution,
NCCL, finite-checkpoint and paired-initialization checks remain mandatory.

The reviewed trainer uses baselinesa3a2119 and Terra46738cde. The frozen teacher,
source, launch files, bank and image hashes passed remote verification before
submission. The evaluation window will start with the allocation, so queue
waiting does not consume its26-hour coverage. Evaluation remains bounded to
both arms at2500/5000/10000/20000updates; no further training jobs are automatic.
See the [live manifest](../../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json)
for current scheduler and monitoring evidence.

## Question and arms

Does a randomly initialized student learn better completion or digging behavior
than a pretrained student when both receive the same policy-KL guidance?

| Arm | Initial model | Optimizer and clocks | Teacher |
| --- | --- | --- | --- |
| `scratch_kl` (primary) | Random full model | Fresh Adam; update 0 | Frozen old zero-cost control u15000 |
| `pretrained_kl` | Old control full model parameters | Fresh Adam; update 0 | Same frozen old control u15000 |

The pretrained treatment copies the actor, critic and shared encoder. It does
not isolate actor-only initialization. Both arms benefit from a pretrained
teacher; “scratch” describes the student's initial parameters. The same KL
coefficient produces stronger initial imitation pressure on the random student,
while the pretrained student initially matches the teacher. That is a mechanism
of this comparison, not evidence of an unmatched hyperparameter.

Use the existing parameters-only `--warm_start_from` for the pretrained arm's
first process. Once initialized, continue each arm from its own native
checkpoint with `--resume_from`, preserving its model, Adam and schedule origin.
The old control's Adam state and its u15000 clock are never imported into either
arm. Environment, RNG and live histories restart at a native segment boundary
as in the existing trainer; continuation is not bit-exact.

## Fixed run contract

| Setting | Value |
| --- | --- |
| Allocation | One CSCS Daint d130 normal node, 24 hours, account lterenzi |
| GPU layout | Two concurrent processes, two GH200 GPUs each; four GPUs total |
| Per-arm batch | 2 × 256 environments = 512 global environments |
| Bank | Same 256 easy-foundation train maps; 64 held-out validation resets |
| Seed | 20260907 for both arms |
| Environment | Terra 46738cde; corrected movement and soil-free chassis protections |
| Frozen teacher | Old zero-cost control u15000, SHA d1a6c07d9d8a40b7b7b60bd0b54313aa46a9b50fb09c789ed4ebffddb2b488b3 |
| PPO workload | 32 rollout steps, two epochs, 32 minibatches; 16,384 transitions and 64 Adam steps per update |
| Advantage normalization | Global minibatch moments across each arm's two GPUs |
| Policy loss | PPO plus beta × KL(teacher || student) on student-visited observations |
| Teacher interface | Frozen parameters; matching executable-dig and other observation selectors |
| KL schedule | beta 1 at update 0, cosine decay to 0 at update 20,000 |
| Value distillation | Disabled |
| Learning rate | Constant 3e-4 in both arms; zero warmup |
| Entropy | Constant 0.02 in both arms |
| Added behavior penalties and ramp | All zero; disabled |
| Checkpoints and target | Every 500 updates, absolute target 500,000, one bounded allocation |
| Evaluation | Both arms at updates 2,500, 5,000, 10,000 and 20,000 |

The common entropy value is a conservative pilot choice that avoids the old
scratch recipe's high initial entropy competing with imitation. It is not an
established Terra optimum. There is no separate entropy or KL-coefficient sweep
in this pair. A single paired seed can screen these two recipes; it cannot
establish general superiority or confirm a historical result.

Teacher-guided policy KL is described in
[Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835).
That motivates trying the method, not a prediction that either initialization
or these coefficients will win. Without a matched no-KL arm, the comparison
cannot isolate the contribution of KL itself.

## Acceptance and evaluation

Before expensive training, each arm must independently complete finite local
updates 1 and 2, with Adam steps 64 and 128, then resume its own u2 checkpoint
through u3 with Adam 192. The actual initialization receipt records model,
optimizer, initial reset and RNG hashes. At startup, require matching optimizer,
reset and RNG hashes, zero clocks, a common teacher hash, random student weights
different from the teacher, and pretrained student weights equal to the teacher.
These checks establish the intended treatment; two updates are not policy
quality evidence.

Inside the CSCS allocation, verify two GH200s per process, disjoint physical GPU
identities, JAX/container versions, convolution backward and NCCL, then repeat
two finite updates at the full 2×256 shape. Both arms must pass the startup and
paired initialization checks before either enters production. A failure stops
the allocation; do not shrink one arm's batch. Sources, teacher, bank, launchers
and image are pinned and hashed. There is no automatic second allocation.

Evaluate matching updates on the same 64-map, greedy, 450-step panel, corresponding
to 40.96M, 81.92M, 163.84M and 327.68M new training transitions. A bounded watcher
can serialize the eight completed evaluations on an idle local GPU; it does not
submit training. Record exact completion, dug and accepted material, productive
base poses, unique area per productive setup, retained-work travel and workspace
adjacency. Compare efficiency on common successful episodes and retain failures
separately, so losing coverage cannot appear to improve efficiency. Wait for
matched complete reports; do not rank from startup or online scalars alone.

The frozen old zero-cost control freshly solves 62/64 under the current rules;
old 2× solves 63/64. Those are easy-bank reference results, not performance on
the full foundation dataset. Keep behavior penalties off until the existing
completion gate passes: two evaluations at least 2,500 updates apart, at least
90% exact completion, and within three percentage points of the zero-cost
reference. Promotion is a separate decision.

## Foundation isolation and previous implementation evidence

The inspected foundation launcher samples only train/all: one curriculum level,
no trench rewards, no mixed-task pool and no cross-process gradients. Trench
alignment is inactive. Shared soil/chassis, dumping, relaxation and movement
repairs still change exploration. Old-policy success proves completion remains
possible, not that random exploration has equal learning difficulty. See the
[foundation isolation audit](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/foundation_isolation_audit.md).

Earlier preparation fixed teacher continuation's coefficient origin, zero-warmup
Optax structure, and executable-dig observation stub. Seven native-teacher tests
and 47 training utility tests passed, along with native u15001/u15002/u15003 GPU
checks. Those remain useful implementation evidence but do not qualify the new
scratch and parameters-only startup paths or the new 2×256 production layout.
The earlier local resume still needed about 206 seconds before its first update;
compilation-cache reuse was not established. New matched initialization gates
and a small opt-in provenance receipt are required for this revised campaign.

## September 14 local qualification

All 57 CPU checks pass: three startup-receipt tests, seven native-teacher tests
and 47 training utility tests. Both actual GPU arms complete updates 1 and 2
with Adam 64/128, then resume their own u2 through u3 with Adam 192. All model,
optimizer, loss and teacher values pass finite checks; material transition
integrity counters are zero, and the bank/environment/R2 contracts match.
Actual initial optimizer, reset, RNG, history and teacher hashes match across
the arms; the pretrained model equals the teacher and the random model differs.
Each resume preserves its own exact model and Adam-moment hashes.

The paired evaluation helper was checked against strong historical policies and
an actual weak 11/64 policy. Maps with no productive digging can have undefined
workspace metrics. These remain unavailable, with explicit matched per-metric
counts; they never become zero or invalidate all-64 coverage. Independent review
reproduces all 14 metrics and missing-value/rejection cases. Wrong-arm checkpoint
labeling is rejected. The new checkpoint also loads and configures successfully
under the pinned historical evaluator 866e8e2.

These are functional local1x32 gates. Each process still spends roughly
196–253 seconds before its first update; compilation reuse and full-size
throughput improvements are not established. Full2x256-per-arm GH200 qualification
is enforced inside the allocation. CSCS SSH last checked at14:40 still rejects
lterenzi with Permission denied(publickey); no paired job has been submitted.

[Current manifest](../../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json) ·
[Current launchers](../../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/launch/) ·
[Superseded, never-submitted proposal](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/manifest.json).
