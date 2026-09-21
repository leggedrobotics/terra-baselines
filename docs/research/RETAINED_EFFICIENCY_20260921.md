# Delayed retained-work efficiency comparison — September 21, 2026

Live check September21 at20:14UTC: CSCS4729577 runs on nid006413, fourGH200s.
Both native first-update/save-resume qualifications, migration parity and
matching initialization pass. Control is aroundu108690 withu108500 saved;
its firstu107500 panel passes at381/384foundations,212/224trenches,31/32roads.
The25% arm qualifies throughu105002 and awaits the control. No treatment
benefit is measured yet. Independent Euler geometry corrections remain
excluded from this matched experiment. Recommended next check23:15UTC;
allocation endsSeptember22 at01:20UTC. No monitor scheduled. Evidence:
`.artifacts/terra_efficiency_p25_20260921/status_20260921_2014/`.

The bounded broad continuation reached u110000. Its final fixed panel triggered
the preregistered trench-retention stop, not a numerical or infrastructure
failure. Select u105000 for a small efficiency comparison; do not promote the
last checkpoint simply because it is newer.

| Native checkpoint | Foundations | Trenches | Road subset | Retention |
| --- | ---: | ---: | ---: | --- |
| u100000 | 378/384 | 209/224 | 30/32 | Reference |
| u105000 | 381/384 | 211/224 | 31/32 | PASS |
| u110000 | 381/384 | 206/224 | 29/32 | STOP: trench net loss 3 > 2 |

CSCS job4725717 exited after 6h44m52s. Its native model and Adam state at
`terra-overnight-20260920/retry_20260921/training/checkpoints/generalist-oracle-combined_update_105000.pkl`
are the common parent. This is the recent broad policy with remaining-time
input and residual actor head; neither imitation nor efficiency costs were
active. Both teachers have already faded to zero.

## Bounded comparison

Use one CSCS node with four GH200 GPUs and an eight-hour allocation ceiling.
Qualify both arms independently through a finite native update and a saved
checkpoint resume before starting production. Then run the arms sequentially.
Both use 4x256 environments, 32 rollout steps, two epochs and 32 minibatches:
32,768 global transitions and 64 Adam steps per update. Their two qualification
updates count toward the 5,000-update budget; production starts at u105002.

The control keeps all six added costs zero. The treatment increases costs
linearly from zero at u105000 to these targets at u107500, then holds them
through u110000:

| Cost | First-stage target |
| --- | ---: |
| Fresh-excavation lateral cost | 0.125 |
| Effective retained-work setup | 0.0025/setup |
| Straight-line transfer between retained work poses | 0.0025/metre |
| Wrapped base-heading change between retained work poses | 0.01/radian |
| Raw chassis travel and rotation | 0 |

This is the proposed 25% stage, not an automatic escalation toward stronger
costs. The ramp is 81.92M transitions and the hold another81.92M per arm.
The upper budget is327.68M new transitions across both arms; no further stage
or continuation is implicit.

Effective work includes excavation, dumping and loose-soil relifting. An
ineffective DO or navigation action creates no retained setup. Pose grouping
uses exact base x,y,heading; cabin swing does not create a setup. Lateral cost
applies only to fresh target excavation, never dumping or relifting. The route
term is a straight-line lower bound and excludes initial approach/final egress.

On the u105000 successful episodes, the undiscounted stage costs total about
0.215 per foundation (p90 0.301, p99 0.407) and0.152 per trench (p90 0.239,
p99 0.346), against success bonus6. These are upper estimates: the evaluator's
heading total includes initial approach while the reward excludes it, an
overestimate of at most0.03142 per episode. No navigation-stack route cost is
claimed.

## Observation and native resume

Retained travel depends on the previous effective work pose. Both arms expose
normalized previous x,y, heading sine/cosine and validity to actor and critic.
Reset context is zero. Two zero-initialized5x704 projections preserve initial
policy/value outputs and every existing model/Adam leaf and clock; new Adam
slots start at zero. Both arms can learn from this context. Subsequent updates
need not match the old input architecture because the new gradients also
participate in global gradient clipping.

The saved six-field ramp restores its original u105000 origin. Evaluation
reads effective costs from the saved environment; checkpoint configuration
records the targets. Historical three-field ramps retain their existing
schema. The training flags are explicit and architecture compatibility rejects
unrequested context changes.

Freeze the same environment runtime used for the u105000 parent panel, plus
the observation-only addition. Do not include the separate September21
float32 metadata, map-edge-clearance or new trench-boundary geometry changes
being evaluated on Euler. They need their own runtime comparison.

## Decision at u107500 and u110000

Run the same608-case fixed panel at the ramp endpoint and after the hold.
Stop a production arm on any integrity failure, foundations below377/384,
trenches below209/224, roads below30/32, or net loss above two in any condition,
all relative to u105000. Report gross gained/lost cases as well as net counts.
A policy-retention stop in one arm does not suppress the other's comparison;
a qualification failure prevents production until repaired.

Compare behavior on the parent/control/treatment common-success intersection.
A repeatable roughly5% reduction in retained distance or increase in unique
area per productive setup is meaningful only with retained completion and no
material continuity decline (review adjacency losses above two percentage
points). Show productive setups separately from all retained work setups.
Lower raw travel alone is not the deployment objective. Completion already
churns without added costs, so a treatment loss must be interpreted against
the matched control. There is no automatic promotion or coefficient increase.

Local artifacts: `.artifacts/terra_efficiency_p25_20260921/`.
CSCS artifacts: `/ritom/scratch/cscs/lterenzi/terra-training/runs/terra-efficiency-p25-20260921/`.
Runtime qualification and scheduler status are recorded in the experiment ledger.

## Submission and validation

CSCS4729577 submitted September21 at17:19UTC and checked17:20UTC:
PENDING/Resources, native cluster startup unverified. Local CPU/migration
checks and the1x32 RTX4090 PPO smoke pass. The smoke found and verified the
fix for missing retained-cost logging keys. Four-GPU preflight, exact parent
output parity and both native update/resume qualifications execute inside the
allocation before production. Recommended check18:15UTC; no monitor scheduled.
