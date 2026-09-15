# Generalist with broad foundation and trench teachers

## Corrected teacher selection

The mixed run should cover all current foundation and trench conditions.
The first candidate foundation teacher was an easy-map specialist. Its 0/384
broad replay did not describe the mature V8 policies. Lorenzo correctly
recalled their broad competence. Pending job 4670716 was held before starting
while those checkpoints were recovered and qualified.

Completed historical promotion results, greedy actions and a 450-step horizon:

| Policy | Foundations | Trenches | Overall |
| --- | ---: | ---: | ---: |
| Feed-forward u86000 | 341/384 | 329/336 | 670/720 |
| GRU u40000 | 343/384 | 334/336 | 677/720 |
| GRU u44000 | 341/384 | 333/336 | 674/720 |

These checkpoints trained on 4,512 maps: 2,400 foundations and 2,112 trenches,
across 25 foundation and 22 trench conditions. The 720 evaluation maps are a
separate promotion panel. Raw reports and actual checkpoint hashes have been
checked under `.artifacts/terra_v8_gru_benchmark_20260820/`.

Use feed-forward u86000 as a temporary broad foundation prior. Its historical
foundation result is close to the GRU result, and its feed-forward interface
supports guidance on shuffled student states without reconstructing recurrent
teacher history. Its checkpoint SHA is
`2fe5d23c86cc7702b188d33ca1ca9a42066a9a2515150e8795f8c640bbbeb4af`.
The trench teacher remains the recovered u86000 specialist,
SHA `de39133d70bb60e716a3ff72966d8efec931874f694eeaf81e402750f7e8d4bf`,
which completed 192/224 current development trenches, including 24/32 roads.

Historical broad results used Terra `25f855db`; the intended runtime is
`46738cde28e455da7c466fc0a2cb64f677d86401`. Therefore qualification replays the
unchanged feed-forward policy on the current 608-map development panel,
explicitly enforcing the current trench gate. It preserves the native policy
inputs, including reset context and carry credit. The 608 development and 720
promotion results are not a paired comparison of environment revisions.

## Full training distribution and native observations

Use the complete finite-metadata training pool: **3,840 original maps**, with
2,400 foundations and 1,440 trenches. Each of the 40 conditions has 96 maps.
Uniform slot sampling gives 62.5% foundation and 37.5% trench resets; actual PPO
transition exposure is measured separately because episode lengths differ.
No easy-map repetition, new map generation or partial-reset curriculum is used.

This includes all 25 foundation conditions and all 15 trench conditions with
persisted finite-section metadata. The seven historical V7 trench conditions
still lack that metadata and cannot enter the current alignment-gated loader.
This is the established current full bank, not a claim that all 47 historical
conditions now pass the new contract. No conditions are removed based on
policy performance. The current loader passes all 3,840 slots; training map,
source and scenario identities are disjoint from all nine broad held-out
manifests. Held-out arrays are not staged for training.

The foundation teacher uses nine local maps, carry credit in `agent_states`,
and `[Q_reset, H_reset/V0]`. It does not use the newer admissible, executable,
relocation-distance or trench-alignment inputs. In particular, full-start reset
context is not generally a zero vector: `H_reset/V0` includes initial material
work. Terra exports these latched quantities regardless of the student's
feature selectors. The teacher consumes its own native preprocessing and
network. The trench teacher retains its legacy admitted-dig vector, reconstructed
before preprocessing from the same pre-action state.

Only a teacher's matching task rows guide the student. The rollout stores
pre-action family IDs through both PPO shuffles. Teacher parameters and logits
remain frozen; the student keeps its current executable digging observations.
Native continuation binds both teacher hashes and family roles.

## Training and evidence gates

The student starts from random parameters, fresh Adam and zero update clocks.
Use one CSCS node with four GH200 GPUs, 256 environments per GPU, rollout 32,
two PPO epochs and 32 minibatches. This is 32,768 transitions per update and
64 Adam steps. Learning rate is 3e-4, entropy coefficient 0.02, and advantage
normalization uses the global minibatch. Keep the current spatial residual
architecture. This run is a capability experiment, not a matched initialization
comparison against the separate two-GPU foundation arms.

Teacher-to-student policy KL starts at 1 and follows the already selected
cosine decay to zero by update 20,000 (655.36 million transitions). No value
distillation or added lateral/base-travel/base-turn costs are used. Existing
reward-v2 task progress, completion and step rewards remain. Efficiency costs
require repeated strong fixed-panel completion before a later stage.

Checkpoint every 500 updates, with absolute target 500,000 beyond one allocation.
The first segment requests 16 hours to fit before CSCS maintenance on September
16, 07:00–19:00 CEST. The nominal continuation segment is 24 hours; no further
allocation or continuation is automatic.
The first allocation repeats four-GPU CUDA, convolution-backward and NCCL
checks, then validates finite u1/u2 checkpoints at the full batch before native
continuation. Before submission, require local native teacher-logit parity,
CPU routing and gradient checks, actual CUDA u1/u2 and native u3, bank integrity,
and independent review of the pinned source, inputs and evidence. The old held
job must not run alongside its replacement.

Primary evaluation is the full 608-map development panel: report 384 foundations
and 224 trenches separately, plus condition and road subsets. Compare continuous
excavation/material progress, completion, productive base poses, unique area
per setup, individual dig area, adjacent workspaces, revisits and travel between
retained work poses. Report successful-episode efficiency separately to avoid
confusing early stalls with efficient work. Raw navigation action counts are
secondary because deployment replans navigation, while workspace continuity
remains relevant. No sealed results guide this selection.

Campaign artifacts, qualification reports and final job receipts:
`/home/lorenzo/moleworks/.artifacts/terra_generalist_broad_teachers_20260915/`.
The initial narrow campaign is preserved in
[its historical design](GENERALIST_TASK_TEACHERS_20260915.md).

## Current-rule qualification and transfer investigation

All three frozen FF u86000 replays are complete on the same 608 development
maps, greedy horizon 450 and seed 20260724, with zero integrity failures:

| Runtime and observation | Foundations | Trenches | Mean foundation excavation |
| --- | ---: | ---: | ---: |
| Historical runtime and native observations | 332/384 | 218/224 | 94.0% |
| Current runtime and native observations | 140/384 | 1/224 | 65.5% |
| Current physics, historical traversability input | 113/384 | 4/224 | 54.2% |

The current-runtime transfer regression is real on the same panel. Restoring
the old observation worsens foundation completion by 27 cases: 37 previously
failed cases become successes, but 64 successes fail. That ablation keeps all
current collision, soil, digging, reward and initial-state generation rules.
It jointly restores the historical footprint transpose and material overwrite
within one policy input; it cannot separate those two observation mechanisms.
The prototype is archived and excluded from the selected training source.

A CPU initial-state audit found that matching map/reset seeds changes the
complete initial Agent on 41/608 cases, including 24/384 foundations. Therefore
old-versus-new runtime is not automatically an identical-start comparison.
At identical old Agent states, **only one of the 23 native input arrays differs**:
the global traversability channel, on all 608 cases. All nine local inputs,
carry credit, reset context and the other global inputs match. Seven old
foundation starts are invalid under the current physical footprint; preserve
and report these rather than silently repairing them. The observation-only
ablation keeps current initial-state generation and all physical state intact.

Among the 360 foundation cases whose complete initial Agent is unchanged, the
historical/current/observation-ablation success counts are 312/132/106. Changed
starts therefore do not explain the gap. Source packages also differ; these
results do not isolate one physical rule as the cause. The footprint, soil-free
chassis and movement repairs affect foundations as well as trenches.

Proceed with native current-runtime observations, the broad FF foundation
prior and the updated trench specialist. The FF teacher's 140/384 completion
does not qualify it as a strong current-physics expert or a deployment policy.
Its role is temporary guidance from a policy trained on 2,400 foundation maps;
fresh PPO must adapt to the changed rules as KL fades to zero. Keep all added
behavior costs off and evaluate the complete broad panel at the planned
milestones. Do not use historical 670/720 as a current-runtime success claim.

The 28 CPU tests and 10 subtests validate routing, gradients and native current
teacher preprocessing, including exact values and logits from the actual broad
checkpoint. Corrected CUDA u1/u2 and native u3 pass with finite student, teacher,
optimizer and loss state, Adam steps 64/128/192 and zero transition integrity
failures. This is a 1x32 diagnostic; four-GPU production-size qualification
remains mandatory in the allocation.

Completion regresses in all24 evaluated foundation conditions. The largest
drops include slab apron-d16 (13/16 to0/16), adjacent bearing walls (14/16
to1/16) and adjacent slabs (13/16 to0/16). Preserve every condition for the
student evaluation; the teacher is not selected by omitting difficult maps.

## Submission and current limits

CSCS job4672272 was submitted at14:08:09CEST on September15. At14:09:04 Slurm
confirms PENDING/Priority, one node/fourGPUs,16hours and lterenzi/d130, with no
estimated start or runtime proof. Narrow job4670716 was canceled before starting;
the existing foundation comparison4665916 remains running.

Source baseline96fcd811e194050323a90018a676c476ebacceee and Terra46738cde are
pinned. Final independent review covers local CUDA/native-resume gates, both
teacher roles/hashes, all3840 bank slots, zero costs and the16-hour override.
All remote bytes pass. The serial verification timeout was repaired with an
8-thread hash helper that preserves every source/input/container check; it is
pinned in the launch manifest and executes before GPU preflight. No training
source or recipe changed in that repair, so the existing local smoke applies.

The next recommended manual check is14:30CEST. No automatic monitor, further
allocation or behavior-cost stage has been created. The frozen manifest retains
its preparation status; campaign job.json and scheduler receipts describe live
state. The first segment can fit before maintenance only if admitted by15:00.

Evidence: `qualification/{runtime_comparison.json,teacher_selection.json,summary.json,protocol.json,legacy_protocol.json,legacy_marker_protocol.json}`
and `qualification/initial_state_audit/comparison.json` in the campaign.
