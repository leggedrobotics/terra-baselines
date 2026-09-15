# Mixed foundations and trenches with separate teachers

## Decision and scope

Lorenzo requested one additional CSCS node for a generalist with teacher
influence. Start a fresh student, with fresh Adam and schedule clocks, on easy
foundations and all fifteen existing trench training conditions. Keep the
foundation initialization comparison on CSCS4665916 unchanged. This allocation
is a mixed-task learning screen; broader foundation generalization remains a
later stage. No automatic extra allocation or behavior-cost increase is enabled.

The foundation teacher is strong on its easy validation distribution, but a
new complete replay under current Terra returned **0/384 harder foundations**
and 0/224 trenches. Guiding the full mixed dataset with that policy would give
unsupported instructions on most tasks. The existing recovered trench teacher
does transfer well to the current trench environment.

| Teacher | Checkpoint | Current fixed evaluation | Native digging input |
| --- | --- | --- | --- |
| Easy foundation control | foundation-b_control_cscs-s20260907, u15000 | 62/64 easy foundations; 0/384 broader foundations | Executable fresh volume |
| Recovered trench specialist | trench_recovery-14055463, u86000 | 192/224 trenches, including24/32 roads; 0/384 foundations | Legacy admitted cell counts |

All reported panels are complete greedy450-step replays under current Terra
`46738cde28e455da7c466fc0a2cb64f677d86401`, with zero transition-integrity
failures. These are development/validation results. The teacher file hashes,
reports, bank builder and launch receipts live in
`/home/lorenzo/moleworks/.artifacts/terra_generalist_teachers_20260915/`.

## Student and teacher contract

The student uses current environment physics and executable digging observations.
Fresh trench alignment, junction admission, soil-free chassis occupancy and
dumping rules come from the pinned Terra source. No Terra code changes are part
of this campaign. Additional lateral digging, base travel and base turning
costs are zero; existing reward-v2 progress, completion and step terms remain.

The teacher sees the student's current state. The pre-action episode family
selects its foundation or trench policy distribution; reset episodes cannot
change the label of the preceding action. Each teacher uses its own stored
network and observation preprocessing. For the legacy trench input, recompute
the admitted-dig vector using Terra's compiled native wrapper before the
teacher's scaling and optional-input ordering. The teacher-only family and
legacy features travel through the same PPO shuffle and never enter the
student's model input.

Both teachers are frozen. Add forward KL from the selected teacher to the
student's normal PPO objective; no teacher value target is used. Cosine weight
starts at1 and reaches0 after20000updates. At that point PPO and constant0.02
entropy continue without teacher guidance. Native continuation preserves the
student weights, Adam state, global update and schedule. It reinitializes live
environments, RNG and history, as the existing Terra continuation does; it is
not a bit-exact interrupted rollout.

Saved training configuration and initialization receipts bind both actual
teacher file hashes and the resolved family IDs. Byte changes, swapped teacher
roles or single/dual mode drift fail native continuation. Paths can relocate
only when the teacher contents match. Evaluation explicitly clears both
teacher flags; inference needs only the student.

## Training distribution and scale

The pooled bank retains256easy foundation training maps and1440existing trench
map identities. Repeat each foundation map six times, retaining1536foundation
and1440trench reset slots:2976total. This yields51.61%/48.39% reset sampling
probabilities. It does **not** guarantee the same PPO transition proportions;
episode lengths change the active-state mixture. Log per-task selected counts
and KL to measure actual exposure.

The bank has1696map identities and1693unique reset-array scenarios: three
scenario pairs already occur in the trench source bank. Preserve them and their
provenance. All arrays and metadata are copied unchanged. Current Terra's full
loader, finite R2 distances, file hashes and disjoint map/source/scenario checks
against eleven held-out manifests pass. No held-out arrays are staged for
training. Distance sidecar SHA:
`8d13b77b38598979129b2be88a1e32188f22352c51f9d5a00679a7446ab785d9`.

One d130 normal allocation uses four GH200 GPUs for24hours. Each GPU has256
environments; rollout32 gives32768global transitions/update. Two PPO epochs and
32minibatches give64Adam updates per rollout,1024global samples/minibatch.
Use global minibatch advantage normalization, constant LR3e-4, constant
entropy0.02 and the existing2,311,701-parameter spatial residual model. KL fades
over655.36M total transitions; per-family exposure is measured separately.

This four-GPU mixed run is not a matched causal comparison against either
two-GPU foundation arm. Task distribution and global batch differ. Compare
held-out competence and behavior, reporting both updates and transitions.

Absolute target500000 safely exceeds one allocation; checkpoint every500.
Resume only from this run's native checkpoint. The local diagnostic uses1GPU
and32environments; the allocation repeats finite startup qualification at the
actual4x256shape before entering the long phase. The two teacher forwards add
work while KL is active. The existing conditional skips them after KL expires;
the legacy feature still incurs rollout work. Measure throughput here rather
than borrowing the single-teacher rate.

## Acceptance and evaluation

Before submission, require focused CPU routing/gradient/observation/resume
checks, CUDA convolution backward, finite actual u1/u2 checkpoints, both tasks
selected, actual fresh initialization, and native u2-to-u3 continuation. On
CSCS also require four GH200 devices and NCCL all-reduce. Independent review
binds the staged source, launch files, teacher/bank inputs and local evidence.
Scheduler RUNNING alone is not a training-health or learning-quality result.

Local qualification completed September15at10:47CEST:43CPU tests and15subtests
passed, with two existing skips. The RTX4090 CUDA backward gate passed; actual
u1/u2/FINAL checkpoints and native u3 pass finite model/Adam/loss and zero
integrity checks. Actual per-update routing counted448foundation and576trench
transitions in this small diagnostic. The native initialization restored u2
and Adam128 exactly; final u3 has Adam192. Diagnostic u2 throughput was1385.55
transitions/s. Fresh and native processes each paid roughly four minutes of
compilation; this does not establish a cross-process cache hit or production
throughput. All receipts are retained under the campaign's `local/` directory.

Evaluate checkpoints2500,5000,10000,20000 and the last complete saved checkpoint
as available. Use the fixed64easy-foundation validation panel and full608
development panel, reporting its224trench cases and road subset separately.
Its384harder foundations are an out-of-distribution diagnostic for this stage.
No sealed-policy selection is permitted. Keep450steps, greedy actions, fixed
reset keys and exact visible-dump completion unchanged.

Report completion plus excavation fraction, terminal/loaded/off-zone soil,
stall duration, entropy, PPO KL/clipping and separate teacher KL/exposure.
For behavior, retain unique excavated area per productive setup, productive
poses, individual dig area, retained setup travel, adjacent workspace fraction,
pose revisits and lateral fresh digging. Raw Terra navigation actions are
secondary: deployment replans navigation between retained work poses, while
workspace continuity remains valuable. Smaller marginal digs can coexist with
larger unique productive work per base pose because digging cones overlap.

Do not add efficiency penalties until repeated fixed-panel completion is
strong. Do not promote the foundation distribution solely on online reward,
entropy or easy-map completion. The next scope decision needs stable trench
competence alongside retained easy-foundation skill and an explicit curriculum
for the harder foundation conditions.
