# Easy-foundation reward screen, September 7, 2026

The user authorized stopping the Euler generalist, leaving the CSCS trench
specialist unchanged, and running up to eight single-GPU training processes on
easy foundations to tune lateral excavation and base-motion costs.

## Shared starting point and task

Euler job 13189428 was cancelled at 19:09:34 CEST after preserving the complete
u5000 checkpoint locally. Its 2,311,701 model parameters and optimizer state are
finite, with Adam step 320000. All arms start from that same checkpoint and
preserve model/Adam/update state. The parent has no adaptive sampler or partial
reset state. Production trench specialist 4617490 is left unchanged.

The new bank uses 64x64 maps, approximately 0.5714 m tiles, ordinary full resets,
unit target depth, square/rectangle/L foundations, no obstacles, and broad
accepted legal dumping outside the foundation. Completion, horizon 450, machine
geometry, foundation edge rules, and per-cell trench admission remain unchanged.
Training, validation and test contain 256, 64 and 64 distinct maps respectively.
Each foundation requires 64–216 cells. All 384 maps passed the exact loader and
distance, capacity, identity and geometry checks. The generated bank contains
the gallery and complete validation results.

All arms use one GPU, 512 environments, 32 rollout steps, 2 PPO epochs and 32
minibatches: 16,384 new transitions and 64 Adam steps per update. Keep the parent's
model/PPO/timing/entropy schedule and input dimensions. The parent's earlier
batch was 65,536 transitions/update; report added data as
`(next_update - 5000) * 16384`, not a recomputed historical total.

## Initial screen

| Arm | Runtime | Seed | Executable observation | Side cost | Travel cost /m | Turn cost /rad |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A legacy baseline | Euler | 20260907 | off | 0 | 0 | 0 |
| B observation control | Euler | 20260907 | on | 0 | 0 | 0 |
| C side only | Euler | 20260907 | on | 0.25 | 0 | 0 |
| D motion only | Euler | 20260907 | on | 0 | 0.005 | 0.02 |
| B runtime control | CSCS | 20260907 | on | 0 | 0 | 0 |
| E combined | CSCS | 20260907 | on | 0.25 | 0.005 | 0.02 |
| F doubled combined | CSCS | 20260907 | on | 0.5 | 0.01 | 0.04 |
| E combined repeat | CSCS | 20260908 | on | 0.25 | 0.005 | 0.02 |

Euler provides four separate one-GPU allocations. Daint's normal partition
allocates exclusive four-GPU nodes: one allocation runs four independent
one-GPU processes with separate run/checkpoint/W&B paths. It does not run one
four-GPU PPO agent. All primary reward contrasts have a zero-cost control on
the same runtime. The cross-runtime interaction and second-seed sensitivity
are exploratory; this is not a fully replicated factorial study.

## Execution and comparison

1. Validate and inspect the bank, and evaluate the frozen parent on validation
   maps with legacy and executable observations. Parent weakness changes the
   interpretation to learning the new task plus efficiency; it is not evidence
   for efficient behavior.
2. Run same-runtime CUDA/convolution preflight and a finite two-update smoke
   for each arm. Start the longer segment only after finite model/Adam/loss and
   transition checks pass. The smoke updates count towards the shared budget.
3. Give each screen a 24-hour allocation, save every 500 updates, and set an
   absolute target beyond one allocation. Compare matched added-update
   milestones +2000, +5000, +10000, +20000; keep wall time and GPU type separate
   from sample efficiency. Timeout with a finite checkpoint is continuable.
4. Evaluate argmax PPO with no MCTS on the same 64-map validation panel, 450-step
   initial-episode horizon and fixed reset seed. Choose on validation only;
   evaluate a selected control/treatment on untouched test maps afterwards.

Keep exact success and terminal progress visible overall and by shape. Compare
travel, productive base stances, area per stance, and lateral digging only on
paired maps solved by both policies. An initial candidate should show no
observed exact-success loss and a material travel/stance improvement (a 10%
screening target), with lower lateral score for a side-cost treatment. These
thresholds are practical screens, not significance tests. Report raw paired
counts/distributions; low travel by a failing or idle policy is not a win.

The easy bank isolates machine positioning/digging preferences. Coefficients
selected here still require testing on constrained dumping and the original
generalist distribution before being adopted there.
