# V8 10M staged curriculum screen

This launcher starts the paired Stage-A capability screen from a healthy,
same-distribution full-V8 compact teacher. Lorenzo explicitly waived formal
teacher mastery for this screen; checkpoint, architecture, sampler, optimizer,
dataset, and finite-state validation remain mandatory.

The two matched arms are:

- `G-V8-XATTN-REWARM-CONTROL`: 2,856,685 parameters;
- `G-V8-10M-XATTN-WARM`: 10,257,209 parameters, channels
  `(64,128,192,256)`.

Both start from the same provisional teacher, use a fresh optimizer, distill
from that frozen teacher, and train on the exact Stage-A capability sampler.
Before PPO, each arm records its logits and values against the teacher on all
720 full-V8 promotion map slots. This diagnostic uses deterministic exact-slot
keys; it is not a behavioral evaluation or a claim that the combined panel's
inherited episode seeds are valid frozen resets.

The declared campaign is:

```text
capability -> nearby -> full                         (maps)
dense_skill -> terminal_margin -> terminal_objective (reward)
```

Only the first dense capability stage is enabled in this revision. The reward
stage does not change until a later fixed full-bank completion gate passes.
Legacy `SPARSE` is never selected.

The current combined V8 main-panel episode seeds must be repaired before the
later full-map or reward-fading gates. This does not block Stage A: its separate
capability promotion and development panels already enumerate their slots
exactly.

Dry run:

```bash
SUBMIT=0 scripts/euler_v8_10m_v1/submit.sh smoke 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
```

Run matched update-1 smokes first. Only after both smokes pass may Stage A be
submitted:

The smokes use four RTX 3090s to avoid the saturated 4090 short queue; the
Stage-A screens remain pinned to four RTX 4090s. Both hardware types execute
the same CUDA, cuDNN, NCCL, finite-state, and checkpoint checks.

```bash
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh smoke 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh screen 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
```

If the smokes are still queued, the screen can be queued safely behind their
exact job IDs. Each screen has an `afterok` dependency on its matching arm and
revalidates the completed smoke receipt before touching the dataset or GPUs:

```bash
SMOKE_REVISION=SMOKE_SOURCE_SHA \
SMOKE_JOB_IDS=CONTROL_SMOKE_JOB,TREATMENT_SMOKE_JOB \
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh screen 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
```

The Stage-A screen targets 4,000 updates at 512 environments/device and
evaluates 1000/2000/3000/4000 on the frozen capability promotion and
development panels. Each arm receives a gate receipt; nearby maps remain
locked unless the latest two checkpoints reach 12/16 exact successes in both
capability conditions on both splits. This launcher never silently changes
reward stage.

The half-sized environment batch is intentional. A real update-1 compile at
1,024 environments/device required a 17.38 GB XLA temporary and did not fit the
24 GB card after other live buffers. Halving environments and minibatches while
doubling update-based schedules preserves total transitions, local minibatch
size, and optimizer-step budget for the matched comparison.
