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
720 exact full-V8 promotion resets.

The declared campaign is:

```text
capability -> nearby -> full                         (maps)
dense_skill -> terminal_margin -> terminal_objective (reward)
```

Only the first dense capability stage is enabled in this revision. The reward
stage does not change until a later fixed full-bank completion gate passes.
Legacy `SPARSE` is never selected.

Dry run:

```bash
SUBMIT=0 scripts/euler_v8_10m_v1/submit.sh smoke 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
```

Run matched update-1 smokes first. Only after both smokes pass may Stage A be
submitted:

```bash
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh smoke 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh screen 20260730 \
  /remote/path/to/teacher_update_008000.pkl \
  /remote/path/to/teacher/run_contract.env
```

The Stage-A screen targets 2,000 updates and evaluates 500/1000/1500/2000 on
the frozen capability promotion and development panels. Each arm receives a
gate receipt; nearby maps remain locked unless the latest two checkpoints reach
12/16 exact successes in both capability conditions on both splits. This
launcher never silently changes reward stage.
