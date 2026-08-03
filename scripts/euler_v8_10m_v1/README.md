# Deferred V8 10M scale screen

This launcher is prepared but must not be submitted until the compact
deep+xattn V8 policy produces a qualified
`terra_v8_dense_reward_gate_v1` receipt. The teacher gate also rechecks the
selected main development evaluation, both all-free capability controls, and
the checkpoint immediately inside Slurm.

The two matched arms are:

- `G-V8-XATTN-REWARM-CONTROL`: 2,856,685 parameters;
- `G-V8-10M-XATTN-WARM`: 10,257,209 parameters, channels
  `(64,128,192,256)`.

Both start from the same qualified teacher, use a fresh optimizer, distill from
that frozen teacher, and train directly on the exact full-V8 sampler. Nothing
else changes. Before PPO, each arm records its logits and values against the
teacher on all 720 exact promotion resets.

Dry run (no SSH or Slurm mutation):

```bash
SUBMIT=0 scripts/euler_v8_10m_v1/submit.sh \
  smoke 20260730 /remote/path/to/dense_reward_gate_receipt.json
```

Once the teacher receipt exists, run matched update-1 smokes first. Only then
may the true 24-hour screen be submitted:

```bash
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh \
  smoke 20260730 /remote/path/to/dense_reward_gate_receipt.json
SUBMIT=1 scripts/euler_v8_10m_v1/submit.sh \
  screen 20260730 /remote/path/to/dense_reward_gate_receipt.json
```

The screen has an intentionally high absolute target of update 20,000 and
checkpoints every 500 updates, so the 24-hour allocation normally ends it by
wall time. A dependent evaluator accepts only clean completion or timeout,
selects updates 500/1000/1500, every 2,000 thereafter, and the latest common
checkpoint, then evaluates both arms on main promotion/development and separate
all-free capability promotion/development. It emits aggregate, family, and
per-condition CSV/JSON/Markdown leaderboards. Any 120-hour continuation requires
a new recorded decision; this launcher never auto-continues or starts a reward
curriculum.
