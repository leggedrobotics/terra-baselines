# V8 recurrent actor pilot (2026-08-17)

## Goal

Implement and run one practical recurrent Terra policy that can represent
phase, repeated failures, and cleanup-cycle history. The active Codex goal is
to research the design, implement sequence-correct recurrent PPO, verify it,
and launch a bounded 4-GPU experiment. This file is the goal's durable contract.

This is a capability experiment. Lorenzo explicitly chose to skip a stateless
GRU control, so a gain cannot be attributed to recurrence alone.

## Design

Use the existing v6.1 same-timestep spatial encoder and feed-forward critic.
Replace only the actor head:

```
fused observation (704)
  -> Dense(160), ReLU          = actor_input
  -> GRU(64), float32          = gru_output          [v2: memory channel only]
  -> concat([gru_output, actor_input])               [v2 concat skip]
  -> Dense(48), ReLU
  -> logits(8)
```

(v1 fed `gru_output` alone into Dense(48); see Execution for why that failed.)

The encoder processes flattened `batch*time` once. Only the small actor head is
scanned. PPO stores the actor carry at the start of each 32-step rollout,
shuffles whole environment sequences with that carry, and resets carry after a
real episode termination. The critic bootstrap is feed-forward and cannot
advance actor memory. MCTS is unsupported because its search nodes do not carry
the GRU state.

The feed-forward path remains a Python-static specialization of the same code;
it retains its original parameter tree and flat-minibatch option. The recurrent
path rejects flat shuffling. For the pilot, `num_steps == num_minibatches == 32`
and 4 x 512 environments keep both the global PPO batch and the encoder's
per-forward batch size at 2,048 and 512/device respectively.

## Why this core

- GRU is the smallest conventional recurrent controller for the observed
  2--18-step Terra cycles and a 32-step PPO rollout.
- LSTM is a reasonable second arm if GRU memory saturates, but adds a second
  carry and an extra gate without evidence Terra needs longer memory.
- GTrXL and linear recurrent units are credible longer-memory alternatives but
  add substantially more implementation and optimization risk for the first
  test.
- ConvLSTM is a poor fit: Terra already has a strong spatial encoder, and
  scanning the large map trunk would waste most of the compute.
- Public Lux RL agents reinforce the spatial-ResNet and curriculum choices, but
  the strongest open examples are feed-forward; they do not identify a better
  recurrent cell for Terra.

Primary references: [Ni et al. 2022](https://proceedings.mlr.press/v162/ni22a.html),
[DRQN](https://arxiv.org/abs/1507.06527),
[GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html),
[Memory Gym](https://www.jmlr.org/papers/v26/24-0043.html), and
[LRU for RL](https://proceedings.mlr.press/v235/lu24h.html).
Relevant Lux implementations are
[Toad Brigade](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) and the
[Deimos Season 2 write-up](https://www.kaggle.com/competitions/lux-ai-season-2/writeups/deimos-10th-place-deimos-s-rl-approach).

## Experiment contract

- Fresh training on the same accepted V8 full-start bank and relay partial-reset
  bank as the active feed-forward run.
- Same reward-v2, Continuous Banded v3 sampler, partial-reset schedule, encoder,
  optimizer, global environments, transitions/update, and global minibatch.
- 4 RTX 4090 GPUs, 512 environments/device, 32 steps, 32 minibatches, 2 epochs.
- No action masking, stall-age scalar, teacher, auxiliary decoder, or warm start.
- The production process itself treats update 1 as the finite smoke, avoiding a
  second multi-hour compile. It checks finite loss, parameters, and optimizer
  state at update 1 and every 10 updates thereafter.

The fixed full-start panel is the scientific readout. Training return and
partial-start success are diagnostics. Also rerun the targeted recurrence panel
to distinguish solved cycles, saturated memory, and remaining planning failures.

## Minimal gates

1. Actor step loop and sequence replay agree, including a selective done reset.
2. One recurrent PPO optimizer update is finite.
3. One exact-shape 4-GPU update completes before the long run starts.

No compatibility modes or additional recurrent architectures are part of this
pilot.

## Execution

### v1 (pure GRU head) — killed 2026-08-17

- Slurm job: `10949597` (`gpuhe.120h`, 4 RTX 4090 GPUs; cancelled at ~u3,800)
- Baselines source: `e5e1c3c50da92636f1be0d8de421e914e34e848f`
- Terra runtime: `25f855db3d913fd638c4e56b1740437a2b7122ca`

Outcome: matched the feed-forward relay to ~u800 (completion ~0.30), then
plateaued flat for 3,000+ updates while the relay broke out via exact-success
bootstrap. Recurrent PPO machinery audited clean and telemetry healthy
(approx_kl/clip/entropy indistinguishable from the relay run); checkpoint
probe showed the update gate never left z~=0.5, so all current-observation
information reached the logits attenuated through the 64-d tanh blend.
Full diagnosis: `V8_RECURRENT_GRU_PLATEAU_DIAGNOSIS_20260817.md`.

Initial job `10949464` failed before model initialization because its committed
source archive lacked an ignored local JAX-check helper. Commit `e5e1c3c`
replaces that dependency with the direct in-job JAX device assertion used by
the replacement job; the failure contains no training or policy evidence.

### v2 (concat-skip head)

Design change, one variable versus v1: the post-GRU MLP consumes
`concat([gru_output(64), actor_input(160)])` instead of the GRU output alone,
so current-step features reach the logits without passing through the gated
state and the GRU only adds memory. +7,680 parameters (delta vs feed-forward
now 46,336). Rollout, replay, storage, and PPO are unchanged.

Gate-bias initialization was considered and deliberately NOT changed: the skip
already guarantees pass-through structurally, a negative update-gate bias
would shorten the memory the pilot exists to test, and keeping the cell at
Flax defaults makes the v1-to-v2 comparison a single-variable change. It stays
on the follow-up list if v2's memory channel remains unused.

Same contract as v1 otherwise: fresh initialization, same banks, sampler,
reward-v2, optimizer, 4 x 512 envs, 32 steps, 32 minibatches, 2 epochs,
seed 20260817, target update 100,000; campaign `terra_v8_relay_gru_v2`.
