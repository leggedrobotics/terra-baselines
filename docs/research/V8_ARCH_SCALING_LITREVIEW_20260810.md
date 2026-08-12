# Scaling neural architectures for on-policy (PPO) RL on grid/spatial-map tasks
### A literature review targeted at the Terra excavation benchmark (64×64 grid, 2.86M → 10M param question)

Date: 2026-08-10. All claims are tied to a named source with a link. Evidence provenance
(on-policy vs value-based/off-policy) is flagged throughout, because it is the single biggest
transfer risk in this literature.

---

## 0. Our case, restated in the units the literature uses

Terra setup under review:

| | 2.86M (incumbent) | 10M (transplant, did not win) |
|---|---|---|
| stage channels | (24, 48, 64, 96) | (64, 128, 192, 256) |
| blocks | (2,2,3,3) | (2,2,3,3) — unchanged |
| final feature map | 8×8×96 | 8×8×256 |
| readout A | flatten(6144) → Dense(192) | flatten(16384) → Dense(192) |
| readout B | 5 latent queries, qkv **96**, over 64 tokens of dim **96** | 5 latent queries, qkv **96**, over 64 tokens of dim **256** |
| encoder output interface | **Dense(160)** | **Dense(160)** — unchanged |
| policy head | (160, 48) | unchanged |
| critic head | (512, 256) | unchanged |
| LR / optimizer | 3e-4, Adam eps 1e-5 | **identical** |

My own parameter/FLOP accounting (reproduces both totals within a few %, so the reading of the
spec is right):

```
2.86M : encoder conv 0.94M | flatten→Dense(192) 1.18M  (≈46% of all params)
        encoder ≈258M MACs; stage-0 (full 64×64) ≈33% of encoder MACs
10M   : encoder conv 7.19M | flatten→Dense(192) 3.15M  (≈29% of all params)
        encoder ≈1944M MACs; stage-0 (full 64×64) ≈31% of encoder MACs
```

Three structural facts follow, and they are exactly the quantities the 2024–2026 scaling
literature says determine whether width helps:

1. **The encoder→head interface is 160-d in both models.** A 2.7× wider trunk is forced through
   an unchanged 160-d bottleneck. Nothing downstream of `Dense(160)` was scaled.
2. **The single largest tensor in both models is a pure compression matrix** (6144→192, then
   16384→192). This is the layer that Sokar & Castro (NeurIPS 2025) name "the bottleneck" and
   identify as *the* limiting factor for scaling pixel-based RL.
3. **The attention readout got relatively narrower under scaling.** Token dim went 96 → 256 while
   qkv stayed 96, so the cross-attention now compresses 2.7:1 per token where before it was 1:1.
   The one component of the readout that could have carried extra width was held fixed.

Also worth noting against benchmark norms: entropy coefficient 0.15→0.02 is 15×–2× the Procgen
PPO default of 0.01 ([Cobbe et al. 2020](https://arxiv.org/abs/1912.01588)); `vf_coef = 2.0` on a
shared trunk is 4× the common 0.5 default; 2 PPO epochs is at the low end (PPO on Procgen uses 3).

---

## 1. When does increasing CNN width/depth help in on-policy RL at fixed sample budget?

### 1.1 The strongest pro-scaling on-policy evidence: Procgen (PPO, IMPALA-CNN width)

[Cobbe et al., "Leveraging Procedural Generation to Benchmark RL" (ICML 2020)](https://arxiv.org/abs/1912.01588)
scaled IMPALA-CNN channels by k ∈ {1,2,4} (params ≈ k²) under PPO, 200M steps, γ=0.999,
entropy 0.01, clip 0.2, LR 5e-4. Findings: *"Larger architectures significantly improve both
sample efficiency and generalization"*, and the small Nature-CNN *"almost completely fails to
train."* Critically for us — **they did not hold the learning rate fixed**: when channels are
scaled by k they scale LR by **1/√k**.

*Applies to us?* Yes in kind (PPO, 64×64 spatial obs, discrete actions), but note two mismatches:
Procgen is a *generalization* benchmark with 200 levels, and the IMPALA-CNN they scaled has the
same flatten→dense pathology discussed in §3 — so "width helped" there was measured *despite* the
bottleneck, at only 4× width and 200M steps.

### 1.2 The scaling law that tells you whether 10M is even the right size

[Hilton, Tang & Schulman, "Scaling laws for single-agent RL" (arXiv 2301.13442)](https://arxiv.org/abs/2301.13442)
introduce **intrinsic performance** (min compute to reach a given return across a model family)
and fit `I^(−β) = (N_c/N)^α_N + (E_c/E)^α_E` over PPO/PPG on Procgen and PPO+LSTM on Dota 2.
Compute-optimal model size scales as `N ∝ C^{1/(1+α_N/α_E)}` with exponents ≈ **0.40–0.80**,
comparable to generative modeling (0.50–0.73)
([summary](https://www.alphaxiv.org/overview/2301.13442)). Horizon length changes the *coefficient*,
not the exponent.

*Applies to us?* Directly — this is PPO on Procgen and Dota. The practical reading: at ~1.3B env
steps we are running ~6.5× the Procgen-hard budget with a model only ~2× the IMPALA τ=2 size, so
we are plausibly on the *under*-parameterized side of the compute-optimal frontier. The law says
more parameters *should* pay — which makes the observed null result evidence about the
*architecture and protocol*, not about parameter count per se. But be careful: Hilton's family
scales the whole network coherently; our 10M scaled only the trunk.

### 1.3 The 2024–2026 architecture-scaling literature (mostly value-based — read critically)

- [Obando-Ceron et al., "Mixtures of Experts Unlock Parameter Scaling for Deep RL" (ICML 2024)](https://arxiv.org/abs/2402.08609)
  — Soft-MoE makes value networks parameter-scalable. **Value-based (Rainbow/DQN).**
- [Sokar et al., "Don't flatten, tokenize!" (ICLR 2025)](https://arxiv.org/abs/2410.01930)
  — the gain from SoftMoE is **tokenization of the encoder output, not the experts**; a single
  appropriately scaled expert reproduces it. Rainbow/DQN/DER on 20- and 60-game Atari, Procgen,
  5 seeds, 200M steps. **Explicit negative on-policy note: SoftMoE "failed to provide similar
  gains in actor-critic algorithms such as PPO … and SAC."** This is an important caveat for us.
- [SimBa (ICLR 2025)](https://arxiv.org/abs/2410.09754) — RSNorm + pre-LN residual MLP blocks +
  post-LN gives a simplicity bias that makes *parameter scaling monotone*; plain MLP SAC
  **degrades** as parameters grow toward 17M, SimBa improves. Includes an **on-policy PPO result
  on Craftax** (1024 parallel envs, 1B steps) where the only change is MLP → SimBa.
  [SimBaV2](https://arxiv.org/abs/2502.15280) replaces LayerNorm with hyperspherical
  normalization + distributional critic and scales critic width and UTD smoothly.
- [BBF (Schwarzer et al., ICML 2023)](https://arxiv.org/abs/2305.19452) — 4× IMPALA-CNN width is
  a *load-bearing* ingredient for human-level Atari 100K, but only alongside shrink-and-perturb
  resets, higher replay ratio, and annealed n-step. **Value-based, sample-efficient regime.**
- [DreamerV3](https://arxiv.org/abs/2301.04104) — 8M→200M sweep, monotone gains in both final
  performance *and* data efficiency. **Model-based**, and world-model losses are supervised, which
  is precisely the regime where scaling is known to be easy.
- [Rybkin, Nauman, Fu, Snell, Abbeel, Levine, Kumar, "Value-Based Deep RL Scales Predictably" (ICML 2025)](https://arxiv.org/abs/2502.04327)
  — predictable data/compute Pareto frontier controlled by UTD; validated on SAC, BRO, PQL.
  **Off-policy only, and the controlling knob (UTD) does not exist in on-policy PPO.** I would not
  transfer its quantitative conclusions to our setting; the qualitative one (fit the frontier from
  small runs before committing) does transfer.
- [Ma et al., "Network Sparsity Unlocks the Scaling Potential of Deep RL" (ICML 2025)](https://arxiv.org/abs/2506.17204)
  — one-shot random pruning before training beats dense scaling, with better parameter efficiency
  and more resistance to plasticity loss and gradient interference.
- [Survey: "Scaling DRL for Decision Making" (arXiv 2508.03194)](https://arxiv.org/html/2508.03194v1)
  — consensus is a **width-favoring trend** in model-free DRL; depth scaling is frequently
  detrimental outside specialized architectures.

### 1.4 Competition evidence (grid worlds, per-cell actions) — the closest task analogue

- **Lux S1, 1st place** — [Isaiah Pressman's writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021):
  fully-convolutional SE-ResNet, **24 residual blocks of 128-channel 5×5 convs, no normalization,
  ~20M params**, board padded to **32×32 and kept at full resolution throughout** (no striding),
  per-cell action heads emitting 32×32×N tensors, IMPALA + UPGO + TD(λ) with KL to a frozen
  teacher. Trained on a **dual-GPU personal machine**. Growth protocol: 8-block → 16-block →
  24-block, *each stage distilled from the previous smaller network as teacher*.
- **Lux S2** — top-5 was mostly rule-based C++/TypeScript; the strongest published RL entry is
  [FLG's 4th-place writeup](https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit)
  (48×48 map, imitation → RL → transfer, ResNet-like). *I could not fetch the body text — Kaggle
  writeups are JS-rendered and not retrievable by fetch — so treat the S2 details as unverified.*
- **Lux S3** — per a third-party Japanese summary of the winners' writeups
  ([kurupical, zenn.dev](https://zenn.dev/kurupical/articles/61dbeedf89a29d)), 1st place
  ("Flat Neurons") used ResNet + ConvLSTM + Transformer at **~200M params**, IMPALA, 3–4 days on
  8×H100, per-cell heads (movement head + position-specific targeting head), self-play with KL to
  historical versions; 2nd place ("Frog Parade") used a ResNet at **~300M params** with **PPO,
  γ≈0.9999**, 8 days on a single-workstation 3090+2070S. **Evidence grade C** — I could not verify
  these parameter counts against the primary Kaggle writeups
  ([1st](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/flat-neurons-1st-place-approach-by-flat-neurons),
  [2nd](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/frog-parade-frog-parade-s-solution)),
  and competition self-play is a fundamentally different (non-stationary, adversarial) objective.

**What the competition evidence actually supports for us:** not "params ≥ 10M is fine" (that is
confounded by self-play and per-cell supervision density), but three concrete design choices —
*full-resolution towers*, *per-cell / dense spatial outputs*, and *distillation-based capacity
growth in stages*.

### 1.5 Counter-evidence and the honest boundary

- The classic value-based counterevidence is [Kumar et al., "Implicit Under-Parameterization"
  (ICLR 2021)](https://arxiv.org/abs/2010.14498): with bootstrapping, *more gradient updates
  reduce* the rank of value features, so bigger nets do not buy expressivity. This is a
  **bootstrapping+regression pathology**; PPO's value head bootstraps through GAE, so a weakened
  form is plausible, but the strong form is off-policy.
- [Obando-Ceron et al., "Small batch deep RL" (NeurIPS 2023)](https://arxiv.org/abs/2310.03882)
  is a reminder that "bigger is better" intuitions from supervised learning routinely invert in
  RL. **Value-based with replay** — does not transfer to on-policy batch sizing.

---

## 2. Known failure mechanisms when naively widening RL nets — which are documented *on-policy*?

### 2.1 The encoder→dense bottleneck (**the mechanism that best matches our null result**)

[Sokar & Castro, "Mind the GAP! The Challenges of Scale in Pixel-based Deep RL" (NeurIPS 2025)](https://arxiv.org/abs/2505.17749):
*"we identify the connection between the output of the encoder … and the ensuing dense layers as
the main underlying factor limiting scaling capabilities; we denote this connection as the
bottleneck."* Bottleneck parameters are `H×W×C×dim(ψ)`; global average pooling reduces it to
`C×dim(ψ)`. Diagnostics: the fully-connected layer *"exhibits the highest percentage of
dormancy"* in scaled baselines; GAP variants show fewer dormant neurons and lower feature norms;
Grad-CAM shows the scaled baseline attends to background instead of task-relevant regions. They
also show prior scaling fixes (SoftMoE-1, pruning/RigL/static sparsity, tokenization) *implicitly*
target the same bottleneck.
**Provenance caveat: Rainbow / DER / SAC only — no on-policy experiments.** I verified this twice.

The independent on-policy replication is [Trumpp et al., "Impoola" (arXiv 2503.05546)](https://arxiv.org/abs/2503.05546),
which *is* PPO on Procgen. Their numbers are strikingly close to our geometry: at 64×64 input the
IMPALA-CNN ends at **64×8×8** and **83.76% of its 626,256 parameters live in the flatten→Linear
layer** (72.75% at τ=2). Replacing flatten with GAP gives **+17% IQM generalization with 35%
fewer parameters**, and Impoola at τ=4 is the best configuration overall while Impala's gains
flatten out past τ=2.

### 2.2 Plasticity loss / dormant neurons

- [Sokar et al., "The Dormant Neuron Phenomenon in Deep RL" (ICML 2023)](https://arxiv.org/abs/2302.12902)
  — dormant fraction grows during training and kills expressivity; ReDo recycles them.
  **Value-based.**
- [Lyle et al., capacity loss / understanding plasticity] and
  [Nikishin et al., primacy bias](https://proceedings.mlr.press/v162/nikishin22a.html) —
  three distinct mechanisms (dead ReLUs, weight-norm growth, feature-rank collapse) that do not
  co-occur. **Off-policy/replay-centric; primacy bias is specifically a replay-buffer artifact and
  does not apply to on-policy PPO with fresh data.**
- **The on-policy study that matters:**
  [Juliani & Ash, "A Study of Plasticity Loss in On-Policy Deep RL" (arXiv 2405.19153)](https://arxiv.org/abs/2405.19153).
  PPO only, on gridworlds (11×11×4), CoinRun/Jumper/Fruitbot (Procgen 64×64×3), Montezuma.
  Result: **plasticity loss is pervasive under domain shift in on-policy RL, and several
  off-policy remedies fail or actively hurt** — reset-final-layer fixed nothing; plasticity
  injection *underperformed the warm-start baseline in all three contexts*; ReDo underperformed
  the warm-start baseline in the expand condition. What worked was the **regenerative family**
  (soft shrink+perturb, L2-to-init regularization); LayerNorm fixed training plasticity but had
  inconsistent generalization effects. Weight magnitude was the strongest correlate of degradation.
  *They did not vary network width, so width×plasticity interaction is not established on-policy.*

### 2.3 Representation collapse specific to PPO

[Moalla, Miele, Pyatko, Grosse, Kaddar, "No Representation, No Trust" (NeurIPS 2024, arXiv 2405.00662)](https://arxiv.org/abs/2405.00662)
is the best PPO-native evidence for the whole §2 family. On Atari-5 + Gravitar and MuJoCo, 5 seeds:
feature-rank deterioration and capacity loss in 5/6 Atari and 7/8 MuJoCo tasks; rising dead
neurons in the penultimate layer; **pre-activation norm explosion precedes rank collapse**;
and — the PPO-specific part — once representations collapse, gradients across states become
colinear so the **clipping ratio no longer enforces a trust region** (measured excess ratios
diverge from the clip limit at collapse). Increasing PPO epochs accelerates the degradation. Fix:
Proximal Feature Optimization, an L2 penalty on the change in pre-activations between consecutive
policies; secondary fixes include Adam moment resets and faster β₂ decay.
**Shared vs separate trunk result is environment-dependent** — see §5.3.

### 2.4 Gradient pathologies at scale

[Creus Castanyer, Obando-Ceron, Li, Bacon, Berseth, Courville, Castro, "Stable Gradients for
Stable Learning at Scale in Deep RL" (NeurIPS 2025, arXiv 2506.15544)](https://arxiv.org/abs/2506.15544):
*"the combination of non-stationarity with gradient pathologies, due to suboptimal architectural
choices, underlie the challenges of scale."* They propose direct interventions that stabilize
gradient flow and report robustness across a range of depths and widths. The abstract does not
enumerate algorithms, so I cannot confirm on-policy coverage from the abstract alone — treat as
architectural guidance, grade B.

### 2.5 LR × width interaction, and µP

- **Everyone who successfully scaled a CNN in RL changed the LR.** Cobbe et al. scale LR by
  **1/√k** for k× channels ([Procgen](https://arxiv.org/abs/1912.01588)). Impoola reports a
  width-dependent rule `η|τ = η|τ=2 · (τ/2)` — note this is *increasing* in width, i.e. the
  opposite sign to Cobbe and to µP; my extraction may have inverted it, so **verify the direction
  in the paper before adopting**. Either way: **our 10M kept LR at exactly 3e-4. That is the one
  hyperparameter every scaling paper in this area says you must not hold fixed.**
- µP itself ([Yang & Hu, Tensor Programs V, arXiv 2203.03466](https://arxiv.org/abs/2203.03466);
  [microsoft/mup](https://github.com/microsoft/mup)) prescribes per-layer LR ∝ 1/fan-in for Adam
  so that features update at commensurate scale as width → ∞, enabling zero-shot LR transfer.
- **Was µP ever validated for PPO?** I found **no published, peer-reviewed µP-for-PPO study.**
  The only direct work is an OpenReview submission, *"μP for RL: Mitigating Feature
  Inconsistencies During Reinforcement Learning"* ([forum](https://openreview.net/forum?id=Wuy631kHwH),
  Oct 2025), which frames the open question exactly right — RL couples learning dynamics to a
  shifting data distribution, so µP's fixed-distribution guarantees are not automatic. I could not
  retrieve its content (OpenReview is behind a browser check). **Evidence grade C/D — treat µP in
  PPO as a plausible but unvalidated prior.**
- Adjacent caution: [Weight Decay may matter more than muP for LR transfer in practice
  (arXiv 2510.19093)](https://arxiv.org/abs/2510.19093).
- PPO-native LR pathology diagnostic: [Fernández-Hernández et al., "When Learning Rates Go Wrong:
  Early Structural Signals in PPO Actor-Critic" (arXiv 2603.09950)](https://arxiv.org/abs/2603.09950)
  — an activation-pattern indicator (OUI) measured at **10% of training already discriminates LR
  regimes** in discrete-control PPO, and *critic and actor prefer different OUI ranges*. Useful as
  a cheap early-abort signal for LR sweeps. They do not study width.

### 2.6 Entropy collapse at scale

There is no clean PPO-width×entropy study I could find. The general result is that PPO's entropy
falls monotonically without intervention and that the entropy coefficient is *highly* sensitive —
too small fails to prevent collapse, too large destabilizes
(surveyed in the RLVR entropy literature, e.g. [The Entropy Mechanism of RL for Reasoning LMs](https://openreview.net/forum?id=vXoksdcfqC),
[Arbitrary Entropy Policy Optimization, arXiv 2510.08141](https://arxiv.org/abs/2510.08141)).
**Relevance to us is indirect but real**: with a wider trunk, the same entropy coefficient
schedule (0.15→0.02) is applied to a policy head whose input distribution has changed scale — the
effective exploration pressure is not held constant by holding the coefficient constant. Grade C
for transfer; worth logging entropy curves in the 2.86M vs 10M comparison before theorizing.

---

## 3. Readout / bottleneck design for spatial policies

### 3.1 Flatten vs global pool vs tokenized/attention readout

This is the most active and most directly relevant question in the 2025–2026 literature, and
**four independent groups converged on the same diagnosis within one year**:

| Work | Readout fix | Setting | On-policy? |
|---|---|---|---|
| [Sokar et al. 2024, "Don't flatten, tokenize!"](https://arxiv.org/abs/2410.01930) | reshape encoder output to `[h·w, d]` (PerConv) or `[d, h·w]` (PerFeat) tokens, per-token projection, then sum/mean | Rainbow/DQN/DER, Atari 20+60, Procgen, 5 seeds | **No** — and explicitly reports SoftMoE gains *did not* materialize for PPO/SAC |
| [Trumpp et al. 2025, Impoola](https://arxiv.org/abs/2503.05546) | flatten → **global average pooling** | **PPO**, Procgen easy(25M/200 lvls) + hard(100M/1000 lvls), 5 seeds | **Yes** |
| [Sokar & Castro 2025, Mind the GAP](https://arxiv.org/abs/2505.17749) | **GAP**, framed as targeting the `H·W·C·dim(ψ)` bottleneck | Rainbow, DER, SAC; Atari/Atari100K/Procgen/DMC | **No** |
| [Kooi, Yang & François-Lavet 2025, Hadamax Encoding](https://arxiv.org/abs/2505.15345) (cited as concurrent by Mind the GAP) | max-pooling of Hadamard products of GELU-activated parallel hidden layers, replacing the flatten pathway | Atari, model-free; **+80% over vanilla PQN**, surpasses Rainbow-DQN, no hyperparameter changes | **No** (PQN/DQN) |

**Direct evidence that a narrow dense bottleneck after a wide trunk wastes width:** Impoola's
parameter table. At 64×64 input the IMPALA-CNN puts **83.76%** of its parameters in the
flatten→Linear layer, and Impoola at τ=4 (GAP, constant-size readout) beats every Impala width
while using 35% fewer parameters. Mind the GAP's dormancy analysis gives the mechanism: the
dense layer after flatten is where dormant neurons concentrate, and it gets worse with width.

**Our readout is this pathology, twice over:** `flatten(8×8×C) → Dense(192)` is 46% of the 2.86M
model and 29% of the 10M model, and the attention branch — the one component that *is* a
tokenized readout in the Sokar sense — kept qkv at 96 while token dim tripled.

### 3.2 The critical caveat for Terra: GAP destroys the position information we need

[Impoola](https://arxiv.org/abs/2503.05546) is explicit that GAP works by **reducing translation
sensitivity** (measured as L1 distance between action distributions under ±8 pixel shifts), and
that gains are *largest in games **without** agent-centered observations* (Bigfish, Chaser, Maze,
Miner, …) while Impala's positional advantage shows up in the agent-centered games (Coinrun,
Caveflyer, Ninja).

Terra sits awkwardly: the observation is a **global, non-agent-centered map** (so Impoola predicts
GAP should help) but the **action space is egocentric** (move/turn/dig relative to the agent), so
the policy must resolve *where the agent is relative to the work*. A naive GAP would erase exactly
that. This is consistent with our own prior internal finding that a global-pool bottleneck killed
spatial layout in a Terra encoder.

**Resolution supported by the literature:** don't choose GAP-vs-flatten; choose **tokenized
readout with agent conditioning**. That is Sokar et al.'s PerConv (keep 64 tokens, per-token
projection, then pool) plus the agent query we already have — i.e. our cross-attention branch,
*widened to match the trunk*, with the flatten branch reduced or removed. This preserves
per-cell identity while removing the `H·W·C×dim` compression matrix.

### 3.3 Per-cell decoders / U-Nets for spatial action spaces

- [Huang et al., Gym-µRTS](https://ieee-cog.org/2021/assets/papers/paper_174.pdf) established
  **GridNet**: an encoder–decoder (image-segmentation style) that emits a sub-action for every
  grid cell, combined with PPO and invalid-action masking. Later µRTS work adds SPP to the PPO
  critic ([Enhancing DRL for scale flexibility in RTS](https://ronaldo.games/assets/pdf/entcom-2024.pdf)).
- All Lux winners use per-cell heads: Pressman emits `32×32×N` logits per unit type
  ([writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)); Lux S3 1st place used a
  movement head plus a *position-specific* targeting head (grade C source).
- [Perceiver IO (Jaegle et al., ICLR 2022)](https://openreview.net/pdf?id=fILj7WpI-g) is the
  general form of our readout: a small latent array cross-attends to a large input, and *outputs*
  are produced by querying the latent with output-specific queries — including StarCraft II.

**Honest mapping to Terra:** Terra has **one** agent and **~9** actions, so a per-cell *policy*
decoder is not directly applicable. What *is* applicable is the same machinery as a **dense
auxiliary head**: predict a per-cell target (remaining work, reachability, next-dig-cell,
dumpability) from the 8×8 or a higher-resolution feature map. This converts a scalar-reward
credit-assignment problem into a dense supervised signal that can actually *consume* trunk width —
the classic argument of [UNREAL (Jaderberg et al., ICLR 2017)](https://arxiv.org/abs/1611.05397),
whose pixel-control head is literally a deconvolutional per-cell decoder and which lifted A3C
from 54% to 87% human-normalized score with ~10× speedup. Grade B for transfer (A3C, 3D nav), but
it is the mechanism most likely to make 10M *useful* rather than merely *present*.

### 3.4 Full-resolution towers vs aggressive striding to 8×8

- Pressman's Lux S1 winner keeps **32×32 full resolution through all 24 residual blocks** — no
  striding at all ([writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)). Same for the
  S2/S3 ResNet towers as far as can be told.
- [Trumpp, Çağatan, Akgün & Caccamo, "Higher Resolution, Better Generalization" (arXiv 2605.10546)](https://arxiv.org/abs/2605.10546)
  is the controlled study, and it is **PPO on Procgen-HD**: resolution 48×48 → 112×112, easy
  (200 levels/25M) and hard (1000 levels/100M). Impoola gains **~18%** going from (64,64,τ=2) to
  (112,112,τ=4) and beats standard Impala by **28%** at best config; train–test gap narrows from
  ~0.30 at 48×48 to 0.23 at 112×112. The key structural point: flattening makes parameters grow
  *quadratically* with resolution (Impala ~2M at 48×48 → ~8M at 112×112) while GAP holds them
  constant (~2M at all resolutions). Gains are largest where the task needs *"precise perception
  of small or distant entities"* and negligible where large static structures dominate.

**Mapping to Terra:** excavation cares about individual cells (dig targets, tile-level dumpability,
one-cell obstacles) — the "small entities" regime. Our aggressive stride to 8×8 means one final-
stage cell covers an 8×8 patch of the 64×64 map; the readout can no longer name a single tile.
Meanwhile my FLOP accounting shows **stage 0 at full 64×64 already consumes ~31–33% of encoder
MACs in both models**, so the 10M variant spent a large share of its extra compute at the
resolution where the receptive field is smallest and the least useful. That is a compute
*allocation* failure orthogonal to the parameter count.

### 3.5 FiLM / agent conditioning

[FiLM (Perez et al., AAAI 2018)](https://dl.acm.org/doi/abs/10.5555/3504035.3504518) — per-channel
affine modulation from a conditioning vector — is the standard cheap alternative to our
agent-query cross-attention, and is used with PPO in robot-learning policies. Also relevant:
egocentric cropping/rotation around the agent is repeatedly shown to improve gridworld
generalization ([Ye et al., "Rotation, Translation, and Cropping for Zero-Shot Generalization",
arXiv 2001.09908](https://arxiv.org/abs/2001.09908);
[Hill et al., arXiv 1910.00571](https://arxiv.org/abs/1910.00571)). For Terra this suggests a
**dual-stream readout**: a translation-invariant pooled/tokenized global stream (GAP-like, scales
with width) plus a small agent-centered crop stream that carries position — rather than one
flatten that must do both.

---

## 4. Warm-starting a bigger net from a smaller one in RL

### 4.1 The precedent that most exactly matches what we did — and it lost

[OpenAI, "Dota 2 with Large Scale Deep RL" (arXiv 1912.06680)](https://arxiv.org/abs/1912.06680)
performed ~20 "surgeries" over 10 months. For additive changes they *"ensure that the new policy
implement exactly the same mathematical function from observations to actions as the old policy"* —
i.e. **function-preserving**. The **LSTM 2048→4096 widening was the one change they could not make
function-preserving**, and their workaround was to *"randomize new weights significantly smaller"*,
with the scale *"set empirically by choosing the highest scale which did not noticeably decrease
the agent's TrueSkill."*

The decisive result is **Rerun**: a from-scratch run on the final architecture only, 2 months and
150±5 PFlop/s·days, which *"continued to improve beyond OpenAI Five's skill, and reached over 98%
winrate against the final version of OpenAI Five."* Their own summary of the surgeried model is
that it *"ultimately plateaued at a weaker skill level than the from-scratch model was able to
achieve."* Surgery was justified purely by *iteration economics* (40 months → 10 months), not by
final quality.

**This is the single most on-point precedent for our case**, and it says: a non-function-preserving
width transplant is expected to underperform a from-scratch run of the same architecture. We have
never run the 10M from scratch, so **the 10M architecture has not actually been tested.**

### 4.2 Function-preserving growth does work — when it is actually function-preserving

- [Chen, Goodfellow & Shlens, "Net2Net" (ICLR 2016)](https://arxiv.org/abs/1511.05641) —
  Net2WiderNet / Net2DeeperNet are exactly function-preserving, so *"the new, larger network
  immediately performs as well as the original"* instead of dipping. Supervised, but this is the
  reference definition.
- [Fehring, Lindauer & Eimer, "Growing with Experience: GrowNN" (arXiv 2506.11706)](https://arxiv.org/abs/2506.11706)
  applies it inside RL: start small, *"add layers without changing the encoded function"*, let
  later updates use the new capacity. **Grown networks beat static counterparts of the same final
  size by up to 48% (MiniHack Room) and 72% (Ant).** Note this is *depth* growth, and the paper's
  abstract does not name the algorithms; grade B.
- [Neuroplastic Expansion (arXiv 2410.07994)](https://arxiv.org/abs/2410.07994) grows from a small
  network to full size during training with dormant-neuron pruning and consolidation.
- Contrast with [plasticity injection (Nikishin et al., NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf),
  which is function-preserving by construction but which
  [Juliani & Ash](https://arxiv.org/abs/2405.19153) found **underperforms even the warm-start
  baseline in all three on-policy conditions**. Function preservation is necessary, not sufficient.

### 4.3 Distillation-based growth (what actually won the closest competitions)

- [Schmitt et al., "Kickstarting Deep RL" (arXiv 1803.03835)](https://arxiv.org/abs/1803.03835):
  auxiliary cross-entropy from teacher to student policy, with **per-teacher coefficients λ_k that
  are annealed by population-based training so the student is released to surpass the teacher.**
  On DMLab-30 the kickstarted agent matched from-scratch performance in ~10× fewer steps and
  **exceeded it by 42%**. *"places no constraints on the architecture of the teacher or student."*
  **The self-annealing λ is the load-bearing detail** — a fixed KL coefficient caps the student at
  the teacher.
- Pressman's Lux S1 chain is the same recipe at competition scale: reward-shaped 8-block →
  16-block → 24-block on sparse reward, *each larger network distilled from the previous smaller
  one* ([writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)). Note that the growth
  steps were **depth**, on a **full-resolution, per-cell** architecture — i.e. the capacity went
  somewhere it could be used.
- [Agarwal et al., "Reincarnating RL" (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)
  formalizes reusing prior computation (their instance: DQN@400M frames → new agent) and is the
  reference framing for "don't retrain from scratch every redesign". **Value-based.**

**Documented failure modes of non-function-preserving transplants**, collected:
(i) immediate function change → the policy that generated the on-policy data no longer matches the
network, so the first PPO updates are effectively off-policy with a broken trust region — the
mechanism [Moalla et al.](https://arxiv.org/abs/2405.00662) show also arises from representation
collapse; (ii) a permanently lower plateau, measured directly by
[OpenAI Five's Rerun](https://arxiv.org/abs/1912.06680); (iii) fixed-coefficient KL distillation
caps the student at the teacher unless annealed
([Kickstarting](https://arxiv.org/abs/1803.03835)); (iv) warm-started nets carry the plasticity
deficits of the parent — [Juliani & Ash](https://arxiv.org/abs/2405.19153) use "warm-start" as
their *degraded* baseline throughout.

---

## 5. Optimizer practice at scale for PPO

### 5.1 Learning rate: the thing we did not change

Covered in §2.5. Summary of the actionable disagreement: Cobbe **1/√k**, µP **1/width** per-layer
for Adam, Impoola a width-proportional rule (direction to be verified). Nobody holds LR fixed.
Any honest 10M attempt must sweep LR over at least {1/2.7, 1/√2.7, 1, ...}×3e-4.

### 5.2 Warmup / optimizer state after growth or transplant

[Moalla et al.](https://arxiv.org/abs/2405.00662) find **Adam moment resets between batches** and
**faster decay of Adam's second moment (β₂ ≈ 0.01 rather than 0.999)** are effective interventions
in PPO when the representation is under stress. This is the closest published analogue to
"warm-up / reset optimizer state after a transplant". Adam ε is known to be set much larger in RL
than in supervised learning and to change behavior materially
([37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/));
our 1e-5 is the standard RL-ish value, and I found **no evidence that ε is the binding constraint
at 10M** — I would not spend a run on it.

### 5.3 Value-loss weighting and shared trunks (**our `vf_coef = 2.0` is a flag**)

- [Cobbe et al., "Phasic Policy Gradient" (ICML 2021)](https://arxiv.org/abs/2009.04416) is
  motivated exactly by this: *"There is a risk that the optimization of one objective will
  interfere with the optimization of the other"*, and their Appendix B shows **separate networks
  outperform shared networks on Procgen**. The asymmetry they establish: *"value function
  optimization … tolerates a significantly higher level of sample reuse than policy optimization"*
  — PPG uses E_π=1, E_V=1, E_aux=6, whereas plain PPO's best is ~3 shared epochs. PPG then
  recovers feature sharing via an auxiliary value head distilled into the policy encoder.
- [Moalla et al.](https://arxiv.org/abs/2405.00662) refine this with a **reward-density
  dependence**: in dense-reward Atari (Phoenix, NameThisGame) trunk sharing acts as a regularizer
  and *helps*; in sparse-reward Gravitar it *catastrophically degrades* — rank collapses, feature
  norms and excess clip ratios explode. They confirm causality by masking 90% of rewards in
  Phoenix and reversing the sign of the effect.

**Mapping to Terra:** Terra's reward is comparatively dense per-episode but the terminal
completion signal is sparse and the horizon is 450 with γ=0.9984 (effective horizon ≈ 625). With
`vf_coef = 2.0`, the value gradient dominates the shared trunk by construction, and the trunk we
widened is the shared one. This is the second-most-likely explanation (after the readout
bottleneck) for why 3.5× more trunk parameters produced no policy improvement: **the extra
capacity was allocated to a trunk whose gradient signal is dominated by value regression.**

### 5.4 Critic vs actor capacity asymmetry

- Pro-asymmetry: [Honey, I Shrunk The Actor (arXiv 2102.11893)](https://arxiv.org/abs/2102.11893)
  reports 70–97% actor size reductions with preserved performance, arguing critic capacity is
  usually the binding constraint. [BRO (NeurIPS 2024, arXiv 2405.16158)](https://arxiv.org/abs/2405.16158)
  scales the *critic* aggressively (BroNet: Dense→LayerNorm→ReLU then LayerNorm'd residual blocks)
  under strong regularization and is the first model-free method to solve Dog/Humanoid. **Off-policy.**
- Anti-asymmetry: [Mastikhina, Sreenivas & Castro, "Optimistic critics can empower small actors"
  (arXiv 2506.01016)](https://arxiv.org/abs/2506.01016) find that shrinking the actor to 1–32% of
  baseline **degrades performance and overfits critics**, root-caused to poor data collection from
  value underestimation. **SAC/DrQ on DMC — off-policy, and the underestimation mechanism (min of
  twin critics) does not exist in PPO.**
- **Net on-policy read:** our critic head (512, 256) is already ~4–5× wider than the policy head
  (160, 48), which is the *right* direction per BRO/PPG. The problem is upstream: both heads sit
  behind the same 160-d encoder output. Widening the critic head further without widening the
  encoder interface will not help.

### 5.5 Batch size and gradient noise

- [McCandlish, Kaplan et al., "An Empirical Model of Large-Batch Training" (arXiv 1812.06162)](https://arxiv.org/abs/1812.06162)
  — the gradient noise scale predicts the largest useful batch size, and **RL agents sit at the
  extreme end** (millions of observations for Dota 2 vs tens of thousands for ImageNet).
- [OpenAI Five](https://arxiv.org/abs/1912.06680) measured this directly: batch size speeds up
  training *sublinearly* up to millions of observations; **data quality dominates** — 8 versions of
  staleness causes significant slowdowns and *"reusing the same data even 2-3 times can cause a
  factor of two slowdown."* They ran staleness 0–1 and sample reuse ≈ 1.0.
- [Beukman, Khetarpal, Zheng, Dabney, Foerster, Dennis & Lyle, "Preventing Learning Stagnation in
  PPO by Scaling to 1 Million Parallel Environments" (arXiv 2603.06009)](https://arxiv.org/abs/2603.06009)
  is the most recent on-policy word. They model PPO's outer loop as stochastic optimization:
  stagnation is **under-regularization of the outer loop** — updates too large relative to gradient
  noise cause thrashing rather than convergence. *"Larger batch sizes are significantly more robust
  to weaker regularisation."* Their scaling recipe: **keep minibatch size and learning rate fixed;
  increase the number of optimization steps proportionally to the number of parallel environments.**
  A single hyperparameter change (reverting IsaacGym minibatch 98,304 → 16,384) beat prior work on
  AllegroKuka / Shadow Hand / Allegro Hand; on Kinetix they sustain improvement to 10¹² transitions
  where standard PPO plateaus at ~10¹⁰. **Caveat: they use a 3-layer, width-256 MLP throughout and
  explicitly do not study model scaling.**

**Mapping to Terra:** batch 65k with 2 epochs and 32 minibatches ⇒ minibatch ≈ 2048 and 64 gradient
steps per update. That is a *small* minibatch relative to batch, which by Beukman et al. is the
robust direction. Our sample reuse of 2 is close to OpenAI Five's recommended ≈1. I see no strong
literature case that batch/epoch settings are what blocked the 10M — but note that a wider model
with an unchanged LR and unchanged minibatch effectively takes *larger* function-space steps, which
is exactly the under-regularized-outer-loop regime Beukman et al. describe.

---

## 6. Mechanism → evidence strength → applies to our case?

Evidence grades: **A** = controlled, multi-seed, *on-policy PPO*, spatial obs. **B** = strong but
value-based/off-policy/model-based, or on-policy but indirect. **C** = competition writeup,
secondary source, or unverified. **D** = open question / no study found.

| # | Mechanism | Key source(s) | Grade | Applies to our case? Why / why not |
|---|---|---|---|---|
| 1 | **Encoder→dense "bottleneck": flatten compression matrix caps usable width** | [Impoola (PPO/Procgen)](https://arxiv.org/abs/2503.05546); [Mind the GAP](https://arxiv.org/abs/2505.17749); [Don't flatten, tokenize!](https://arxiv.org/abs/2410.01930) | **A** (Impoola) + B (others, value-based) | **Yes, strongest match.** Our flatten→Dense(192) is 46%/29% of params and feeds an *unchanged* 160-d interface. Impoola's 64×64→8×8×64 geometry is nearly identical to ours. |
| 2 | **Fixed encoder-output width while trunk widens (no interface scaling)** | Same as #1; [Perceiver IO](https://openreview.net/pdf?id=fILj7WpI-g) | **A/B** | **Yes.** 160-d in both models; qkv held at 96 while token dim tripled. This alone can explain a null result. |
| 3 | **LR must be scaled with width** | [Cobbe/Procgen 1/√k](https://arxiv.org/abs/1912.01588); [Impoola](https://arxiv.org/abs/2503.05546); [µP](https://arxiv.org/abs/2203.03466) | **A** (Cobbe, Impoola) | **Yes.** LR was held at 3e-4 across a 2.7× channel scale. Direction of the correction is contested (1/√k vs µP 1/width vs Impoola's reported ∝τ) — must be swept, not assumed. |
| 4 | **Non-function-preserving transplant plateaus below from-scratch** | [OpenAI Five surgery + Rerun](https://arxiv.org/abs/1912.06680); [Net2Net](https://arxiv.org/abs/1511.05641) | **A** (OpenAI Five is on-policy PPO at scale) | **Yes, exactly.** Our 10M was a channel transplant + KL distillation, never from scratch. OpenAI Five's own conclusion: surgeried model plateaued lower; Rerun beat it 98:2. |
| 5 | **Fixed-λ KL distillation caps the student at the teacher** | [Kickstarting](https://arxiv.org/abs/1803.03835); [Pressman Lux S1](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) | **B/C** | **Likely yes.** Kickstarting's λ is *annealed by PBT* precisely to let the student surpass. If our KL coefficient was constant, "did not outperform the 2.86M teacher" is the predicted outcome. |
| 6 | **Value-loss domination of a shared trunk** | [PPG](https://arxiv.org/abs/2009.04416); [No Representation, No Trust](https://arxiv.org/abs/2405.00662) | **A** (both on-policy PPO) | **Yes, plausible.** `vf_coef=2.0` (4× the common default) on a shared trunk; Moalla et al. show sparse-reward + shared trunk ⇒ rank collapse. The widened part is the shared part. |
| 7 | **Aggressive striding to 8×8 loses tile-level detail; full-res towers win** | [Higher Resolution (PPO/Procgen-HD)](https://arxiv.org/abs/2605.10546); Lux winners keep 32×32 full-res | **A** (Trumpp 2026) + C (Lux) | **Yes.** Excavation is a "small entities" task (per-tile targets). Also a compute-allocation issue: ~31–33% of encoder MACs sit at 64×64 stage 0 in both models. |
| 8 | **Plasticity loss / dormant neurons under widening** | [Dormant neurons](https://arxiv.org/abs/2302.12902); [Mind the GAP dormancy in the dense layer](https://arxiv.org/abs/2505.17749); [on-policy study](https://arxiv.org/abs/2405.19153) | **B** | **Partly.** The on-policy study confirms plasticity loss *exists* in PPO but does **not** vary width, and finds several off-policy remedies (ReDo, plasticity injection, layer resets) *fail or hurt* on-policy. Do not import ReDo. |
| 9 | **Feature-rank collapse / implicit under-parameterization** | [Kumar et al.](https://arxiv.org/abs/2010.14498) (off-policy); [Moalla et al.](https://arxiv.org/abs/2405.00662) (PPO) | B (Kumar) / **A** (Moalla) | **Yes via Moalla**, not via Kumar. Kumar's mechanism is bootstrapped regression with replay; Moalla's PPO version (rank collapse + pre-activation norm explosion + broken trust region) is the transferable one. Worth *measuring* (effective rank of the 160-d output, dead-unit fraction) before more redesign. |
| 10 | **Primacy bias** | [Nikishin et al.](https://proceedings.mlr.press/v162/nikishin22a.html) | B | **No.** It is a replay-buffer oversampling artifact; on-policy PPO with fresh 65k-transition batches has no analogue. |
| 11 | **MoE unlocks parameter scaling** | [Obando-Ceron et al.](https://arxiv.org/abs/2402.08609); [Don't flatten, tokenize!](https://arxiv.org/abs/2410.01930) | B, with a **negative on-policy note** | **Weakly / probably not as MoE.** The follow-up paper attributes the gain to tokenization, not experts, and states SoftMoE *"failed to provide similar gains"* for PPO and SAC. Take the tokenization, skip the MoE. |
| 12 | **Simplicity-bias architecture (LayerNorm + residual + input norm) enables monotone scaling** | [SimBa](https://arxiv.org/abs/2410.09754); [SimBaV2](https://arxiv.org/abs/2502.15280); [BRO](https://arxiv.org/abs/2405.16158) | B (mostly SAC) with **one PPO/Craftax data point** | **Probably yes, cheaply.** Our SE-ResNet has residuals; the question is whether it has normalization. If it does not (like Pressman's Lux net), adding pre-LN + a post-block LN is the cheapest scaling enabler in the literature. |
| 13 | **Gradient pathologies at depth/width; gradient-clip interaction** | [Stable Gradients at Scale](https://arxiv.org/abs/2506.15544) | B | **Plausible, unquantified.** Global-norm clip 0.5 on a 3.5× larger model clips a *different* fraction of updates. Cheap to instrument (log pre-clip grad norm and clip-hit rate for both models). |
| 14 | **µP / per-module LR for RL** | [Tensor Programs V](https://arxiv.org/abs/2203.03466); [µP for RL (OpenReview submission)](https://openreview.net/forum?id=Wuy631kHwH) | **D** | **Unvalidated for PPO.** No peer-reviewed RL-µP result found. Use as a *prior on LR direction* to seed a sweep, not as a settled method. |
| 15 | **Entropy collapse interacting with scale** | RLVR entropy literature; PPO entropy sensitivity | C | **Unclear.** No PPO width×entropy study found. Our entropy schedule (0.15→0.02) is unusually strong vs Procgen's 0.01; hold it fixed across the size comparison and *measure*, don't theorize. |
| 16 | **Adam ε sensitivity** | [37 Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | C | **No evidence it binds here.** Do not spend a run on it. |
| 17 | **Batch/minibatch/outer-loop regularization at scale** | [Beukman et al. 2026](https://arxiv.org/abs/2603.06009); [McCandlish](https://arxiv.org/abs/1812.06162); [OpenAI Five](https://arxiv.org/abs/1912.06680) | **A** (Beukman, PPO) | **Second-order.** Our minibatch 2048 / reuse 2 is already in the recommended regime. But a wider net at the same LR takes larger *function-space* steps — the exact under-regularized-outer-loop failure they describe. |
| 18 | **Dense per-cell auxiliary supervision to make width usable** | [UNREAL](https://arxiv.org/abs/1611.05397); [GridNet/µRTS](https://ieee-cog.org/2021/assets/papers/paper_174.pdf); Lux per-cell heads | B/C | **Yes, high upside.** With one agent and 9 actions, our *only* dense signal into a 10M trunk would be auxiliary. This is the strongest "make the capacity earn its keep" lever. |

---

## 7. Ranked, literature-backed interventions to make ~10M actually beat 2.86M

Ranked by (expected effect × directness of evidence) ÷ cost. Interventions 1–3 are, in my reading,
prerequisites — running any of the later ones without them repeats the same experiment.

---

**#1 — Widen the *readout interface*, not just the trunk. Replace `flatten→Dense(192)` with a
tokenized/pooled readout whose output width scales with the trunk.**

Concretely: keep the 64 tokens of the final stage, project per token (PerConv-style), pool, and let
the encoder output be ~384–512-d instead of 160-d; scale attention qkv with token dim (96→256) and
scale the fusion `Dense(160)` accordingly. Kill or shrink the flatten branch.

*Precedent:* [Impoola](https://arxiv.org/abs/2503.05546) — **PPO on Procgen**, 83.76% of Impala's
64×64 parameters live in the flatten→Linear layer; GAP readout gives **+17% IQM generalization with
35% fewer parameters** and is the only variant that keeps improving to τ=4.
[Mind the GAP](https://arxiv.org/abs/2505.17749) — the encoder→dense connection is *"the main
underlying factor limiting scaling capabilities"*, and dormancy concentrates in that dense layer.
[Don't flatten, tokenize!](https://arxiv.org/abs/2410.01930) — tokenization, not experts, is the
active ingredient.
*Caveat:* pure GAP will destroy agent-position information that Terra's egocentric action space
needs — Impoola itself shows GAP's benefit comes from *reduced translation sensitivity* and is
largest in **non**-agent-centered games. Use tokenized+agent-query pooling, keep the coordinate
channels, and A/B against a small agent-centered-crop side stream.
*Cost:* one architecture change, from-scratch run.

---

**#2 — Run the 10M architecture from scratch on the full schedule. Stop evaluating transplants.**

*Precedent:* [OpenAI Five](https://arxiv.org/abs/1912.06680): the LSTM 2048→4096 widening is
explicitly the one surgery they could **not** make function-preserving; the accumulated-surgery
model *"plateaued at a weaker skill level than the from-scratch model"*, and the from-scratch
**Rerun reached >98% winrate against it**. [Net2Net](https://arxiv.org/abs/1511.05641) defines what
"function-preserving" means and why non-preserving growth costs a recovery period.
*Reading for us:* the 10M **architecture** has not been falsified — only one particular *transplant
protocol* has. Any claim of the form "10M does not help on Terra" is currently unsupported.
*Cost:* one full 1.3B-step run. This is the run that decides the question.

---

**#3 — Scale the learning rate with width and sweep the direction; add a short warmup after any
warm start.**

Candidate LRs for a 2.7× channel scale: 3e-4 ÷ 2.7 (µP-flavoured), 3e-4 ÷ √2.7 ≈ 1.8e-4 (Cobbe),
3e-4 (control). Use [OUI](https://arxiv.org/abs/2603.09950) or simple entropy/KL/clip-fraction
curves at 10% of the schedule to abort losers cheaply.

*Precedent:* [Cobbe et al.](https://arxiv.org/abs/1912.01588) scale LR by **1/√k** for k× IMPALA
channels — the very experiment that established "wider helps" in PPO did *not* hold LR fixed.
[Impoola](https://arxiv.org/abs/2503.05546) uses an explicit width-dependent rule.
[µP](https://arxiv.org/abs/2203.03466) prescribes per-layer LR ∝ 1/fan-in for Adam.
[Moalla et al.](https://arxiv.org/abs/2405.00662) support Adam moment resets / faster β₂ decay when
representations are stressed — the right analogue of "warm-up after transplant".
*Caveat:* **µP has never been validated for PPO** (the only work is an unreviewed OpenReview
submission), and the three rules disagree on magnitude and possibly sign. Sweep, don't assume.
*Cost:* 3 short runs (≈15–20% of schedule) before committing.

---

**#4 — Add a dense per-cell auxiliary decoder so the extra capacity has something to fit.**

E.g. a small deconv/segmentation head off the 16×16 or 32×32 stage predicting per-tile remaining
work, dig-target validity, reachability, or dumpability, with a modest loss weight.

*Precedent:* [UNREAL](https://arxiv.org/abs/1611.05397) — pixel-control is literally a per-cell
deconvolutional head; A3C 54% → 87% human-normalized, ~10× faster.
[GridNet / Gym-µRTS](https://ieee-cog.org/2021/assets/papers/paper_174.pdf) — encoder–decoder
per-cell outputs with PPO are the standard for grid RTS.
[Lux winners](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) all use per-cell heads; the S3
1st place added *next-step prediction outputs (enemy-unit probability, vision prediction) fed into
the action heads* (grade C).
*Why it matters specifically for us:* Terra has **one agent and 9 actions**. A 10M trunk receives
almost no bits per transition from a 9-way categorical. Auxiliary spatial targets are the only
literature-backed way to make a wide trunk trainable in a low-dimensional-action task.
*Cost:* one head + one loss term; can be combined with #1.

---

**#5 — Reduce striding / raise effective spatial resolution at the readout (e.g. read out at 16×16
and drop one stride, or reallocate stage-0 compute to later stages).**

*Precedent:* [Trumpp et al. 2026, "Higher Resolution, Better Generalization"](https://arxiv.org/abs/2605.10546)
— **PPO on Procgen-HD**, +18% from (64,64,τ=2)→(112,112,τ=4), +28% over Impala at best config,
train–test gap 0.30→0.23; gains concentrate where the task requires *"precise perception of small
or distant entities"*. Crucially, this only works once the readout is resolution-decoupled (GAP),
because flatten makes parameters grow quadratically with resolution.
Lux winners keep **32×32 full resolution through 24 residual blocks** with no striding at all.
*Our specific finding:* ~31–33% of encoder MACs already sit in the full-res 64×64 stage 0 in both
models, so the 10M variant paid a large FLOP premium at the least informative resolution. Consider
shifting channels away from stage 0 and toward a shallower stride (final 16×16).
*Order matters:* do this **after** #1, or the flatten cost explodes 4×.

---

**#6 — Decouple the value objective from the widened trunk: lower `vf_coef`, or move to
separate towers / PPG-style auxiliary distillation.**

*Precedent:* [PPG](https://arxiv.org/abs/2009.04416) — separate networks outperform shared on
Procgen (Appendix B); the value function *"tolerates a significantly higher level of sample reuse
than policy optimization"*, which is why PPG splits phases (E_π=1, E_V=1, E_aux=6).
[Moalla et al.](https://arxiv.org/abs/2405.00662) — trunk sharing helps in *dense*-reward tasks and
**catastrophically degrades in sparse-reward** ones (rank collapse, exploding feature norms and
excess clip ratios); verified causally by masking 90% of rewards.
*Our flag:* `vf_coef = 2.0` is 4× the common default, applied to the shared trunk we widened.
Cheapest test: `vf_coef ∈ {0.5, 2.0}` × {2.86M, 10M} — a 2×2 that directly tests whether value
domination is what neutralized the extra width.
*Cost:* hyperparameter only, or one architecture change for separate towers.

---

**#7 — Add normalization for scaling stability (pre-LN inside residual blocks + post-block LN),
i.e. adopt the SimBa/BroNet recipe in the trunk.**

*Precedent:* [SimBa](https://arxiv.org/abs/2410.09754) — plain MLP SAC **degrades** toward 17M
params while SimBa improves monotonically; includes an on-policy **PPO/Craftax** result where the
only change is the architecture. [SimBaV2](https://arxiv.org/abs/2502.15280) extends with
hyperspherical normalization. [BRO](https://arxiv.org/abs/2405.16158) — LayerNorm'd residual blocks
are what make critic scaling safe. Survey confirms the **width-favoring, normalization-dependent**
consensus ([2508.03194](https://arxiv.org/html/2508.03194v1)).
*Caveat:* mostly SAC/off-policy; and [Juliani & Ash](https://arxiv.org/abs/2405.19153) found
LayerNorm alone fixed *training* plasticity but had **inconsistent generalization effects** in
on-policy RL — pair it with a regenerative regularizer rather than expecting it to stand alone.
Note Pressman's winning Lux net used **no normalization at all** at 20M params, so this is not
strictly necessary at our scale.
*Cost:* small; fold into #2's from-scratch run.

---

**#8 — If you keep any distillation, fix the protocol: anneal the KL coefficient to zero and let
the student be released.**

*Precedent:* [Kickstarting](https://arxiv.org/abs/1803.03835) — per-teacher coefficients λ_k are
annealed by population-based training, and the kickstarted agent **surpassed from-scratch by 42%**
while matching it in ~10× fewer steps. [Pressman's Lux S1 chain](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)
grew 8→16→24 blocks with each smaller net as teacher, on a full-resolution per-cell architecture.
[Reincarnating RL](https://arxiv.org/abs/2206.01626) is the general framing.
*Reading for us:* "distilled from a 2.86M teacher and did not outperform it" is the *expected*
outcome of a non-annealed KL. Also note both successful chains grew **depth** on architectures with
no readout bottleneck — depth growth is closer to function-preserving than a channel transplant.
*Cost:* schedule change only; but I would rank this **below** #2 — the from-scratch run is the
cleaner experiment.

---

**#9 — Function-preserving growth (Net2WiderNet / GrowNN) instead of transplant, if a
from-scratch run is genuinely unaffordable.**

*Precedent:* [Net2Net](https://arxiv.org/abs/1511.05641) — the widened net *"immediately performs
as well as the original"*. [GrowNN (arXiv 2506.11706)](https://arxiv.org/abs/2506.11706) — RL-native
function-preserving *depth* growth, beating same-size static nets by up to **48% (MiniHack Room)**
and **72% (Ant)**. [Neuroplastic Expansion](https://arxiv.org/abs/2410.07994) for width.
*Caveat:* function preservation is necessary but not sufficient —
[plasticity injection](https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf)
is function-preserving by construction yet [Juliani & Ash](https://arxiv.org/abs/2405.19153) found
it *underperformed the warm-start baseline in all three on-policy conditions*.

---

**#10 — Instrument before iterating: effective rank of the 160-d encoder output, dormant/dead-unit
fraction per layer, pre-clip gradient norm and clip-hit rate, PPO clip fraction and KL, entropy —
for 2.86M vs 10M side by side.**

*Precedent:* [Moalla et al.](https://arxiv.org/abs/2405.00662) show pre-activation norm explosion
*precedes* rank collapse and that excess clip ratios diverge at collapse — i.e. these are leading
indicators, not post-hoc explanations. [Mind the GAP](https://arxiv.org/abs/2505.17749) localizes
dormancy to the post-flatten dense layer specifically. [OUI](https://arxiv.org/abs/2603.09950)
discriminates LR regimes at 10% of training.
*Cost:* near-zero, and it converts the next 3 runs from guesses into measurements.

---

**Explicitly deprioritized (with reasons):**
- **MoE.** [Don't flatten, tokenize!](https://arxiv.org/abs/2410.01930) shows the gain is
  tokenization, and states SoftMoE *"failed to provide similar gains in actor-critic algorithms
  such as PPO … and SAC."* Take #1, skip MoE.
- **ReDo / plasticity injection / layer resets.** [Juliani & Ash](https://arxiv.org/abs/2405.19153)
  found these fail or underperform the warm-start baseline **specifically on-policy**.
- **Adam ε tuning.** No evidence it binds; would consume a run.
- **Batch/minibatch resizing.** We are already in the regime [Beukman et al.](https://arxiv.org/abs/2603.06009)
  recommend; second-order relative to #1–#3.
- **Per-cell *policy* decoder.** Terra has one agent and 9 actions — the Lux/µRTS per-cell head has
  no direct analogue. Use it as an *auxiliary* head (#4), not as the policy.

---

## 8. Papers that directly contradict or complicate our scaling setup

1. **[Trumpp et al., Impoola (arXiv 2503.05546)](https://arxiv.org/abs/2503.05546) — the sharpest
   contradiction.** PPO on Procgen, same 64×64 input, same 8×8 final feature map. It reports that
   **83.76% of the parameters** sit in the flatten→Linear readout at exactly our geometry, that
   Impala's width scaling flattens out past τ=2, and that a *constant-size, resolution-decoupled*
   readout beats every wider flatten-based model with 35% fewer parameters. Our 10M is the Impala
   arm of their experiment.
2. **[Sokar & Castro, Mind the GAP! (NeurIPS 2025)](https://arxiv.org/abs/2505.17749)** — names the
   encoder→dense connection as *the* factor limiting scaling and shows dormancy concentrates there.
   Our 10M widened everything *except* that interface. (Value-based only — but it is the mechanism
   Impoola independently confirms under PPO.)
3. **[OpenAI, Dota 2 at scale (arXiv 1912.06680)](https://arxiv.org/abs/1912.06680)** — the
   non-function-preserving width surgery is the one they flag as problematic, and their from-scratch
   Rerun beat the surgeried model **>98:2**. Our 10M was produced by exactly the protocol their own
   ablation says loses.
4. **[Cobbe et al., Procgen (ICML 2020)](https://arxiv.org/abs/1912.01588)** — the canonical
   "wider helps in PPO" result **scales the learning rate by 1/√k**. We held LR fixed at 3e-4
   across a 2.7× channel scale, so we did not reproduce their protocol.
5. **[Sokar et al., Don't flatten, tokenize! (ICLR 2025)](https://arxiv.org/abs/2410.01930)** —
   complicates rather than contradicts: it is the best mechanistic account of why the readout is
   the bottleneck, *and* it explicitly reports that the SoftMoE remedy **did not transfer to PPO**.
   Anyone proposing MoE for our case has to answer this.
6. **[Cobbe et al., PPG (ICML 2021)](https://arxiv.org/abs/2009.04416) +
   [Moalla et al. (NeurIPS 2024)](https://arxiv.org/abs/2405.00662)** — both argue that a shared
   trunk under a heavy value objective is where PPO representations degrade. Our `vf_coef = 2.0`
   on a shared trunk is 4× the usual default, and the trunk is the thing we widened.
7. **[Juliani & Ash (arXiv 2405.19153)](https://arxiv.org/abs/2405.19153)** — contradicts the
   *remedies* half of the plasticity literature for on-policy settings: ReDo, plasticity injection
   and final-layer resets fail or underperform warm-start under PPO. Do not import off-policy fixes.
8. **[Beukman et al. (arXiv 2603.06009)](https://arxiv.org/abs/2603.06009)** — a caution in the
   other direction: they get PPO from a ~10¹⁰ plateau to 10¹² transitions using **a width-256,
   3-layer MLP** and pure outer-loop regularization. Their existence proof is that PPO plateaus are
   often an *optimization* problem, not a *capacity* problem. If our 2.86M plateau is outer-loop
   thrashing, no amount of width will fix it.

---

## 9. What I could not verify

- **Kaggle writeups (Lux S2 FLG, Lux S3 1st/2nd) are JS-rendered and not retrievable by fetch.**
  The Lux S3 parameter counts (200M / 300M) come from a third-party Japanese summary
  ([kurupical, zenn.dev](https://zenn.dev/kurupical/articles/61dbeedf89a29d)). Grade C — do not
  build an argument on those numbers without opening the Kaggle pages in a browser. The Lux **S1**
  details are primary and verified ([GitHub writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)).
- **µP-for-RL** is an unreviewed OpenReview submission behind a browser check
  ([forum](https://openreview.net/forum?id=Wuy631kHwH)); content not retrieved.
- **Impoola's learning-rate rule direction** (`η|τ = η|τ=2 · (τ/2)`, i.e. LR *increasing* with
  width) is the opposite of Cobbe's 1/√k and of µP. My extraction may have inverted it — check the
  paper before adopting.
- **Hilton et al.'s per-environment α_N / α_E** were not recoverable from accessible sources; only
  the aggregate compute-optimal exponent range (0.40–0.80) is cited here
  ([alphaXiv overview](https://www.alphaxiv.org/overview/2301.13442)).
- **Stable Gradients at Scale (arXiv 2506.15544)** — the abstract does not enumerate algorithms, so
  on-policy coverage is unconfirmed.

---

## 10. Source list

**On-policy / PPO scaling (grade A–B)**
- Cobbe, Hesse, Hilton, Schulman — *Leveraging Procedural Generation to Benchmark RL* (ICML 2020) — https://arxiv.org/abs/1912.01588
- Hilton, Tang, Schulman — *Scaling laws for single-agent RL* (2023) — https://arxiv.org/abs/2301.13442
- Trumpp, Schäfftlein, Theile, Caccamo — *Impoola* (2025) — https://arxiv.org/abs/2503.05546
- Trumpp, Çağatan, Akgün, Caccamo — *Higher Resolution, Better Generalization* (2026) — https://arxiv.org/abs/2605.10546
- Moalla et al. — *No Representation, No Trust* (NeurIPS 2024) — https://arxiv.org/abs/2405.00662
- Cobbe, Hilton, Klimov, Schulman — *Phasic Policy Gradient* (ICML 2021) — https://arxiv.org/abs/2009.04416
- Juliani, Ash — *A Study of Plasticity Loss in On-Policy Deep RL* (2024) — https://arxiv.org/abs/2405.19153
- Beukman, Khetarpal, Zheng, Dabney, Foerster, Dennis, Lyle — *Preventing Learning Stagnation in PPO by Scaling to 1M Parallel Envs* (2026) — https://arxiv.org/abs/2603.06009
- Andrychowicz et al. — *What Matters In On-Policy RL?* (2020) — https://arxiv.org/abs/2006.05990
- Huang et al. — *Gym-µRTS* (CoG 2021) — https://ieee-cog.org/2021/assets/papers/paper_174.pdf
- Fernández-Hernández et al. — *When Learning Rates Go Wrong: Early Structural Signals in PPO Actor-Critic* (2026) — https://arxiv.org/abs/2603.09950
- *The 37 Implementation Details of PPO* — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

**Large-scale RL systems / model surgery**
- OpenAI — *Dota 2 with Large Scale Deep RL* (2019) — https://arxiv.org/abs/1912.06680
- Vinyals et al. — *Grandmaster level in StarCraft II* (Nature 2019) — https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- Schmitt et al. — *Kickstarting Deep RL* (2018) — https://arxiv.org/abs/1803.03835
- Agarwal et al. — *Reincarnating RL* (NeurIPS 2022) — https://arxiv.org/abs/2206.01626
- Chen, Goodfellow, Shlens — *Net2Net* (ICLR 2016) — https://arxiv.org/abs/1511.05641
- Fehring, Lindauer, Eimer — *Growing with Experience (GrowNN)* (2025) — https://arxiv.org/abs/2506.11706
- *Neuroplastic Expansion in Deep RL* (2024) — https://arxiv.org/abs/2410.07994

**Architecture scaling (mostly value-based / off-policy — read with the caveat)**
- Sokar, Obando-Ceron, Courville, Larochelle, Castro — *Don't flatten, tokenize!* (ICLR 2025) — https://arxiv.org/abs/2410.01930
- Sokar, Castro — *Mind the GAP!* (NeurIPS 2025) — https://arxiv.org/abs/2505.17749
- Obando-Ceron et al. — *MoEs Unlock Parameter Scaling for Deep RL* (ICML 2024) — https://arxiv.org/abs/2402.08609
- Lee et al. — *SimBa* (ICLR 2025) — https://arxiv.org/abs/2410.09754 ; *SimBaV2* — https://arxiv.org/abs/2502.15280
- Nauman et al. — *BRO* (NeurIPS 2024) — https://arxiv.org/abs/2405.16158
- Schwarzer et al. — *BBF* (ICML 2023) — https://arxiv.org/abs/2305.19452
- Hafner et al. — *DreamerV3* (2023) — https://arxiv.org/abs/2301.04104
- Rybkin, Nauman, Fu, Snell, Abbeel, Levine, Kumar — *Value-Based Deep RL Scales Predictably* (ICML 2025) — https://arxiv.org/abs/2502.04327
- Ma et al. — *Network Sparsity Unlocks the Scaling Potential of Deep RL* (ICML 2025) — https://arxiv.org/abs/2506.17204
- Creus Castanyer et al. — *Stable Gradients for Stable Learning at Scale* (NeurIPS 2025) — https://arxiv.org/abs/2506.15544
- Kooi, Yang, François-Lavet — *Hadamax Encoding* (NeurIPS 2025) — https://arxiv.org/abs/2505.15345
- *Scaling DRL for Decision Making: A Survey* (2025) — https://arxiv.org/html/2508.03194v1

**Plasticity / representation pathologies**
- Sokar, Agarwal, Castro, Evci — *The Dormant Neuron Phenomenon* (ICML 2023) — https://arxiv.org/abs/2302.12902
- Kumar, Agarwal, Ghosh, Levine — *Implicit Under-Parameterization* (ICLR 2021) — https://arxiv.org/abs/2010.14498
- Nikishin et al. — *Primacy Bias* (ICML 2022) — https://proceedings.mlr.press/v162/nikishin22a.html
- Nikishin et al. — *Plasticity Injection* (NeurIPS 2023) — https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf

**Readout / conditioning / auxiliary tasks**
- Jaegle et al. — *Perceiver IO* (ICLR 2022) — https://openreview.net/pdf?id=fILj7WpI-g
- Jaderberg et al. — *RL with Unsupervised Auxiliary Tasks (UNREAL)* (ICLR 2017) — https://arxiv.org/abs/1611.05397
- Perez et al. — *FiLM* (AAAI 2018) — https://dl.acm.org/doi/abs/10.5555/3504035.3504518
- Ye et al. — *Rotation, Translation, and Cropping for Zero-Shot Generalization* (2020) — https://arxiv.org/abs/2001.09908

**Optimization / batch size**
- McCandlish, Kaplan, Amodei et al. — *An Empirical Model of Large-Batch Training* (2018) — https://arxiv.org/abs/1812.06162
- Yang, Hu et al. — *Tensor Programs V (µP / µTransfer)* (2022) — https://arxiv.org/abs/2203.03466 ; https://github.com/microsoft/mup
- *µP for RL: Mitigating Feature Inconsistencies During RL* (OpenReview submission, 2025) — https://openreview.net/forum?id=Wuy631kHwH
- Obando-Ceron, Bellemare, Castro — *Small batch deep RL* (NeurIPS 2023) — https://arxiv.org/abs/2310.03882
- *Weight Decay may matter more than muP for LR Transfer in Practice* (2025) — https://arxiv.org/abs/2510.19093

**Competition writeups**
- Isaiah Pressman — *Lux AI 2021 (S1) 1st place* — https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- FLG — *Lux S2, 4th place, deep RL* — https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit
- Flat Neurons — *Lux S3 1st place* — https://www.kaggle.com/competitions/lux-ai-season-3/writeups/flat-neurons-1st-place-approach-by-flat-neurons
- Frog Parade — *Lux S3 2nd place* — https://www.kaggle.com/competitions/lux-ai-season-3/writeups/frog-parade-frog-parade-s-solution
- kurupical — *Lux S3 RL solutions summary (JA)* — https://zenn.dev/kurupical/articles/61dbeedf89a29d
