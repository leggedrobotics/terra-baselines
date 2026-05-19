# PPO Scaling Brief for Terra Map Policies

Source: Oracle browser Deep Research report saved by the user as
`~/Downloads/terra_deep_research.md` on `2026-05-19`.

Scope for the next Oracle/code review:

- Keep the algorithm PPO. Do not recommend switching to IMPALA, Dreamer, or
  another RL algorithm.
- Treat Lux/Kaggle material as architecture/training evidence for grid-map
  policies, not as a reason to introduce self-play or league machinery into
  Terra.
- Review the current Terra implementation against this brief, especially the
  `medium_deep` and `large_deep` presets, rollout shape, PPO minibatch/epoch
  choices, imitation-to-PPO handoff, critic sizing, and value/entropy
  diagnostics.
- Produce concrete changes or ablations that can be tested with the current
  Terra PPO stack.

Local Terra update after the first review pass:

- The Deep Research recommendation for action masking is conditional: use a
  mask only for actions that are truly impossible and have identical semantics
  at deployment time.
- Terra's tested coarse actor mask does not meet that bar as a default PPO
  change. Forced-mask rollouts on the current ResMap/R1/R2 lineage remove
  invalid selections but consistently push probability mass into `DO_NOTHING`
  and reduce productive `DO`; they did not recover completions in local probes.
- The larger-network PPO runs therefore stay unmasked. The next fixes should
  prioritize rollout length, PPO minibatch/epoch shape, critic sizing and
  diagnostics, and a softer teacher handoff before revisiting any exact
  physical-invalid mask.

## Bottom line

The strongest evidence points to a **PPO optimization mismatch**, not a fundamental capacity limit, as the most likely reason your ~10M `large_deep` policy is lagging after the imitation phase. Three patterns show up repeatedly across PPO studies and Lux-style grid agents: sample efficiency drops when on-policy collection uses **too many environments with very short rollout fragments**; large symmetric actor-critic models often waste capacity because the **critic benefits more from size than the actor**; and large visual policies usually need a **softer imitation-to-RL handoff** than a hard BC stop followed by full-strength PPO. In parallel, the Lux/MicroRTS competition literature suggests that strong map agents usually scale by **deeper residual trunks, explicit map channels, squeeze-excitation, exact action masking, and bottlenecked receptive-field growth**, not just by making the last feature dimension fatter. citeturn5view0turn23view0turn34view0turn14view0turn18view4turn15view0turn13view1

My read, adapted to Terra, is that the next best tests are: **use longer rollouts at the same total batch size, make the large policy more depth-/critic-focused rather than simply wider, extend and soften the teacher handoff, and rebalance PPO toward larger minibatches plus slightly more reuse with stronger critic hygiene**. Those changes stay fully within PPO. citeturn5view1turn23view0turn21view0turn22view0

## Prioritized recommendations

**First, trade some parallelism for temporal depth.** Your current batch is 4096 envs × 32 steps. For long-horizon excavation, that is a lot of breadth and not much temporal continuity. In a large PPO ablation, increasing the number of parallel environments hurt sample efficiency on several tasks because rollouts became shorter and value bootstrapping happened earlier; the same study recommends tuning the number of transitions per iteration and notes that more environments are mostly a wall-clock win, not a learning win. A Lux JAX writeup also reports using 512–1024 environments rather than thousands, while keeping large batch updates. For Terra, the clean test is to keep roughly the same total transitions per update but switch to **2048×64** or **1024×128** for the large model first. That is the single highest-confidence adjustment for delayed terminal success. citeturn5view1turn27search0

**Second, stop scaling the actor symmetrically with the critic.** The best broad PPO ablation in on-policy RL found that separate policy and value networks usually beat shared ones, that a **wider value network** often helps more than a wider policy, and that the best policy width depends on task complexity. PPG then sharpened the same point: shared parameters create policy/value interference, and value learning usually tolerates more reuse than policy learning. A separate case study across actor-critic algorithms, including PPO, found that actors can often be made much smaller than critics without hurting performance, because value estimation carries more modeling burden. For Terra, that means the next “large” model should not just be “medium but fatter everywhere”; it should be **critic-heavy or encoder-heavy, with an actor no larger than necessary**. citeturn4view0turn23view0turn34view0

**Third, scale the encoder by depth and receptive field before raw width.** Lux writeups are remarkably consistent here. Deimos used an **8-layer, 64-wide ResNet**. Pressman’s strong Lux Season 1 visual agent used a **24-block fully convolutional ResNet with squeeze-excitation, no normalization, and ~20M parameters**. The later competition-winning RTS paper reports the **DoubleCone** backbone, which keeps a ResNet-like structure but **downscales the middle residual stack by 4×** to grow receptive field and reduce compute; the same paper also notes that larger maps needed even more aggressive downscaling rather than plain widening. That pattern argues for testing a 10M Terra encoder that adds **residual depth and one strong bottleneck/downsample stage**, while keeping the actor head relatively restrained, instead of only pushing `map_feature_dim` from 224 to 512. citeturn14view0turn18view4turn15view0turn17view0

**Fourth, soften the BC-to-PPO handoff.** Kickstarting work showed that teacher-guided training can improve data efficiency dramatically while still letting the student exceed the teacher. In Lux Season 1, Pressman kept a **frozen teacher KL** during RL and explicitly credits it with stabilizing behavior and preventing strategic cycles; he also trained **smaller models first, then larger models with the smaller models as teachers**. A recent PPO-specific robotics preprint goes one step further and reports that **pretraining both actor and critic** beats actor-only pretraining on simulated robotics tasks. In Terra terms, 100 updates of imitation is probably reasonable for ~2M, but it is a plausible under-warm-start for ~10M. The next test should be **300–1000 warmup updates for the large model**, plus a **small teacher KL / BC regularizer that anneals to zero over the first chunk of PPO**, instead of an abrupt full handoff. citeturn22view0turn18view4turn21view0

**Fifth, rebalance PPO optimization for the large model toward larger minibatches and slightly more reuse.** PPO’s original paper exists because the clipped objective allows **multiple epochs of minibatch updates**. Andrychowicz et al. found that going over the data multiple times is crucial, that stale advantages hurt, and that recomputing advantages once per pass helps. In your current setup, the large model uses **twice as many minibatches as the medium model**, which means smaller optimization batches and noisier gradients. For the next large-model run, I would test **32–64 minibatches instead of 128**, so the minibatch size rises from 1024 to 2048–4096, and pair that with **3–4 PPO epochs** rather than 2 if you can recompute advantages each pass. If you change optimization batch size, the NeurIPS 2022 batch-scaling analysis suggests Adam often follows a **square-root LR scaling** rule in the small-batch regime, though PPO still needs KL monitoring because it is not perfectly batch-invariant. citeturn36view0turn5view1turn29view0

**Sixth, harden the critic before touching the reward design.** The on-policy ablation recommends **observation normalization**, checking **value normalization**, using **MSE rather than Huber**, and **not using PPO-style value clipping**, which hurt performance in that study. Lux training logs also show that value loss spikes are common during exploration, but the authors still treat persistent KL spikes as a sign of overly large policy steps and forgetting. For Terra, if terminal success is missing and the large model is slow after PPO starts, the critic is a prime suspect. The next run should explicitly verify: no value clipping, normalized observations, a test with value normalization or PopArt-style target normalization if reward scales drift, and separate logging of actor and critic gradient norms. citeturn4view0turn4view1turn5view0turn9view0turn30search3

**Seventh, add exact invalid-action masking if Terra has truly impossible actions.** The masking paper gives a theoretical justification for masking in policy-gradient methods and shows it matters more as the invalid fraction grows. Lux PPO-style agents repeatedly used hard masks to zero logits and gradients for actions that cannot execute. If Terra has actions that are genuinely impossible in a given state—rather than merely bad—this is one of the few changes that can simultaneously improve exploration efficiency and reduce gradient noise. The important caveat is to mask only what is **semantically invalid**, not what a handcrafted heuristic merely dislikes. citeturn24view0turn8view4turn15view0

## What not to change

Do **not** collapse back to a shared actor/critic trunk while debugging the large model. Both the large on-policy study and PPG point the other way: separate networks usually work better, and shared parameters can create interference between policy and value learning. If anything, the critic deserves **more** dedicated capacity or update budget, not less. citeturn4view0turn23view0

Do **not** remove explicit terrain-derived channels or global context in the hope that a larger ResNet will “figure it out.” Lux agents that worked on hard grid domains did the opposite: they fed rich stacks of map features, encoded global phase/context, and broadcast global information into the spatial tensor. That is especially relevant for robotics maps with hard geometry and phase-dependent behavior. citeturn14view0turn8view4turn18view3

Do **not** add BatchNorm as a default fix. The strongest Lux visual writeup explicitly used a deep SE-ResNet with **no normalization layers** in the residual body, and the other major Lux backbones described publicly also rely on residual structure, feature engineering, and input/value normalization rather than batch-dependent normalization inside the map encoder. If you need a normalization experiment, reserve it for **LayerNorm or GroupNorm**, and treat it as a secondary test, not the default path. citeturn18view4turn13view1turn31search4

Do **not** enable PPO-style value clipping if it is currently off. The largest on-policy ablation found it hurt across the tested environments, and the same study also found Huber worse than MSE for the value head. Undoing a good critic to “stabilize” it is the classic PPO trap. citeturn5view0

Do **not** increase environment count further or shorten rollouts further. For your current symptoms, that would be exactly the wrong direction. citeturn5view1

## Suggested ranges for the next sweep

These are **practical starting grids**, adapted from the PPO studies above and from Lux-style map encoders. They are not single-paper defaults.

For the **~2M policy**, keep it in the “fast learner” role. A good envelope is **6–10 residual blocks**, mostly **64–96 channels**, with **SE in every block or every other block**, no BatchNorm, and at most **one** delayed 2–4× bottleneck/downsample stage. Keep the actor modest: something like **actor feature dim 192–256** and **critic feature dim 224–320** is more defensible than scaling both to the ceiling. For PPO, test **2048–4096 envs × 32–64 steps**, **2–4 epochs**, and **32–64 minibatches** so the optimization minibatch lands around **2048–4096 samples**. Use **Adam with β1≈0.9**, **global grad clip 0.5–1.0**, and **LR around 2e-4 to 4e-4** with linear decay. If you are warm-starting from the teacher, keep BC/teacher regularization short-to-moderate: **100–300 imitation updates**, then optionally anneal a teacher KL over the first **500–1000 PPO updates**. citeturn14view0turn18view4turn4view0turn5view1turn29view0

For the **~10M policy**, move it into the “slow but high-capacity critic / deep encoder” role. Prefer **10–16 residual blocks** at **96–128 channels**, or a **DoubleCone-style** trunk with one strong 4× bottleneck in the middle, over a simple head-width jump. Keep the actor narrower than the critic; a sensible target is **actor feature dim 224–384** and **critic feature dim 512–768**. The collection setup I would test first is **1024–2048 envs × 64–128 steps**, keeping roughly the same transitions per update as today but making the data less myopic. For optimization, use **32–64 minibatches**, **3–4 epochs**, and recompute advantages each pass if feasible. Start **below or around the classic PPO LR**, roughly **1e-4 to 3e-4**, then adjust with KL; if you substantially increase optimization minibatch size, modest LR increases are defensible under the Adam square-root rule, but PPO still needs clip/KL restraint. I would also make the post-BC PPO phase more conservative than the medium model: smaller clip range early, larger minibatches, weaker early entropy, and a slower teacher anneal. citeturn15view0turn17view0turn5view1turn29view0turn34view0

For **entropy/exploration**, I would not reuse the same schedule blindly across model sizes. A high-capacity student coming out of imitation already has structured behavior; the large on-policy ablation found entropy regularization was often not very helpful when policy initialization was already careful, and Lux RL agents often used teacher stabilization or reward curriculum rather than brute-force entropy to stay exploratory. That makes me suspicious of a very high initial entropy coefficient immediately after BC for the 10M model. A practical test is to use a **lower entropy coefficient during the first PPO phase after imitation**, then decay to a small floor instead of starting with a very strong exploration push. That recommendation is a task-specific inference, but it is consistent with the cited evidence. citeturn5view1turn18view4turn22view0

## Risks and diagnostics

When **value loss spikes** but returns also improve, that is not automatically bad; Lux training logs explicitly note that such spikes can mean the agent has reached transitions whose value it cannot yet estimate. The red flag is different: **value loss rising while explained variance stays near zero or goes negative, with no corresponding policy improvement**. In that regime, the critic is adding noise rather than reducing it. Explained variance is best at 1, worse as it falls, and recent PPO analysis argues that **zero or negative EV** is a meaningful sign that the critic is not helping. citeturn9view0turn35search3turn25search12

When **mean KL spikes abruptly**, treat it as a high-priority failure signal, not a curiosity. The Lux PPO paper calls abrupt KL spikes undesirable because they indicate oversized learning steps that can cause forgetting. In practice, if the large model shows KL spikes right after PPO starts, the response order should be: increase optimization minibatch size, reduce LR, reduce epochs or clip range for the handoff phase, and lower early entropy pressure if BC just ended. citeturn9view0turn36view0

When **entropy collapses early** without a step change in success metrics, suspect teacher forgetting or invalid action burden before you suspect “the agent has converged.” This is where annealed teacher KL, exact action masking, and larger PPO minibatches help the most. Pressman’s writeup is especially relevant: he used teacher KL specifically to stabilize behavior during RL, not just to get a better starting point. citeturn18view4turn24view0

For larger visual PPO models, also watch for **representation collapse / plasticity loss** rather than only reward charts. A NeurIPS 2024 study found that PPO can suffer feature-rank deterioration and collapse, which in turn undermines the trust-region logic that is supposed to keep updates safe. If you cannot afford full representation-rank diagnostics, at least log per-layer activation variance, dead-channel fraction, and feature cosine drift across checkpoints; if those flatten while returns stall, the encoder—not the policy head—is probably the bottleneck. citeturn33search0turn33search2

## Lux AI lessons adapted to Terra

The **directly transferable** Lux lessons are about representation and optimization, not about self-play itself. Lux agents that actually worked on large, variable maps used **explicit map tensors**, **global context features**, **residual convolutional encoders**, and often **squeeze-excitation**. They also frequently simplified or factorized the action space and used **hard masks** for impossible actions. That maps well to Terra’s single-agent excavation setting: explicit terrain/process channels are good, residual map encoders are good, and exact masks are good if the environment exposes genuinely impossible moves. citeturn14view0turn8view4turn18view3turn24view0

The Lux competition evidence also favors **depth and bottlenecks over naive width scaling**. Deimos stayed relatively modest at 8 layers × 64 width; Pressman went very deep with 24 SE residual blocks; the later RTS competition-winning DRL paper and follow-up writeups emphasize **DoubleCone-style mid-network downscaling** to grow receptive field efficiently. For Terra, that argues for trying a deeper, bottlenecked 10M model before trying an even wider feature head. citeturn14view0turn18view4turn15view0turn26search10

The main Lux lesson that is **only partially transferable** is self-play stabilization. PFSP, opponent pools, and replay-state resets are crucial in competitive environments, but Terra is single-agent. What still transfers is the meta-lesson: hard environments often need a **curriculum or a teacher bridge**. In Lux, teams used dense-to-sparse reward transitions, frozen-teacher KL, or both. In Terra, the analogous move is not a league; it is a **longer IL warm-start plus an annealed imitation regularizer**, with the possibility of a stronger critic pretrain. citeturn8view4turn14view0turn18view4turn21view0

## Source links

- **PPO original paper** — why multiple minibatch epochs are even legal in PPO. citeturn36view0
- **What Matters for On-Policy Deep Actor-Critic Methods?** — separate networks, value normalization, no value clipping, Adam defaults, data-pass recommendations. citeturn4view0turn5view1
- **Phasic Policy Gradient** — policy/value interference and the observation that critic optimization tolerates more sample reuse than policy optimization. citeturn23view0
- **Batch Size-Invariance for Policy Optimization** — guidance for how PPO/Adam behavior changes when optimization batch sizes change. citeturn29view0
- **A Closer Look at Invalid Action Masking in Policy Gradient Algorithms** — theoretical and empirical basis for hard invalid-action masking. citeturn24view0
- **Honey, I Shrunk the Actor** — evidence that actor and critic should not be sized symmetrically by default. citeturn34view0
- **Centralized Control for Multi-Agent RL in a Complex RTS Game** — Lux PPO map encoder, feature maps, masking, and training diagnostics that transfer to grid robotics. citeturn8view4turn9view0
- **Deimos Lux Season 2 writeup** — 8-layer 64-wide ResNet, explicit map features, and dense-to-sparse curriculum. citeturn14view0
- **Pressman Lux Season 1 writeup** — deep SE-ResNet without normalization, teacher KL during RL, staged larger-student training. citeturn18view3turn18view4
- **Competition-winning RTS DRL paper** — DoubleCone bottleneck backbone, exact masking, multi-value-head reward transition ideas. citeturn15view0turn17view0
- **Kickstarting Deep Reinforcement Learning** — teacher-guided training as a bridge that improves sample efficiency without capping student performance. citeturn22view0
- **Actor-Critic Pretraining for PPO** — recent robotics evidence that pretraining the critic as well as the actor improves PPO sample efficiency. citeturn21view0
- **No Representation, No Trust** — PPO can fail through representation collapse, not only through bad reward design. citeturn33search0turn33search2

## Open questions and limitations

A few important knobs were not in the prompt, so the recommendations above are necessarily conditional: your current **learning rate, clip range, value-loss coefficient, GAE λ / γ, whether value clipping is already disabled, whether observation/value normalization are already enabled, and whether advantages are recomputed after each epoch**. Those details matter enough that I would instrument them before a large sweep. The most likely “gotcha” is that the large model is being blamed for what is really a **short-rollout, noisy-minibatch, hard-handoff PPO configuration**.
