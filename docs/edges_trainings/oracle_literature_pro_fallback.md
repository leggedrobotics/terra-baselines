## Verdict

Your manual literature note is **directionally correct and actionable**. The main gap is that it frames the problem as “add action masks” too broadly. In Terra, there are at least three distinct issues:

1. **Primitive action validity**: movement, cabin, arm, dump, and `DO` actions that are physically impossible or guaranteed no-op.
2. **Tile-level feasibility hidden inside `DO`**: `DO` may dig core tiles, dig legal edge tiles, or silently fail on edge tiles because the pose/alignment precondition is false.
3. **Critic aliasing**: visually similar states can have very different returns depending on hidden edge legality, remaining edge count, timeout proximity, and terminal bonus structure.

So the best reading is: **Terra now has a state-dependent feasible-action MDP with hidden affordances, rare endgame success, and phase-dependent return distributions**. Masking is necessary, but not sufficient.

---

## 1. Audit of the current note

| Area                     |        Current note | Assessment                                                                                                                                     | What to add                                                                                                     |
| ------------------------ | ------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Invalid action masking   |              Strong | Correct first intervention. Huang & Ontañón is the right anchor.                                                                               | Separate primitive masks from tile-level dig feasibility; cite action elimination and action-space shaping too. |
| 2026 masking paper       |              Useful | Good fit, but it is a very recent preprint, so treat as supporting evidence, not the only basis.                                               | Use it mainly for “valid-action suppression” and feasibility-classification motivation.                         |
| Edge affordance features |              Strong | This is likely as important as masking.                                                                                                        | Add auxiliary prediction losses and explicit legality/progress labels.                                          |
| Critic instability       | Good but incomplete | Notes PopArt/UVFA, but should also cite PPO implementation sensitivity, time-limit handling, MAPPO value normalization, and multi-critic work. | Add phase-conditioned/multi-head critic, value normalization, timeout bootstrapping, terminal reward audit.     |
| Curriculum               |              Strong | Reverse curriculum is exactly relevant because failure is near the end.                                                                        | Add GoalGAN, demonstrations, self-imitation, HER-style caveats.                                                 |
| Hierarchy/options        |                Good | Correctly delayed until masking/observability are fixed.                                                                                       | Add MAXQ and practical scripted options before learned Option-Critic.                                           |
| Reward shaping           |                Good | Needs stricter potential-based formulation and caveats.                                                                                        | Use `F(s,s') = gamma Phi(s') - Phi(s)`; avoid raw alignment bonuses unless knowingly changing the objective.    |

---

## 2. RL taxonomy of the Terra failure

**State-dependent feasible action set.** The environment has an implicit `A_valid(s)`. The current PPO samples from the full action set, so it wastes probability mass on actions that are invalid or no-op in the current state. Invalid action masking is directly relevant here; Huang and Ontañón show that masking is a valid policy-gradient modification and becomes more important as the number of invalid actions grows. ([arXiv][1])

**Hidden affordance / partial observability.** The simulator computes edge legality from border direction, base proximity, and arm/cabin alignment, but the policy and critic do not see those signals explicitly. This creates state aliasing: two observations may look similar, but `DO` has different consequences.

**Phased multi-task problem.** Core digging and edge finishing are different subtasks. Core digging is “cover target tile with cone”; edge digging is “navigate to boundary, align, then dig.” This fits goal-conditioned or task-conditioned value prediction, as in Universal Value Function Approximators, and multi-task critic normalization. ([Proceedings of Machine Learning Research][2])

**Rare high-value endgame.** Successful edge finishing is sparse and late. GAE targets can become high variance near terminal success, especially with a large terminal reward and mixed timeout/success endings. PPO and GAE are sensitive to implementation choices, value normalization, clipping, and bootstrapping details. ([arXiv][3])

**Long-horizon temporally abstract task.** “Move to edge → align cabin/arm → dig clean edge” is naturally an option-like behavior, but hierarchy should come after the validity and observability bugs are fixed.

---

## 3. Missing or weak references to add

### Invalid action masking and action elimination

* **“A Closer Look at Invalid Action Masking in Policy Gradient Algorithms” — Shengyi Huang, Santiago Ontañón.** Core reference for masking in policy-gradient methods; directly supports masking invalid Terra actions before sampling and PPO log-prob computation. ([arXiv][1])

* **“Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms” — Renos Zabounidis et al.** Very relevant to Terra because it argues that unmasked policy gradients with shared parameters can suppress valid actions in states with state-dependent action validity. It also motivates feasibility prediction. Treat as a recent preprint, not settled consensus. ([arXiv][4])

* **“Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning” — Tom Zahavy et al.** Useful because Terra has an oracle-like invalid-action signal from the simulator. It supports learning or using an elimination signal to remove actions known to be invalid. ([NeurIPS Proceedings][5])

* **“Action Space Shaping in Deep Reinforcement Learning” — Anssi Kanervisto, Christian Scheller, Ville Hautamäki.** Supports the broader point that task-specific action-space restrictions can be critical for deep RL performance. ([arXiv][6])

* **SB3-Contrib Maskable PPO documentation.** Not a paper, but useful as an implementation reference: mask the logits before constructing the distribution, and use the same masking in evaluation and training. ([Stable Baselines3 Contrib Docs][7])

### Hidden feasibility, affordances, and auxiliary losses

* **“Reinforcement Learning with Unsupervised Auxiliary Tasks” — Max Jaderberg et al.** Good support for adding auxiliary prediction/control heads when the main reward is sparse or delayed. For Terra, predict edge legality, blocked-edge count, legal `DO` count, or next progress class. ([arXiv][8])

* **“Universal Value Function Approximators” — Tom Schaul et al.** Supports conditioning the value function on goals or task descriptors. Terra can condition on phase, remaining edge count, edge side, or target tile class. ([Proceedings of Machine Learning Research][2])

* **“Successor Features for Transfer in Reinforcement Learning” — André Barreto et al.** Relevant if Terra wants reusable representations for core excavation, edge finishing, dumping, and map variations. More research-heavy than the first fixes. ([arXiv][9])

### Critic stability, value normalization, and phase-conditioned critics

* **“Learning values across many orders of magnitude” — Hado van Hasselt et al.** PopArt reference. Relevant because Terra mixes dense step rewards with sparse terminal bonuses and different phase return scales. ([arXiv][10])

* **“Multi-task Deep Reinforcement Learning with PopArt” — Matteo Hessel et al.** Stronger multi-task reference for scale-normalized value prediction across tasks with different reward statistics. ([arXiv][11])

* **“Multi-Critic Actor Learning: Teaching Reinforcement Learning Policies to Act with Style” — Siddharth Mysore et al.** Supports separate critics when a single critic must fit different objectives or return structures. Terra’s core/edge phases are a good candidate. ([OpenReview][12])

* **“The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games” — Chao Yu et al.** Useful for the multi-agent Terra setting; its implementation study highlights value normalization as an important PPO stabilization detail. ([OpenReview][13])

* **“Time Limits in Reinforcement Learning” — Fabio Pardo et al.** Important if Terra mixes task success, failure, and timeout under one `done`. Incorrect timeout handling can create value bias and instability. ([Proceedings of Machine Learning Research][14])

* **“Implementation Matters in Deep Policy Gradients” — Logan Engstrom et al.** Supports auditing PPO details before attributing the issue only to exploration. Terra’s high `vf_coef`, large clip range, terminal backfill, and value clipping deserve ablation. ([arXiv][15])

* **“What Matters in On-Policy Reinforcement Learning?” — Marcin Andrychowicz et al.** Useful practical reference for normalization, clipping, and PPO implementation choices. ([arXiv][16])

### Curriculum, reverse curriculum, demonstrations, and rare endgames

* **“Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey” — Sanmit Narvekar et al.** Good survey anchor for staged task sequencing. ([Journal of Machine Learning Research][17])

* **“Reverse Curriculum Generation for Reinforcement Learning” — Carlos Florensa et al.** Directly relevant: start near successful end states, then gradually move the reset distribution farther away. Terra should start from almost-finished edge states. ([arXiv][18])

* **“Automatic Goal Generation for Reinforcement Learning Agents” — Carlos Florensa et al.** Useful if you want automatic sampling of edge-start states at intermediate difficulty. ([arXiv][19])

* **“Hindsight Experience Replay” — Marcin Andrychowicz et al.** Relevant conceptually for sparse goal completion, but less directly plug-and-play with on-policy PPO unless Terra changes training style or uses relabeling only in auxiliary/off-policy components. ([NeurIPS Proceedings][20])

* **“Deep Q-learning from Demonstrations” — Todd Hester et al.** Supports using a small set of scripted or teleoperated edge-finishing demonstrations to overcome rare successful exploration. ([AAAI Publications][21])

* **“Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations” — Aravind Rajeswaran et al.** Strong robotics reference for demonstration-accelerated RL, especially where precise manipulation is hard to discover by exploration. ([arXiv][22])

* **“Self-Imitation Learning” — Junhyuk Oh et al.** Relevant if rare successful edge trajectories appear but are not reinforced enough; use past high-return behavior as an auxiliary imitation target. ([Proceedings of Machine Learning Research][23])

### Options, hierarchy, and shaping

* **“Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning” — Richard Sutton, Doina Precup, Satinder Singh.** Foundational options reference. Terra edge finishing is naturally a temporally extended skill. ([ScienceDirect][24])

* **“The Option-Critic Architecture” — Pierre-Luc Bacon, Jean Harb, Doina Precup.** Relevant if you want learned options and learned termination, but likely not the first intervention. ([arXiv][25])

* **“Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition” — Thomas Dietterich.** Supports decomposing the task into subtasks with separate value structure. Terra has obvious subtasks: core dig, move, align, edge dig, dump. ([JAIR][26])

* **“Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping” — Andrew Ng, Daishi Harada, Stuart Russell.** Required reference for potential-based shaping. It supports shaping only when the added reward has the potential-difference form. ([People @ EECS][27])

* **“Potential-Based Shaping and Q-Value Initialization are Equivalent” — Eric Wiewiora.** Useful for interpreting shaping as initialization-like guidance rather than arbitrary reward hacking. ([arXiv][28])

---

## 4. Prioritized Terra recommendation plan

### P0 — Instrument and mask real invalid actions

**Terra failure mode:** PPO samples `DO` even when edge tiles are blocked by hidden feasibility; many actions become deterministic no-ops, so the policy gradient learns slowly and the critic sees noisy returns.

**Recommendation:**

* Implement a real `valid_action_mask` in the environment.
* Apply it before `tfp.distributions.Categorical`, both during rollout sampling and PPO update log-prob recomputation.
* Use the same masked logits for entropy.
* Keep at least one fallback valid action, such as no-op.
* Do not mask repositioning actions just because they do not immediately dig; mask actions that are physically invalid or guaranteed no-op.
* For `DO`, distinguish:

  * digs at least one core tile,
  * digs at least one legal edge tile,
  * touches edge tiles but all are blocked,
  * digs nothing.

For `DO`, the safest first rule is: **mask only when the action would dig zero useful tiles and has no other intended effect**. If `DO` digs core tiles while edge tiles are blocked, it should remain valid, but the observation/logs should expose the blocked edge count.

**Diagnostics to log:**

* mask cardinality per state,
* invalid/no-op action rate before and after masking,
* `DO` effect count,
* legal edge `DO` count,
* blocked edge tile count,
* entropy over valid actions only,
* policy probability on `DO` when edge legality is true vs false.

---

### P1 — Expose edge affordances in the observation

**Terra failure mode:** the simulator knows edge legality, but the policy and critic must infer it indirectly from maps and pose. This creates avoidable state aliasing.

**Recommendation:**

Add explicit low-dimensional features and/or map channels:

* local tile class: core target, edge target, non-target,
* nearest remaining edge segment direction, encoded as `sin/cos`,
* base-to-edge distance,
* signed or absolute alignment error between cabin/arm and edge tangent/normal,
* whether current cone touches edge tiles,
* number of legal edge tiles in current cone,
* number of blocked edge tiles in current cone,
* remaining core tile count,
* remaining edge tile count,
* phase flag: core-heavy, edge-setup, edge-dig-ready, near-complete, timeout-risk.

Add auxiliary heads:

* predict valid primitive action mask,
* predict `DO` legality,
* predict number of core/edge tiles removed by each `DO`,
* predict phase,
* predict next-step progress bucket.

Keep auxiliary loss weights small at first. These heads should improve the shared representation without dominating PPO.

**Diagnostics:**

* auxiliary accuracy/AUC for edge legality,
* critic error split by edge-legal vs edge-blocked states,
* value prediction split by phase,
* policy `DO` probability conditioned on edge-ready states.

---

### P2 — Fix critic structure before adding heavy hierarchy

**Terra failure mode:** one scalar value head must fit core excavation, edge setup, legal edge digging, near-success terminal reward, and timeout states. This can cause value spikes and poor GAE targets.

**Recommendation order:**

1. **Add phase features to the critic input.**
   This is the lowest-risk UVFA-style change.

2. **Add phase-conditioned or multi-head value prediction.**
   Example heads:

   * core excavation,
   * edge setup,
   * edge dig-ready,
   * dumping/repositioning,
   * terminal/near-timeout.

   Use either hard selection from known phase labels or a soft gating network. Start with hard labels because Terra already computes structured state.

3. **Normalize value targets.**
   Try running return normalization or PopArt. This is especially relevant if terminal rewards are much larger than dense step rewards.

4. **Audit PPO value loss settings.**
   Current `vf_coef=5.0` and `clip_eps=0.5` are aggressive enough to deserve ablation. Try lower `vf_coef`, separate value clip range, Huber value loss, and separate actor/critic trunks.

5. **Fix termination semantics.**
   Separate:

   * `done_task`,
   * timeout/truncation,
   * failure termination.

   Bootstrap on timeout if the episode is truncated rather than truly terminal. Disable or separately ablate the mixed-agent terminal reward backfill until value behavior is stable.

**Diagnostics:**

* explained variance by phase,
* value target histogram by phase,
* GAE/TD-error histogram by phase,
* value clipping fraction,
* critic gradient norm vs actor gradient norm,
* predicted value vs empirical return for fixed edge-start states,
* terminal cause counts.

---

### P3 — Add reverse curriculum for edge endgames

**Terra failure mode:** edge completion is rare, so PPO receives too few successful examples of “align near edge then dig.” Waiting for full-task exploration to discover this is wasteful.

**Recommendation:**

Create reset distributions for:

1. **edge-only finishing**: core already dug, several edge tiles remain;
2. **one-side remaining**: only one boundary side remains;
3. **corner remaining**: corners are often the hardest alignment cases;
4. **almost complete**: one or two edge tiles remain;
5. **full task**: original distribution.

Train with a mixture, then gradually increase the full-task fraction. Also consider staged constraint hardening:

* wide proximity tolerance → real tolerance,
* loose alignment tolerance → real tolerance,
* soft edge mask/shaping → hard edge mask,
* simple rectangles → irregular foundations.

This is a good fit for reverse curriculum: start near successful edge states and move backward toward the full problem.

**Diagnostics:**

* success rate on each reset suite,
* edge completion percentage,
* legal edge `DO` attempts per episode,
* steps from edge-ready state to edge dig,
* full-task performance retention,
* corner-specific failure rate.

---

### P4 — Use potential-based shaping, not arbitrary alignment bonuses

**Terra failure mode:** edge setup has delayed reward. The agent may need many movement/cabin actions before a legal edge `DO`, so PPO sees weak credit assignment.

**Recommendation:**

Use potential-based shaping:

[
r'(s,a,s') = r(s,a,s') + \gamma \Phi(s') - \Phi(s)
]

Candidate Terra potentials:

[
\Phi(s) =
-\alpha \cdot N_\text{remaining edge}
-\beta \cdot N_\text{remaining core}
-\eta \cdot e_\text{nearest legal edge pose}
]

where `e_nearest legal edge pose` combines base-to-edge distance and arm/cabin alignment error for the nearest remaining edge segment.

Important details:

* use the same `gamma` as PPO;
* define `Phi` from Markov state only;
* set terminal potential consistently, usually zero;
* ablate shaping on/off;
* log each shaping component separately.

Avoid raw rewards like “+0.1 for being aligned” unless you are willing to change the optimal policy. Potential-based shaping is the safer default because it is designed to preserve optimal policies under standard assumptions. ([People @ EECS][27])

---

### P5 — Add hierarchy/options only after P0–P3

**Terra failure mode:** edge finishing is temporally extended, but learned hierarchy will not fix hidden legality or invalid action sampling.

**Practical hierarchy:**

High-level mode chooses among:

* core dig,
* move to edge,
* align to edge,
* dig edge,
* dump/reposition.

Low-level controller executes primitive actions. Start with simple scripted or supervised edge-alignment options before full Option-Critic. For example:

* initiation: `remaining_edge_tiles > 0`;
* option target: nearest remaining edge segment;
* termination: `legal_edge_dig_count > 0` or timeout;
* policy: move base near boundary, align cabin/arm, then allow `DO`.

This is likely more robust than immediately learning options end-to-end.

---

## 5. Concrete ablation plan

Run these in order, with fixed maps and fixed evaluation seeds:

| Experiment | Change                                             | Expected result                                                                          |
| ---------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| A          | Current baseline + new diagnostics only            | Confirms no-op `DO`, blocked edge tiles, and phase-specific value spikes.                |
| B          | Valid action masking only                          | Lower no-op rate, better sample efficiency, but may still struggle with edge setup.      |
| C          | Edge affordance observations only                  | Better critic calibration and edge alignment behavior, but still wastes invalid samples. |
| D          | Masking + edge affordances                         | First likely strong improvement. This should be the main quick fix.                      |
| E          | D + phase-conditioned critic                       | Lower value spikes and better explained variance by phase.                               |
| F          | E + value normalization / PopArt / lower `vf_coef` | More stable critic targets and less critic domination.                                   |
| G          | F + reverse edge curriculum                        | Higher edge completion rate and better rare-endgame behavior.                            |
| H          | G + potential-based shaping                        | Faster learning of alignment/setup, with shaping ablated for final validation.           |
| I          | H + hierarchy/options                              | Only worth it if flat policy still struggles with long edge setup.                       |

Use separate evaluation suites:

* full foundation,
* core-only,
* edge-only,
* one side remaining,
* corners remaining,
* near-timeout,
* multi-agent alternating case.

Report:

* task completion rate,
* final core completion,
* final edge completion,
* no-op `DO` rate,
* blocked-edge `DO` rate,
* legal-edge `DO` rate,
* mean remaining edge tiles at timeout,
* value explained variance by phase,
* value spike frequency,
* return variance,
* PPO KL,
* entropy over valid actions,
* actor/critic gradient norm ratio.

---

## 6. Recommendation mapped to Terra failure modes

| Terra symptom                           | Likely cause                                                   | Best intervention                                                                          |
| --------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| PPO finishes less often after edge rule | Edge `DO` requires rare pose/alignment sequence                | Reverse edge curriculum + edge affordance observations                                     |
| Many `DO` actions do nothing            | Unmasked categorical samples invalid/no-op actions             | Real action mask in rollout and PPO update                                                 |
| Value function small spikes             | Mixed return distributions and hidden legality                 | Phase-conditioned critic + value normalization + timeout audit                             |
| Critic cannot predict near-finish value | Terminal reward sparse and edge legality hidden                | Edge features, phase labels, return normalization, fixed termination semantics             |
| Agent digs core but leaves border       | Core and edge are different subtasks                           | Separate edge phase features, edge reset curriculum, optional edge option                  |
| Agent fails to align before edge dig    | Alignment progress is delayed                                  | Potential-based shaping over distance/alignment error, plus auxiliary legality prediction  |
| Multi-agent value instability           | Terminal reward backfill and alternating turns distort targets | Disable/ablate backfill, phase-specific value diagnostics, MAPPO-style value normalization |

---

## 7. Caveats

Masking is not enough. It prevents invalid sampling, but it does not teach the policy how to make `DO` become valid. Terra also needs edge affordance features or auxiliary legality prediction.

Do not mask all “currently unproductive” actions. Movement and cabin rotations may be necessary setup actions. Mask physical impossibilities and guaranteed no-effect primitives, not exploration steps.

Potential-based shaping only preserves the optimal policy under the standard MDP assumptions. Multi-agent alternating rewards, terminal backfill, or shaping based on non-observed simulator internals can break that guarantee.

HER, self-imitation, and demonstrations are relevant, but they are not clean drop-ins for pure on-policy PPO. The lowest-friction version is: use reset curricula, add a small behavior-cloning auxiliary loss from scripted edge-alignment traces, and optionally add self-imitation on rare successful PPO rollouts.

Distributional critics may help if returns remain multimodal, but they are not the first fix. First remove hidden legality, fix masks, split phases, and normalize value targets.

---

## Bottom-line plan

Implement in this order:

1. **Real action masks + detailed edge/no-op logging.**
2. **Expose edge legality, alignment error, base-to-edge distance, edge/core counts, and phase flags.**
3. **Add auxiliary edge-legality/progress prediction.**
4. **Use a phase-conditioned or multi-head critic with value normalization.**
5. **Audit terminal reward, timeout bootstrapping, and mixed-agent terminal backfill.**
6. **Train with reverse edge curricula and edge-start reset distributions.**
7. **Add potential-based shaping for edge pose/alignment progress.**
8. **Only then consider hierarchy/options for edge finishing.**

The strongest near-term experiment is **masking + edge affordance observations + phase-conditioned critic**. That directly targets the Terra-specific failure: the simulator enforces edge feasibility, but the actor and critic currently do not receive or use that feasibility structure.

[1]: https://arxiv.org/abs/2006.14171?utm_source=chatgpt.com "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"
[2]: https://proceedings.mlr.press/v37/schaul15.html?utm_source=chatgpt.com "Universal Value Function Approximators"
[3]: https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com "Proximal Policy Optimization Algorithms"
[4]: https://arxiv.org/abs/2603.09090?utm_source=chatgpt.com "Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms"
[5]: https://proceedings.neurips.cc/paper/2018/hash/645098b086d2f9e1e0e939c27f9f2d6f-Abstract.html?utm_source=chatgpt.com "Action Elimination with Deep Reinforcement Learning"
[6]: https://arxiv.org/abs/2004.00980?utm_source=chatgpt.com "Action Space Shaping in Deep Reinforcement Learning"
[7]: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html?utm_source=chatgpt.com "Maskable PPO - Stable Baselines3 Contrib docs!"
[8]: https://arxiv.org/abs/1611.05397?utm_source=chatgpt.com "Reinforcement Learning with Unsupervised Auxiliary Tasks"
[9]: https://arxiv.org/abs/1606.05312?utm_source=chatgpt.com "Successor Features for Transfer in Reinforcement Learning"
[10]: https://arxiv.org/abs/1602.07714?utm_source=chatgpt.com "Learning values across many orders of magnitude"
[11]: https://arxiv.org/abs/1809.04474?utm_source=chatgpt.com "Multi-task Deep Reinforcement Learning with PopArt"
[12]: https://openreview.net/forum?id=rJvY_5OzoI&utm_source=chatgpt.com "Multi-Critic Actor Learning: Teaching RL Policies to Act with ..."
[13]: https://openreview.net/forum?id=YVXaxB6L2Pl&utm_source=chatgpt.com "The Surprising Effectiveness of PPO in Cooperative Multi- ..."
[14]: https://proceedings.mlr.press/v80/pardo18a.html?utm_source=chatgpt.com "Time Limits in Reinforcement Learning"
[15]: https://arxiv.org/abs/2005.12729?utm_source=chatgpt.com "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO"
[16]: https://arxiv.org/abs/2006.05990?utm_source=chatgpt.com "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
[17]: https://jmlr.org/papers/v21/20-212.html?utm_source=chatgpt.com "Curriculum Learning for Reinforcement Learning Domains"
[18]: https://arxiv.org/abs/1707.05300?utm_source=chatgpt.com "Reverse Curriculum Generation for Reinforcement Learning"
[19]: https://arxiv.org/abs/1705.06366?utm_source=chatgpt.com "Automatic Goal Generation for Reinforcement Learning Agents"
[20]: https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html?utm_source=chatgpt.com "Hindsight Experience Replay"
[21]: https://ojs.aaai.org/index.php/AAAI/article/view/11757?utm_source=chatgpt.com "Deep Q-learning From Demonstrations"
[22]: https://arxiv.org/abs/1709.10087?utm_source=chatgpt.com "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"
[23]: https://proceedings.mlr.press/v80/oh18b.html?utm_source=chatgpt.com "Self-Imitation Learning"
[24]: https://www.sciencedirect.com/science/article/pii/S0004370299000521?utm_source=chatgpt.com "A framework for temporal abstraction in reinforcement ..."
[25]: https://arxiv.org/abs/1609.05140?utm_source=chatgpt.com "The Option-Critic Architecture"
[26]: https://www.jair.org/index.php/jair/article/view/10266?utm_source=chatgpt.com "Hierarchical Reinforcement Learning with the MAXQ Value ..."
[27]: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf?utm_source=chatgpt.com "Policy invariance under reward transformations"
[28]: https://arxiv.org/abs/1106.5267?utm_source=chatgpt.com "Potential-Based Shaping and Q-Value Initialization are Equivalent"
