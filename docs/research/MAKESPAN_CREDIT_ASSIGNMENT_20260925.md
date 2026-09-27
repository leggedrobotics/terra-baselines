# Makespan credit for multi-machine teams: literature and design (2026-09-25)

Context: the two-excavator team (see `MULTI_AGENT_JOINT_POLICY_20260923.md`)
reaches a median executed-plan speedup of about 1.5 over one excavator and
stops improving after u15000, while its speedup in rounds keeps rising. The
round reward charges a move and a full-workspace dig the same, so it does not
see the site time. The remaining limit is load balance: the busier machine
loads 62% of the soil, and one machine loads more than 80% on 19% of maps.

## The problem

Site time is a makespan: the time until the last machine finishes,
`T = max_i T_i`. A max is not a sum. Its value, and so any reward built on it,
changes only when the busiest machine works. Our first makespan cost charged
the growth of `max_i W_i` (W_i: a machine's executed-plan time so far) and
inherited the two standard failures:

- Free riding: the machine that is not the busiest could add setups, travel or
  double handling at no cost.
- Charging necessary work: the first machine to dig was charged immediately,
  even when the dig was part of an even split.

The same difficulty appears as min-max objectives (min-max routing, multiple
travelling salesmen), makespan in scheduling, equitable workload in
multi-robot task allocation, and credit assignment for non-decomposable team
objectives.

## What the literature does

1. **A dense reward from a lower bound on the final makespan (scheduling).**
   L2D (Zhang et al., NeurIPS 2020) rewards the decrease of a makespan lower
   bound, `R(a_t, s_t) = H(s_t) − H(s_{t+1})`, where `H(s_t)` is the largest
   lower-bound completion time over all operations, scheduled or not. With
   γ = 1 the rewards sum to `H(s_0) − C_max`: the reward is dense and still
   exactly makespan minimization. For identical parallel machines the classical
   bound (Graham's list-scheduling analysis, 1969) is
   `max(max_i W_i, (Σ_i W_i + remaining work) / m)`: the busiest machine so
   far, or a perfect split of all work done and left.
2. **The sparse final makespan with variance reduction (min-max routing).**
   DAN (Cao, Sun, Sartoretti) gives only the negative longest tour at the end of
   an episode, shares one network across agents that decide one after another,
   and trains REINFORCE with a greedy-rollout baseline. ScheduleNet (Park,
   Bakhtiyar, Park) normalizes the makespan by a baseline policy's,
   `(M(π_θ) − M(π_b)) / M(π_b)`, and uses clipped REINFORCE without a learned
   value function, which they found unreliable for makespan.
3. **Showing the policy the balance (equity context).** Equity-Transformer (Son
   et al., AAAI 2024) generates the agents' tours as one sequence and feeds, at
   every step, the active agent's current tour length plus the remaining
   distance, and the remaining cities per unused agent, so the policy can see
   who is behind the fair share.
4. **Per-agent credit.** Difference rewards (Agogino & Tumer, 2008) score agent
   i by `G(z) − G(z_{−i})`, the team objective minus its value without i's
   contribution; they stay aligned with the team objective and are easier to
   learn from. COMA's counterfactual baseline is the actor-critic form. For
   a makespan, the counterfactual needs an estimate of the plan without agent
   i, which is costly to compute.
5. **Smooth aggregates between sum and max.** Welfare functions such as the
   generalized Gini function or lexicographic maximin (Zimmer et al., ICML
   2021) trade efficiency against equity; a LogSumExp or p-norm soft maximum
   gives every agent a signal weighted by its load. Both approximate the
   makespan instead of optimizing it.
6. **Divide first, then plan each robot.** DARP (Kapoutsis et al., 2017)
   partitions the area into equal, connected regions, one per robot, before
   each robot plans its own coverage path. For excavation this is a strong
   non-learned baseline: split the dig volume into balanced regions and run the
   single-excavator policy in each.
7. **Shaping versus changing the objective.** Potential-based shaping preserves
   the equilibria of a multi-agent system (Devlin & Kudenko, 2011) when the
   potential is fixed at terminal states. The makespan potential ends at the
   final makespan, so it changes the objective to makespan, deliberately.

## Design adopted

Following items 1 and 3 (terra `State._makespan_terms`, `_get_reward_v2`;
`docs/ENVIRONMENT.md`):

- `W_i`: executed-plan seconds of each active tracked excavator, accumulated on
  its effective DO events: loaded units × 18.7 s (a 0.187 m³ cubic cell at
  30 s per 0.3 m³), travel from its previous work pose at 0.5 m/s, and
  `makespan_setup_s` per new work pose (workspace).
- `R`: remaining loading time, `(unexcavated required units + loose soil
  outside the accepted region) × 18.7 s`. Carried loads need no more loading.
- `A`: number of such machines; `T_job`: the single-machine loading time of
  the job.
- Bound `B = max(max_i W_i, (Σ_i W_i + R) / A) / T_job`; reward per round
  `−makespan_cost × (B' − B)`. At reset `B = 1/A`; summed over an episode the
  reward is `−makespan_cost × (final makespan / T_job − 1/A)`, 0 for a
  perfect split without overhead.
- Observation (`agent_states[..., 9:11]`): each machine's `W_i / T_job` (own
  first) and the fair share `(Σ_i W_i + R) / (A · T_job)`.

Behavior of the reward:

| Event | Charge |
|---|---|
| Machines load in step (balanced) | none: R falls as fast as Σ W grows |
| Either machine adds a setup, travel or a relift | charged when it raises B; time / A while the average-work bound is active, possibly zero while the busiest machine dominates |
| Staging soil outside the accepted region | R grows: the future relift is charged now |
| One machine passes the fair share | the excess of the busiest machine |
| One machine alone | only its overhead (loading moves W and R equally) |

Not modeled: waiting for and interference between machines, dump time
beyond the bucket cycle, and skid-steer pickups (FORWARD). Overhead on a less-busy machine can remain uncharged while the busiest
machine dominates the bound. The telescoping identity above is an undiscounted
successful-episode identity; PPO discounting and unfinished terminal states
qualify its interpretation. The cost is 0 by
default and then leaves the reward bitwise unchanged (tested); the frozen
benchmark hash excludes the inert defaults.

## Run and evaluation

Fine-tune the team from u20000 with `makespan_cost = 4`,
`makespan_setup_s = 30` and the new observation (a checkpoint without it is
migrated with zero input weights: the policy and value are unchanged, tested).
Evaluate at u22500, u25000, u27500 and u30000 against the single excavator on
the same 512 maps, with `compare.py --setup-s 0` and `--setup-s 30`. The old
recipe plateaued at about 1.52 between u15000 and u20000. An executed-plan
speedup clearly above that, at equal success, supports further investigation;
a matched old-reward continuation is needed to attribute it causally to the
new reward; the busier machine's share should move toward 0.5. The run record is
in `MULTI_AGENT_JOINT_POLICY_20260923.md`.

If balance stays poor: per-agent counterfactual credit (item 4) for the actor,
and the DARP-style split (item 6) as a baseline.

## September 27: smallest useful paper ablation

The cost4 run reached u30000. Its u22500 candidate improves modeled time but
loses held-out reliability: on the same 512 initial development episodes,
old u20000 versus cost4 u22500 succeeds 508/501 sampled and 499/484 greedy.
On common successes the cost4 candidate is about 8% faster with 30-second
setups. This is a speed/completion tradeoff, not a replacement for the old team.

Lorenzo requested fast results with limited compute and enough ablation evidence
for the paper. The next run is therefore **one cost0 continuation**, reusing
the existing cost4 u22500 checkpoint and evaluation panels. Both continue
native u20000 for 2,500 updates with workload observations enabled, setup30,
four GH200s x256 environments x32 rounds, seed20260915, gamma0.9984 and constant
entropy0.02. Model, Adam state and absolute schedules are preserved. The new
allocation is capped at3h, including sampled/greedy held-out evaluation; a
timeout evaluates the latest valid checkpoint and explicitly marks unequal
training budgets. A cost0/cost2 pair was prepared but superseded before submission.

This isolates the added makespan penalty from more training and the added
observations. It is one training seed on a development panel, not evidence of
seed robustness or a final untouched test. Report full-panel completion first,
paired success gains/losses, then time and workload balance on the fixed
common-success intersection. The old u20000 remains the reliability reference.
Defer another reward weight, a new critic, and heterogeneous-team training until
this comparison changes a decision. A balanced connected-area split with the
solo planner is the next useful baseline without new policy training.

The underlying reward remains material-progress shaping, success +6,
horizon failure -1, a total step cost of1 over the450-round horizon, and the
inherited local behavior costs. The makespan term is additional; its weight
is the only intervention in this comparison.

The large-batch CSCS solo evaluation has a separate completion discrepancy:
`dump_mask_integrity` reports0 although saved maps have no dump/obstacle overlap.
Do not use the invalid solo panel as a speedup denominator. The team-only
ablation uses the existing successful two-agent path; any evaluation workaround
must be explicit and checked independently.

That reference was recovered in GPU job4811857 by replacing the Boolean
not-any reduction with an equivalent int32 overlap count. Solo excavator now
succeeds503/512 sampled and505/512 greedy. Replaying old-team greedy with the
workaround gives exactly the same ten per-episode arrays as before, including
success, steps, soil and travel. Seven CPU completion-contract tests pass,
including rejection of obstructed declared dump cells. Local Terra fix04a64d98
does not alter the frozen training snapshot used for the control.

The repaired comparison gives modeled solo/team speedups1.563/1.766 for old
team/cost4 sampled, and1.556/1.757 greedy, with30-second setups, on fixed
three-policy common successes (n492/n473). Completion counts above remain the
primary qualification. Large-job sampled speedups are1.75/1.83, versus1.26/1.60
on small jobs; most cost4 reliability loss is in the small-job stratum. These
volume thirds are descriptive development-set analysis, not a trained fleet
selection rule. Report and raw panels are under `heldout_repaired/` below.

Control job4811962 is submitted with the3h cap; qualification4811852 passed
finite updates, saving, native resume and optimizer continuity in12m50s.
At07:22UTC on September27 it was pending with an11:40CEST start estimate.

Launch/evidence directory:
`/home/lorenzo/moleworks/.artifacts/terra_multiagent_parallel_20260927/`.

## Sources

- Zhang, Song, Cao, Zhang, Tan, Xu. Learning to Dispatch for Job Shop
  Scheduling via Deep Reinforcement Learning. NeurIPS 2020.
  https://arxiv.org/abs/2010.12367
- Graham. Bounds on Multiprocessing Timing Anomalies. SIAM Journal on Applied
  Mathematics 17(2):416–429, 1969. https://doi.org/10.1137/0117039
- Cao, Sun, Sartoretti. DAN: Decentralized Attention-based Neural Network for
  the MinMax Multiple Traveling Salesman Problem. https://arxiv.org/abs/2109.04205
- Park, Bakhtiyar, Park. ScheduleNet: Learn to solve multi-agent scheduling
  problems with reinforcement learning. https://arxiv.org/abs/2106.03051
- Son, Kim, Choi, Kim, Park. Equity-Transformer: Solving NP-Hard Min-Max
  Routing Problems as Sequential Generation with Equity Context. AAAI 2024.
  https://arxiv.org/abs/2306.02689
- Agogino, Tumer. Analyzing and visualizing multiagent rewards in dynamic and
  stochastic domains. Autonomous Agents and Multi-Agent Systems, 2008.
  https://doi.org/10.1007/s10458-008-9046-9
- Foerster et al. Counterfactual Multi-Agent Policy Gradients. AAAI 2018.
  https://doi.org/10.1609/aaai.v32i1.11794
- Zimmer, Glanois, Siddique, Weng. Learning Fair Policies in Decentralized
  Cooperative Multi-Agent Reinforcement Learning. ICML 2021.
  https://arxiv.org/abs/2012.09421
- Kapoutsis, Chatzichristofis, Kosmatopoulos. DARP: Divide Areas Algorithm for
  Optimal Multi-Robot Coverage Path Planning. Journal of Intelligent & Robotic
  Systems 86:663–680, 2017. https://doi.org/10.1007/s10846-016-0461-x
- Devlin, Kudenko. Theoretical considerations of potential-based reward
  shaping for multi-agent systems. AAMAS 2011.
  https://doi.org/10.5555/2030470.2030503
