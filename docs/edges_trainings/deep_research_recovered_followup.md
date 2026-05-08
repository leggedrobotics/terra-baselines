Terra Edge-Digging RL Research Report
1. Problem Taxonomy

The new edge rule turns Terra from a mostly homogeneous gridworld digging task into a state-dependent feasible-action RL problem with a hidden or underexposed affordance structure. The primitive DO action is not uniformly meaningful: for core tiles it can remove target cells whenever the workspace cone covers them, but for edge tiles it only works when base proximity and cabin/arm alignment constraints are satisfied. In RL terms, this creates a variable valid-action set, rarely valid high-impact action effects, and many no-op or partial no-op transitions that are visually similar unless the observation exposes edge feasibility.

The resulting task is also naturally phase structured:

Core excavation: dense progress; dig feasibility mostly depends on workspace coverage.

Edge setup: navigation and orientation toward a legal boundary pose.

Edge finishing: sparse, precise, high-value progress.

Terminal completion: large, thresholded return changes near episode end.

This is close to a multi-task / hierarchical MDP with shared dynamics but different affordances, rewards, and value scales across phases. PPO with one unmasked categorical policy and one scalar value head must simultaneously learn core digging, edge alignment, and near-terminal completion. That combination explains the observed symptoms: lower completion rate, many ineffective DO samples, high-variance GAE targets, and local value spikes.

2. Relevant Literature and Why It Applies
Invalid action masking and state-dependent feasibility

Huang & Ontañón, “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms,” FLAIRS 2022
. Shows that invalid action masking gives theoretically valid policy-gradient updates and becomes important as the invalid-action fraction grows; directly relevant because Terra currently samples from an unmasked categorical even though the simulator has deterministic feasibility rules. 
Florida Journals

Huang’s invalid action masking blog
. Useful implementation-level explanation: invalid logits are masked before the categorical distribution, and gradients for invalid logits are zeroed under the masked distribution; relevant for ensuring PPO sampling, log-prob recomputation, and entropy all use the same mask. 
Costa Huang's Website

Zabounidis et al., “Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms,” arXiv 2026
. Identifies a failure mode where unmasked policy gradients suppress actions that are invalid in visited states but valid in rare unvisited states; this maps well to Terra’s edge dig actions, which are invalid/no-op in many states but critical near the boundary. 
Cool Papers

Stable-Baselines3 Contrib MaskablePPO documentation
. Not a paper, but a useful reference implementation: masks are used during rollout collection, action sampling, action evaluation, and masked evaluation; relevant because Terra must use the same mask in rollout and PPO update, not only at sampling time. 
Stable Baselines3 Contrib Docs

Zahavy et al., “Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning,” NeurIPS 2018
. Learns an action-elimination network from an external invalid-action signal; relevant if Terra later wants a learned feasibility predictor or auxiliary invalid-action classifier rather than relying only on simulator masks. 
NeurIPS Papers

PPO, GAE, and value instability

Schulman et al., “Proximal Policy Optimization Algorithms,” 2017
. PPO’s clipped surrogate is robust but still sensitive to rollout distribution, advantage scale, entropy, and value loss; relevant because masking changes the behavior policy and must be reflected consistently in PPO ratios. 
Hugging Face

Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” 2015
. GAE trades bias and variance through gamma and lambda; relevant because rare edge completions plus high terminal reward can make high-variance advantage targets and local value spikes. 
Hugging Face

Engstrom et al., “Implementation Matters in Deep RL: A Case Study on PPO and TRPO,” ICLR 2020
. Shows PPO behavior can change strongly due to implementation details; relevant for Terra because mask consistency, value clipping, reward scaling, shared-vs-separate networks, and timeout handling can dominate apparent algorithmic effects. 
OpenReview

Huang et al., “The 37 Implementation Details of Proximal Policy Optimization,” ICLR Blog Track 2022
. Useful engineering reference; relevant notes include value clipping, advantage normalization, entropy handling, reward scaling, separate actor/critic networks, and evaluation details. 
ICLR Blog Track

Pardo et al., “Time Limits in Reinforcement Learning,” ICML 2018
. Shows that treating time-limit truncations like true terminal states can cause state aliasing and instability; relevant if Terra episodes end by timeout and the critic receives abrupt zero/bootstrap targets at nonterminal states. 
Proceedings of Machine Learning Research

Value normalization, multi-task critics, and return scale

van Hasselt et al., “Learning Values Across Many Orders of Magnitude,” NeurIPS 2016
. Introduces adaptive target normalization for value learning; relevant because Terra’s dense excavation rewards and terminal completion rewards may live on different scales. 
NeurIPS Papers

Hessel et al., “Multi-task Deep Reinforcement Learning with PopArt,” AAAI 2019
. Extends PopArt to actor-critic multi-task RL and motivates scale-invariant value learning; relevant because core digging and edge finishing behave like tasks/phases with different return distributions. 
Google Research

Bellemare, Dabney & Munos, “A Distributional Perspective on Reinforcement Learning,” ICML 2017
. Argues that learning a return distribution can be more informative than learning only expected value; relevant as a research-heavy option if near-edge states have multimodal returns: timeout/failure versus large completion reward. 
Proceedings of Machine Learning Research

Goal conditioning, task conditioning, and successor features

Schaul et al., “Universal Value Function Approximators,” ICML 2015
. Conditions value functions on goals; relevant because Terra can condition the critic/policy on goal-like descriptors such as remaining core tiles, remaining edge tiles, active edge segment, or current phase. 
Proceedings of Machine Learning Research

Barreto et al., “Successor Features for Transfer in Reinforcement Learning,” NeurIPS 2017
. Decouples dynamics features from reward weights for transfer across related tasks; relevant if Terra has repeated foundation layouts where core, edge, and dump objectives share dynamics but differ in reward emphasis. 
NeurIPS Papers

Auxiliary losses and hidden feasibility

Jaderberg et al., “Reinforcement Learning with Unsupervised Auxiliary Tasks,” ICLR 2017
. Shows auxiliary prediction/control tasks can improve representation learning when extrinsic rewards are sparse; relevant for adding heads that predict edge legality, alignment error, proximity error, touched border direction, or next-step dig success. 
OpenReview

Zabounidis et al., 2026
, again, is especially relevant because it proposes feasibility classification to learn validity-discriminating features when masks are available during training but may not be available at deployment. In Terra, the simulator already computes the legality signal, so this is a low-cost auxiliary target. 
ResearchGate

Curriculum and rare endgame tasks

Narvekar et al., “Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey,” JMLR 2020
. Surveys sequencing tasks or samples to improve learning when the final task is too hard from scratch; relevant for staged edge constraints, easier border tolerances, and edge-state oversampling. 
Journal of Machine Learning Research

Florensa et al., “Reverse Curriculum Generation for Reinforcement Learning,” CoRL 2017
. Trains from states near the goal and gradually expands the start-state distribution; highly relevant because Terra’s hard part appears to be the rare endgame edge finishing phase. 
Proceedings of Machine Learning Research

Florensa et al., “Automatic Goal Generation for Reinforcement Learning Agents,” ICML 2018
. Generates goals at the right difficulty level; relevant if Terra later wants automatic generation of partial foundation states, edge-only tasks, or layout variants. 
Proceedings of Machine Learning Research

Andrychowicz et al., “Hindsight Experience Replay,” NeurIPS 2017
. Useful for sparse goal completion in off-policy settings; less directly compatible with on-policy PPO, but relevant if Terra introduces goal-conditioned off-policy training or relabels partial excavation goals. 
NeurIPS Papers

Oh et al., “Self-Imitation Learning,” ICML 2018
. Reuses past high-return trajectories to improve exploration; relevant because successful edge completions are rare and should not be discarded after one PPO update. 
Proceedings of Machine Learning Research

Hester et al., “Deep Q-learning from Demonstrations,” AAAI 2018
. Shows demonstrations can accelerate RL when early performance is poor; relevant if scripted or expert edge-finishing trajectories are available. 
AAAI

Hierarchical RL and options

Sutton, Precup & Singh, “Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning,” Artificial Intelligence 1999
. Introduces options as temporally extended actions; relevant because “navigate to edge,” “align cabin/arm,” and “dig edge segment” are natural options. 
ECE UVic

Bacon, Harb & Precup, “The Option-Critic Architecture,” AAAI 2017
. Learns option policies and termination functions end-to-end; relevant as a research-heavy route if hand-designed phases are too rigid. 
CiNii

Kulkarni et al., “Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation,” 2016
. Uses high-level goals and low-level controllers in sparse-reward settings; relevant for a high-level policy that chooses subgoals such as core excavation, repositioning, edge alignment, and edge digging. 
ResearchGate

Potential-based reward shaping

Ng, Harada & Russell, “Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping,” ICML 1999
. Establishes that shaping of the form F(s,a,s') = gamma * Phi(s') - Phi(s) preserves optimal policies under standard assumptions; relevant for shaping edge alignment/proximity progress without changing the intended optimum. 
UT Austin Computer Science

Wiewiora, “Potential-Based Shaping and Q-Value Initialization are Equivalent,” JAIR 2003
. Shows a link between potential shaping and value initialization; relevant because shaping may help exploration but should be treated as a learning aid, not as a hidden change to the task objective. 
CMU School of Computer Science

3. Main Diagnosis for Terra

The edge constraint likely caused a compound failure, not a single bug:

Unmasked policy sampling wastes probability mass. The categorical policy keeps sampling actions whose useful effect is impossible in the current state. For edge tiles, the invalid action is not just illegal; it can look like a plausible DO action but produce no or partial progress.

The critic sees aliased states. Without explicit edge feasibility features, two observations can look similar while one permits edge progress and the other produces no-op digging. This violates the practical assumptions needed for smooth value approximation.

Rare successful edge states create high-variance value targets. The critic must fit ordinary core progress, failed edge attempts, successful edge finishing, timeouts, and terminal bonuses with one scalar head.

The problem is phase-structured. Core excavation and edge finishing differ in preconditions, reward density, action meaning, and return scale.

The terminal reward may create sharp local discontinuities. If completion reward is thresholded/exponential and backfilled near alternating multi-agent turns, near-finish edge states can have value jumps that are not well explained by the current observation.

4. Recommendations
4.1 Actor and Action Masking
Recommendation A1: Add real invalid-action masking to PPO

Use a boolean valid-action mask in both rollout sampling and PPO update. For invalid actions, set logits to a large negative value before constructing the categorical distribution. Compute sampling, log-probs, entropy, KL, and PPO ratio from the same masked distribution.

For Terra, the first mask should include:

DO invalid when the action cannot remove any valid tile.

Edge-dig invalid when the workspace only touches edge tiles and edge proximity/alignment fails.

Movement invalid if blocked or impossible.

Cabin/arm actions invalid if they violate discrete constraints or have no effect, if applicable.

A guaranteed fallback action, such as NOOP or a safe wait, to avoid all-invalid states.

Why: Huang & Ontañón show masking is a valid policy-gradient operation and that penalizing invalid actions scales poorly when invalid actions are common. Terra has deterministic state-dependent invalidity, so a simulator-derived mask is exactly the setting where masking is preferable to learning invalidity through penalties. 
Florida Journals

Recommendation A2: Do not only mask during sampling

Masking only at action selection but recomputing PPO log-probs from unmasked logits creates a mismatch between the behavior policy and update policy. Store or recompute the same valid-action mask for PPO update. Use the masked categorical for old log-probs, new log-probs, entropy, and KL diagnostics.

Why: MaskablePPO implementations explicitly carry masks through rollout collection and action evaluation; this is necessary for consistent policy-gradient ratios. 
Stable Baselines3 Contrib Docs

Recommendation A3: Prefer masking over invalid-action penalties for hard deterministic constraints

Invalid/no-op penalties are still useful as diagnostics, but they should not be the main way to teach edge legality. With a rare edge action, penalties can suppress DO globally before the agent reaches states where edge DO is valid. This is the “valid action suppression” failure mode described by Zabounidis et al. 
Cool Papers

4.2 Observations and Auxiliary Losses
Recommendation O1: Expose edge affordance features

Add explicit features already available from the simulator:

local edge/core target channel;

local border band mask;

nearest border segment tangent and/or normal;

signed or absolute base-to-border distance;

arm/cabin-to-border alignment error;

boolean edge_dig_valid_now;

count of remaining core tiles and edge tiles;

fraction of edge tiles completed;

phase label: core_remaining, edge_remaining, edge_only, completion_ready;

valid-action mask, either as policy input or only as distribution mask.

Why: If feasibility is hidden, the agent must infer a geometric predicate from sparse no-op outcomes. Auxiliary and explicit affordance signals reduce state aliasing and make the critic’s target smoother.

Recommendation O2: Add an auxiliary edge-feasibility prediction head

Train a supervised head from simulator labels:

can_DO_remove_any_tile;

can_DO_remove_edge_tile;

edge_alignment_ok;

edge_proximity_ok;

nearest edge direction class;

predicted number of tiles removed by DO.

This can share the encoder with actor and critic but should have a small loss coefficient. It should be logged separately from PPO losses.

Why: UNREAL supports auxiliary tasks as a way to improve representation learning under sparse extrinsic reward, and the newer action-validity literature specifically motivates feasibility classification for state-dependent action validity. 
OpenReview
+1

Recommendation O3: Include remaining time or timeout status

If the episode has a hard time limit, expose normalized remaining time or step budget. If timeout is only a training truncation and not part of the task, bootstrap through timeout rather than treating it as terminal.

Why: Pardo et al. show that mishandling time limits can cause state aliasing and training instability. 
Proceedings of Machine Learning Research

4.3 Critic and Value Stabilization
Recommendation C1: Add phase-conditioned value prediction

Start simple:

Add phase/progress features to the critic input.

Then test two or three value heads selected by phase:

core excavation;

edge setup / edge finishing;

terminal/completion-ready.

Keep the actor head shared initially.

Why: A single value head is fitting multiple return regimes. UVFA-style conditioning supports value functions that condition on task/goal descriptors, and PopArt/multi-task RL supports handling tasks with different value scales. 
Proceedings of Machine Learning Research
+1

Recommendation C2: Separate actor and critic trunks or reduce trunk sharing

The current shared trunk can force policy and value gradients to compete. Test:

shared trunk baseline;

separate MLP heads only;

partially separate late layers;

fully separate actor and critic encoders if compute allows.

Why: PPO implementation studies show architecture and “small” implementation choices can materially affect PPO behavior. This matters when critic gradients are large or noisy. 
ICLR
+1

Recommendation C3: Normalize or rescale value targets

Test:

reward scaling;

return normalization;

PopArt value normalization;

lower terminal reward magnitude with equivalent success ranking;

separate logging of dense return and terminal return.

Why: PopArt was designed for value targets with changing or very different scales, and multi-task PopArt specifically addresses actor-critic learning across tasks with different reward scales. 
NeurIPS Papers
+1

Recommendation C4: Make value loss less brittle

Test:

lower vf_coef, especially if currently high;

Huber value loss instead of MSE;

smaller value clipping range than policy clip range;

value target clipping only as an ablation;

monitor explained variance by phase, not only globally.

Why: Value spikes can come from rare high-return states, large terminal jumps, and one value head overfitting local targets. These are not fixed by masking alone.

Recommendation C5: Tune GAE for the edge phase

Run a small sweep over:

gamma: current high value versus slightly lower values;

gae_lambda: 0.90, 0.95, 0.97;

terminal reward scale;

timeout bootstrapping.

Why: GAE explicitly trades bias and variance. Rare edge rewards and long horizons may need a different bias-variance setting than core excavation. 
Hugging Face

Recommendation C6: Consider distributional critic only after simpler fixes

A distributional critic can represent multimodal outcomes near the edge: timeout/fail versus successful terminal bonus. This is research-heavy for PPO and should come after masking, observation, and phase-conditioned critic ablations.

Why: Distributional RL motivates modeling the full return distribution rather than only the mean, but transferring this cleanly into your PPO code is more invasive. 
Proceedings of Machine Learning Research

4.4 Reward Shaping
Recommendation R1: Use potential-based shaping for edge progress

Use shaping of the form:

F(s, a, s') = gamma * Phi(s') - Phi(s)

Candidate potentials:

Phi_edge_count(s)      = - remaining_edge_tiles(s)
Phi_core_count(s)      = - remaining_core_tiles(s)
Phi_proximity(s)       = - min_distance_to_legal_edge_pose(s)
Phi_alignment(s)       = - min_alignment_error_to_relevant_edge(s)
Phi_legal_pose(s)      = + indicator(edge_DO_valid_now)
Phi_completion(s)      = weighted progress toward all tiles removed

A practical Terra potential can combine edge terms:

Phi(s) =
  w1 * normalized_completed_edge_tiles
- w2 * normalized_base_to_edge_distance
- w3 * normalized_alignment_error
+ w4 * indicator(edge_dig_valid_now)

Why: Potential-based shaping preserves the optimal policy under standard assumptions, unlike arbitrary dense rewards that may teach the robot to “pose nicely” without finishing the edge. 
UT Austin Computer Science

Recommendation R2: Keep shaping separate from evaluation

Report both:

environment return without shaping;

training return with shaping;

task success/completion;

edge completion.

Do not select models only on shaped return.

Recommendation R3: Avoid permanent penalties for failed edge DO as the main signal

A small diagnostic penalty for no-op may reduce dithering, but using penalties to teach edge feasibility can suppress rare valid edge digging. Masking plus feasibility features is safer.

4.5 Curriculum and Data Generation
Recommendation K1: Build an edge-finishing reset curriculum

Create reset distributions with partially completed foundations:

only one edge segment remains;

one side edge remains;

corners remain;

all core done, all edge remains;

mixed core and edge remaining;

full task.

Train with a mixture of full episodes and curriculum resets. Anneal toward full episodes only after edge success is reliable.

Why: Reverse curriculum starts near the goal and expands outward, which matches Terra’s rare endgame edge-completion problem. 
Proceedings of Machine Learning Research

Recommendation K2: Stage edge constraint hardness

Use a staged schedule:

no edge alignment constraint;

soft proximity/alignment reward but no hard mask;

hard mask with generous tolerance;

hard mask with target tolerance;

randomized tolerance/layouts.

This should be an experimental curriculum, not the final evaluation setting.

Why: Curriculum RL supports sequencing easier tasks before the final task when learning from scratch is too sparse. 
Journal of Machine Learning Research

Recommendation K3: Oversample rare edge states during training diagnostics

For on-policy PPO, this means reset-state sampling rather than replay. Oversample states where:

core is mostly complete;

edge tiles remain;

the agent is near but misaligned;

a legal edge dig is one or two actions away;

successful completion is possible within a short horizon.

Recommendation K4: Add demonstrations if scripted edge behavior exists

Even a small set of scripted edge-align-and-dig trajectories can be used for behavior cloning warm start or auxiliary imitation loss. Keep it as a separate ablation.

Why: Demonstration methods reduce poor early exploration, and Terra’s edge preconditions are easy for a planner or scripted controller to demonstrate. 
AAAI

Recommendation K5: Use self-imitation for rare successful completions

Store successful edge-finishing trajectories and add an auxiliary imitation or advantage-weighted behavior cloning loss. This is more invasive than reset curricula but can help once success is nonzero.

Why: Self-Imitation Learning is designed to exploit past good trajectories to improve exploration and performance. 
Proceedings of Machine Learning Research

4.6 Hierarchical / Options Approach
Recommendation H1: Start with hand-defined options, not end-to-end option discovery

A practical hierarchy for Terra:

core_dig_option: dig reachable core tiles;

navigate_to_edge_option: move base near a selected edge segment;

align_to_edge_option: rotate cabin/arm to match segment direction;

edge_dig_option: execute legal edge DO;

dump_option: handle dumping if needed.

The high-level policy chooses option/subgoal; the low-level controller executes primitive actions for a short horizon or until termination.

Why: The options framework formalizes temporally extended actions, and Terra’s edge finishing has clear subgoal boundaries. 
ECE UVic

Recommendation H2: Learn high-level selection after primitive masking works

Do not use hierarchy to hide missing feasibility information. Options still need edge affordance observations and valid-action masks. Otherwise, the low-level edge option will fail for the same reason the flat policy fails.

Recommendation H3: Try Option-Critic only as a later research track

Option-Critic can learn options and terminations end-to-end, but it adds instability and interpretability burden. It is less likely to be the fastest fix than masks, affordance observations, and curricula.

Why: Option-Critic is powerful but more complex; Terra already has obvious semantic phases, so hand-specified options are the lower-risk first step. 
CiNii

5. Prioritized Experiment Plan
Phase 0: Diagnostics Before Changing Learning
Goal

Confirm whether the edge rule creates invalid/no-op action concentration and phase-specific value errors.

Add logs

DO_selected_count

DO_effective_count

DO_noop_count

DO_partial_noop_count

edge_tiles_touched_by_workspace

edge_tiles_removed

core_tiles_removed

edge_alignment_ok

edge_proximity_ok

edge_dig_valid_now

valid-action count per state

invalid action probability mass under unmasked policy

entropy over all actions and entropy over valid actions

value prediction by phase

TD error / GAE advantage by phase

explained variance by phase

timeout vs true terminal

completion percentage at episode end

edge completion percentage at episode end

Success criterion

You can answer: “Is PPO failing because it never reaches edge-valid states, because it reaches them but does not choose DO, or because the critic destabilizes after rare successes?”

Phase 1: Quick Low-Risk Changes
Experiment 1.1: Real PPO action masking

Change: Implement valid-action masking for sampling and PPO update.

Ablations:

baseline unmasked PPO;

mask only DO;

mask all deterministic invalid primitive actions;

mask with and without no-op fallback.

Expected improvement:

fewer no-op digs;

higher edge progress;

lower invalid probability mass;

cleaner policy entropy;

less wasted exploration.

Failure modes:

mask is too strict and removes useful exploratory actions;

all actions masked in some states;

training/eval mismatch if eval does not use masks;

PPO ratio/log-prob mismatch if masks differ between rollout and update.

Primary metrics:

completion rate;

edge completion rate;

DO_effective / DO_selected;

invalid action probability mass;

KL and entropy over masked distribution;

value spikes by phase.

Experiment 1.2: Expose simple edge affordance features

Change: Add low-dimensional features:

remaining core count;

remaining edge count;

edge/core phase;

edge_dig_valid_now;

alignment error;

base-to-edge distance.

Ablations:

mask only;

mask + counts;

mask + counts + alignment/proximity;

mask + full edge affordance features.

Expected improvement:

critic value becomes smoother around edge states;

policy learns setup behavior faster;

fewer near-edge dithering loops.

Failure modes:

features leak too much task-specific geometry and reduce generalization;

feature normalization wrong;

phase label too coarse and causes brittle switching.

Primary metrics:

edge setup success: reaches valid edge pose;

mean steps from edge-only state to completion;

value explained variance in edge phase;

alignment/proximity error histograms.

Experiment 1.3: Reward/terminal diagnostics and scale sweep

Change: Keep reward semantics but sweep terminal reward scale and value coefficient.

Ablations:

current terminal reward;

terminal reward scaled down;

terminal reward separated from dense reward for logging;

vf_coef lower values;

Huber value loss.

Expected improvement:

fewer value spikes;

lower critic loss outliers;

less policy collapse after rare completions.

Failure modes:

terminal reward too small and agent optimizes dense digging without finishing;

lower value coefficient increases policy-gradient variance.

Primary metrics:

max/percentile value error;

value target distribution by phase;

success rate;

final completion percentage.

Phase 2: Medium Architecture and Curriculum Changes
Experiment 2.1: Phase-conditioned critic

Change: Add phase/progress inputs to critic, then test multi-head value prediction.

Ablations:

one value head, no phase;

one value head with phase features;

two heads: core vs edge;

three heads: core, edge setup, completion-ready;

PopArt or return normalization.

Expected improvement:

lower critic error in edge phase;

fewer small value spikes;

better advantage estimates near terminal states.

Failure modes:

phase boundaries create discontinuities;

wrong phase selection causes bad bootstrap;

multi-head critic has less data per head.

Primary metrics:

value explained variance per phase;

TD-error percentiles per phase;

policy update KL after edge successes;

completion and edge completion rate.

Experiment 2.2: Edge reset / reverse curriculum

Change: Add reset states near edge completion and train with mixed start distribution.

Curriculum schedule:

edge-only, one segment remaining;

edge-only, one side remaining;

edge-only, all border remaining;

mixed core/edge partially dug;

full foundation.

Expected improvement:

nonzero edge completion early;

faster learning of alignment/proximity sequence;

reduced dependence on rare full-episode successes.

Failure modes:

policy overfits to artificial reset states;

agent performs edge finishing only when core is already done;

distribution shift when returning to full episodes.

Primary metrics:

success from each reset bucket;

full-episode success;

edge completion conditional on core completion;

number of valid edge poses reached per episode.

Experiment 2.3: Auxiliary feasibility prediction

Change: Add supervised auxiliary heads from simulator labels.

Targets:

valid action mask;

DO removes core/edge/any tile;

edge proximity ok;

edge alignment ok;

touched edge direction.

Expected improvement:

better shared representation;

improved edge validity prediction;

lower critic aliasing.

Failure modes:

auxiliary loss dominates PPO;

feasibility prediction is perfect but policy still ignores edge due to reward scale;

learned features overfit to current map generation.

Primary metrics:

auxiliary accuracy/F1;

calibration of edge_dig_valid_now;

policy probability of setup actions near edge;

value error before/after valid edge pose.

Phase 3: Research-Heavy Options
Experiment 3.1: Potential-based edge shaping

Change: Add potential-based shaping for edge distance/alignment/progress.

Ablations:

no shaping;

edge tile count potential;

alignment/proximity potential;

combined potential;

non-potential dense shaping as a negative control.

Expected improvement:

faster edge setup;

fewer random-walk alignment attempts;

higher edge completion without changing final evaluation objective.

Failure modes:

shaping weights too large and dominate terminal objective;

agent learns to hover near legal poses without digging;

potential is not Markov under current observation.

Primary metrics:

unshaped return;

shaped return;

edge completion;

time spent valid-but-not-digging;

alignment/proximity progress curves.

Experiment 3.2: Hand-designed options

Change: Add high-level choices for core_dig, go_to_edge, align_edge, dig_edge.

Expected improvement:

shorter effective horizon;

clearer credit assignment;

better endgame behavior.

Failure modes:

bad option termination;

high-level policy selects edge option too early or too late;

low-level option inherits feasibility blindness.

Primary metrics:

option selection frequency by phase;

option success rate;

option duration;

edge completion after entering edge option;

full task success.

Experiment 3.3: Self-imitation or demonstrations

Change: Use successful edge completion trajectories for imitation.

Expected improvement:

rare successes become reusable learning signal;

faster stabilization after first successes.

Failure modes:

imitates suboptimal long trajectories;

reduces exploration;

stale demonstrations conflict with changing policy distribution.

Primary metrics:

success after first successful trajectory;

action KL to demonstration on edge states;

full task success;

edge finishing time.

6. Recommended First Ablation Matrix

Use a fixed set of maps/seeds and report mean plus confidence intervals.

ID	Mask	Edge obs	Critic	Curriculum	Shaping	Purpose
B0	no	no	current	no	no	current baseline
A1	DO only	no	current	no	no	isolate invalid edge dig masking
A2	all invalid primitives	no	current	no	no	full action-mask effect
O1	all	simple counts/phase	current	no	no	test phase observability
O2	all	full affordance	current	no	no	test hidden feasibility
C1	all	full affordance	phase-conditioned value	no	no	test critic aliasing
C2	all	full affordance	PopArt/normalized value	no	no	test value scale
K1	all	full affordance	best critic	edge resets	no	test rare endgame data
R1	all	full affordance	best critic	edge resets	potential	test shaping
H1	all	full affordance	best critic	edge resets	optional	test hierarchy

Run B0, A1, A2, O1, O2 first. Do not start hierarchy before A2/O2.

7. Metrics That Decide Whether Each Idea Helped
Completion metrics

full task completion rate;

final completion percentage;

final edge completion percentage;

success conditional on core completion;

success from edge-only reset states.

Action feasibility metrics

valid-action set size distribution;

probability mass assigned to invalid actions before masking;

DO_noop_rate;

DO_partial_noop_rate;

legal edge DO opportunities per episode;

probability of selecting DO when edge DO is legal.

Geometry metrics

base-to-nearest-edge distance;

arm/cabin alignment error;

time spent near edge but misaligned;

time spent aligned but too far;

time spent in valid edge pose before digging.

Critic metrics

value explained variance globally and by phase;

value prediction histograms by phase;

TD-error and GAE-advantage percentiles by phase;

critic loss outliers;

value change before/after terminal reward;

value error at timeout versus true terminal.

PPO metrics

masked entropy;

unmasked entropy, for diagnostics only;

approximate KL;

clip fraction;

policy loss;

value loss;

gradient norm;

fraction of minibatches dominated by edge-reset samples.

8. Expected Outcomes

The highest expected return-on-effort is:

Real action masking. This directly fixes sampling from invalid/no-op actions.

Explicit edge affordance observations. This fixes hidden feasibility and critic aliasing.

Phase-conditioned critic. This addresses mixed return distributions.

Reverse/edge curriculum. This makes edge completion frequent enough to learn.

Potential-based shaping. This helps exploration if kept policy-invariant and evaluated unshaped.

Options/hierarchy. This is likely useful, but only after the primitive edge affordance problem is solved.

9. Caveats

Invalid action masking assumes the mask is correct. If the mask removes actions that can produce useful progress, it biases learning by construction.

Masking can reduce exploration over invalid actions, which is good for hard physical constraints but bad if “invalid” really means “currently not useful but sometimes informative.”

Potential-based shaping preserves optimal policies under standard MDP assumptions, but only if the potential is a function of Markov state. If Terra’s observation omits edge geometry, shaping may be policy-invariant for the simulator state but not for the agent’s observation.

Reverse curricula require reset states that are physically and distributionally plausible. If reset states are unrealistic, the agent may learn edge finishing that does not transfer to full episodes.

Multi-head critics can reduce interference but can also fragment data. Use phase-conditioned single-head critic before a full multi-head design.

Distributional critics and learned options are plausible research directions, but they are not the first fix for a hidden deterministic feasibility predicate.

Literature on games and Atari does not transfer perfectly to deterministic gridworld excavation. The strongest transfer is conceptual: variable action sets, sparse endgame success, state aliasing, and value-scale mismatch.
