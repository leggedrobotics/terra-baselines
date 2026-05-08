# Terra edge-constraint RL notes

Date: 2026-05-08

Browser Deep Research Oracle did not produce a usable report:
- `terra-edge-rl-research` timed out before completion.
- `terra-edge-rl-research-retry` completed, but the captured answer was minified
  web-app JavaScript, not a research report.
- `terra-edge-literature-compact` was retried with the reusable manual-login
  Chrome profile and a tighter prompt. It also completed, but again saved the
  Deep Research React/web-app JavaScript instead of the report text.
- The completed Deep Research conversation was then reopened and a plain
  follow-up asked ChatGPT to paste/reconstruct the report body as normal
  Markdown. That succeeded.
- Recovered output path:
  `.codex_tmp/terra_edge_rl_research/deep_research_recovered_followup.md`.

This note is a manual, source-backed fallback based on the current `terra` and
`terra-baselines` code on `multi-agent`.

Repo-attached Oracle feedback:
- `terra-source-edge-audit` succeeded in normal browser Pro mode with source and
  config files from both `terra` and `terra-baselines`.
- Output path:
  `.codex_tmp/terra_edge_rl_research/oracle_repo_edge_review_manual_login.md`.
- Its highest-priority findings match this note: wire real action masks into PPO,
  fix existing action-mask bugs before using them, disable/fix mixed terminal
  reward backfill, expose legal edge-dig affordances, and add phase features or
  simple gated value heads before heavier architecture changes.

Literature Oracle fallback:
- Since browser Deep Research repeatedly saved UI JavaScript instead of the
  report, a normal browser Pro literature-review fallback was run with the brief
  and this note attached.
- Output path:
  `.codex_tmp/terra_edge_rl_research/oracle_literature_pro_fallback.md`.
- That report adds references for action elimination, action-space shaping,
  PPO implementation details, time-limit handling, MAPPO-style value
  normalization, demonstrations/self-imitation, MAXQ, and practical
  phase-conditioned critic diagnostics.

## Local diagnosis

The edge rule turns digging into a state-dependent feasible-action problem:

- `terra/terra/config.py:204` enables `enforce_foundation_border_alignment`.
- `terra/terra/state.py:1229` computes a foundation border alignment mask from
  touched border segment, arm angle, and base proximity.
- `terra/terra/state.py:2022` and `terra/terra/state.py:2065` apply that mask to
  the dig mask.
- `terra/terra/state.py:3538` says `action_mask` is informational only, and
  `terra/terra/state.py:3541` returns all-zero action masks.
- `terra-baselines/utils/utils_ppo.py:49` samples from
  `tfp.distributions.Categorical(logits=logits_pi)` with no legality mask.

The critic is also asked to fit a mixed-return problem with one scalar head:

- `terra-baselines/utils/models.py:430` defines one value MLP and one policy MLP.
- `terra-baselines/utils/models.py:525` returns a single `v` and `xpi`.
- `terra-baselines/train.py:41` uses `clip_eps=0.5`.
- `terra-baselines/train.py:45` uses `vf_coef=5.0`.
- `terra-baselines/train_mixed.py:875` spreads terminal bonus backward over the
  last alternating agent steps, which can make near-terminal edge states much
  higher variance than ordinary core excavation states.

The most likely failure mode is therefore not just "edge constraints are hard".
It is a combination of:

1. The simulator has a hard deterministic edge feasibility rule.
2. The policy is not constrained by that feasibility rule at sampling/update time.
3. The observation lacks explicit edge feasibility/proximity/alignment features.
4. A single value head must model both dense core excavation and rare high-return
   edge-completion states.

## Literature-grounded recommendations

### 1. Add real invalid-action masking first

Huang and Ontanon show that invalid action masking is theoretically valid for
policy gradients and empirically important as invalid action count grows:
https://arxiv.org/abs/2006.14171

More recent work on unmasked policy gradients argues that state-dependent action
validity can suppress valid actions under shared parameters, and that action
masking avoids this class of failure:
https://arxiv.org/abs/2603.09090

For Terra, implement a real action mask that reflects movement/cabin/dig legality
and pass it into PPO before constructing the categorical distribution:

- Add `-inf` or a very large negative value to invalid logits before sampling.
- Use the same masked logits for log-probs and entropy in PPO updates.
- Keep at least one valid fallback action, e.g. no-op, to avoid all-masked states.
- Log invalid dig attempts and mask cardinality per phase.

This should be the first experiment because it directly addresses the current
code mismatch.

### 2. Expose edge affordance features to the model

Masking alone prevents bad sampling, but it does not necessarily teach the model
why an action is valid. The 2026 masking paper explicitly motivates feasibility
classification as an auxiliary representation target:
https://arxiv.org/abs/2603.09090

The older UNREAL auxiliary-task result is not about action masks specifically,
but it supports the broader pattern of using auxiliary control/prediction losses
to make sparse-reward RL learn useful representations faster:
https://arxiv.org/abs/1611.05397

For Terra, add observation channels or low-dimensional features for:

- border/core tile type in local target map,
- nearest border segment direction,
- base-to-border signed/absolute distance,
- arm-to-border alignment error,
- boolean "current dig cone has legal edge tile",
- phase label such as no-border-left / border-present / completion-ready.

Optionally add an auxiliary head predicting edge dig legality from observation.

### 3. Separate or condition value prediction by phase

The code currently has one scalar value head for qualitatively different modes:
core excavation, maneuvering to boundary, and edge finishing. Multi-task value
normalization work motivates scale-aware value prediction when one model spans
different return distributions:
https://arxiv.org/abs/1809.04474

Universal value functions motivate conditioning value prediction on goal/task
descriptors:
https://proceedings.mlr.press/v37/schaul15.html

For Terra, compare these in order:

- Add phase features into the critic input.
- Add small separate value heads for core vs edge phase, selected by a known
  phase indicator.
- Add PopArt or return normalization if value targets have large scale shifts.
- Reduce `vf_coef` and/or value clip range if critic loss dominates policy.

### 4. Use a targeted edge curriculum

Curriculum RL surveys frame task sequencing as a standard way to speed learning
on hard target tasks:
https://www.jmlr.org/papers/volume21/20-212/20-212.pdf

Reverse curriculum generation is particularly relevant when the hard part is the
end of the episode:
https://arxiv.org/abs/1707.05300

For Terra, add reset distributions that start from partially dug foundations:

- edge-only finishing states,
- one side remaining,
- corners remaining,
- full task with increasing edge width/tolerance strictness.

Train or mix these with full episodes until edge completion is no longer rare.

### 5. Consider a hierarchical edge-finishing option after masking works

The options framework formalizes temporally extended actions:
https://www.ece.uvic.ca/~bctill/papers/learning/Sutton_etal_1999.pdf

Option-Critic learns option policies and terminations end-to-end:
https://arxiv.org/abs/1609.05140

For Terra, a practical hierarchy would be:

- high level: choose core dig / reposition to edge / align along edge / dig edge,
- low level: current primitive action policy or short scripted primitives.

Do this after action masks and edge observations, because hierarchy will not fix
hidden legality by itself.

### 6. Use policy-invariant shaping for edge progress

Potential-based reward shaping preserves optimal policies when shaping is the
difference of a potential function over states:
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/potential_based_shaping_equals_initialization.pdf

For Terra, candidate potentials:

- negative remaining edge tile count,
- negative minimum alignment/proximity error for remaining edge tiles,
- progress toward a legal edge dig pose.

Keep shaping diagnostic and ablated; it should guide exploration without making
the final objective ambiguous.

## First experiment set

1. Wire real action masks into PPO and log invalid-action rates.
2. Add edge-affordance observation features/channels already computed in
   `terra/terra/state.py`.
3. Add a small phase-conditioned critic or two value heads.
4. Add edge-finishing curriculum resets and evaluate completion rate separately
   for core vs edge remaining states.
5. Only then tune hierarchy/options or more aggressive reward shaping.
