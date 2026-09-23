"""Start a jointly controlled team from a single-agent policy.

The team network is the single-agent network applied to every agent's view
plus the zero-output ``intent_decoder``. Single-agent training only ever saw
the acting agent in agent slot 0, so the fused-feature weights that read slots
1..3 never received gradient and still hold their random initialization.
Zeroing them makes the migrated team policy equal, on every view, to the
single-agent policy on the same inputs; teammates then enter through learning.
"""

import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze

from utils.models import validate_model_params_match

# Heads whose first layer reads the fused feature vector. Its leading block is
# the four agent-slot embeddings, acting agent first.
_FUSED_INPUT_HEADS = ("mlp_pi", "mlp_v", "actor_residual_head")
_AGENT_SLOTS = 4


def agent_embedding_width(params) -> int:
    """Output width of ``AgentStateNet`` (one agent slot of the fused vector)."""
    net = params["params"]["agent_state_net"]
    width = 0
    for mlp in ("mlp_one_hot", "mlp_two_hot", "mlp_continuous", "mlp_agent_type"):
        layers = net[mlp]
        last = max(layers, key=lambda name: int(name.rsplit("_", 1)[1]))
        layer = layers[last]
        width += int(layer["bias"].shape[0] if "bias" in layer else layer["layers_0"]["bias"].shape[0])
    return width


def _first_dense(head):
    layer = head[min(head, key=lambda name: int(name.rsplit("_", 1)[1]))]
    return layer if "kernel" in layer else layer["layers_0"]


def team_params_from_single_agent(single_params, team_params):
    """Function-preserving team parameters from a single-agent checkpoint."""
    params = unfreeze(single_params)
    team = unfreeze(team_params)
    if "intent_decoder" in params["params"]:
        raise ValueError("the checkpoint already controls a team")
    params["params"]["intent_decoder"] = team["params"]["intent_decoder"]
    width = agent_embedding_width(params)
    teammates = slice(width, _AGENT_SLOTS * width)
    for name in _FUSED_INPUT_HEADS:
        if name in params["params"]:
            dense = _first_dense(params["params"][name])
            dense["kernel"] = jnp.asarray(dense["kernel"]).at[teammates].set(0.0)
    validate_model_params_match(team, params, "team warm start")
    return params


def _adam_states(tree):
    return [
        node for node in jax.tree_util.tree_leaves(
            tree, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState)
        )
        if isinstance(node, optax.ScaleByAdamState)
    ]


def team_optimizer_state(single_opt_state, team_opt_state):
    """Carry the parent's Adam moments and step count over to the team.

    A fresh Adam moves every weight by about the learning rate in its first
    steps regardless of gradient scale, which disturbs a converged policy.
    Shared parameters keep their moments; the intent decoder starts at zero.
    """
    (parent,) = _adam_states(single_opt_state)

    def graft(fresh, saved):
        fresh, saved = unfreeze(fresh), unfreeze(saved)
        return {"params": {
            name: saved["params"][name] if name in saved["params"] else leaf
            for name, leaf in fresh["params"].items()
        }}

    def replace(node):
        if not isinstance(node, optax.ScaleByAdamState):
            return node
        return node._replace(
            count=parent.count,
            mu=graft(node.mu, parent.mu),
            nu=graft(node.nu, parent.nu),
        )

    return jax.tree_util.tree_map(
        replace, team_opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState)
    )
