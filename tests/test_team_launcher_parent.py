"""A specialist may seed a shared team; unsupported parent embodiments fail early."""

from types import SimpleNamespace

import pytest

from scripts.team.run import validate_parent_config


@pytest.mark.parametrize("agent_types", [(0,), (2,), None])
@pytest.mark.parametrize("action_types", [(0,), None])
def test_tracked_single_specialist_parent(agent_types, action_types):
    validate_parent_config(SimpleNamespace(
        agent_types_override=agent_types, action_types_override=action_types,
    ))


@pytest.mark.parametrize("agent_types,action_types", [
    ((0, 0), (0, 0)), ((2, 2), (0, 0)), ((0, 2), (0, 0)),
    ((1,), (0,)), ((0,), (1,)), ((2,), (1,)), ((2,), (0, 0)),
])
def test_unsupported_parent_rejected(agent_types, action_types):
    with pytest.raises(ValueError, match="single tracked excavator or skid steer"):
        validate_parent_config(SimpleNamespace(
            agent_types_override=agent_types, action_types_override=action_types,
        ))
