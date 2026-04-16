"""Control outcome normalization for policy/actions."""

from __future__ import annotations

from typing import Iterable, List

from omega.interfaces.contracts_v1 import OffAction


_OUTCOME_PRECEDENCE: List[str] = [
    "TOOL_FREEZE",
    "HUMAN_ESCALATE",
    "SOURCE_QUARANTINE",
    "SOFT_BLOCK",
    "REQUIRE_APPROVAL",
    "WARN",
]


def control_outcome_from_action_types(action_types: Iterable[str]) -> str:
    seen = {str(item).strip().upper() for item in action_types if str(item).strip()}
    for candidate in _OUTCOME_PRECEDENCE:
        if candidate in seen:
            return candidate
    return "ALLOW"


def control_outcome_from_actions(actions: Iterable[OffAction]) -> str:
    return control_outcome_from_action_types(action.type for action in actions)
