"""Shared constants/helpers for config validators."""

from __future__ import annotations

from typing import Tuple

WALLS_V1_ORDER: Tuple[str, str, str, str] = (
    "override_instructions",
    "secret_exfiltration",
    "tool_or_action_abuse",
    "policy_evasion",
)

