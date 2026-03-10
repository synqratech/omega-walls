"""Tool gateway implementation (fail closed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from omega.interfaces.contracts_v1 import OffAction, ToolDecision, ToolRequest


_DEFAULT_CAPABILITIES = {
    "retrieval_readonly": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "summarize": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "echo": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "write_file": {"mode": "write", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
    "network_post": {"mode": "dangerous", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
}


@dataclass(frozen=True)
class ToolCapability:
    mode: str
    allowed_when: Set[str]
    requires_human_approval: bool


def _normalize_tool_mode(value: str) -> str:
    norm = str(value).strip().lower()
    if norm not in {"read_only", "write", "dangerous"}:
        raise ValueError(f"Unsupported tool capability mode: {value}")
    return norm


def _normalize_allowed_when(values: List[str] | None) -> Set[str]:
    raw = values or ["NO_OFF", "OFF"]
    out = {str(v).strip().upper() for v in raw}
    if not out:
        out = {"NO_OFF", "OFF"}
    if not out.issubset({"NO_OFF", "OFF"}):
        raise ValueError("allowed_when must contain only NO_OFF|OFF")
    return out


@dataclass
class ToolGatewayV1:
    config: Optional[Dict] = None

    def __post_init__(self) -> None:
        tools_cfg = (self.config or {}).get("tools", {})
        self.unknown_tool_policy = str(tools_cfg.get("unknown_tool_policy", "DENY")).upper()
        if self.unknown_tool_policy != "DENY":
            raise ValueError("tools.unknown_tool_policy must be DENY in v1")
        self.freeze_read_only_exception = bool(tools_cfg.get("freeze", {}).get("read_only_exception", True))

        raw_caps = tools_cfg.get("capabilities", {}) or _DEFAULT_CAPABILITIES
        self.capabilities: Dict[str, ToolCapability] = {}
        for tool_name, cap in raw_caps.items():
            cap_dict = cap or {}
            mode = _normalize_tool_mode(cap_dict.get("mode", "dangerous"))
            requires_human_approval = bool(cap_dict.get("requires_human_approval", False))
            if mode in {"write", "dangerous"} and not requires_human_approval:
                raise ValueError(f"Tool capability '{tool_name}' mode={mode} must require human approval")
            self.capabilities[str(tool_name)] = ToolCapability(
                mode=mode,
                allowed_when=_normalize_allowed_when(cap_dict.get("allowed_when")),
                requires_human_approval=requires_human_approval,
            )

    def _find_freeze(self, current_actions: List[OffAction]) -> Optional[OffAction]:
        for action in current_actions:
            if action.type == "TOOL_FREEZE":
                return action
        return None

    @staticmethod
    def _is_off_state(current_actions: List[OffAction]) -> bool:
        return any(action.type in {"SOFT_BLOCK", "HUMAN_ESCALATE", "SOURCE_QUARANTINE", "TOOL_FREEZE"} for action in current_actions)

    def find_freeze(self, current_actions: List[OffAction]) -> Optional[OffAction]:
        return self._find_freeze(current_actions)

    def is_off_state(self, current_actions: List[OffAction]) -> bool:
        return self._is_off_state(current_actions)

    def capability_for(self, tool_name: str) -> Optional[ToolCapability]:
        return self.capabilities.get(tool_name)

    def ensure_tool_coverage(self, tool_names: List[str]) -> None:
        missing = [name for name in tool_names if name not in self.capabilities]
        if missing:
            raise ValueError(f"Missing capability profiles for tools: {', '.join(sorted(missing))}")

    def enforce(self, request: ToolRequest, current_actions: List[OffAction]) -> ToolDecision:
        freeze_action = self._find_freeze(current_actions)
        off_state = self._is_off_state(current_actions)
        capability = self.capabilities.get(request.tool_name)

        if capability is None and self.unknown_tool_policy == "DENY":
            return ToolDecision(allowed=False, mode="TOOLS_DISABLED", reason="UNKNOWN_TOOL", logged=True)

        mode = freeze_action.tool_mode if freeze_action is not None else "TOOLS_DISABLED"
        cap_mode = capability.mode if capability is not None else "dangerous"

        if freeze_action is not None:
            mode = freeze_action.tool_mode or "TOOLS_DISABLED"
            if mode == "TOOLS_DISABLED":
                if self.freeze_read_only_exception and cap_mode != "read_only":
                    return ToolDecision(allowed=False, mode=mode, reason="TOOL_FREEZE_ACTIVE", logged=True)
                if not (self.freeze_read_only_exception and cap_mode == "read_only"):
                    return ToolDecision(allowed=False, mode=mode, reason="TOOL_FREEZE_ACTIVE", logged=True)
            elif mode == "TOOLS_ALLOWLIST":
                allowlist = freeze_action.allowlist or []
                if request.tool_name not in allowlist:
                    return ToolDecision(allowed=False, mode=mode, reason="NOT_IN_ALLOWLIST", logged=True)
                if self.freeze_read_only_exception and cap_mode != "read_only":
                    return ToolDecision(allowed=False, mode=mode, reason="TOOL_FREEZE_ACTIVE", logged=True)
            else:
                return ToolDecision(allowed=False, mode=mode, reason="POLICY_BLOCK", logged=True)

        if capability is not None:
            if off_state and "OFF" not in capability.allowed_when:
                return ToolDecision(allowed=False, mode=mode, reason="OFF_STATE_BLOCK", logged=True)
            if capability.requires_human_approval and not bool(request.args.get("human_approved", False)):
                return ToolDecision(allowed=False, mode=mode, reason="HUMAN_APPROVAL_REQUIRED", logged=True)

        return ToolDecision(allowed=True, mode=mode, reason="OK", logged=True)
