"""Tool gateway implementation (fail closed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from omega.interfaces.contracts_v1 import OffAction, ToolDecision, ToolRequest
from omega.tools.arg_validation import ToolArgValidationConfig, validate_tool_args


_DEFAULT_CAPABILITIES = {
    "retrieval_readonly": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "summarize": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "echo": {"mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "write_file": {"mode": "write", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
    "network_post": {"mode": "dangerous", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
}
_VALID_CAPABILITY_CLASSES = {"READ_SAFE", "WRITE_LOW_RISK", "DESTRUCTIVE", "EXEC_DEPLOY", "PRIV_ESC"}
_DEFAULT_RISK_BY_CLASS = {
    "READ_SAFE": "low",
    "WRITE_LOW_RISK": "medium",
    "DESTRUCTIVE": "high",
    "EXEC_DEPLOY": "high",
    "PRIV_ESC": "critical",
}


@dataclass(frozen=True)
class ToolCapability:
    mode: str
    allowed_when: Set[str]
    requires_human_approval: bool
    capability_class: str
    risk_level: str


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


def _default_capability_class(mode: str) -> str:
    if mode == "read_only":
        return "READ_SAFE"
    if mode == "write":
        return "WRITE_LOW_RISK"
    return "EXEC_DEPLOY"


def _normalize_capability_class(value: str, mode: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        normalized = _default_capability_class(mode)
    if normalized not in _VALID_CAPABILITY_CLASSES:
        raise ValueError(f"Unsupported capability_class: {value}")
    return normalized


def _normalize_risk_level(value: str, capability_class: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = _DEFAULT_RISK_BY_CLASS.get(capability_class, "medium")
    return normalized


def _as_upper_set(values: List[str] | None) -> Set[str]:
    return {str(x).strip().upper() for x in list(values or []) if str(x).strip()}


def _as_lower_set(values: List[str] | None) -> Set[str]:
    return {str(x).strip().lower() for x in list(values or []) if str(x).strip()}


@dataclass
class ToolGatewayV1:
    config: Optional[Dict] = None

    def __post_init__(self) -> None:
        tools_cfg = (self.config or {}).get("tools", {})
        self.unknown_tool_policy = str(tools_cfg.get("unknown_tool_policy", "DENY")).upper()
        if self.unknown_tool_policy != "DENY":
            raise ValueError("tools.unknown_tool_policy must be DENY in v1")
        self.freeze_read_only_exception = bool(tools_cfg.get("freeze", {}).get("read_only_exception", True))
        self.arg_validation = ToolArgValidationConfig.from_tools_config(tools_cfg.get("arg_validation", {}))
        autonomy_cfg = tools_cfg.get("autonomy_soft", {}) or {}
        self.autonomy_soft_enabled = bool(autonomy_cfg.get("enabled", False))
        self.environment_tag = str(autonomy_cfg.get("environment_tag", "dev")).strip().lower()
        self.prod_tags = {
            str(x).strip().lower()
            for x in list(autonomy_cfg.get("prod_tags", ["prod", "production"]))
            if str(x).strip()
        }
        self.approval_classes_prod = {
            str(x).strip().upper()
            for x in list(
                autonomy_cfg.get(
                    "require_approval_classes_prod",
                    ["DESTRUCTIVE", "EXEC_DEPLOY", "PRIV_ESC"],
                )
            )
            if str(x).strip()
        }
        self.approval_classes_non_prod = {
            str(x).strip().upper()
            for x in list(autonomy_cfg.get("require_approval_classes_non_prod", []))
            if str(x).strip()
        }
        self.tool_allowlist = _as_lower_set(autonomy_cfg.get("tool_allowlist", []))
        self.tool_denylist = _as_lower_set(autonomy_cfg.get("tool_denylist", []))
        self.class_allowlist = _as_upper_set(autonomy_cfg.get("class_allowlist", []))
        self.class_denylist = _as_upper_set(autonomy_cfg.get("class_denylist", []))
        env_overrides = autonomy_cfg.get("env_overrides", {}) or {}
        self.autonomy_env_overrides: Dict[str, Dict] = {}
        if isinstance(env_overrides, dict):
            for env_name, env_cfg in env_overrides.items():
                if not isinstance(env_cfg, dict):
                    continue
                self.autonomy_env_overrides[str(env_name).strip().lower()] = dict(env_cfg)
        self.enforce_capability_requires_human_approval = bool(
            autonomy_cfg.get("enforce_capability_requires_human_approval", not self.autonomy_soft_enabled)
        )
        backup_cfg = autonomy_cfg.get("backup_safety", {}) or {}
        self.backup_safety_enabled = bool(backup_cfg.get("enabled", False))
        self.backup_precondition_required = bool(backup_cfg.get("require_immutable_backup_precondition", False))
        self.backup_precondition_ready = bool(backup_cfg.get("immutable_backup_ready", True))
        self.backup_sensitive_classes = {
            str(x).strip().upper()
            for x in list(backup_cfg.get("sensitive_classes", ["DESTRUCTIVE"]))
            if str(x).strip()
        }

        raw_caps = tools_cfg.get("capabilities", {}) or _DEFAULT_CAPABILITIES
        self.capabilities: Dict[str, ToolCapability] = {}
        for tool_name, cap in raw_caps.items():
            cap_dict = cap or {}
            mode = _normalize_tool_mode(cap_dict.get("mode", "dangerous"))
            requires_human_approval = bool(cap_dict.get("requires_human_approval", False))
            capability_class = _normalize_capability_class(cap_dict.get("capability_class", ""), mode)
            risk_level = _normalize_risk_level(cap_dict.get("risk_level", ""), capability_class)
            if (
                mode in {"write", "dangerous"}
                and not requires_human_approval
                and self.enforce_capability_requires_human_approval
            ):
                raise ValueError(f"Tool capability '{tool_name}' mode={mode} must require human approval")
            self.capabilities[str(tool_name)] = ToolCapability(
                mode=mode,
                allowed_when=_normalize_allowed_when(cap_dict.get("allowed_when")),
                requires_human_approval=requires_human_approval,
                capability_class=capability_class,
                risk_level=risk_level,
            )

    def _is_prod_environment(self) -> bool:
        return self.environment_tag in self.prod_tags

    def _env_override(self) -> Dict:
        return self.autonomy_env_overrides.get(self.environment_tag, {}) or {}

    def _effective_tool_allowlist(self) -> Set[str]:
        env_cfg = self._env_override()
        env_set = _as_lower_set(env_cfg.get("tool_allowlist", []))
        return self.tool_allowlist.union(env_set)

    def _effective_tool_denylist(self) -> Set[str]:
        env_cfg = self._env_override()
        env_set = _as_lower_set(env_cfg.get("tool_denylist", []))
        return self.tool_denylist.union(env_set)

    def _effective_class_allowlist(self) -> Set[str]:
        env_cfg = self._env_override()
        env_set = _as_upper_set(env_cfg.get("class_allowlist", []))
        return self.class_allowlist.union(env_set)

    def _effective_class_denylist(self) -> Set[str]:
        env_cfg = self._env_override()
        env_set = _as_upper_set(env_cfg.get("class_denylist", []))
        return self.class_denylist.union(env_set)

    def _approval_required_for_capability(self, capability: ToolCapability) -> bool:
        if not self.autonomy_soft_enabled:
            return bool(capability.requires_human_approval)
        env_cfg = self._env_override()
        classes = self.approval_classes_prod if self._is_prod_environment() else self.approval_classes_non_prod
        env_override_classes = _as_upper_set(env_cfg.get("require_approval_classes", []))
        if env_override_classes:
            classes = env_override_classes
        if capability.capability_class in classes:
            return True
        if self.enforce_capability_requires_human_approval:
            return bool(capability.requires_human_approval)
        return False

    def _find_freeze(self, current_actions: List[OffAction]) -> Optional[OffAction]:
        for action in current_actions:
            if action.type == "TOOL_FREEZE":
                return action
        return None

    @staticmethod
    def _find_require_approval(current_actions: List[OffAction]) -> Optional[OffAction]:
        for action in current_actions:
            if action.type == "REQUIRE_APPROVAL" and str(action.target).upper() == "TOOLS":
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
        require_approval_action = self._find_require_approval(current_actions)
        off_state = self._is_off_state(current_actions)
        capability = self.capabilities.get(request.tool_name)
        capability_class = capability.capability_class if capability is not None else None
        risk_level = capability.risk_level if capability is not None else None

        if capability is None and self.unknown_tool_policy == "DENY":
            return ToolDecision(
                allowed=False,
                mode="TOOLS_DISABLED",
                reason="UNKNOWN_TOOL",
                logged=True,
                validation_status="skipped",
                validation_reason=None,
                capability_class=None,
                risk_level=None,
                approval_required=False,
            )

        mode = freeze_action.tool_mode if freeze_action is not None else "TOOLS_DISABLED"
        cap_mode = capability.mode if capability is not None else "dangerous"

        if self.autonomy_soft_enabled:
            tool_name_norm = str(request.tool_name).strip().lower()
            effective_tool_denylist = self._effective_tool_denylist()
            effective_tool_allowlist = self._effective_tool_allowlist()
            effective_class_denylist = self._effective_class_denylist()
            effective_class_allowlist = self._effective_class_allowlist()

            if tool_name_norm in effective_tool_denylist:
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="AUTONOMY_TOOL_DENYLIST",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=False,
                )
            if effective_tool_allowlist and tool_name_norm not in effective_tool_allowlist:
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="AUTONOMY_TOOL_NOT_ALLOWLISTED",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=False,
                )
            if capability is not None:
                if capability.capability_class in effective_class_denylist:
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="AUTONOMY_CLASS_DENYLIST",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )
                if effective_class_allowlist and capability.capability_class not in effective_class_allowlist:
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="AUTONOMY_CLASS_NOT_ALLOWLISTED",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )

        if freeze_action is not None:
            mode = freeze_action.tool_mode or "TOOLS_DISABLED"
            if mode == "TOOLS_DISABLED":
                if self.freeze_read_only_exception and cap_mode != "read_only":
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="TOOL_FREEZE_ACTIVE",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )
                if not (self.freeze_read_only_exception and cap_mode == "read_only"):
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="TOOL_FREEZE_ACTIVE",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )
            elif mode == "TOOLS_ALLOWLIST":
                allowlist = freeze_action.allowlist or []
                if request.tool_name not in allowlist:
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="NOT_IN_ALLOWLIST",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )
                if self.freeze_read_only_exception and cap_mode != "read_only":
                    return ToolDecision(
                        allowed=False,
                        mode=mode,
                        reason="TOOL_FREEZE_ACTIVE",
                        logged=True,
                        validation_status="skipped",
                        validation_reason=None,
                        capability_class=capability_class,
                        risk_level=risk_level,
                        approval_required=False,
                    )
            else:
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="POLICY_BLOCK",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=False,
                )

        # REQUIRE_APPROVAL is weaker than TOOL_FREEZE and applied in chokepoint before execution.
        if require_approval_action is not None:
            allowlist = [str(x) for x in (require_approval_action.allowlist or []) if str(x).strip()]
            applies = True if not allowlist else (request.tool_name in allowlist)
            if applies and not bool(request.args.get("human_approved", False)):
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="REQUIRE_APPROVAL_PENDING",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=True,
                )

        if capability is not None:
            if (
                self.autonomy_soft_enabled
                and self.backup_safety_enabled
                and self.backup_precondition_required
                and (not self.backup_precondition_ready)
                and capability.capability_class in self.backup_sensitive_classes
            ):
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="BACKUP_POLICY_PRECONDITION",
                    logged=True,
                    validation_status="skipped",
                    validation_reason="immutable_backup_precondition_not_ready",
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=False,
                )
            if off_state and "OFF" not in capability.allowed_when:
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="OFF_STATE_BLOCK",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=False,
                )
            approval_required = self._approval_required_for_capability(capability)
            if approval_required and not bool(request.args.get("human_approved", False)):
                return ToolDecision(
                    allowed=False,
                    mode=mode,
                    reason="HUMAN_APPROVAL_REQUIRED",
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability_class,
                    risk_level=risk_level,
                    approval_required=True,
                )

        validation = validate_tool_args(request.tool_name, request.args, self.arg_validation)
        if validation.checked and not validation.allowed:
            return ToolDecision(
                allowed=False,
                mode=mode,
                reason=str(validation.reason_code or "INVALID_TOOL_ARGS_SCHEMA"),
                logged=True,
                validation_status="failed",
                validation_reason=validation.reason,
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=False,
            )
        status = "passed" if validation.checked else "not_checked"
        return ToolDecision(
            allowed=True,
            mode=mode,
            reason="OK",
            logged=True,
            validation_status=status,
            validation_reason=None,
            capability_class=capability_class,
            risk_level=risk_level,
            approval_required=False,
        )
