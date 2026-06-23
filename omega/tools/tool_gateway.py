"""Tool gateway implementation (fail closed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from omega.interfaces.contracts_v1 import OffAction, ToolDecision, ToolRequest
from omega.tools.arg_validation import ToolArgValidationConfig, validate_tool_args
from omega.tools.approval import tool_args_sha256, tool_intent_id
from omega.notifications.models import utc_now_iso


_BUILTIN_DISABLED_TOOLS = frozenset({"write_file", "network_post"})


_DEFAULT_CAPABILITIES = {
    "retrieval_readonly": {"enabled": True, "mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "summarize": {"enabled": True, "mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    "echo": {"enabled": True, "mode": "read_only", "allowed_when": ["NO_OFF", "OFF"], "requires_human_approval": False},
    # Built-in side-effecting adapters are intentionally unavailable in v1.
    "write_file": {"enabled": False, "mode": "write", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
    "network_post": {"enabled": False, "mode": "dangerous", "allowed_when": ["NO_OFF"], "requires_human_approval": True},
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
    enabled: bool
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
    approval_store: Optional[Any] = None

    def __post_init__(self) -> None:
        tools_cfg = (self.config or {}).get("tools", {})
        self.unknown_tool_policy = str(tools_cfg.get("unknown_tool_policy", "DENY")).upper()
        if self.unknown_tool_policy != "DENY":
            raise ValueError("tools.unknown_tool_policy must be DENY in v1")
        self.freeze_read_only_exception = bool(tools_cfg.get("freeze", {}).get("read_only_exception", False))
        if self.freeze_read_only_exception:
            raise ValueError("tools.freeze.read_only_exception must be false; use TOOLS_ALLOWLIST for explicit exceptions")
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
            normalized_tool_name = str(tool_name).strip().lower()
            # These legacy built-ins are removed from the executable registry and
            # cannot be re-enabled through profile/env/request configuration.
            enabled = bool(cap_dict.get("enabled", True)) and normalized_tool_name not in _BUILTIN_DISABLED_TOOLS
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
            self.capabilities[normalized_tool_name] = ToolCapability(
                enabled=enabled,
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

    def bind_approval_store(self, store: Any) -> None:
        self.approval_store = store

    def _approval_decision(
        self,
        *,
        request: ToolRequest,
        mode: str,
        capability_class: Optional[str],
        risk_level: Optional[str],
        validation_status: str,
        missing_reason: str,
    ) -> Optional[ToolDecision]:
        try:
            expected_intent = tool_intent_id(request)
            args_hash = tool_args_sha256(request.args)
        except ValueError as exc:
            return ToolDecision(
                allowed=False,
                mode=mode,
                reason="INVALID_TOOL_INTENT",
                logged=True,
                validation_status=validation_status,
                validation_reason=str(exc),
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=True,
            )
        request.intent_id = expected_intent
        approval_id = str(request.approval_id or "").strip()
        if not approval_id:
            return ToolDecision(
                allowed=False,
                mode=mode,
                reason=str(missing_reason),
                logged=True,
                validation_status=validation_status,
                validation_reason=None,
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=True,
                intent_id=expected_intent,
            )
        if self.approval_store is None:
            return ToolDecision(
                allowed=False,
                mode=mode,
                reason="APPROVAL_STORE_UNAVAILABLE",
                logged=True,
                validation_status=validation_status,
                validation_reason=None,
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=True,
                intent_id=expected_intent,
                approval_id=approval_id,
            )
        record = self.approval_store.consume_tool_approval(
            approval_id=approval_id,
            tenant_id=str(request.tenant_id or ""),
            session_id=str(request.session_id or ""),
            tool_name=str(request.tool_name or "").strip().lower(),
            tool_args_sha256=args_hash,
            tool_intent_id=expected_intent,
            step=int(request.step),
            now_iso=utc_now_iso(),
        )
        if record is None:
            return ToolDecision(
                allowed=False,
                mode=mode,
                reason="APPROVAL_INVALID_EXPIRED_OR_CONSUMED",
                logged=True,
                validation_status=validation_status,
                validation_reason=None,
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=True,
                intent_id=expected_intent,
                approval_id=approval_id,
            )
        return None

    def enforce(self, request: ToolRequest, current_actions: List[OffAction]) -> ToolDecision:
        freeze_action = self._find_freeze(current_actions)
        require_approval_action = self._find_require_approval(current_actions)
        off_state = self._is_off_state(current_actions)
        tool_name = str(request.tool_name or "").strip().lower()
        request.tool_name = tool_name
        capability = self.capabilities.get(tool_name)
        capability_class = capability.capability_class if capability is not None else None
        risk_level = capability.risk_level if capability is not None else None
        mode = freeze_action.tool_mode if freeze_action is not None else "TOOLS_ENABLED"

        def deny(reason: str, *, validation_status: str = "skipped", validation_reason: Optional[str] = None, approval_required: bool = False) -> ToolDecision:
            return ToolDecision(
                allowed=False,
                mode=str(mode or "TOOLS_DISABLED"),
                reason=reason,
                logged=True,
                validation_status=validation_status,
                validation_reason=validation_reason,
                capability_class=capability_class,
                risk_level=risk_level,
                approval_required=approval_required,
                intent_id=request.intent_id,
                approval_id=request.approval_id,
            )

        if capability is None:
            return deny("UNKNOWN_TOOL")
        if not capability.enabled:
            return deny("TOOL_DISABLED_BY_CONFIG")

        if self.autonomy_soft_enabled:
            effective_tool_denylist = self._effective_tool_denylist()
            effective_tool_allowlist = self._effective_tool_allowlist()
            effective_class_denylist = self._effective_class_denylist()
            effective_class_allowlist = self._effective_class_allowlist()
            if tool_name in effective_tool_denylist:
                return deny("AUTONOMY_TOOL_DENYLIST")
            if effective_tool_allowlist and tool_name not in effective_tool_allowlist:
                return deny("AUTONOMY_TOOL_NOT_ALLOWLISTED")
            if capability.capability_class in effective_class_denylist:
                return deny("AUTONOMY_CLASS_DENYLIST")
            if effective_class_allowlist and capability.capability_class not in effective_class_allowlist:
                return deny("AUTONOMY_CLASS_NOT_ALLOWLISTED")

        if freeze_action is not None:
            mode = freeze_action.tool_mode or "TOOLS_DISABLED"
            if mode == "TOOLS_DISABLED":
                return deny("TOOL_FREEZE_ACTIVE")
            if mode == "TOOLS_ALLOWLIST":
                allowlist = {str(name).strip().lower() for name in (freeze_action.allowlist or [])}
                if tool_name not in allowlist:
                    return deny("NOT_IN_ALLOWLIST")
            else:
                return deny("POLICY_BLOCK")

        if (
            self.autonomy_soft_enabled
            and self.backup_safety_enabled
            and self.backup_precondition_required
            and not self.backup_precondition_ready
            and capability.capability_class in self.backup_sensitive_classes
        ):
            return deny("BACKUP_POLICY_PRECONDITION", validation_reason="immutable_backup_precondition_not_ready")
        if off_state and "OFF" not in capability.allowed_when:
            return deny("OFF_STATE_BLOCK")

        # Validate the exact arguments before asking for or consuming an approval.
        validation = validate_tool_args(tool_name, request.args, self.arg_validation)
        if validation.checked and not validation.allowed:
            return deny(
                str(validation.reason_code or "INVALID_TOOL_ARGS_SCHEMA"),
                validation_status="failed",
                validation_reason=validation.reason,
            )
        validation_status = "passed" if validation.checked else "not_checked"

        approval_required = self._approval_required_for_capability(capability)
        if require_approval_action is not None:
            allowlist = {str(x).strip().lower() for x in (require_approval_action.allowlist or []) if str(x).strip()}
            if not allowlist or tool_name in allowlist:
                approval_required = True
        if approval_required:
            blocked = self._approval_decision(
                request=request,
                mode=str(mode or "TOOLS_ENABLED"),
                capability_class=capability_class,
                risk_level=risk_level,
                validation_status=validation_status,
                missing_reason=("REQUIRE_APPROVAL_PENDING" if require_approval_action is not None else "HUMAN_APPROVAL_REQUIRED"),
            )
            if blocked is not None:
                return blocked

        return ToolDecision(
            allowed=True,
            mode=str(mode or "TOOLS_ENABLED"),
            reason="OK",
            logged=True,
            validation_status=validation_status,
            validation_reason=None,
            capability_class=capability_class,
            risk_level=risk_level,
            approval_required=False,
            intent_id=request.intent_id,
            approval_id=request.approval_id,
        )
