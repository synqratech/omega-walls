"""Core adapter contract and runtime for framework integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Mapping, Optional
import uuid

from omega.config.loader import config_refs_from_snapshot, load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OffAction, ToolRequest
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.rag.harness import OmegaRAGHarness
from omega.tools.tool_gateway import ToolGatewayV1


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _reason_codes(reasons: Any) -> List[str]:
    out: List[str] = []
    if bool(getattr(reasons, "reason_spike", False)):
        out.append("reason_spike")
    if bool(getattr(reasons, "reason_wall", False)):
        out.append("reason_wall")
    if bool(getattr(reasons, "reason_sum", False)):
        out.append("reason_sum")
    if bool(getattr(reasons, "reason_multi", False)):
        out.append("reason_multi")
    return out


class _AdapterMockLLM:
    def generate(self, prompt: str) -> Dict[str, Any]:
        del prompt
        return {"text": "adapter_noop_response"}


@dataclass(frozen=True)
class AdapterSessionContext:
    session_id: str
    actor_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterDecision:
    session_id: str
    step: int
    off: bool
    control_outcome: str
    actions: List[OffAction]
    reason_codes: List[str]
    trace_id: str
    decision_id: str
    risk_score: Optional[float] = None


@dataclass(frozen=True)
class ToolGateDecision:
    allowed: bool
    reason: str
    mode: str
    tool_name: str
    decision_ref: AdapterDecision
    executed: bool
    gateway_coverage: float
    orphan_executions: int


@dataclass(frozen=True)
class MemoryWriteDecision:
    allowed: bool
    mode: str
    reason: str
    source_id: str
    source_type: str
    source_trust: str
    tags: Dict[str, Any]
    decision_ref: AdapterDecision


class OmegaBlockedError(RuntimeError):
    """Raised when a model step is blocked by Omega policy."""

    def __init__(self, message: str, decision: AdapterDecision):
        self.decision = decision
        super().__init__(str(message))


class OmegaToolBlockedError(RuntimeError):
    """Raised when a tool call is blocked by Omega tool gate."""

    def __init__(self, message: str, gate_decision: ToolGateDecision):
        self.gate_decision = gate_decision
        super().__init__(str(message))


class OmegaAdapterRuntime:
    """Framework-agnostic adapter runtime backed by the canonical Omega harness."""

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        config_dir: Optional[str] = None,
        projector_mode: Optional[str] = None,
        api_model: Optional[str] = None,
        cli_overrides: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
        default_source_type: str = "other",
        default_trust: str = "untrusted",
        max_chars: int = 8000,
    ) -> None:
        effective_overrides: Dict[str, Any] = {}
        if isinstance(cli_overrides, Mapping):
            effective_overrides = dict(cli_overrides)

        projector_override: Dict[str, Any] = {}
        if projector_mode is not None:
            projector_override.setdefault("projector", {})["mode"] = str(projector_mode)
        if api_model is not None:
            projector_override.setdefault("projector", {}).setdefault("api_perception", {})["model"] = str(api_model)
        if projector_override:
            effective_overrides = _deep_merge(effective_overrides, projector_override)

        # Adapter runtime must preflight tool path only, never execute real tools.
        runtime_override = {"tools": {"execution_mode": "DRY_RUN"}}
        effective_overrides = _deep_merge(effective_overrides, runtime_override)

        snapshot = load_resolved_config(
            config_dir=str(config_dir) if config_dir else None,
            profile=str(profile),
            cli_overrides=effective_overrides or None,
            env=dict(env) if env is not None else None,
        )
        cfg = _deep_merge(snapshot.resolved, {})

        # Keep adapter runtime state isolated from other local runs.
        sqlite_suffix = uuid.uuid4().hex[:10]
        sqlite_rel = f"artifacts/state/adapter_runtime_{sqlite_suffix}.db"
        cfg.setdefault("off_policy", {}).setdefault("cross_session", {})["sqlite_path"] = sqlite_rel

        self.profile = str(profile)
        self.config = cfg
        self.snapshot = snapshot
        self.config_refs = config_refs_from_snapshot(snapshot, code_commit="local")
        self.default_source_type = str(default_source_type or "other")
        self.default_trust = str(default_trust or "untrusted")
        self.max_chars = max(256, int(max_chars))
        self._harness_by_session: Dict[str, OmegaRAGHarness] = {}

    def _create_harness(self, *, session_id: str, actor_id: str) -> OmegaRAGHarness:
        harness = OmegaRAGHarness(
            projector=build_projector(self.config),
            omega_core=OmegaCoreV1(omega_params_from_config(self.config)),
            off_policy=OffPolicyV1(self.config),
            tool_gateway=ToolGatewayV1(self.config),
            config=self.config,
            llm_backend=_AdapterMockLLM(),
        )
        harness.reset_state(session_id=session_id, actor_id=actor_id)
        return harness

    def _get_harness(self, ctx: AdapterSessionContext) -> OmegaRAGHarness:
        session_id = str(ctx.session_id)
        actor_id = str(ctx.actor_id or ctx.session_id)
        harness = self._harness_by_session.get(session_id)
        if harness is None:
            harness = self._create_harness(session_id=session_id, actor_id=actor_id)
            self._harness_by_session[session_id] = harness
        return harness

    def _clip(self, text: str) -> str:
        return str(text or "")[: self.max_chars]

    def _build_item(self, *, text: str, source_id: str) -> ContentItem:
        return ContentItem(
            doc_id=f"adapter-{uuid.uuid4().hex[:12]}",
            source_id=str(source_id),
            source_type=self.default_source_type,
            trust=self.default_trust,
            text=self._clip(text),
            meta={"adapter_runtime": True},
        )

    def _to_adapter_decision(self, out: Dict[str, Any]) -> AdapterDecision:
        step_result = out["step_result"]
        decision = out["decision"]
        risk_value = getattr(step_result, "m", None)
        risk_score: Optional[float] = None
        if risk_value is not None:
            try:
                risk_score = float(risk_value)
            except Exception:
                risk_score = None
        return AdapterDecision(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            off=bool(step_result.off),
            control_outcome=str(decision.control_outcome),
            actions=list(decision.actions),
            reason_codes=_reason_codes(step_result.reasons),
            trace_id=str(out.get("trace_id", "")),
            decision_id=str(out.get("decision_id", "")),
            risk_score=risk_score,
        )

    @staticmethod
    def _normalize_trust(source_trust: str) -> str:
        raw = str(source_trust or "untrusted").strip().lower()
        if raw in {"trusted", "internal", "allowlisted"}:
            return "trusted"
        if raw in {"mixed", "partially_trusted", "semi_trusted"}:
            return "mixed"
        return "untrusted"

    def check_model_input(self, messages_text: str, ctx: AdapterSessionContext) -> AdapterDecision:
        if not str(ctx.session_id or "").strip():
            raise ValueError("AdapterSessionContext.session_id is required")
        harness = self._get_harness(ctx)
        actor_id = str(ctx.actor_id or ctx.session_id)

        item = self._build_item(text=messages_text, source_id="adapter:model_input")
        out = harness.run_step(
            user_query=self._clip(messages_text),
            packet_items=[item],
            actor_id=actor_id,
            config_refs=self.config_refs,
        )
        return self._to_adapter_decision(out)

    def check_tool_call(self, tool_name: str, tool_args: Dict[str, Any], ctx: AdapterSessionContext) -> ToolGateDecision:
        if not str(ctx.session_id or "").strip():
            raise ValueError("AdapterSessionContext.session_id is required")
        if not str(tool_name or "").strip():
            raise ValueError("tool_name is required")

        harness = self._get_harness(ctx)
        actor_id = str(ctx.actor_id or ctx.session_id)

        serialized_args = json.dumps(tool_args or {}, ensure_ascii=True, sort_keys=True, default=str)
        item = self._build_item(
            text=f"tool_name={tool_name}\nargs={serialized_args}",
            source_id="adapter:tool_call",
        )
        request = ToolRequest(
            tool_name=str(tool_name),
            args={"tool_args": dict(tool_args or {}), "request_origin": "explicit"},
            session_id=str(ctx.session_id),
            step=int(getattr(harness.state, "step", 0)) + 1,
        )
        out = harness.run_step(
            user_query=f"Tool preflight for {tool_name}",
            packet_items=[item],
            tool_requests=[request],
            actor_id=actor_id,
            config_refs=self.config_refs,
        )
        decision_ref = self._to_adapter_decision(out)

        tool_decisions = list(out.get("tool_decisions", []))
        tool_executions = list(out.get("tool_executions", []))
        gateway_events = list(out.get("tool_gateway_events", []))
        explicit_requests = 1.0
        gateway_coverage = float(len(gateway_events) / explicit_requests) if explicit_requests > 0 else 1.0
        orphan_executions = int(
            sum(
                1
                for exec_, dec in zip(tool_executions, tool_decisions)
                if bool(getattr(exec_, "executed", False)) and not bool(getattr(dec, "allowed", False))
            )
        )

        if not tool_decisions:
            return ToolGateDecision(
                allowed=False,
                reason="MISSING_GATE_DECISION",
                mode="unknown",
                tool_name=str(tool_name),
                decision_ref=decision_ref,
                executed=False,
                gateway_coverage=gateway_coverage,
                orphan_executions=orphan_executions,
            )

        first = tool_decisions[0]
        first_exec = tool_executions[0] if tool_executions else None
        return ToolGateDecision(
            allowed=bool(getattr(first, "allowed", False)),
            reason=str(getattr(first, "reason", "unknown")),
            mode=str(getattr(first, "mode", "unknown")),
            tool_name=str(tool_name),
            decision_ref=decision_ref,
            executed=bool(getattr(first_exec, "executed", False)),
            gateway_coverage=gateway_coverage,
            orphan_executions=orphan_executions,
        )

    def check_memory_write(
        self,
        *,
        memory_text: str,
        source_id: str,
        source_type: str = "other",
        source_trust: str = "untrusted",
        ctx: AdapterSessionContext,
        source_tags: Optional[Mapping[str, Any]] = None,
    ) -> MemoryWriteDecision:
        if not str(source_id or "").strip():
            raise ValueError("source_id is required")

        trust = self._normalize_trust(source_trust)
        decision = self.check_model_input(messages_text=memory_text, ctx=ctx)
        action_types = {str(action.type).upper() for action in decision.actions}
        outcome = str(decision.control_outcome).upper()
        reason_codes = {str(code).lower() for code in decision.reason_codes}

        hard_block = bool(
            decision.off
            or outcome in {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}
            or bool(action_types & {"SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE", "REQUIRE_APPROVAL"})
        )
        quarantine = bool(action_types & {"SOURCE_QUARANTINE"}) or bool(
            trust != "trusted"
            and (
                "reason_multi" in reason_codes
                or "reason_spike" in reason_codes
                or (
                    decision.risk_score is not None
                    and float(decision.risk_score) >= 0.7
                )
            )
        )

        if hard_block:
            mode = "deny"
            allowed = False
            reason = "BLOCKING_POLICY_SIGNAL"
        elif quarantine:
            mode = "quarantine"
            allowed = False
            reason = "UNTRUSTED_OR_QUARANTINED_SOURCE"
        else:
            mode = "allow"
            allowed = True
            reason = "ALLOW_WRITE"

        tags = dict(source_tags or {})
        tags.setdefault("source_id", str(source_id))
        tags.setdefault("source_type", str(source_type or "other"))
        tags.setdefault("source_trust", trust)

        return MemoryWriteDecision(
            allowed=allowed,
            mode=mode,
            reason=reason,
            source_id=str(source_id),
            source_type=str(source_type or "other"),
            source_trust=trust,
            tags=tags,
            decision_ref=decision,
        )
