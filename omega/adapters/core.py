"""Core adapter contract and runtime for framework integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import threading
from typing import Any, Dict, List, Mapping, Optional
import uuid

from omega.config.loader import config_refs_from_snapshot, load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OffAction, ToolRequest
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.rag.harness import OmegaRAGHarness
from omega.runtime.integrity_policy import assess_runtime_artifact, build_runtime_artifact
from omega.runtime.operation_gate import evaluate_operation_gate
from omega.runtime.artifacts import OperationIntent
from omega.tools.adapters import extract_tool_output_text
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    incident_id: Optional[str] = None
    policy_id: Optional[str] = None
    fallback_hint: Optional[str] = None
    provider_id: Optional[str] = None
    health_state: Optional[str] = None
    llm_fallback_active: Optional[bool] = None
    fallback_level: Optional[str] = None
    boundary_mode: Optional[str] = None
    coverage_status: Optional[Dict[str, str]] = None
    segment_stats: Optional[Dict[str, int]] = None
    pressure_dedupe: Optional[Dict[str, Any]] = None
    effect_wall_candidate: Optional[Dict[str, Any]] = None
    effect_policy_gate: Optional[Dict[str, Any]] = None
    artifact_assessment_summary: Optional[Dict[str, Any]] = None
    operation_gate_events: Optional[List[Dict[str, Any]]] = None
    skillbox_status: Optional[str] = None
    skillbox_verification: Optional[Dict[str, Any]] = None
    skillbox_ledger_hit: Optional[bool] = None
    skillbox_content_sha256: Optional[str] = None
    skillbox_capabilities: Optional[List[str]] = None
    skillbox_gate_decision: Optional[str] = None


@dataclass(frozen=True)
class AdapterSegment:
    role: str
    origin: str
    trust: str
    text: str
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    derived_from: Optional[List[str]] = None


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
    operation_gate: Optional[Dict[str, Any]] = None


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
    operation_gate: Optional[Dict[str, Any]] = None


class OmegaBlockedError(RuntimeError):
    """Raised when a model step is blocked by Omega policy."""

    def __init__(self, message: str, decision: AdapterDecision):
        self.decision = decision
        super().__init__(str(message))

    def to_structured_payload(self) -> Dict[str, Any]:
        policy_id = str(self.decision.policy_id or "").strip() or None
        fallback_hint = str(self.decision.fallback_hint or "").strip() or None
        reason = str(self.decision.control_outcome or "BLOCK")
        if self.decision.reason_codes:
            reason = str(self.decision.reason_codes[0])
        return {
            "action": str(self.decision.control_outcome or "BLOCK"),
            "reason": reason,
            "policy_id": policy_id,
            "fallback_hint": fallback_hint,
            "incident_id": str(self.decision.incident_id or "").strip() or None,
            "trace_id": str(self.decision.trace_id or ""),
            "decision_id": str(self.decision.decision_id or ""),
        }


class OmegaToolBlockedError(RuntimeError):
    """Raised when a tool call is blocked by Omega tool gate."""

    def __init__(self, message: str, gate_decision: ToolGateDecision):
        self.gate_decision = gate_decision
        super().__init__(str(message))

    def to_structured_payload(self) -> Dict[str, Any]:
        decision = self.gate_decision.decision_ref
        policy_id = str(decision.policy_id or "").strip() or None
        fallback_hint = str(decision.fallback_hint or "").strip() or None
        action = OmegaAdapterRuntime.resolve_tool_block_action(self.gate_decision)
        return {
            "action": str(action),
            "reason": str(self.gate_decision.reason or "TOOL_BLOCKED"),
            "policy_id": policy_id,
            "fallback_hint": fallback_hint,
            "incident_id": str(decision.incident_id or "").strip() or None,
            "trace_id": str(decision.trace_id or ""),
            "decision_id": str(decision.decision_id or ""),
        }


class OmegaAdapterRuntime:
    """Framework-agnostic adapter runtime backed by the canonical Omega harness."""

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        config_dir: Optional[str] = None,
        projector_mode: Optional[str] = None,
        api_model: Optional[str] = None,
        api_provider: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_base_url: Optional[str] = None,
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
        if api_provider is not None:
            projector_override.setdefault("projector", {}).setdefault("api_perception", {})["provider"] = str(api_provider)
        if api_key_env is not None:
            projector_override.setdefault("projector", {}).setdefault("api_perception", {})["api_key_env"] = str(api_key_env)
        if api_base_url is not None:
            projector_override.setdefault("projector", {}).setdefault("api_perception", {})["base_url"] = str(api_base_url)
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
        self._harness_lock = threading.Lock()
        self._boundary_observed_by_session: Dict[str, Dict[str, bool]] = {}
        self._last_boundary_mode_by_session: Dict[str, str] = {}
        self.boundary_mode = "blob_fallback"
        self._coverage_status_blob: Dict[str, str] = {
            "before_model_call": "partial",
            "tool_preflight": "full",
            "tool_output_reentry": "missing",
            "memory_write": "partial",
            "memory_read": "missing",
            "agent_message": "missing",
        }
        self.coverage_status: Dict[str, str] = dict(self._coverage_status_blob)

    @staticmethod
    def recommended_boundary_mode_for_profile(profile: str) -> str:
        env = str(profile or "").strip().lower()
        if env in {"prod", "pilot", "pilot_canonical"}:
            return "segmented"
        return "blob_fallback"

    @classmethod
    def resolve_boundary_mode(cls, mode: str, *, profile: str = "quickstart") -> str:
        raw = str(mode or "").strip().lower()
        if raw in {"blob_fallback", "segmented"}:
            return raw
        if raw in {"recommended", "auto", "recommended_prod", "production"}:
            return cls.recommended_boundary_mode_for_profile(profile)
        if raw in {"compatibility", "legacy"}:
            return "blob_fallback"
        return "blob_fallback"

    @classmethod
    def policy_presets(cls, *, profile: str = "quickstart") -> Dict[str, Dict[str, Any]]:
        recommended = cls.recommended_boundary_mode_for_profile(profile)
        return {
            "compatibility": {
                "boundary_mode": "blob_fallback",
                "description": "Legacy-compatible path with prompt blob precheck.",
            },
            "recommended": {
                "boundary_mode": str(recommended),
                "description": (
                    "Profile-aware recommendation: segmented for prod/pilot, "
                    "blob_fallback for quickstart/dev."
                ),
            },
            "production_segmented": {
                "boundary_mode": "segmented",
                "description": "Production-first segmented trust-boundary mode.",
            },
        }

    def _coverage_status_for_mode(self, mode: str) -> Dict[str, str]:
        base = dict(self._coverage_status_blob)
        if str(mode) == "segmented":
            base["before_model_call"] = "full"
        return base

    def _ensure_boundary_observed_flags(self, session_id: str) -> Dict[str, bool]:
        sid = str(session_id or "").strip()
        if sid not in self._boundary_observed_by_session:
            self._boundary_observed_by_session[sid] = {
                "memory_write_seen": False,
                "memory_read_seen": False,
                "tool_output_reentry_seen": False,
            }
        return self._boundary_observed_by_session[sid]

    def _mark_boundary_observed(self, ctx: AdapterSessionContext, key: str) -> None:
        if not str(ctx.session_id or "").strip():
            return
        flags = self._ensure_boundary_observed_flags(str(ctx.session_id))
        flags[str(key)] = True

    def _coverage_status_for_ctx(self, ctx: AdapterSessionContext, mode: str) -> Dict[str, str]:
        base = self._coverage_status_for_mode(mode)
        if not str(ctx.session_id or "").strip():
            return base
        flags = self._ensure_boundary_observed_flags(str(ctx.session_id))
        if bool(flags.get("memory_write_seen")):
            base["memory_write"] = "full"
        if bool(flags.get("memory_read_seen")):
            base["memory_read"] = "full"
        if bool(flags.get("tool_output_reentry_seen")):
            base["tool_output_reentry"] = "full"
        return base

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
        with self._harness_lock:
            harness = self._harness_by_session.get(session_id)
            if harness is None:
                harness = self._create_harness(session_id=session_id, actor_id=actor_id)
                self._harness_by_session[session_id] = harness
            return harness

    @staticmethod
    def _coverage_grade(coverage_status: Mapping[str, str]) -> str:
        values = [str(v).strip().lower() for v in dict(coverage_status or {}).values()]
        if values and all(v == "full" for v in values):
            return "full"
        if any(v == "full" for v in values):
            return "partial"
        return "minimal"

    def get_boundary_coverage_report(self, *, session_id: Optional[str] = None) -> Dict[str, Any]:
        if str(session_id or "").strip():
            session_ids = [str(session_id).strip()]
        else:
            session_ids = sorted(
                {
                    *[str(x) for x in self._harness_by_session.keys()],
                    *[str(x) for x in self._boundary_observed_by_session.keys()],
                }
            )

        sessions: List[Dict[str, Any]] = []
        for sid in session_ids:
            flags = self._ensure_boundary_observed_flags(sid)
            mode = str(
                self._last_boundary_mode_by_session.get(
                    sid,
                    self.recommended_boundary_mode_for_profile(self.profile),
                )
            )
            coverage_status = self._coverage_status_for_ctx(
                AdapterSessionContext(session_id=sid, actor_id=sid),
                mode,
            )
            missing = [k for k, v in coverage_status.items() if str(v).strip().lower() != "full"]
            sessions.append(
                {
                    "session_id": sid,
                    "boundary_mode": mode,
                    "coverage_status": dict(coverage_status),
                    "coverage_grade": self._coverage_grade(coverage_status),
                    "missing_boundaries": sorted(missing),
                    "observed_flags": {
                        "memory_write_seen": bool(flags.get("memory_write_seen")),
                        "memory_read_seen": bool(flags.get("memory_read_seen")),
                        "tool_output_reentry_seen": bool(flags.get("tool_output_reentry_seen")),
                    },
                }
            )

        full_count = sum(1 for row in sessions if str(row.get("coverage_grade")) == "full")
        partial_count = sum(1 for row in sessions if str(row.get("coverage_grade")) == "partial")
        minimal_count = max(0, int(len(sessions) - full_count - partial_count))
        return {
            "event": "omega_boundary_coverage_report_v1",
            "schema_version": "1.0",
            "timestamp": _utc_now_iso(),
            "profile": str(self.profile),
            "recommended_boundary_mode": self.recommended_boundary_mode_for_profile(self.profile),
            "session_count": int(len(sessions)),
            "summary": {
                "full": int(full_count),
                "partial": int(partial_count),
                "minimal": int(minimal_count),
            },
            "sessions": sessions,
        }

    def _clip(self, text: str) -> str:
        return str(text or "")[: self.max_chars]

    @staticmethod
    def _normalize_origin(origin: Optional[str]) -> str:
        raw = str(origin or "").strip().lower()
        if raw in {
            "retrieval",
            "tool_output",
            "memory",
            "memory_write",
            "memory_read",
            "agent_message",
            "user",
            "system",
            "developer",
            "policy",
            "config",
            "trusted_control",
            "trusted_user",
            "model_input",
            "tool_call",
            "sanitizer_output",
            "api_request",
            "other",
        }:
            return raw
        return "unknown"

    @classmethod
    def _normalize_trust(cls, source_trust: str, *, origin: Optional[str] = None) -> str:
        origin_norm = cls._normalize_origin(origin)
        if origin_norm == "unknown":
            return "untrusted"
        raw = str(source_trust or "untrusted").strip().lower()
        if raw in {"trusted_control", "trusted_user"}:
            return raw
        if origin_norm in {"system", "developer", "policy", "config", "trusted_control"} and raw in {
            "trusted",
            "internal",
            "allowlisted",
        }:
            return "trusted_control"
        if origin_norm in {"user", "trusted_user"} and raw in {"trusted", "internal", "allowlisted"}:
            return "trusted_user"
        if raw in {"trusted", "internal", "allowlisted"}:
            return "trusted"
        if raw in {"mixed", "partially_trusted", "semi_trusted"}:
            return "mixed"
        return "untrusted"

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _next_boundary_step(harness: OmegaRAGHarness) -> int:
        return int(getattr(harness.state, "step", 0)) + 1

    def _build_item(
        self,
        *,
        text: str,
        source_id: str,
        source_type: Optional[str] = None,
        source_trust: Optional[str] = None,
        origin: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        boundary_step: Optional[int] = None,
        extra_meta: Optional[Mapping[str, Any]] = None,
    ) -> ContentItem:
        clipped = self._clip(text)
        origin_norm = self._normalize_origin(origin)
        trust = self._normalize_trust(source_trust or self.default_trust, origin=origin_norm)
        artifact_id = f"art-{uuid.uuid4().hex[:16]}"
        content_hash = self._content_hash(clipped)
        lineage = [str(x) for x in list(derived_from or []) if str(x).strip()]
        meta: Dict[str, Any] = {
            "adapter_runtime": True,
            "boundary_mode": self.boundary_mode,
            "coverage_status": dict(self.coverage_status),
            "artifact_id": artifact_id,
            "origin": origin_norm,
            "content_hash": content_hash,
        }
        if lineage:
            meta["derived_from"] = list(lineage)
        if boundary_step is not None:
            meta["boundary_step"] = int(boundary_step)
        if isinstance(extra_meta, Mapping):
            meta.update(dict(extra_meta))
        return ContentItem(
            doc_id=f"adapter-{uuid.uuid4().hex[:12]}",
            source_id=str(source_id),
            source_type=str(source_type or self.default_source_type),
            trust=trust,
            text=clipped,
            artifact_id=artifact_id,
            origin=origin_norm,
            derived_from=list(lineage) if lineage else None,
            content_hash=content_hash,
            boundary_step=(int(boundary_step) if boundary_step is not None else None),
            meta=meta,
        )

    def _run_items_check(
        self,
        *,
        ctx: AdapterSessionContext,
        user_query: str,
        packet_items: List[ContentItem],
        boundary_mode: str,
        segment_stats: Optional[Mapping[str, Any]] = None,
        tool_requests: Optional[List[ToolRequest]] = None,
        return_raw: bool = False,
    ) -> AdapterDecision | tuple[AdapterDecision, Dict[str, Any]]:
        if not str(ctx.session_id or "").strip():
            raise ValueError("AdapterSessionContext.session_id is required")
        harness = self._get_harness(ctx)
        actor_id = str(ctx.actor_id or ctx.session_id)
        prev_mode = str(self.boundary_mode)
        prev_coverage = dict(self.coverage_status)
        try:
            self.boundary_mode = str(boundary_mode)
            self.coverage_status = self._coverage_status_for_ctx(ctx, str(boundary_mode))
            self._last_boundary_mode_by_session[str(ctx.session_id)] = str(boundary_mode)
            out = harness.run_step(
                user_query=self._clip(user_query),
                packet_items=list(packet_items),
                tool_requests=list(tool_requests or []) or None,
                actor_id=actor_id,
                config_refs=self.config_refs,
            )
            if isinstance(segment_stats, Mapping):
                out["segment_stats"] = dict(segment_stats)
            decision = self._to_adapter_decision(out)
            if bool(return_raw):
                return decision, out
            return decision
        finally:
            self.boundary_mode = prev_mode
            self.coverage_status = prev_coverage

    def _to_adapter_decision(self, out: Dict[str, Any]) -> AdapterDecision:
        step_result = out["step_result"]
        decision = out["decision"]
        policy_decision = out.get("policy_decision")
        incident_id = str(out.get("incident_artifact_id", "") or "").strip() or None
        risk_value = getattr(step_result, "m_next", None)
        if risk_value is None:
            risk_value = getattr(step_result, "m", None)
        risk_score: Optional[float] = None
        if risk_value is not None:
            try:
                if hasattr(risk_value, "__len__") and not isinstance(risk_value, (str, bytes)):
                    seq = [float(x) for x in list(risk_value)]
                    risk_score = float(max(seq)) if seq else 0.0
                else:
                    risk_score = float(risk_value)
            except Exception:
                risk_score = None
        policy_id = None
        fallback_hint = None
        if policy_decision is not None:
            policy_id = (
                str(getattr(policy_decision, "policy_id", "") or "").strip()
                or str(getattr(policy_decision, "rule_id", "") or "").strip()
                or None
            )
            fallback_hint = str(getattr(policy_decision, "fallback_hint", "") or "").strip() or None
        api_status = {}
        try:
            harness = self._harness_by_session.get(str(step_result.session_id))
            if harness is not None:
                api_status_fn = getattr(harness.projector, "api_perception_status", None)
                if callable(api_status_fn):
                    raw = api_status_fn()
                    if isinstance(raw, dict):
                        api_status = dict(raw)
        except Exception:
            api_status = {}
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
            incident_id=incident_id,
            policy_id=policy_id,
            fallback_hint=fallback_hint,
            provider_id=str(api_status.get("provider_id", "") or "").strip() or None,
            health_state=str(api_status.get("health_state", "") or "").strip() or None,
            llm_fallback_active=(
                bool(api_status.get("llm_fallback_active"))
                if "llm_fallback_active" in api_status
                else None
            ),
            fallback_level=str(api_status.get("fallback_level", "") or "").strip() or None,
            boundary_mode=str(self.boundary_mode),
            coverage_status=dict(self.coverage_status),
            segment_stats=(dict(out.get("segment_stats", {})) if isinstance(out.get("segment_stats"), Mapping) else None),
            pressure_dedupe=(dict(out.get("pressure_dedupe", {})) if isinstance(out.get("pressure_dedupe"), Mapping) else None),
            effect_wall_candidate=(
                dict(out.get("effect_wall_candidate", {}))
                if isinstance(out.get("effect_wall_candidate"), Mapping)
                else None
            ),
            effect_policy_gate=(
                dict(out.get("effect_policy_gate", {}))
                if isinstance(out.get("effect_policy_gate"), Mapping)
                else None
            ),
            artifact_assessment_summary=(
                dict(out.get("artifact_assessment_summary", {}))
                if isinstance(out.get("artifact_assessment_summary"), Mapping)
                else None
            ),
            operation_gate_events=(
                [dict(x) for x in list(out.get("operation_gate_events", [])) if isinstance(x, Mapping)]
                if isinstance(out.get("operation_gate_events"), list)
                else None
            ),
            skillbox_status=str(out.get("skillbox_status", "disabled")),
            skillbox_verification=(
                dict(out.get("skillbox_verification", {}))
                if isinstance(out.get("skillbox_verification"), Mapping)
                else None
            ),
            skillbox_ledger_hit=(
                bool(out.get("skillbox_ledger_hit", False))
                if "skillbox_ledger_hit" in out
                else None
            ),
            skillbox_content_sha256=(
                str(out.get("skillbox_content_sha256"))
                if out.get("skillbox_content_sha256") is not None
                else None
            ),
            skillbox_capabilities=(
                [str(x) for x in list(out.get("skillbox_capabilities", []) or [])]
                if "skillbox_capabilities" in out
                else None
            ),
            skillbox_gate_decision=(
                str(out.get("skillbox_gate_decision"))
                if out.get("skillbox_gate_decision") is not None
                else None
            ),
        )

    @staticmethod
    def build_block_contract_from_decision(decision: AdapterDecision) -> Dict[str, Any]:
        reason = str(decision.control_outcome or "BLOCK")
        if decision.reason_codes:
            reason = str(decision.reason_codes[0])
        return {
            "action": str(decision.control_outcome or "BLOCK"),
            "reason": reason,
            "policy_id": str(decision.policy_id or "").strip() or None,
            "fallback_hint": str(decision.fallback_hint or "").strip() or None,
            "incident_id": str(decision.incident_id or "").strip() or None,
            "trace_id": str(decision.trace_id or ""),
            "decision_id": str(decision.decision_id or ""),
        }

    @classmethod
    def build_block_contract_from_tool_gate(cls, gate_decision: ToolGateDecision) -> Dict[str, Any]:
        payload = cls.build_block_contract_from_decision(gate_decision.decision_ref)
        payload["action"] = cls.resolve_tool_block_action(gate_decision)
        payload["reason"] = str(gate_decision.reason or payload["reason"])
        return payload

    @staticmethod
    def resolve_tool_block_action(gate_decision: ToolGateDecision) -> str:
        if bool(getattr(gate_decision, "allowed", False)):
            decision = gate_decision.decision_ref
            return str(getattr(decision, "control_outcome", "ALLOW") or "ALLOW")
        decision = gate_decision.decision_ref
        outcome = str(getattr(decision, "control_outcome", "") or "").strip().upper()
        # Tool deny must never look like ALLOW in external contracts.
        if outcome and outcome not in {"ALLOW", "WARN"}:
            return outcome
        return "TOOL_FREEZE"

    @staticmethod
    def build_siem_boundary_event(
        decision: AdapterDecision,
        *,
        phase: str = "decision",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        coverage_status = dict(decision.coverage_status or {})
        missing_boundaries = sorted([k for k, v in coverage_status.items() if str(v).strip().lower() != "full"])
        out: Dict[str, Any] = {
            "event": "omega_boundary_event_v1",
            "schema_version": "1.0",
            "timestamp": _utc_now_iso(),
            "phase": str(phase),
            "session_id": str(decision.session_id),
            "step": int(decision.step),
            "trace_id": str(decision.trace_id or ""),
            "decision_id": str(decision.decision_id or ""),
            "control_outcome": str(decision.control_outcome or "ALLOW"),
            "off": bool(decision.off),
            "risk_score": (float(decision.risk_score) if decision.risk_score is not None else None),
            "reason_codes": [str(x) for x in list(decision.reason_codes or []) if str(x).strip()],
            "boundary_mode": str(decision.boundary_mode or "blob_fallback"),
            "coverage_status": coverage_status,
            "coverage_grade": OmegaAdapterRuntime._coverage_grade(coverage_status),
            "missing_boundaries": missing_boundaries,
        }
        if isinstance(decision.segment_stats, Mapping):
            out["segment_stats"] = {str(k): int(v) for k, v in dict(decision.segment_stats).items()}
        if isinstance(decision.pressure_dedupe, Mapping):
            pd = dict(decision.pressure_dedupe)
            out["pressure_dedupe"] = {
                "input_count": int(pd.get("input_count", 0)),
                "kept_count": int(pd.get("kept_count", 0)),
                "deduped_count": int(pd.get("deduped_count", 0)),
                "deduped_by_artifact_id": int(pd.get("deduped_by_artifact_id", 0)),
                "deduped_by_content_hash_source": int(pd.get("deduped_by_content_hash_source", 0)),
            }
        if isinstance(decision.effect_wall_candidate, Mapping):
            out["effect_wall_candidate"] = dict(decision.effect_wall_candidate)
        if isinstance(decision.effect_policy_gate, Mapping):
            out["effect_policy_gate"] = dict(decision.effect_policy_gate)
        if isinstance(decision.artifact_assessment_summary, Mapping):
            out["artifact_assessment_summary"] = dict(decision.artifact_assessment_summary)
        if isinstance(decision.operation_gate_events, list):
            out["operation_gate_events"] = [
                dict(x) for x in list(decision.operation_gate_events) if isinstance(x, Mapping)
            ]
        if decision.skillbox_status is not None:
            out["skillbox_status"] = str(decision.skillbox_status)
        if isinstance(decision.skillbox_verification, Mapping):
            out["skillbox_verification"] = dict(decision.skillbox_verification)
        if decision.skillbox_ledger_hit is not None:
            out["skillbox_ledger_hit"] = bool(decision.skillbox_ledger_hit)
        if decision.skillbox_content_sha256 is not None:
            out["skillbox_content_sha256"] = str(decision.skillbox_content_sha256)
        if isinstance(decision.skillbox_capabilities, list):
            out["skillbox_capabilities"] = [str(x) for x in list(decision.skillbox_capabilities)]
        if decision.skillbox_gate_decision is not None:
            out["skillbox_gate_decision"] = str(decision.skillbox_gate_decision)
        if isinstance(extra, Mapping):
            out.update(dict(extra))
        return out

    @staticmethod
    def build_security_metadata(
        decision: AdapterDecision,
        *,
        phase: str = "decision",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        mode = "allow"
        outcome = str(decision.control_outcome or "ALLOW").upper()
        if decision.off or outcome in {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}:
            mode = "deny"
        elif outcome in {"WARN", "SOFT_BLOCK", "SOURCE_QUARANTINE", "HUMAN_ESCALATE", "REQUIRE_APPROVAL"}:
            mode = "degraded"
        out: Dict[str, Any] = {
            "phase": str(phase),
            "mode": mode,
            "risk": (float(decision.risk_score) if decision.risk_score is not None else None),
            "action": str(decision.control_outcome or "ALLOW"),
            "trace_id": str(decision.trace_id or ""),
            "decision_id": str(decision.decision_id or ""),
            "provider_id": str(decision.provider_id or "").strip() or None,
            "health_state": str(decision.health_state or "").strip() or None,
            "llm_fallback_active": bool(decision.llm_fallback_active) if decision.llm_fallback_active is not None else None,
            "fallback_level": str(decision.fallback_level or "").strip() or None,
            "boundary_mode": str(decision.boundary_mode or "blob_fallback"),
            "coverage_status": dict(decision.coverage_status or {}),
        }
        if isinstance(decision.segment_stats, Mapping):
            out["segment_stats"] = {str(k): int(v) for k, v in dict(decision.segment_stats).items()}
        if isinstance(decision.pressure_dedupe, Mapping):
            pd = dict(decision.pressure_dedupe)
            out["pressure_dedupe"] = {
                "input_count": int(pd.get("input_count", 0)),
                "kept_count": int(pd.get("kept_count", 0)),
                "deduped_count": int(pd.get("deduped_count", 0)),
                "deduped_by_artifact_id": int(pd.get("deduped_by_artifact_id", 0)),
                "deduped_by_content_hash_source": int(pd.get("deduped_by_content_hash_source", 0)),
            }
        if isinstance(decision.effect_wall_candidate, Mapping):
            out["effect_wall_candidate"] = dict(decision.effect_wall_candidate)
        if isinstance(decision.effect_policy_gate, Mapping):
            out["effect_policy_gate"] = dict(decision.effect_policy_gate)
        if isinstance(decision.artifact_assessment_summary, Mapping):
            out["artifact_assessment_summary"] = dict(decision.artifact_assessment_summary)
        if isinstance(decision.operation_gate_events, list):
            out["operation_gate_events"] = [
                dict(x) for x in list(decision.operation_gate_events) if isinstance(x, Mapping)
            ]
        if decision.skillbox_status is not None:
            out["skillbox_status"] = str(decision.skillbox_status)
        if isinstance(decision.skillbox_verification, Mapping):
            out["skillbox_verification"] = dict(decision.skillbox_verification)
        if decision.skillbox_ledger_hit is not None:
            out["skillbox_ledger_hit"] = bool(decision.skillbox_ledger_hit)
        if decision.skillbox_content_sha256 is not None:
            out["skillbox_content_sha256"] = str(decision.skillbox_content_sha256)
        if isinstance(decision.skillbox_capabilities, list):
            out["skillbox_capabilities"] = [str(x) for x in list(decision.skillbox_capabilities)]
        if decision.skillbox_gate_decision is not None:
            out["skillbox_gate_decision"] = str(decision.skillbox_gate_decision)
        out["siem_boundary_event"] = OmegaAdapterRuntime.build_siem_boundary_event(
            decision,
            phase=phase,
            extra=extra,
        )
        if isinstance(extra, Mapping):
            out.update(dict(extra))
        return out

    @staticmethod
    def _segment_from_mapping(raw: Mapping[str, Any]) -> AdapterSegment:
        return AdapterSegment(
            role=str(raw.get("role", "other") or "other"),
            origin=str(raw.get("origin", "unknown") or "unknown"),
            trust=str(raw.get("trust", "untrusted") or "untrusted"),
            text=str(raw.get("text", "") or ""),
            source_id=(str(raw.get("source_id")).strip() if raw.get("source_id") is not None else None),
            source_type=(str(raw.get("source_type")).strip() if raw.get("source_type") is not None else None),
            derived_from=(
                [str(x) for x in list(raw.get("derived_from", [])) if str(x).strip()]
                if isinstance(raw.get("derived_from"), list)
                else None
            ),
        )

    def check_model_input(
        self,
        messages_text: str,
        ctx: AdapterSessionContext,
        *,
        source_id: str = "adapter:model_input",
        source_type: Optional[str] = None,
        source_trust: Optional[str] = None,
        origin: str = "model_input",
        derived_from: Optional[List[str]] = None,
    ) -> AdapterDecision:
        if not str(ctx.session_id or "").strip():
            raise ValueError("AdapterSessionContext.session_id is required")
        harness = self._get_harness(ctx)
        item = self._build_item(
            text=messages_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            origin=origin,
            derived_from=derived_from,
            boundary_step=self._next_boundary_step(harness),
        )
        return self._run_items_check(
            ctx=ctx,
            user_query=messages_text,
            packet_items=[item],
            boundary_mode="blob_fallback",
        )

    def check_model_segments(
        self,
        segments: List[Mapping[str, Any]],
        ctx: AdapterSessionContext,
        *,
        source_type: Optional[str] = None,
    ) -> AdapterDecision:
        if not str(ctx.session_id or "").strip():
            raise ValueError("AdapterSessionContext.session_id is required")
        if not isinstance(segments, list) or not segments:
            raise ValueError("segments must be a non-empty list")

        harness = self._get_harness(ctx)
        boundary_step = self._next_boundary_step(harness)

        segment_objs: List[AdapterSegment] = []
        for raw in segments:
            if isinstance(raw, Mapping):
                segment_objs.append(self._segment_from_mapping(raw))

        packet_items: List[ContentItem] = []
        projected_trusts = {"untrusted", "tainted_internal", "semi_trusted", "mixed"}
        unknown_origin_count = 0

        for idx, seg in enumerate(segment_objs):
            origin_norm = self._normalize_origin(seg.origin)
            if origin_norm == "unknown":
                unknown_origin_count += 1
            trust_norm = self._normalize_trust(seg.trust, origin=origin_norm)

            if trust_norm not in projected_trusts:
                continue

            item_source_id = (
                str(seg.source_id)
                if str(seg.source_id or "").strip()
                else f"adapter:segment:{origin_norm}:{idx}"
            )
            item_source_type = str(seg.source_type or source_type or self.default_source_type)
            packet_items.append(
                self._build_item(
                    text=seg.text,
                    source_id=item_source_id,
                    source_type=item_source_type,
                    source_trust=trust_norm,
                    origin=origin_norm,
                    derived_from=list(seg.derived_from or []),
                    boundary_step=boundary_step,
                )
            )

        blob_user_query = self._clip(
            "\n".join(str(seg.text or "") for seg in segment_objs if str(seg.text or "").strip())
        )
        if not blob_user_query:
            blob_user_query = "segmented_input"

        return self._run_items_check(
            ctx=ctx,
            user_query=blob_user_query,
            packet_items=packet_items,
            boundary_mode="segmented",
            segment_stats={
                "total_segments": int(len(segment_objs)),
                "projected_segments": int(len(packet_items)),
                "skipped_trusted_segments": int(max(0, len(segment_objs) - len(packet_items))),
                "unknown_origin_to_untrusted": int(unknown_origin_count),
            },
        )

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
            origin="tool_call",
            boundary_step=self._next_boundary_step(harness),
        )
        request = ToolRequest(
            tool_name=str(tool_name),
            args={**dict(tool_args or {}), "tool_args": dict(tool_args or {}), "request_origin": "explicit"},
            session_id=str(ctx.session_id),
            step=int(getattr(harness.state, "step", 0)) + 1,
        )
        decision_ref, out = self._run_items_check(
            ctx=ctx,
            user_query=f"Tool preflight for {tool_name}",
            packet_items=[item],
            boundary_mode="blob_fallback",
            tool_requests=[request],
            return_raw=True,
        )

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
            operation_gate=(
                dict((out.get("operation_gate_events") or [None])[0])
                if isinstance((out.get("operation_gate_events") or [None])[0], Mapping)
                else None
            ),
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

        self._mark_boundary_observed(ctx, "memory_write_seen")
        trust = self._normalize_trust(source_trust, origin="memory_write")
        try:
            decision = self.check_model_input(
                memory_text,
                ctx,
                source_id=str(source_id),
                source_type=str(source_type or "other"),
                source_trust=str(source_trust or trust),
                origin="memory_write",
            )
        except TypeError:
            # Keep older monkeypatch-style tests and lightweight stubs compatible.
            decision = self.check_model_input(memory_text, ctx)
        item = self._build_item(
            text=memory_text,
            source_id=str(source_id),
            source_type=str(source_type or "other"),
            source_trust=str(source_trust or trust),
            origin="memory_write",
            boundary_step=int(decision.step),
            extra_meta={
                "runtime_integrity_operation": "memory_write",
                "source_tags": dict(source_tags or {}),
            },
        )
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
        operation_gate_decision = evaluate_operation_gate(
            config=self.config,
            intent=OperationIntent(
                operation_type="memory_write",
                target=str(source_id),
                source_artifact_ids=[str(item.artifact_id or "")],
                source_trust_states=[str(trust)],
                metadata={},
            ),
            artifact_assessments=[
                {
                    "artifact_id": str(item.artifact_id or ""),
                    "shadow_verdict": (
                        "quarantine" if (hard_block or quarantine) else "allow"
                    ),
                    "hard_invariant_hits": (
                        ["quarantined_source_artifact"] if (hard_block or quarantine) else []
                    ),
                }
            ],
        )
        if operation_gate_decision.status == "deny" and bool(operation_gate_decision.would_enforce):
            mode = "deny"
            allowed = False
            reason = str(operation_gate_decision.reason_code)

        return MemoryWriteDecision(
            allowed=allowed,
            mode=mode,
            reason=reason,
            source_id=str(source_id),
            source_type=str(source_type or "other"),
            source_trust=trust,
            tags=tags,
            decision_ref=decision,
            operation_gate=operation_gate_decision.to_dict(),
        )

    def check_memory_read(
        self,
        *,
        memory_text: str,
        source_id: str,
        source_type: str = "memory",
        source_trust: str = "tainted_internal",
        ctx: AdapterSessionContext,
        derived_from: Optional[List[str]] = None,
    ) -> AdapterDecision:
        if not str(source_id or "").strip():
            raise ValueError("source_id is required")
        self._mark_boundary_observed(ctx, "memory_read_seen")
        return self.check_model_input(
            messages_text=memory_text,
            ctx=ctx,
            source_id=str(source_id),
            source_type=str(source_type or "memory"),
            source_trust=str(source_trust or "tainted_internal"),
            origin="memory_read",
            derived_from=list(derived_from or []),
        )

    def check_tool_output_reentry(
        self,
        *,
        tool_name: str,
        output: Any,
        ctx: AdapterSessionContext,
        target: str,
        source_id: Optional[str] = None,
        source_type: str = "tool_output",
        source_trust: str = "untrusted",
        derived_from: Optional[List[str]] = None,
    ) -> Optional[AdapterDecision]:
        target_norm = str(target or "").strip().lower()
        # Reingest only when output flows into context/memory/agent-message.
        if target_norm not in {"context", "memory", "agent_message"}:
            return None

        output_text = extract_tool_output_text(output, max_chars=self.max_chars)
        if not str(output_text).strip():
            return None

        self._mark_boundary_observed(ctx, "tool_output_reentry_seen")
        if target_norm == "memory":
            self._mark_boundary_observed(ctx, "memory_write_seen")
        if target_norm == "agent_message":
            origin = "agent_message"
            default_trust = "tainted_internal"
        else:
            origin = "tool_output"
            default_trust = "untrusted"
        harness = self._get_harness(ctx)
        item = self._build_item(
            text=output_text,
            source_id=str(source_id or f"adapter:tool_output:{str(tool_name or 'unknown')}"),
            source_type=str(source_type or "tool_output"),
            source_trust=str(source_trust or default_trust),
            origin=origin,
            derived_from=list(derived_from or []),
            boundary_step=self._next_boundary_step(harness),
            extra_meta={
                "integrity_reentry_scanned": True,
                "runtime_integrity_operation": f"tool_output_reentry:{target_norm}",
            },
        )
        return self._run_items_check(
            ctx=ctx,
            user_query=output_text,
            packet_items=[item],
            boundary_mode="blob_fallback",
        )
