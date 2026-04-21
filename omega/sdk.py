"""Public SDK facade for frictionless Omega Walls integration."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import uuid

import numpy as np

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.errors import (
    OmegaAPIError,
    OmegaConfigError,
    OmegaInitializationError,
    OmegaMissingDependencyError,
    OmegaRuntimeError,
    OmegaSDKError,
)
from omega.interfaces.contracts_v1 import ContentItem, OffAction, OmegaState
from omega.log_contract import make_log_event
from omega.monitoring.collector import build_monitor_collector_from_config
from omega.monitoring.enrichment import build_downstream_summary, build_redacted_fragments
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.monitoring.models import MonitorEvent, utc_now_iso
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.structured_logging import build_structured_emitter_from_config, engine_version
from omega.sdk_types import DetectionResult, GuardAction, GuardDecision, OmegaDetectionResult


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


def _to_missing_dependency_error(exc: ModuleNotFoundError) -> OmegaMissingDependencyError:
    dep = str(getattr(exc, "name", "") or "unknown")
    return OmegaMissingDependencyError(
        dep,
        hint="Install required dependencies (for example: pip install omega-walls or omega-walls[api]).",
    )


def _is_api_error_message(message: str) -> bool:
    text = str(message or "").lower()
    markers = (
        "missing_api_key",
        "missing_env:",
        "api_call_failed:",
        "url_error:",
        "api_adapter",
        "schema_error:",
    )
    return any(m in text for m in markers)


def _map_exception(*, exc: Exception, phase: str) -> OmegaSDKError:
    if isinstance(exc, OmegaSDKError):
        return exc
    if isinstance(exc, ModuleNotFoundError):
        return _to_missing_dependency_error(exc)

    msg = str(exc)
    if phase == "config":
        return OmegaConfigError(f"Failed to load/validate config: {msg}")

    # API adapter may raise APIRequestError with code/body fields without exposing class publicly.
    if hasattr(exc, "code") and hasattr(exc, "body"):
        code = getattr(exc, "code", None)
        body = getattr(exc, "body", None)
        return OmegaAPIError(f"API request failed (code={code})", code=code, details={"body": body})
    if _is_api_error_message(msg):
        return OmegaAPIError(msg)

    if phase == "init":
        return OmegaInitializationError(f"Failed to initialize OmegaWalls: {msg}")
    return OmegaRuntimeError(f"Detection runtime failed: {msg}")


class OmegaWalls:
    """Stable public facade for text-level detection in 5-10 lines of code."""

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        config_dir: Optional[str] = None,
        projector_mode: Optional[str] = None,
        api_model: Optional[str] = None,
        cli_overrides: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
        default_trust: str = "untrusted",
        default_source_type: str = "other",
        default_session_id: str = "omega-sdk-default",
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

        try:
            snapshot = load_resolved_config(
                config_dir=str(config_dir) if config_dir else None,
                profile=str(profile),
                cli_overrides=effective_overrides or None,
                env=dict(env) if env is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="config") from exc

        cfg = snapshot.resolved
        try:
            params = omega_params_from_config(cfg)
            self._snapshot = snapshot
            self._config = cfg
            self._projector = build_projector(cfg)
            self._core = OmegaCoreV1(params)
            self._off_policy = OffPolicyV1(cfg)
            self._states: Dict[str, OmegaState] = {}
            self._guard_mode = resolve_guard_mode(cfg)
            self._monitor_collector = build_monitor_collector_from_config(
                config=cfg,
                force_enable=(self._guard_mode == GuardMode.MONITOR),
            )
            self._structured_emitter = build_structured_emitter_from_config(config=cfg, logger_name="omega.sdk")
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="init") from exc

        self.profile = str(profile)
        self.walls = list(params.walls)
        self.default_trust = str(default_trust or "untrusted")
        self.default_source_type = str(default_source_type or "other")
        self.default_session_id = str(default_session_id or "omega-sdk-default")

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def reset_session(self, session_id: Optional[str] = None) -> None:
        sid = str(session_id or self.default_session_id)
        self._states.pop(sid, None)

    def reset_all_sessions(self) -> None:
        self._states.clear()

    def analyze_text(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        source_id: str = "sdk:text",
        source_type: Optional[str] = None,
        trust: Optional[str] = None,
        doc_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        reset_session: bool = False,
    ) -> DetectionResult:
        sid = str(session_id or self.default_session_id)
        if bool(reset_session) or sid not in self._states:
            self._states[sid] = OmegaState(
                session_id=sid,
                m=np.zeros(len(self.walls), dtype=float),
                step=0,
            )
        state = self._states[sid]

        item = ContentItem(
            doc_id=str(doc_id or f"sdk-{uuid.uuid4().hex[:12]}"),
            source_id=str(source_id),
            source_type=str(source_type or self.default_source_type),
            trust=str(trust or self.default_trust),
            text=str(text or ""),
            meta=dict(meta or {}),
        )
        try:
            projection = self._projector.project(item)
            step_result = self._core.step(state=state, items=[item], projections=[projection])
            intended_decision = self._off_policy.select_actions(step_result=step_result, items=[item])
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="runtime") from exc

        monitor_enabled = bool(self._guard_mode == GuardMode.MONITOR)
        if monitor_enabled:
            decision = GuardDecision(
                off=bool(step_result.off),
                severity=str(intended_decision.severity),
                control_outcome="ALLOW",
                reason_codes=_reason_codes(step_result.reasons),
                top_docs=[str(x) for x in list(step_result.top_docs)],
                actions=[],
            )
            intended_action = str(intended_decision.control_outcome)
            actual_action = "ALLOW"
        else:
            decision = GuardDecision(
                off=bool(step_result.off),
                severity=str(intended_decision.severity),
                control_outcome=str(intended_decision.control_outcome),
                reason_codes=_reason_codes(step_result.reasons),
                top_docs=[str(x) for x in list(step_result.top_docs)],
                actions=[self._action_to_guard_action(action) for action in list(intended_decision.actions)],
            )
            intended_action = str(intended_decision.control_outcome)
            actual_action = str(intended_decision.control_outcome)

        wall_scores = {
            wall: float(step_result.p[idx])
            for idx, wall in enumerate(self.walls)
        }
        memory_scores = {
            wall: float(step_result.m_next[idx])
            for idx, wall in enumerate(self.walls)
        }
        walls_triggered = sorted(
            wall
            for idx, wall in enumerate(self.walls)
            if float(step_result.p[idx]) > 0.0 or float(step_result.m_next[idx]) > 0.0
        )
        max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
        sum_m = float(np.sum(step_result.m_next)) if len(step_result.m_next) else 0.0
        sum_ratio = min(1.0, sum_m / max(float(self._core.params.off_Sigma), 1e-6))
        severity_score = {"L1": 0.0, "L2": 0.5, "L3": 1.0}.get(str(intended_decision.severity), 0.0)
        risk_score = float(max(0.0, min(1.0, 0.60 * max_p + 0.30 * sum_ratio + 0.10 * severity_score)))
        fp_hint = infer_false_positive_hint(
            risk_score=float(risk_score),
            intended_action=str(intended_action),
            reason_codes=_reason_codes(step_result.reasons),
            triggered_rules=list(walls_triggered),
            attribution=[
                {
                    "doc_id": str(item.doc_id),
                    "source_id": str(item.source_id),
                    "trust": str(item.trust),
                    "contribution": 1.0,
                }
            ],
            config=self._config,
        )
        monitor_attribution = [
            {
                "doc_id": str(item.doc_id),
                "source_id": str(item.source_id),
                "trust": str(item.trust),
                "contribution": 1.0,
            }
        ]
        monitor_fragments = build_redacted_fragments(
            attribution_rows=monitor_attribution,
            item_text_by_doc={str(item.doc_id): str(item.text)},
            max_fragments=1,
            max_chars=240,
        )
        intended_action_types = sorted({str(a.type) for a in list(intended_decision.actions)})
        intended_blocked_doc_ids = sorted(
            {
                str(doc_id)
                for action in list(intended_decision.actions)
                if str(action.type) == "SOFT_BLOCK"
                for doc_id in list(action.doc_ids or [])
            }
        )
        intended_quarantined_sources = sorted(
            {
                str(source)
                for action in list(intended_decision.actions)
                if str(action.type) == "SOURCE_QUARANTINE"
                for source in list(action.source_ids or [])
            }
        )
        intended_prevented_tools = ["*"] if "TOOL_FREEZE" in set(intended_action_types) else []
        monitor_downstream = build_downstream_summary(
            intended_action=str(intended_action),
            action_types=list(intended_action_types),
            blocked_doc_ids=intended_blocked_doc_ids,
            quarantined_source_ids=intended_quarantined_sources,
            prevented_tools=intended_prevented_tools,
        )
        monitor_rules = {
            "triggered_rules": list(walls_triggered),
            "reason_codes": _reason_codes(step_result.reasons),
        }
        monitor_payload = {
            "enabled": bool(monitor_enabled),
            "guard_mode": str(self._guard_mode.value).lower(),
            "intended_action": str(intended_action),
            "actual_action": str(actual_action),
            "triggered_rules": list(walls_triggered),
            "rules": monitor_rules,
            "fragments": monitor_fragments,
            "downstream": monitor_downstream,
            "false_positive_hint": fp_hint,
        }
        self._structured_emitter.emit(
            make_log_event(
                event="risk_assessed",
                session_id=str(sid),
                mode=str(self._guard_mode.value).lower(),
                engine_version=engine_version(),
                risk_score=float(risk_score),
                intended_action_native=str(intended_action),
                actual_action_native=str(actual_action),
                action_types=list(intended_action_types),
                triggered_rules=list(walls_triggered),
                attribution_rows=list(monitor_attribution),
                fragments=list(monitor_fragments),
                fp_hint=(str(fp_hint) if fp_hint else None),
                ts=utc_now_iso(),
                trace_id=f"sdk:{sid}:{int(step_result.step)}",
                decision_id=f"sdk_dec:{sid}:{int(step_result.step)}:{str(actual_action)}",
                surface="sdk",
                input_type="prompt",
                input_length=len(str(text or "")),
                source_type=str(item.source_type),
            )
        )
        trace_id = f"sdk:{sid}:{int(step_result.step)}"
        decision_id = f"sdk_dec:{sid}:{int(step_result.step)}:{str(actual_action)}"
        if bool(self._monitor_collector.enabled):
            self._monitor_collector.emit(
                MonitorEvent(
                    ts=utc_now_iso(),
                    surface="sdk",
                    session_id=str(sid),
                    actor_id=str(sid),
                    mode=str(self._guard_mode.value).lower(),
                    risk_score=float(risk_score),
                    intended_action=str(intended_action),
                    actual_action=str(actual_action),
                    triggered_rules=list(walls_triggered),
                    attribution=monitor_attribution,
                    reason_codes=_reason_codes(step_result.reasons),
                    rules=monitor_rules,
                    fragments=monitor_fragments,
                    downstream=monitor_downstream,
                    trace_id=str(trace_id),
                    decision_id=str(decision_id),
                    false_positive_hint=(str(fp_hint) if fp_hint else None),
                    metadata={"source_id": str(item.source_id), "source_type": str(item.source_type)},
                )
            )

        return DetectionResult(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            walls_triggered=walls_triggered,
            wall_scores=wall_scores,
            memory_scores=memory_scores,
            decision=decision,
            monitor=monitor_payload,
        )

    def _action_to_guard_action(self, action: OffAction) -> GuardAction:
        return GuardAction(
            type=str(action.type),
            target=str(action.target),
            doc_ids=list(action.doc_ids) if action.doc_ids else None,
            source_ids=list(action.source_ids) if action.source_ids else None,
            tool_mode=str(action.tool_mode) if action.tool_mode else None,
            allowlist=list(action.allowlist) if action.allowlist else None,
            horizon_steps=int(action.horizon_steps) if action.horizon_steps is not None else None,
            incident_packet=dict(action.incident_packet) if action.incident_packet else None,
        )
