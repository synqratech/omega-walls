"""End-to-end Omega harness with pluggable LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np

from omega.effects.runtime import evaluate_typed_effect_shadow
from omega.interfaces.contracts_v1 import (
    ContentItem,
    OffAction,
    OmegaOffReasons,
    OmegaState,
    ToolDecision,
    ToolRequest,
)
from omega.log_contract import make_log_event
from omega.monitoring.collector import build_monitor_collector_from_config
from omega.monitoring.enrichment import (
    build_downstream_summary,
    build_redacted_fragments,
)
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.monitoring.models import MonitorEvent
from omega.notifications.dispatcher import (
    build_dispatcher_from_config,
    infer_major_triggers,
)
from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso
from omega.notifications.startup_flow import run_startup_notifications
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.policy.enforcement_state import EnforcementStateManager
from omega.rag.context_builder import ContextBuilder
from omega.rag.attachment_ingestion import extract_attachment
from omega.runtime.integrity_policy import (
    assess_runtime_artifact,
    build_runtime_artifact,
    runtime_integrity_enabled,
    summarize_artifact_assessments,
)
from omega.runtime.operation_gate import evaluate_operation_gate
from omega.runtime.artifacts import OperationIntent
from omega.runtime.skillbox import SkillBox, evaluate_skillbox_shadow
from omega.runtime.scan_pipeline import (
    apply_semantic_failure_policy_to_actions,
    compose_control_outcome_state,
    compose_effective_actions as shared_compose_effective_actions,
    compose_enforcement_phase,
    project_items_phase,
    projection_semantic_failed as shared_projection_semantic_failed,
    run_omega_step_phase,
    semantic_failure_policy_from_config as shared_semantic_failure_policy_from_config,
)
from omega.telemetry.events import (
    build_enforcement_step_event,
    build_evidence_debug_event,
    build_off_event,
    build_policy_decision_event,
    build_step_event,
    build_tool_gateway_step_event,
)
from omega.telemetry.ids import build_decision_id, build_trace_id_runtime
from omega.telemetry.incident_artifact import (
    build_incident_artifact,
    should_capture_incident_text,
    should_emit_incident_artifact,
)
from omega.telemetry.anonymous import AnonymousTelemetryService, build_telemetry_event
from omega.structured_logging import (
    build_structured_emitter_from_config,
    engine_version,
)
from omega.tools.adapters import ToolExecution, build_default_tool_registry

LOGGER = logging.getLogger(__name__)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MockLLM:
    def generate(self, prompt: str) -> Dict[str, Any]:
        tool_intents: List[str] = []
        if "CALL_TOOL(" in prompt:
            tool_intents.append("detected_inline_tool_intent")
        return {"text": "mocked_response", "tool_intents": tool_intents}


class OmegaRAGHarness:
    def __init__(
        self,
        projector,
        omega_core,
        off_policy,
        tool_gateway,
        config: Dict[str, Any],
        llm_backend: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        system_prompt: str = "You are a secure assistant.",
    ) -> None:
        self.projector = projector
        self.omega_core = omega_core
        self.off_policy = off_policy
        self.tool_gateway = tool_gateway
        self.config = config
        self.context_builder = ContextBuilder()
        self.llm = llm_backend or MockLLM()
        self.tool_registry = tool_registry or build_default_tool_registry()
        self.system_prompt = system_prompt
        self.state = OmegaState(
            session_id=f"sess-local-{uuid.uuid4().hex[:10]}",
            m=np.zeros(4, dtype=float),
            step=0,
        )
        self.enforcement = EnforcementStateManager.from_config(config)
        self.cross_session = CrossSessionStateManager.from_config(config)
        self.skillbox = SkillBox.from_config(config)
        if hasattr(self.tool_registry, "list_tools"):
            self.tool_gateway.ensure_tool_coverage(
                list(self.tool_registry.list_tools())
            )
        self._default_actor_id: Optional[str] = None
        self._warned_actor_fallback: bool = False
        tuning_cfg = (self.config.get("off_policy", {}) or {}).get(
            "stateful_support_tuning", {}
        ) or {}
        self._support_tuning_enabled = bool(tuning_cfg.get("enabled", False))
        self._support_tuning_cfg = (
            dict(tuning_cfg) if isinstance(tuning_cfg, dict) else {}
        )
        self._support_combo_streak = 0
        self._support_continuity_hits = 0
        self._support_sq_streak = 0
        self.guard_mode = resolve_guard_mode(self.config)
        self.monitor_collector = build_monitor_collector_from_config(
            config=self.config,
            force_enable=(self.guard_mode == GuardMode.MONITOR),
        )
        self.notification_dispatcher = build_dispatcher_from_config(config=self.config)
        if self.notification_dispatcher is not None and hasattr(
            self.tool_gateway, "bind_approval_store"
        ):
            self.tool_gateway.bind_approval_store(self.notification_dispatcher.store)
        self.telemetry_service = AnonymousTelemetryService(
            config=self.config,
            dispatcher=self.notification_dispatcher,
            surface="runtime",
            start_worker=False,
        )
        self._last_trusted_control_alert_step: int = 0
        self.structured_emitter = build_structured_emitter_from_config(
            config=self.config, logger_name="omega.runtime"
        )
        profile_name = str(
            (
                (self.config.get("profiles", {}) or {}).get("env", "")
                or (self.config.get("runtime", {}) or {}).get("mode", "custom")
            )
        )
        self.startup_summary = run_startup_notifications(
            config=self.config,
            profile=profile_name,
            surface="runtime",
            projector=self.projector,
            dispatcher=self.notification_dispatcher,
        )

    def close(self) -> None:
        if getattr(self, "telemetry_service", None) is not None:
            self.telemetry_service.close()
        if self.notification_dispatcher is not None:
            self.notification_dispatcher.close()

    def reset_state(
        self, session_id: Optional[str] = None, actor_id: Optional[str] = None
    ) -> None:
        self.state.m = np.zeros_like(self.state.m)
        self.state.step = 0
        self.enforcement.reset()
        self._support_combo_streak = 0
        self._support_continuity_hits = 0
        self._support_sq_streak = 0
        if session_id is not None:
            self.state.session_id = session_id
        self._default_actor_id = actor_id

    def _resolve_actor_id(self, actor_id: Optional[str]) -> str:
        if actor_id:
            return actor_id
        if self._default_actor_id:
            return self._default_actor_id

        fallback_enabled = bool(
            getattr(self.cross_session, "fallback_actor_to_session", True)
        )
        if fallback_enabled:
            if not self._warned_actor_fallback:
                LOGGER.warning(
                    "actor_id is missing; fallback to session_id for cross-session state"
                )
                self._warned_actor_fallback = True
            return self.state.session_id
        raise ValueError(
            "actor_id is required when cross_session.fallback_actor_to_session=false"
        )

    @staticmethod
    def _compose_effective_actions(
        policy_actions: List[OffAction], active_actions: List[OffAction]
    ) -> List[OffAction]:
        out: List[OffAction] = []
        out.extend(
            action
            for action in policy_actions
            if action.type in {"SOFT_BLOCK", "HUMAN_ESCALATE"}
        )
        out.extend(action for action in active_actions if action.type == "TOOL_FREEZE")
        out.extend(
            action for action in active_actions if action.type == "SOURCE_QUARANTINE"
        )
        out.extend(
            action
            for action in policy_actions
            if action.type
            not in {"SOFT_BLOCK", "HUMAN_ESCALATE", "TOOL_FREEZE", "SOURCE_QUARANTINE"}
        )
        return out

    @staticmethod
    def _extract_tool_requests_from_text(
        text: str, session_id: str, step: int
    ) -> List[ToolRequest]:
        pattern = re.compile(
            r"CALL_TOOL\s*\(\s*[\"']([a-zA-Z0-9_:-]+)[\"']\s*(?:,\s*(.*?))?\)",
            re.DOTALL,
        )
        requests: List[ToolRequest] = []
        for idx, match in enumerate(pattern.finditer(text), start=1):
            tool_name = match.group(1)
            raw_args = (match.group(2) or "").strip()
            requests.append(
                ToolRequest(
                    tool_name=tool_name,
                    args={
                        "raw_args": raw_args,
                        "intent_id": idx,
                        "request_origin": "inferred",
                    },
                    session_id=session_id,
                    step=step,
                )
            )
        return requests

    @staticmethod
    def _marker_hits(text_norm: str, markers: List[str]) -> List[str]:
        return sorted(
            {
                str(m).strip().lower()
                for m in markers
                if str(m).strip() and str(m).strip().lower() in text_norm
            }
        )

    def _build_risk_event(
        self,
        *,
        control_outcome: str,
        action_types: List[str],
        trace_id: str,
        decision_id: str,
        step: int,
        session_id: str,
        actor_id: str,
        severity: str,
        incident_artifact_id: Optional[str],
        reason_flags: List[str],
        risk_score: float,
    ) -> RiskEvent:
        semantic_active = bool(getattr(self.projector, "semantic_active", True))
        api_status_fn = getattr(self.projector, "api_perception_status", None)
        api_status = api_status_fn() if callable(api_status_fn) else {}
        api_fallback = bool((api_status or {}).get("llm_fallback_active", False))
        fallback_active = (not semantic_active) or api_fallback
        triggers = infer_major_triggers(
            control_outcome=str(control_outcome),
            action_types=action_types,
            fallback_active=fallback_active,
        )
        return RiskEvent(
            event_id=new_event_id(),
            timestamp=utc_now_iso(),
            surface="runtime",
            control_outcome=str(control_outcome),
            triggers=triggers,
            reasons=list(reason_flags),
            action_types=list(action_types),
            trace_id=str(trace_id),
            decision_id=str(decision_id),
            incident_artifact_id=str(incident_artifact_id or ""),
            tenant_id="runtime",
            session_id=str(session_id),
            actor_id=str(actor_id),
            step=int(step),
            severity=str(severity),
            risk_score=float(max(0.0, min(1.0, risk_score))),
            payload_redacted={
                "control_outcome": str(control_outcome),
                "action_types": list(action_types),
                "reasons": list(reason_flags),
                "trace_id": str(trace_id),
                "decision_id": str(decision_id),
                "incident_artifact_id": str(incident_artifact_id or ""),
                "session_id": str(session_id),
                "actor_id": str(actor_id),
            },
        )

    def _monitor_attribution(
        self,
        *,
        step_result: Any,
        items: List[ContentItem],
    ) -> List[Dict[str, Any]]:
        item_by_id = {str(item.doc_id): item for item in items}
        top_docs = {str(x) for x in list(step_result.top_docs)}
        rows: List[Dict[str, Any]] = []
        for contrib in list(step_result.contribs):
            doc_id = str(getattr(contrib, "doc_id", ""))
            if top_docs and doc_id not in top_docs:
                continue
            item = item_by_id.get(doc_id)
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_id": str(getattr(contrib, "source_id", "")),
                    "trust": str(getattr(item, "trust", "untrusted"))
                    if item is not None
                    else "untrusted",
                    "contribution": float(getattr(contrib, "c", 0.0)),
                }
            )
        rows.sort(
            key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", "")))
        )
        return rows[:8]

    @staticmethod
    def _item_origin(item: ContentItem) -> str:
        origin = (
            str(item.origin or ((item.meta or {}).get("origin", "")) or "")
            .strip()
            .lower()
        )
        return origin or "unknown"

    @staticmethod
    def _trust_label(item: ContentItem) -> str:
        return str(item.trust or "untrusted").strip().lower() or "untrusted"

    def _is_trusted_control_item(self, item: ContentItem) -> bool:
        trust = self._trust_label(item)
        origin = self._item_origin(item)
        if trust == "trusted_control":
            return True
        if trust == "trusted" and origin in {
            "system",
            "developer",
            "policy",
            "config",
            "trusted_control",
        }:
            return True
        return False

    @staticmethod
    def _content_hash(item: ContentItem) -> str:
        return str(item.content_hash or "") or _sha256_hex(str(item.text or ""))

    def _resolve_item_boundary_step(self, item: ContentItem) -> int:
        if item.boundary_step is not None:
            return int(item.boundary_step)
        meta = item.meta or {}
        meta_step = meta.get("boundary_step")
        if meta_step is not None:
            return int(meta_step)
        # Keep dedupe local to current harness step when boundary metadata is absent.
        return int(self.state.step) + 1

    def _dedupe_pressure_items_step_local(
        self,
        items: List[ContentItem],
    ) -> Tuple[List[ContentItem], Dict[str, Any]]:
        dedupe_seen: Set[str] = set()
        kept: List[ContentItem] = []
        dropped_rows: List[Dict[str, Any]] = []
        dropped_by_artifact = 0
        dropped_by_content = 0

        for item in list(items or []):
            boundary_step = self._resolve_item_boundary_step(item)
            artifact_id = str(
                item.artifact_id or ((item.meta or {}).get("artifact_id", "")) or ""
            ).strip()
            content_hash = self._content_hash(item)
            source_id = str(item.source_id or "")

            dedupe_key = ""
            dedupe_basis = "none"
            if artifact_id:
                dedupe_key = f"{boundary_step}|artifact:{artifact_id}"
                dedupe_basis = "artifact_id"
            else:
                # Preserve source-level signal: same hash from different sources is not deduped.
                dedupe_key = f"{boundary_step}|hash:{content_hash}|source:{source_id}"
                dedupe_basis = "content_hash_source"

            if dedupe_key in dedupe_seen:
                if dedupe_basis == "artifact_id":
                    dropped_by_artifact += 1
                else:
                    dropped_by_content += 1
                dropped_rows.append(
                    {
                        "doc_id": str(item.doc_id),
                        "artifact_id": (artifact_id or None),
                        "content_hash": content_hash,
                        "source_id": source_id,
                        "boundary_step": int(boundary_step),
                        "dedupe_basis": dedupe_basis,
                    }
                )
                continue

            dedupe_seen.add(dedupe_key)
            kept.append(item)

        return kept, {
            "enabled": True,
            "input_count": int(len(items or [])),
            "kept_count": int(len(kept)),
            "deduped_count": int(len(dropped_rows)),
            "deduped_by_artifact_id": int(dropped_by_artifact),
            "deduped_by_content_hash_source": int(dropped_by_content),
            "dropped": dropped_rows[:50],
        }

    def _trusted_control_audit_rows(
        self, items: List[ContentItem]
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in list(items):
            rows.append(
                {
                    "doc_id": str(item.doc_id),
                    "artifact_id": str(
                        item.artifact_id
                        or ((item.meta or {}).get("artifact_id", ""))
                        or ""
                    )
                    or None,
                    "source_id": str(item.source_id),
                    "source_type": str(item.source_type),
                    "trust": str(item.trust),
                    "origin": self._item_origin(item),
                    "content_hash": self._content_hash(item),
                    "text_sha256": _sha256_hex(str(item.text or "")),
                    "text_len": int(len(str(item.text or ""))),
                    "boundary_step": (
                        int(item.boundary_step)
                        if item.boundary_step is not None
                        else (
                            int((item.meta or {}).get("boundary_step"))
                            if (item.meta or {}).get("boundary_step") is not None
                            else None
                        )
                    ),
                    "excluded_from_pressure": True,
                }
            )
        return rows

    def _trusted_control_guard(
        self,
        *,
        total_docs: int,
        trusted_control_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cfg = (
            (
                (self.config.get("off_policy", {}) or {}).get("trust_boundary", {})
                or {}
            ).get("trusted_control_guard", {})
        ) or {}
        enabled = bool(cfg.get("enabled", True))
        min_total_docs = max(1, int(cfg.get("min_total_docs_for_ratio", 6)))
        warn_ratio_gte = float(cfg.get("warn_ratio_gte", 0.75))
        warn_count_gte = max(1, int(cfg.get("warn_count_gte", 12)))
        cooldown_steps = max(0, int(cfg.get("alert_cooldown_steps", 5)))
        emit_structured_alert = bool(cfg.get("emit_structured_alert", True))
        emit_notification_alert = bool(cfg.get("emit_notification_alert", False))
        raw_policy_action = (
            str(cfg.get("policy_action_on_trigger", "none")).strip().lower()
        )
        policy_action_on_trigger = (
            raw_policy_action
            if raw_policy_action in {"none", "warn", "human_escalate"}
            else "none"
        )

        trusted_control_count = int(len(trusted_control_rows))
        pressure_count = max(0, int(total_docs) - trusted_control_count)
        trusted_control_ratio = (
            float(trusted_control_count / float(total_docs))
            if int(total_docs) > 0
            else 0.0
        )

        reasons: List[str] = []
        if (
            enabled
            and int(total_docs) >= min_total_docs
            and trusted_control_ratio >= warn_ratio_gte
        ):
            reasons.append("trusted_control_ratio_high")
        if enabled and trusted_control_count >= warn_count_gte:
            reasons.append("trusted_control_count_high")
        triggered = bool(reasons)
        alert_suppressed_by_cooldown = bool(
            triggered
            and cooldown_steps > 0
            and int(self.state.step) > 0
            and (int(self.state.step) - int(self._last_trusted_control_alert_step))
            < cooldown_steps
        )
        should_alert = bool(triggered and not alert_suppressed_by_cooldown)
        return {
            "enabled": enabled,
            "triggered": triggered,
            "reasons": sorted(set(reasons)),
            "reason_code": ("trusted_control_overuse" if triggered else None),
            "trusted_control_count": trusted_control_count,
            "pressure_count": pressure_count,
            "total_docs": int(total_docs),
            "trusted_control_ratio": float(trusted_control_ratio),
            "thresholds": {
                "min_total_docs_for_ratio": int(min_total_docs),
                "warn_ratio_gte": float(warn_ratio_gte),
                "warn_count_gte": int(warn_count_gte),
            },
            "alert_cooldown_steps": int(cooldown_steps),
            "alert_suppressed_by_cooldown": bool(alert_suppressed_by_cooldown),
            "should_alert": bool(should_alert),
            "emit_structured_alert": bool(emit_structured_alert),
            "emit_notification_alert": bool(emit_notification_alert),
            "policy_action_on_trigger": str(policy_action_on_trigger),
            "policy_action_applied": "none",
        }

    def _apply_cross_session_carryover_signal(
        self,
        *,
        user_query: str,
        carryover_applied: bool,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "hit": False,
            "order_hits": [],
            "action_hits": [],
            "walls_applied": {},
        }
        cs_cfg = self.config.get("off_policy", {}).get("cross_session", {})
        sig_cfg = cs_cfg.get("carryover_signal", {}) if isinstance(cs_cfg, dict) else {}
        if (
            not isinstance(sig_cfg, dict)
            or not bool(sig_cfg.get("enabled", False))
            or not carryover_applied
        ):
            return out

        text_norm = " ".join(str(user_query or "").lower().split())
        order_tokens = [str(x).lower() for x in sig_cfg.get("order_tokens", [])]
        action_markers = [str(x).lower() for x in sig_cfg.get("action_markers", [])]
        order_hits = self._marker_hits(text_norm, order_tokens)
        action_hits = self._marker_hits(text_norm, action_markers)
        min_order_hits = max(1, int(sig_cfg.get("min_order_hits", 2)))
        min_action_hits = max(1, int(sig_cfg.get("min_action_hits", 1)))
        if len(order_hits) < min_order_hits or len(action_hits) < min_action_hits:
            return out

        floor_by_wall = sig_cfg.get("floor_by_wall", {})
        if not isinstance(floor_by_wall, dict):
            return out
        walls = list(self.config.get("omega", {}).get("walls", []))
        walls_applied: Dict[str, float] = {}
        for wall, floor in floor_by_wall.items():
            wall_name = str(wall)
            if wall_name not in walls:
                continue
            try:
                floor_val = max(0.0, min(1.0, float(floor)))
            except (TypeError, ValueError):
                continue
            idx = walls.index(wall_name)
            before = float(self.state.m[idx])
            self.state.m[idx] = max(self.state.m[idx], floor_val)
            if float(self.state.m[idx]) > before:
                walls_applied[wall_name] = float(self.state.m[idx])

        out["order_hits"] = order_hits
        out["action_hits"] = action_hits
        out["walls_applied"] = walls_applied
        out["hit"] = True
        return out

    def _contains_marker(self, text_norm: str, markers: List[str]) -> bool:
        for marker in markers:
            mk = str(marker).strip().lower()
            if mk and mk in text_norm:
                return True
        return False

    def _semantic_failure_policy(self) -> str:
        return str(shared_semantic_failure_policy_from_config(self.config))

    @staticmethod
    def _projection_semantic_failed(projections: List[Any]) -> bool:
        return bool(shared_projection_semantic_failed(projections))

    def _apply_stateful_support_tuning(
        self, *, user_query: str, step_result: Any
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "enabled": bool(self._support_tuning_enabled),
            "applied": False,
            "continuity_hit": False,
            "review_bypass_hit": False,
            "combo_hit": False,
            "dynamic_tau": None,
            "combo_streak": int(self._support_combo_streak),
            "source_quarantine_streak": int(self._support_sq_streak),
        }
        if not self._support_tuning_enabled:
            return out

        text_norm = " ".join(str(user_query or "").lower().split())
        cfg = self._support_tuning_cfg
        continuity_markers = list(
            cfg.get(
                "continuity_markers",
                [
                    "continuity",
                    "earlier case",
                    "prior case",
                    "reference case",
                    "legacy service path",
                    "same service footprint",
                ],
            )
        )
        action_markers = list(
            cfg.get(
                "action_markers",
                [
                    "please send",
                    "please apply",
                    "restore",
                    "resend",
                    "switch",
                    "update",
                    "service credit",
                    "archived statements",
                ],
            )
        )
        review_bypass_markers = list(
            cfg.get(
                "review_bypass_markers",
                [
                    "queue can catch up",
                    "review queue can catch up",
                    "before month-end",
                    "close the books today",
                    "pending review",
                ],
            )
        )
        continuity_hit = self._contains_marker(
            text_norm, continuity_markers
        ) and self._contains_marker(text_norm, action_markers)
        review_bypass_hit = self._contains_marker(
            text_norm, review_bypass_markers
        ) and self._contains_marker(text_norm, action_markers)
        out["continuity_hit"] = bool(continuity_hit)
        out["review_bypass_hit"] = bool(review_bypass_hit)

        if continuity_hit or review_bypass_hit:
            self._support_continuity_hits += 1
        else:
            self._support_continuity_hits = max(
                0, int(self._support_continuity_hits) - 1
            )

        walls = list((self.config.get("omega", {}) or {}).get("walls", []))
        idx_by_wall = {str(w): i for i, w in enumerate(walls)}
        participating = {
            str(walls[i])
            for i in range(min(len(walls), len(step_result.p), len(step_result.m_next)))
            if float(step_result.p[i]) > 0.0 or float(step_result.m_next[i]) > 0.0
        }
        combo_walls = list(
            cfg.get("combo_walls", ["override_instructions", "tool_or_action_abuse"])
        )
        combo_hit = all(str(w) in participating for w in combo_walls)
        out["combo_hit"] = bool(combo_hit)
        if combo_hit:
            self._support_combo_streak += 1
        else:
            self._support_combo_streak = 0

        min_hits_for_boost = max(1, int(cfg.get("min_hits_for_boost", 2)))
        if self._support_continuity_hits >= min_hits_for_boost:
            boost_walls = list(
                cfg.get("boost_walls", ["policy_evasion", "tool_or_action_abuse"])
            )
            boost_delta = float(cfg.get("boost_m_next_delta", 0.16))
            for wall in boost_walls:
                idx = idx_by_wall.get(str(wall))
                if idx is None:
                    continue
                step_result.m_next[idx] = float(step_result.m_next[idx]) + float(
                    boost_delta
                )
            self.state.m = np.asarray(step_result.m_next, dtype=float)
            out["applied"] = True

        tau = float(self.omega_core.params.off_tau)
        combo_min_streak = max(1, int(cfg.get("combo_min_streak", 2)))
        combo_tau_delta = float(cfg.get("combo_tau_delta", 0.25))
        combo_tau_floor = float(cfg.get("combo_tau_floor", 0.62))
        if self._support_combo_streak >= combo_min_streak:
            tau = max(float(combo_tau_floor), float(tau - combo_tau_delta))
            out["applied"] = True

        sq_min_streak = max(1, int(cfg.get("sq_min_streak", 2)))
        sq_tau_override = float(cfg.get("sq_tau_override", 0.62))
        sq_min_p = float(cfg.get("sq_min_p", 0.18))
        max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
        if self._support_sq_streak >= sq_min_streak and max_p >= sq_min_p:
            tau = min(float(tau), float(sq_tau_override))
            out["applied"] = True

        reason_spike = bool(max_p >= tau)
        reason_wall = bool(
            float(np.max(step_result.m_next)) >= float(self.omega_core.params.off_Theta)
        )
        reason_sum = bool(
            float(np.sum(step_result.m_next)) >= float(self.omega_core.params.off_Sigma)
        )
        reason_multi = bool(
            int(
                np.count_nonzero(
                    np.asarray(step_result.m_next, dtype=float)
                    >= float(self.omega_core.params.off_theta)
                )
            )
            >= int(self.omega_core.params.off_N)
        )
        step_result.reasons = OmegaOffReasons(
            reason_spike=bool(reason_spike),
            reason_wall=bool(reason_wall),
            reason_sum=bool(reason_sum),
            reason_multi=bool(reason_multi),
        )
        step_result.off = bool(
            reason_spike or reason_wall or reason_sum or reason_multi
        )
        out["dynamic_tau"] = float(tau)
        out["combo_streak"] = int(self._support_combo_streak)
        out["source_quarantine_streak"] = int(self._support_sq_streak)
        return out

    def _phase_preprocess_inputs(
        self, packet_items: List[ContentItem]
    ) -> Dict[str, Any]:
        packet_items = list(packet_items or [])
        trusted_control_items = [
            item for item in packet_items if self._is_trusted_control_item(item)
        ]
        trusted_control_doc_ids = {str(item.doc_id) for item in trusted_control_items}
        pressure_items_raw = [
            item
            for item in packet_items
            if str(item.doc_id) not in trusted_control_doc_ids
        ]
        pressure_items, pressure_dedupe = self._dedupe_pressure_items_step_local(
            pressure_items_raw
        )
        trusted_control_audit = self._trusted_control_audit_rows(trusted_control_items)
        trusted_control_guard = self._trusted_control_guard(
            total_docs=len(packet_items),
            trusted_control_rows=trusted_control_audit,
        )
        return {
            "packet_items": packet_items,
            "trusted_control_items": trusted_control_items,
            "trusted_control_doc_ids": trusted_control_doc_ids,
            "pressure_items": pressure_items,
            "pressure_dedupe": pressure_dedupe,
            "trusted_control_audit": trusted_control_audit,
            "trusted_control_guard": trusted_control_guard,
        }

    def _phase_artifact_integrity(
        self,
        *,
        packet_items: List[ContentItem],
        trusted_control_doc_ids: Set[str],
        effect_shadow: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not runtime_integrity_enabled(self.config):
            return {
                "artifacts": [],
                "artifact_assessments": [],
                "artifact_assessment_summary": {
                    "artifact_count": 0,
                    "assessment_count": 0,
                    "kind_counts": {},
                    "trust_counts": {},
                    "shadow_verdict_counts": {},
                    "hard_invariant_hits": [],
                },
                "assessment_by_doc_id": {},
            }
        cfg = ((self.config or {}).get("runtime_integrity", {}) or {})
        emit_artifact_trace = bool(cfg.get("emit_artifact_trace", True))
        artifacts = []
        assessments = []
        assessment_by_doc_id: Dict[str, Dict[str, Any]] = {}
        for item in list(packet_items or []):
            artifact = build_runtime_artifact(
                item,
                trusted_control_excluded=(str(item.doc_id) in trusted_control_doc_ids),
            )
            assessment = assess_runtime_artifact(artifact, effect_shadow=effect_shadow)
            artifacts.append(artifact)
            assessments.append(assessment)
            assessment_by_doc_id[str(item.doc_id)] = assessment.to_dict()
        summary = summarize_artifact_assessments(
            artifacts=artifacts,
            assessments=assessments,
        )
        summary["packet_effect_signal"] = {
            "effect_forecast_status": str(
                effect_shadow.get("effect_forecast_status", "disabled")
            ),
            "effect_policy_gate_status": str(
                effect_shadow.get("effect_policy_gate_status", "disabled")
            ),
            "has_effect_candidate": bool(
                isinstance(effect_shadow.get("effect_wall_candidate"), dict)
            ),
        }
        summary["named_skill_invocation"] = (
            dict(effect_shadow.get("named_skill_invocation", {}))
            if isinstance(effect_shadow.get("named_skill_invocation"), dict)
            else None
        )
        summary["skill_provenance_assessment"] = (
            dict(effect_shadow.get("skill_provenance_assessment", {}))
            if isinstance(effect_shadow.get("skill_provenance_assessment"), dict)
            else None
        )
        summary["skillbox_status"] = str(effect_shadow.get("skillbox_status", "disabled"))
        summary["skillbox_verification"] = (
            dict(effect_shadow.get("skillbox_verification", {}))
            if isinstance(effect_shadow.get("skillbox_verification"), dict)
            else None
        )
        summary["skillbox_ledger_hit"] = bool(effect_shadow.get("skillbox_ledger_hit", False))
        summary["skillbox_content_sha256"] = effect_shadow.get("skillbox_content_sha256")
        summary["skillbox_capabilities"] = [
            str(x) for x in list(effect_shadow.get("skillbox_capabilities", []) or [])
        ]
        summary["skillbox_gate_decision"] = str(
            effect_shadow.get("skillbox_gate_decision", "disabled")
        )
        return {
            "artifacts": (
                [artifact.to_dict() for artifact in artifacts]
                if emit_artifact_trace
                else []
            ),
            "artifact_assessments": (
                [assessment.to_dict() for assessment in assessments]
                if emit_artifact_trace
                else []
            ),
            "artifact_assessment_summary": summary,
            "assessment_by_doc_id": assessment_by_doc_id,
        }

    @staticmethod
    def _operation_type_for_tool_request(
        request: ToolRequest,
        *,
        capability_class: str,
    ) -> str:
        if str(capability_class or "").upper() == "PRIV_ESC":
            return "privilege_change"
        if bool((request.args or {}).get("resource_heavy", False)):
            return "resource_heavy_action"
        if str((request.args or {}).get("request_origin", "")).strip().lower() == "skill_install":
            return "skill_install"
        return "tool_call"

    def _phase_hydrate_cross_session(
        self, *, resolved_actor_id: str, user_query: str
    ) -> Dict[str, Any]:
        cross_hydrated = self.cross_session.hydrate_actor_state(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
        )
        self.state.m = np.maximum(
            self.state.m, cross_hydrated.carried_scars_after_decay
        )
        carryover_signal = self._apply_cross_session_carryover_signal(
            user_query=user_query,
            carryover_applied=bool(cross_hydrated.carryover_applied),
        )
        return {
            "cross_hydrated": cross_hydrated,
            "carryover_signal": carryover_signal,
        }

    def _phase_projection_and_policy(
        self, *, pressure_items: List[ContentItem], user_query: str
    ) -> Dict[str, Any]:
        projection_phase = project_items_phase(
            projector=self.projector,
            cfg=self.config,
            items=pressure_items,
        )
        projections = list(projection_phase.projections)
        step_result = run_omega_step_phase(
            omega_core=self.omega_core,
            state=self.state,
            items=pressure_items,
            projections=projections,
        )
        support_tuning = self._apply_stateful_support_tuning(
            user_query=user_query, step_result=step_result
        )
        policy_decision = self.off_policy.select_actions(
            step_result=step_result, items=pressure_items
        )
        return {
            "projection_phase": projection_phase,
            "projections": projections,
            "step_result": step_result,
            "support_tuning": support_tuning,
            "policy_decision": policy_decision,
        }

    def run_attachment_step(
        self,
        *,
        user_query: str,
        content: bytes,
        filename: str,
        mime: str,
        tenant_id: str = "harness",
        data_region: str = "unspecified",
        source_id: Optional[str] = None,
        trust: str = "untrusted",
        actor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a document/image through the same multimodal packet contract as the API."""
        raw = bytes(content)
        request_id = f"harness-attachment-{uuid.uuid4().hex}"
        attachment_cfg = (
            (self.config.get("retriever", {}) or {}).get("sqlite_fts", {}) or {}
        ).get("attachments", {}) or {}
        try:
            extracted = extract_attachment(
                content_bytes=raw,
                filename=str(filename),
                mime=str(mime),
                cfg=attachment_cfg,
            )
            register = getattr(self.projector, "register_image_blob", None)
            variants: list[dict[str, Any]] = []
            for asset in list(extracted.visual_assets or []):
                if not callable(register):
                    raise RuntimeError("projector_missing_multimodal_blob_boundary")
                payload = asset.decode()
                variants.append(
                    {
                        "mime": asset.mime,
                        "sha256": asset.sha256,
                        "bytes_ref": register(
                            scope_id=request_id,
                            data=payload,
                            mime=asset.mime,
                            expected_sha256=asset.sha256,
                        ),
                        "size_bytes": asset.size_bytes,
                        "role": asset.role,
                        "width": asset.width or None,
                        "height": asset.height or None,
                        "asset_id": asset.asset_id,
                        "source_kind": asset.source_kind,
                        "page_number": asset.page_number,
                        "embedded_index": asset.embedded_index,
                    }
                )
            text = str(extracted.text or "").strip() or (
                "[attachment_visual_only]" if variants else "[attachment_text_empty]"
            )
            meta: dict[str, Any] = {
                "tenant_id": str(tenant_id),
                "data_region": str(data_region),
                "request_id": request_id,
                "attachment_format": str(extracted.format),
                "ingestion_flags": list(extracted.warnings),
                "visual_status": str(extracted.visual_status),
                "visual_asset_manifest": [
                    {k: v for k, v in row.items() if k != "bytes_ref"}
                    for row in variants
                ],
            }
            if variants:
                meta["semantic_image"] = (
                    variants[0] if len(variants) == 1 else {"variants": variants}
                )
            item = ContentItem(
                doc_id=f"harness-att-{uuid.uuid4().hex[:12]}",
                source_id=str(
                    source_id
                    or f"harness:{extracted.format}:{hashlib.sha256(raw).hexdigest()[:16]}"
                ),
                source_type=str(extracted.format or "attachment"),
                trust=str(trust),
                text=text,
                meta=meta,
            )
            return self.run_step(
                user_query=user_query, packet_items=[item], actor_id=actor_id
            )
        finally:
            release = getattr(self.projector, "release_image_scope", None)
            if callable(release):
                release(request_id)

    def run_step(
        self,
        user_query: str,
        packet_items: List[ContentItem],
        tool_requests: Optional[List[ToolRequest]] = None,
        actor_id: Optional[str] = None,
        config_refs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        enforcement_mode = str(
            self.config.get("off_policy", {}).get("enforcement_mode", "ENFORCE")
        ).upper()
        tools_execution_mode = str(
            self.config.get("tools", {}).get("execution_mode", "ENFORCE")
        ).upper()
        guard_mode = self.guard_mode
        monitor_enabled = bool(guard_mode == GuardMode.MONITOR)
        resolved_actor_id = self._resolve_actor_id(actor_id)
        input_phase = self._phase_preprocess_inputs(packet_items)
        packet_items = list(input_phase["packet_items"])
        trusted_control_doc_ids = set(input_phase["trusted_control_doc_ids"])
        pressure_items = list(input_phase["pressure_items"])
        pressure_dedupe = dict(input_phase["pressure_dedupe"])
        trusted_control_audit = list(input_phase["trusted_control_audit"])
        trusted_control_guard = dict(input_phase["trusted_control_guard"])

        cross_phase = self._phase_hydrate_cross_session(
            resolved_actor_id=resolved_actor_id,
            user_query=user_query,
        )
        carryover_signal = dict(cross_phase["carryover_signal"])

        proj_phase = self._phase_projection_and_policy(
            pressure_items=pressure_items,
            user_query=user_query,
        )
        projection_phase = proj_phase["projection_phase"]
        step_result = proj_phase["step_result"]
        support_tuning = dict(proj_phase["support_tuning"])
        policy_decision = proj_phase["policy_decision"]
        skillbox_shadow = evaluate_skillbox_shadow(
            config=self.config,
            items=pressure_items,
            user_query=user_query,
            source_meta={
                "session_id": str(self.state.session_id),
                "actor_id": str(resolved_actor_id),
                "surface": "runtime",
            },
            skillbox=self.skillbox,
        )
        effect_shadow = evaluate_typed_effect_shadow(
            config=self.config,
            projector=self.projector,
            items=pressure_items,
            user_query=user_query,
            source_meta={
                "session_id": str(self.state.session_id),
                "actor_id": str(resolved_actor_id),
                "surface": "runtime",
            },
            forecaster=getattr(self, "effect_forecaster", None),
            skillbox=self.skillbox,
        )
        effect_shadow = {
            **dict(effect_shadow),
            **{
                key: value
                for key, value in dict(skillbox_shadow).items()
                if key
                in {
                    "named_skill_invocation",
                    "skill_provenance_assessment",
                    "skillbox_status",
                    "skillbox_verification",
                    "skillbox_ledger_hit",
                    "skillbox_content_sha256",
                    "skillbox_capabilities",
                    "skillbox_gate_decision",
                }
            },
        }
        artifact_integrity = self._phase_artifact_integrity(
            packet_items=packet_items,
            trusted_control_doc_ids=trusted_control_doc_ids,
            effect_shadow=effect_shadow,
        )
        assessment_by_doc_id = dict(artifact_integrity["assessment_by_doc_id"])

        self.cross_session.record_step(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
            step_result=step_result,
            policy_actions=policy_decision.actions,
            packet_items=pressure_items,
        )
        cross_active_actions = self.cross_session.active_actions(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
            step=step_result.step,
        )
        cross_snapshot = self.cross_session.snapshot(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
            step=step_result.step,
        )
        cross_block = cross_snapshot.get("cross_session", {})
        if isinstance(cross_block, dict):
            cross_block["carryover_signal_hit"] = bool(
                carryover_signal.get("hit", False)
            )
            cross_block["carryover_signal_walls_applied"] = dict(
                carryover_signal.get("walls_applied", {})
            )
            cross_block["carryover_signal_order_hits"] = list(
                carryover_signal.get("order_hits", [])
            )
            cross_block["carryover_signal_action_hits"] = list(
                carryover_signal.get("action_hits", [])
            )
            cross_block["stateful_support_tuning"] = dict(support_tuning)

        effective_actions = shared_compose_effective_actions(
            policy_actions=policy_decision.actions,
            cross_active_actions=cross_active_actions,
        )
        effective_actions = apply_semantic_failure_policy_to_actions(
            actions=effective_actions,
            semantic_phase=projection_phase,
            session_id=str(step_result.session_id),
            step=int(step_result.step),
        )
        trusted_control_policy_action = (
            str(trusted_control_guard.get("policy_action_on_trigger", "none"))
            .strip()
            .lower()
        )
        if bool(trusted_control_guard.get("triggered", False)):
            if trusted_control_policy_action == "warn":
                if not any(str(a.type) == "WARN" for a in list(effective_actions)):
                    effective_actions.append(OffAction(type="WARN", target="SESSION"))
                    trusted_control_guard["policy_action_applied"] = "warn"
            elif trusted_control_policy_action == "human_escalate":
                if not any(
                    str(a.type) == "HUMAN_ESCALATE" for a in list(effective_actions)
                ):
                    effective_actions.append(
                        OffAction(
                            type="HUMAN_ESCALATE",
                            target="AGENT",
                            incident_packet={
                                "reason": "trusted_control_overuse_policy_action",
                                "session_id": str(step_result.session_id),
                                "step": int(step_result.step),
                            },
                        )
                    )
                    trusted_control_guard["policy_action_applied"] = "human_escalate"
        enforcement_phase = compose_enforcement_phase(
            policy_decision=policy_decision,
            effective_actions=effective_actions,
            monitor_enabled=monitor_enabled,
            enforcement_mode=enforcement_mode,
            tools_execution_mode=tools_execution_mode,
        )
        intended_decision = enforcement_phase.intended_decision
        decision = enforcement_phase.decision
        enforcement_actions = list(enforcement_phase.enforcement_actions)
        tools_execution_mode = str(enforcement_phase.tools_execution_mode)

        if any(str(a.type) == "SOURCE_QUARANTINE" for a in enforcement_actions):
            self._support_sq_streak += 1
        else:
            self._support_sq_streak = 0

        trace_id = build_trace_id_runtime(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            doc_ids=sorted({str(item.doc_id) for item in pressure_items}),
        )
        intended_action_types = list(enforcement_phase.intended_action_types)
        action_types = list(enforcement_phase.action_types)
        intended_action = str(enforcement_phase.intended_action)
        actual_action = str(enforcement_phase.actual_action)
        decision_id = build_decision_id(
            trace_id=trace_id,
            control_outcome=str(
                intended_action if monitor_enabled else decision.control_outcome
            ),
            action_types=intended_action_types if monitor_enabled else action_types,
            severity=str(decision.severity),
            off=bool(step_result.off),
        )

        blocked = set()
        for action in enforcement_actions:
            if action.type == "SOFT_BLOCK" and action.doc_ids:
                blocked.update(action.doc_ids)
        for action in enforcement_actions:
            if action.type != "SOURCE_QUARANTINE" or not action.source_ids:
                continue
            blocked.update(
                item.doc_id
                for item in pressure_items
                if item.source_id in set(action.source_ids)
            )
        intended_blocked = set()
        intended_quarantined_source_ids = sorted(
            {
                source_id
                for action in list(intended_decision.actions)
                if str(action.type) == "SOURCE_QUARANTINE"
                for source_id in list(action.source_ids or [])
            }
        )
        for action in list(intended_decision.actions):
            if str(action.type) == "SOFT_BLOCK" and action.doc_ids:
                intended_blocked.update(action.doc_ids)
            if str(action.type) == "SOURCE_QUARANTINE" and action.source_ids:
                intended_blocked.update(
                    item.doc_id
                    for item in pressure_items
                    if item.source_id in set(action.source_ids)
                )

        allowed_pressure_ids = {
            str(item.doc_id)
            for item in pressure_items
            if str(item.doc_id) not in blocked
        }
        allowed_items = [
            item
            for item in packet_items
            if str(item.doc_id) in trusted_control_doc_ids
            or str(item.doc_id) in allowed_pressure_ids
        ]
        context = self.context_builder.build_context(
            system_prompt=self.system_prompt,
            user_query=user_query,
            allowed_items=allowed_items,
            diagnostics={
                "off": step_result.off,
                "reasons": step_result.reasons.__dict__,
                "top_docs": step_result.top_docs,
            },
        )
        llm_response = self.llm.generate(context)

        inferred_requests = self._extract_tool_requests_from_text(
            llm_response.get("text", ""),
            session_id=self.state.session_id,
            step=step_result.step,
        )
        latest_approval_status = "none"
        merged_requests: List[tuple[ToolRequest, str]] = []
        for request in list(tool_requests or []):
            request.tenant_id = "runtime"
            request.actor_id = str(resolved_actor_id)
            if "request_origin" not in request.args:
                request.args["request_origin"] = "explicit"
            merged_requests.append((request, "explicit"))
        for request in inferred_requests:
            request.tenant_id = "runtime"
            request.actor_id = str(resolved_actor_id)
            merged_requests.append((request, "inferred"))

        tool_decisions = []
        tool_executions: List[ToolExecution] = []
        tool_gateway_events: List[Dict[str, Any]] = []
        operation_gate_events: List[Dict[str, Any]] = []
        skillbox_verification = effect_shadow.get("skillbox_verification")
        if isinstance(skillbox_verification, dict):
            operation_gate_events.append(
                {
                    "status": str(effect_shadow.get("skillbox_gate_decision", "shadow_only")),
                    "reason_code": str(skillbox_verification.get("reason_code", "")),
                    "shadow_only": True,
                    "would_enforce": False,
                    "requires_approval": bool(
                        skillbox_verification.get("requires_approval", False)
                    ),
                    "hard_invariant_hits": [],
                    "details": {
                        "operation_type": "skill_run",
                        "target": str(skillbox_verification.get("skill_name", "") or "unknown"),
                        "provenance_status": str(
                            skillbox_verification.get("verification_status", "")
                        ),
                        "simulated_block": bool(
                            skillbox_verification.get("simulated_block", False)
                        ),
                        "ledger_hit": bool(effect_shadow.get("skillbox_ledger_hit", False)),
                    },
                }
            )
        else:
            skill_provenance_assessment = effect_shadow.get("skill_provenance_assessment")
            if isinstance(skill_provenance_assessment, dict):
                operation_gate_events.append(
                    {
                        "status": "shadow_only",
                        "reason_code": str(skill_provenance_assessment.get("reason_code", "")),
                        "shadow_only": True,
                        "would_enforce": False,
                        "requires_approval": bool(
                            skill_provenance_assessment.get("requires_approval", False)
                        ),
                        "hard_invariant_hits": [],
                        "details": {
                            "operation_type": "skill_run",
                            "target": str(skill_provenance_assessment.get("skill_name", "") or "unknown"),
                            "provenance_status": str(skill_provenance_assessment.get("status", "")),
                            "simulated_block": bool(
                                skill_provenance_assessment.get("simulated_block", False)
                            ),
                        },
                    }
                )
        off_state = bool(self.tool_gateway.is_off_state(enforcement_actions))
        freeze_active = bool(
            self.tool_gateway.find_freeze(enforcement_actions) is not None
        )
        actor_hash = str(
            (cross_snapshot.get("cross_session") or {}).get("actor_hash", "n/a")
        )
        source_ids_seen = sorted({item.source_id for item in allowed_items})[:16]

        for request, request_origin in merged_requests:
            capability_obj = self.tool_gateway.capability_for(request.tool_name)
            capability = {
                "mode": capability_obj.mode
                if capability_obj is not None
                else "unknown",
                "requires_human_approval": capability_obj.requires_human_approval
                if capability_obj is not None
                else False,
                "capability_class": capability_obj.capability_class
                if capability_obj is not None
                else "unknown",
                "risk_level": capability_obj.risk_level
                if capability_obj is not None
                else "unknown",
            }
            source_artifact_ids = []
            source_trust_states = []
            source_assessments = []
            for item in list(allowed_items):
                artifact_id = str(
                    item.artifact_id or ((item.meta or {}).get("artifact_id", "")) or ""
                ).strip()
                if artifact_id:
                    source_artifact_ids.append(artifact_id)
                source_trust_states.append(str(item.trust or "untrusted"))
                row = assessment_by_doc_id.get(str(item.doc_id))
                if isinstance(row, dict):
                    source_assessments.append(row)
            operation_intent = OperationIntent(
                operation_type=self._operation_type_for_tool_request(
                    request,
                    capability_class=capability.get("capability_class", "unknown"),
                ),
                target=str(request.tool_name),
                source_artifact_ids=sorted(set(source_artifact_ids)),
                source_trust_states=sorted(set(source_trust_states)),
                capability_class=str(capability.get("capability_class", "unknown")),
                risk_level=str(capability.get("risk_level", "unknown")),
                approval_present=bool(getattr(request, "approval_id", None)),
                metadata={
                    "request_origin": str(request_origin),
                    "resource_heavy": bool((request.args or {}).get("resource_heavy", False)),
                    "budget_available": bool((request.args or {}).get("budget_available", False)),
                    "source_mismatch": bool((request.args or {}).get("source_mismatch", False)),
                    "skillbox_verification_status": (
                        str((effect_shadow.get("skillbox_verification") or {}).get("verification_status", ""))
                        if isinstance(effect_shadow.get("skillbox_verification"), dict)
                        else ""
                    ),
                    "skillbox_gate_decision": str(effect_shadow.get("skillbox_gate_decision", "")),
                },
            )
            operation_gate = evaluate_operation_gate(
                config=self.config,
                intent=operation_intent,
                artifact_assessments=source_assessments,
            )
            operation_gate_events.append(operation_gate.to_dict())
            if operation_gate.status == "deny" and bool(operation_gate.would_enforce):
                decision_out = ToolDecision(
                    allowed=False,
                    mode="TOOLS_DISABLED",
                    reason=str(operation_gate.reason_code),
                    logged=True,
                    validation_status="skipped",
                    validation_reason=None,
                    capability_class=capability.get("capability_class"),
                    risk_level=capability.get("risk_level"),
                    approval_required=bool(operation_gate.requires_approval),
                    intent_id=request.intent_id,
                    approval_id=request.approval_id,
                )
            else:
                decision_out = self.tool_gateway.enforce(request, enforcement_actions)
            tool_decisions.append(decision_out)
            adapter_present = bool(self.tool_registry.has(request.tool_name))
            human_approved = bool(decision_out.allowed and decision_out.approval_id)
            executed = False

            if not decision_out.allowed:
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=False,
                        executed=False,
                        reason=decision_out.reason,
                    )
                )
                tool_gateway_events.append(
                    build_tool_gateway_step_event(
                        session_id=self.state.session_id,
                        step=step_result.step,
                        tool_name=request.tool_name,
                        request_origin=request_origin,
                        intent_id=request.args.get("intent_id"),
                        decision={
                            "allowed": decision_out.allowed,
                            "reason": decision_out.reason,
                            "mode": decision_out.mode,
                            "off_state": off_state,
                            "freeze_active": freeze_active,
                            "validation_status": decision_out.validation_status,
                            "validation_reason": decision_out.validation_reason,
                            "capability_class": decision_out.capability_class,
                            "risk_level": decision_out.risk_level,
                            "approval_required": decision_out.approval_required,
                            "operation_gate": operation_gate.to_dict(),
                        },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
                        control_outcome=decision.control_outcome,
                        trace_id=trace_id,
                        decision_id=decision_id,
                    )
                )
                continue

            if tools_execution_mode == "DRY_RUN":
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=True,
                        executed=False,
                        reason="DRY_RUN_MODE",
                        output={"status": "dry_run", "tool_name": request.tool_name},
                    )
                )
                tool_gateway_events.append(
                    build_tool_gateway_step_event(
                        session_id=self.state.session_id,
                        step=step_result.step,
                        tool_name=request.tool_name,
                        request_origin=request_origin,
                        intent_id=request.args.get("intent_id"),
                        decision={
                            "allowed": decision_out.allowed,
                            "reason": decision_out.reason,
                            "mode": decision_out.mode,
                            "off_state": off_state,
                            "freeze_active": freeze_active,
                    "validation_status": decision_out.validation_status,
                    "validation_reason": decision_out.validation_reason,
                    "capability_class": decision_out.capability_class,
                    "risk_level": decision_out.risk_level,
                    "approval_required": decision_out.approval_required,
                    "operation_gate": operation_gate.to_dict(),
                },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
                        control_outcome=decision.control_outcome,
                        trace_id=trace_id,
                        decision_id=decision_id,
                    )
                )
                continue

            if not adapter_present:
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=True,
                        executed=False,
                        reason="NO_ADAPTER",
                        error=f"No adapter registered for {request.tool_name}",
                    )
                )
                tool_gateway_events.append(
                    build_tool_gateway_step_event(
                        session_id=self.state.session_id,
                        step=step_result.step,
                        tool_name=request.tool_name,
                        request_origin=request_origin,
                        intent_id=request.args.get("intent_id"),
                        decision={
                            "allowed": decision_out.allowed,
                            "reason": decision_out.reason,
                            "mode": decision_out.mode,
                            "off_state": off_state,
                            "freeze_active": freeze_active,
                            "validation_status": decision_out.validation_status,
                            "validation_reason": decision_out.validation_reason,
                            "capability_class": decision_out.capability_class,
                            "risk_level": decision_out.risk_level,
                            "approval_required": decision_out.approval_required,
                        },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
                        control_outcome=decision.control_outcome,
                        trace_id=trace_id,
                        decision_id=decision_id,
                    )
                )
                continue

            try:
                output = self.tool_registry.execute(
                    request,
                    context={
                        "allowed_items": allowed_items,
                        "session_id": self.state.session_id,
                        "step": step_result.step,
                        "tool_output_dir": self.config.get("tools", {}).get(
                            "output_dir", "artifacts/tools"
                        ),
                        "tenant_id": "runtime",
                        "actor_id": str(resolved_actor_id),
                    },
                )
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=True,
                        executed=True,
                        reason=decision_out.reason,
                        output=output,
                    )
                )
                executed = True
            except Exception as exc:  # pragma: no cover
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=True,
                        executed=False,
                        reason="ADAPTER_ERROR",
                        error=str(exc),
                    )
                )
            tool_gateway_events.append(
                build_tool_gateway_step_event(
                    session_id=self.state.session_id,
                    step=step_result.step,
                    tool_name=request.tool_name,
                    request_origin=request_origin,
                    intent_id=request.args.get("intent_id"),
                    decision={
                        "allowed": decision_out.allowed,
                        "reason": decision_out.reason,
                        "mode": decision_out.mode,
                        "off_state": off_state,
                        "freeze_active": freeze_active,
                        "validation_status": decision_out.validation_status,
                        "validation_reason": decision_out.validation_reason,
                        "capability_class": decision_out.capability_class,
                        "risk_level": decision_out.risk_level,
                        "approval_required": decision_out.approval_required,
                    },
                    capability=capability,
                    human_approved=human_approved,
                    executed=executed,
                    adapter_present=adapter_present,
                    execution_mode=tools_execution_mode,
                    actor_hash=actor_hash,
                    source_ids_seen=source_ids_seen,
                    control_outcome=decision.control_outcome,
                    trace_id=trace_id,
                    decision_id=decision_id,
                )
            )

        freeze_block = cross_snapshot.get("freeze", {})
        if isinstance(freeze_block, dict):
            freeze_action = next(
                (a for a in enforcement_actions if str(a.type) == "TOOL_FREEZE"), None
            )
            if freeze_action is not None:
                freeze_block["freeze_stage"] = (
                    int(freeze_action.freeze_stage)
                    if freeze_action.freeze_stage is not None
                    else None
                )
                freeze_block["stage_reason"] = (
                    str(freeze_action.stage_reason)
                    if freeze_action.stage_reason is not None
                    else None
                )
                freeze_block["escalation_required"] = bool(
                    freeze_action.escalation_required
                )
            else:
                freeze_block.setdefault("freeze_stage", None)
                freeze_block.setdefault("stage_reason", None)
                freeze_block.setdefault("escalation_required", False)

        step_event = build_step_event(
            step_result, trace_id=trace_id, decision_id=decision_id
        )
        enforcement_event = build_enforcement_step_event(
            session_id=step_result.session_id,
            step=step_result.step,
            enforcement_snapshot=cross_snapshot,
            active_actions=enforcement_actions,
            control_outcome=decision.control_outcome,
            cross_session=cross_snapshot.get("cross_session"),
            trace_id=trace_id,
            decision_id=decision_id,
        )
        off_event = None
        if step_result.off:
            off_event = build_off_event(
                step_result=step_result,
                decision=decision,
                items=packet_items,
                config_refs=config_refs or {"code_commit": "local"},
                thresholds={
                    "epsilon": self.config["omega"]["epsilon"],
                    "alpha": self.config["omega"]["alpha"],
                    "beta": self.config["omega"]["beta"],
                    "lambda": self.config["omega"]["lambda"],
                    "off": self.config["omega"]["off"],
                    "attrib_gamma": self.config["omega"]["attribution"]["gamma"],
                },
                capture_text=self.config.get("logging", {}).get(
                    "capture_text", "NEVER"
                ),
                max_text_chars=int(
                    self.config.get("logging", {}).get("max_text_chars", 800)
                ),
                trace_id=trace_id,
                decision_id=decision_id,
            )

        extra_reason_flags: List[str] = []
        if bool(trusted_control_guard.get("triggered", False)):
            extra_reason_flags.append("trusted_control_overuse")
            for extra_reason in list(trusted_control_guard.get("reasons", []) or []):
                if str(extra_reason).strip():
                    extra_reason_flags.append(str(extra_reason))
        phase_state = compose_control_outcome_state(
            walls=self.config["omega"]["walls"],
            step_result=step_result,
            policy_action_types=[str(a.type) for a in list(intended_decision.actions)],
            semantic_phase=projection_phase,
            extra_reason_flags=extra_reason_flags,
        )
        reason_flags = list(phase_state.reason_flags)
        walls_triggered = list(phase_state.walls_triggered)
        top_doc_set = set(step_result.top_docs)
        top_docs_summary = [
            {
                "doc_id": str(contrib.doc_id),
                "source_id": str(contrib.source_id),
                "contrib_c": float(contrib.c),
                "active_walls": [
                    str(self.config["omega"]["walls"][idx])
                    for idx, score in enumerate(list(contrib.v))
                    if float(score) > 0.0
                ],
            }
            for contrib in step_result.contribs
            if str(contrib.doc_id) in top_doc_set
        ]
        signal_hits: Dict[str, int] = {}
        for contrib in step_result.contribs:
            if str(contrib.doc_id) not in top_doc_set:
                continue
            for key, value in dict(contrib.evidence.matches or {}).items():
                if isinstance(value, bool) and value:
                    signal_hits[str(key)] = signal_hits.get(str(key), 0) + 1
                elif isinstance(value, (int, float)) and float(value) > 0.0:
                    signal_hits[str(key)] = signal_hits.get(str(key), 0) + 1
        projector_signal_summary = {
            "top_signal_hits": [
                {"signal": str(name), "hits": int(hits)}
                for name, hits in sorted(
                    signal_hits.items(), key=lambda row: (-int(row[1]), str(row[0]))
                )[:20]
            ],
            "top_docs_count": int(len(top_docs_summary)),
        }
        policy_decision_event = build_policy_decision_event(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            trace_id=trace_id,
            decision_id=decision_id,
            control_outcome=str(decision.control_outcome),
            off=bool(step_result.off),
            severity=str(decision.severity),
            action_types=action_types,
            actions=list(decision.actions),
            refs={
                **dict(config_refs or {"code_commit": "local"}),
                "guard_mode": str(guard_mode.value).lower(),
                "intended_action": str(intended_action),
                "actual_action": str(actual_action),
            },
        )
        evidence_debug_event = build_evidence_debug_event(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            trace_id=trace_id,
            decision_id=decision_id,
            walls=list(self.config["omega"]["walls"]),
            reasons=reason_flags,
            walls_triggered=walls_triggered,
            top_docs_summary=top_docs_summary,
            projector_signal_summary=projector_signal_summary,
        )

        incident_artifact = None
        incident_artifact_id = None
        if should_emit_incident_artifact(
            config=self.config, control_outcome=decision.control_outcome
        ):
            capture_incident_text = should_capture_incident_text(config=self.config)
            top_docs_lookup = {item.doc_id: item for item in packet_items}
            top_docs: List[Dict[str, Any]] = []
            for doc_id in list(step_result.top_docs):
                item = top_docs_lookup.get(doc_id)
                if item is None:
                    continue
                row: Dict[str, Any] = {
                    "doc_id": item.doc_id,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "trust": item.trust,
                    "text_sha256": _sha256_hex(str(item.text)),
                }
                if capture_incident_text:
                    row["text"] = str(item.text)
                top_docs.append(row)
            walls_triggered = [
                str(self.config["omega"]["walls"][idx])
                for idx, value in enumerate(step_result.p)
                if float(value) > 0.0 or float(step_result.m_next[idx]) > 0.0
            ]
            quarantined_source_ids = sorted(
                {
                    source_id
                    for action in enforcement_actions
                    if action.type == "SOURCE_QUARANTINE"
                    for source_id in list(action.source_ids or [])
                }
            )
            incident_artifact = build_incident_artifact(
                config=self.config,
                surface="runtime",
                session_id=str(step_result.session_id),
                step=int(step_result.step),
                control_outcome=str(decision.control_outcome),
                off=bool(step_result.off),
                severity=str(decision.severity),
                actions=list(decision.actions),
                reason_flags=[
                    key
                    for key, value in step_result.reasons.__dict__.items()
                    if bool(value)
                ],
                contributing_signals={
                    "max_p": float(np.max(step_result.p))
                    if len(step_result.p)
                    else 0.0,
                    "sum_m_next": float(np.sum(step_result.m_next))
                    if len(step_result.m_next)
                    else 0.0,
                    "walls_triggered": walls_triggered,
                    "top_docs_count": int(len(step_result.top_docs)),
                },
                top_docs=top_docs,
                blocked_doc_ids=sorted(blocked),
                quarantined_source_ids=quarantined_source_ids,
                tool_gateway_events=tool_gateway_events,
                context_total_docs=len(packet_items),
                context_allowed_docs=len(allowed_items),
                config_refs=config_refs or {"code_commit": "local"},
                refs={
                    "off_event_present": bool(off_event is not None),
                    "enforcement_event_present": bool(enforcement_event is not None),
                    "tool_gateway_events_count": int(len(tool_gateway_events)),
                },
                trace_id=trace_id,
                decision_id=decision_id,
            )
            incident_artifact_id = str(
                incident_artifact.get("incident_artifact_id", "")
            )

        max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
        sum_m = float(np.sum(step_result.m_next)) if len(step_result.m_next) else 0.0
        sum_ratio = min(1.0, sum_m / max(float(self.omega_core.params.off_Sigma), 1e-6))
        severity_score = {"L1": 0.0, "L2": 0.5, "L3": 1.0}.get(
            str(decision.severity), 0.0
        )
        risk_score = float(
            max(0.0, min(1.0, 0.60 * max_p + 0.30 * sum_ratio + 0.10 * severity_score))
        )
        monitor_attribution = self._monitor_attribution(
            step_result=step_result, items=packet_items
        )
        fp_hint = infer_false_positive_hint(
            risk_score=float(risk_score),
            intended_action=str(intended_action),
            reason_codes=list(reason_flags),
            triggered_rules=list(walls_triggered),
            attribution=monitor_attribution,
            config=self.config,
        )
        monitor_fragments = build_redacted_fragments(
            attribution_rows=monitor_attribution,
            item_text_by_doc={
                str(item.doc_id): str(item.text) for item in packet_items
            },
            item_meta_by_doc={
                str(item.doc_id): (
                    dict(item.meta) if isinstance(item.meta, dict) else {}
                )
                for item in packet_items
            },
            max_fragments=4,
            max_chars=240,
        )
        prevented_tools = sorted(
            {
                str(name)
                for name in (
                    [request.tool_name for request, _ in merged_requests]
                    if "TOOL_FREEZE" in intended_action_types
                    else []
                )
                if str(name).strip()
            }
        )
        monitor_downstream = build_downstream_summary(
            intended_action=str(intended_action),
            action_types=list(intended_action_types),
            blocked_doc_ids=sorted(str(x) for x in intended_blocked),
            quarantined_source_ids=list(intended_quarantined_source_ids),
            prevented_tools=prevented_tools,
        )
        monitor_rules = {
            "triggered_rules": list(walls_triggered),
            "reason_codes": list(reason_flags),
        }
        monitor_payload = {
            "enabled": bool(monitor_enabled),
            "guard_mode": str(guard_mode.value).lower(),
            "intended_action": str(intended_action),
            "actual_action": str(actual_action),
            "triggered_rules": list(walls_triggered),
            "rules": monitor_rules,
            "fragments": monitor_fragments,
            "downstream": monitor_downstream,
            "false_positive_hint": fp_hint,
            "semantic_projection": {
                "status": str(phase_state.semantic_failure_status),
                "policy": str(phase_state.semantic_failure_policy),
                "policy_branch": str(phase_state.semantic_policy_branch),
            },
            "trust_boundary": {
                "trusted_control_excluded_count": int(len(trusted_control_audit)),
                "trusted_control_segments": list(trusted_control_audit),
                "guard": dict(trusted_control_guard),
                "pressure_dedupe": dict(pressure_dedupe),
            },
        }
        self.structured_emitter.emit(
            make_log_event(
                event="risk_assessed",
                session_id=str(step_result.session_id),
                mode=str(guard_mode.value).lower(),
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
                trace_id=str(trace_id),
                decision_id=str(decision_id),
                surface="runtime",
                input_type="context_chunk",
                input_length=sum(
                    len(str(item.text or "")) for item in list(packet_items)
                ),
                source_type=(
                    str(packet_items[0].source_type)
                    if len(packet_items) == 1
                    else ("mixed" if len(packet_items) > 1 else None)
                ),
            )
        )
        if monitor_enabled:
            self.monitor_collector.emit(
                MonitorEvent(
                    ts=utc_now_iso(),
                    surface="runtime",
                    session_id=str(step_result.session_id),
                    actor_id=str(resolved_actor_id),
                    mode=str(guard_mode.value).lower(),
                    risk_score=float(risk_score),
                    intended_action=str(intended_action),
                    actual_action=str(actual_action),
                    triggered_rules=list(walls_triggered),
                    attribution=list(monitor_attribution),
                    reason_codes=list(reason_flags),
                    rules=monitor_rules,
                    fragments=monitor_fragments,
                    downstream=monitor_downstream,
                    trace_id=str(trace_id),
                    decision_id=str(decision_id),
                    false_positive_hint=(str(fp_hint) if fp_hint else None),
                    metadata={
                        "step": int(step_result.step),
                        "severity": str(decision.severity),
                    },
                )
            )
        risk_event = self._build_risk_event(
            control_outcome=str(decision.control_outcome),
            action_types=action_types,
            trace_id=str(trace_id),
            decision_id=str(decision_id),
            step=int(step_result.step),
            session_id=str(step_result.session_id),
            actor_id=str(resolved_actor_id),
            severity=str(decision.severity),
            incident_artifact_id=incident_artifact_id,
            reason_flags=list(reason_flags),
            risk_score=risk_score,
        )
        api_status_fn = getattr(self.projector, "api_perception_status", None)
        api_status = api_status_fn() if callable(api_status_fn) else {}
        fallback_active = (
            not bool(getattr(self.projector, "semantic_active", True))
        ) or bool((api_status or {}).get("llm_fallback_active", False))
        fallback_level = str((api_status or {}).get("fallback_level", "none") or "none")
        provenance_type = (
            str(packet_items[0].source_type)
            if len(packet_items) == 1
            else ("mixed" if len(packet_items) > 1 else "unknown")
        )
        orchestrator_cfg = (
            (self.config.get("projector", {}) or {}).get("api_perception", {}) or {}
        ).get("orchestrator", {}) or {}
        module_flags = {
            "orchestrator": bool(orchestrator_cfg.get("enabled", False)),
            "monitoring": bool(
                ((self.config.get("monitoring", {}) or {}).get("enabled", False))
            ),
            "notifications": bool(
                ((self.config.get("notifications", {}) or {}).get("enabled", False))
            ),
        }
        self.telemetry_service.emit_event(
            build_telemetry_event(
                surface="runtime",
                control_outcome=str(decision.control_outcome),
                severity=str(decision.severity),
                walls_triggered=list(walls_triggered),
                reason_codes=list(reason_flags),
                action_types=list(action_types),
                risk_score=float(risk_score),
                fallback_active=bool(fallback_active),
                fallback_level=str(fallback_level),
                accumulation_steps=int(step_result.step),
                provenance_type=str(provenance_type),
                module_flags=module_flags,
                fp_reported=False,
            )
        )
        if bool(trusted_control_guard.get("should_alert", False)):
            if bool(trusted_control_guard.get("emit_structured_alert", True)):
                self.structured_emitter.emit(
                    make_log_event(
                        event="trust_boundary_guard_alert",
                        session_id=str(step_result.session_id),
                        mode=str(guard_mode.value).lower(),
                        engine_version=engine_version(),
                        risk_score=float(
                            min(
                                1.0,
                                max(
                                    0.0,
                                    float(
                                        trusted_control_guard.get(
                                            "trusted_control_ratio", 0.0
                                        )
                                    ),
                                ),
                            )
                        ),
                        intended_action_native="WARN",
                        actual_action_native="WARN",
                        action_types=["SOURCE_QUARANTINE"],
                        triggered_rules=list(
                            sorted(
                                set(
                                    ["trusted_control_overuse"]
                                    + [
                                        str(x)
                                        for x in list(
                                            trusted_control_guard.get("reasons", [])
                                            or []
                                        )
                                    ]
                                )
                            )
                        ),
                        attribution_rows=[
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "source_id": str(row.get("source_id", "")),
                                "contribution": 1.0,
                            }
                            for row in list(trusted_control_audit)[:8]
                        ],
                        fragments=[],
                        fp_hint="trusted_control_overuse",
                        ts=utc_now_iso(),
                        trace_id=str(trace_id),
                        decision_id=str(decision_id),
                        surface="runtime",
                        input_type="context_chunk",
                        input_length=sum(
                            int(row.get("text_len", 0))
                            for row in list(trusted_control_audit)
                        ),
                        source_type="trusted_control",
                    )
                )
            if bool(trusted_control_guard.get("emit_notification_alert", False)):
                alert_event = self._build_risk_event(
                    control_outcome="WARN",
                    action_types=["SOURCE_QUARANTINE"],
                    trace_id=str(trace_id),
                    decision_id=str(decision_id),
                    step=int(step_result.step),
                    session_id=str(step_result.session_id),
                    actor_id=str(resolved_actor_id),
                    severity="L1",
                    incident_artifact_id=None,
                    reason_flags=list(
                        sorted(
                            set(
                                ["trusted_control_overuse"]
                                + [
                                    str(x)
                                    for x in list(
                                        trusted_control_guard.get("reasons", []) or []
                                    )
                                ]
                            )
                        )
                    ),
                    risk_score=float(
                        min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    trusted_control_guard.get(
                                        "trusted_control_ratio", 0.0
                                    )
                                ),
                            ),
                        )
                    ),
                )
                self.notification_dispatcher.emit_risk_event(alert_event)
            self._last_trusted_control_alert_step = int(step_result.step)
        self.notification_dispatcher.emit_risk_event(risk_event)
        policy_approval_required = ("HUMAN_ESCALATE" in action_types) or (
            "REQUIRE_APPROVAL" in action_types
        )
        gateway_approval_required = any(
            (not td.allowed) and bool(td.approval_required) for td in tool_decisions
        )
        approval_required = bool(policy_approval_required or gateway_approval_required)
        approval_id: Optional[str] = None
        tool_approvals: List[Dict[str, str]] = []
        timeout_sec = int(
            (
                (self.config.get("notifications", {}) or {}).get("approvals", {}) or {}
            ).get("timeout_sec", 900)
        )
        if policy_approval_required:
            required_action = (
                "HUMAN_ESCALATE"
                if "HUMAN_ESCALATE" in action_types
                else "REQUIRE_APPROVAL"
            )
            approval = self.notification_dispatcher.create_action_request(
                risk_event=risk_event,
                required_action=required_action,
                timeout_sec=max(10, timeout_sec),
            )
            approval_id = str(approval.approval_id)
            latest_approval_status = str(approval.status)
        for (request, _origin), tool_decision in zip(merged_requests, tool_decisions):
            if tool_decision.allowed or not bool(tool_decision.approval_required):
                continue
            approval = self.notification_dispatcher.create_tool_action_request(
                risk_event=risk_event,
                tool_request=request,
                timeout_sec=max(10, timeout_sec),
            )
            request.approval_id = str(approval.approval_id)
            request.intent_id = str(approval.tool_intent_id)
            tool_approvals.append(
                {
                    "approval_id": str(approval.approval_id),
                    "tool_name": str(approval.tool_name),
                    "tool_intent_id": str(approval.tool_intent_id),
                    "status": str(approval.status),
                }
            )
            if approval_id is None:
                approval_id = str(approval.approval_id)
            latest_approval_status = str(approval.status)

        return {
            "trace_id": trace_id,
            "decision_id": decision_id,
            "step_result": step_result,
            "control_outcome": decision.control_outcome,
            "decision": decision,
            "policy_decision": policy_decision,
            "allowed_items": allowed_items,
            "context": context,
            "llm_response": llm_response,
            "tool_decisions": tool_decisions,
            "tool_executions": tool_executions,
            "inferred_tool_requests": inferred_requests,
            "tool_gateway_events": tool_gateway_events,
            "step_event": step_event,
            "enforcement_event": enforcement_event,
            "policy_decision_event": policy_decision_event,
            "evidence_debug_event": evidence_debug_event,
            "off_event": off_event,
            "incident_artifact_id": incident_artifact_id,
            "incident_artifact": incident_artifact,
            "approval_required": bool(approval_required),
            "approval_id": approval_id,
            "approval_status": str(latest_approval_status),
            "tool_approvals": tool_approvals,
            "monitor": monitor_payload,
            "notification_metrics": self.notification_dispatcher.metrics_snapshot(),
            "monitoring_metrics": self.monitor_collector.health_snapshot(),
            "trusted_control_segments": trusted_control_audit,
            "trusted_control_guard": trusted_control_guard,
            "pressure_dedupe": pressure_dedupe,
            "pressure_items_count": int(len(pressure_items)),
            "artifacts": list(artifact_integrity.get("artifacts", [])),
            "artifact_assessments": list(
                artifact_integrity.get("artifact_assessments", [])
            ),
            "artifact_assessment_summary": dict(
                artifact_integrity.get("artifact_assessment_summary", {})
            ),
            "operation_gate_events": list(operation_gate_events),
            "operation_gate_summary": {
                "event_count": int(len(operation_gate_events)),
                "deny_count": int(
                    sum(1 for row in list(operation_gate_events) if str(row.get("status", "")) == "deny")
                ),
                "require_approval_count": int(
                    sum(
                        1
                        for row in list(operation_gate_events)
                        if bool(row.get("requires_approval", False))
                    )
                ),
            },
            "effect_wall_candidate": effect_shadow.get("effect_wall_candidate"),
            "effect_policy_gate": effect_shadow.get("effect_policy_gate"),
            "effect_policy_gate_status": str(
                effect_shadow.get("effect_policy_gate_status", "disabled")
            ),
            "effect_forecast": effect_shadow.get("effect_forecast"),
            "effect_forecast_status": str(
                effect_shadow.get("effect_forecast_status", "disabled")
            ),
            "named_skill_invocation": effect_shadow.get("named_skill_invocation"),
            "skill_provenance_assessment": effect_shadow.get("skill_provenance_assessment"),
            "skillbox_status": str(effect_shadow.get("skillbox_status", "disabled")),
            "skillbox_verification": effect_shadow.get("skillbox_verification"),
            "skillbox_ledger_hit": bool(effect_shadow.get("skillbox_ledger_hit", False)),
            "skillbox_content_sha256": effect_shadow.get("skillbox_content_sha256"),
            "skillbox_capabilities": list(effect_shadow.get("skillbox_capabilities", []) or []),
            "skillbox_gate_decision": str(
                effect_shadow.get("skillbox_gate_decision", "disabled")
            ),
            "semantic_failure_status": str(phase_state.semantic_failure_status),
            "semantic_failure_policy": str(phase_state.semantic_failure_policy),
            "semantic_failure_policy_branch": str(phase_state.semantic_policy_branch),
        }
