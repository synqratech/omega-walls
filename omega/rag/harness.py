"""End-to-end Omega harness with pluggable LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaOffReasons, OmegaState, ToolRequest
from omega.log_contract import make_log_event
from omega.monitoring.collector import build_monitor_collector_from_config
from omega.monitoring.enrichment import build_downstream_summary, build_redacted_fragments
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.monitoring.models import MonitorEvent
from omega.notifications.dispatcher import build_dispatcher_from_config, infer_major_triggers
from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso
from omega.notifications.startup_flow import run_startup_notifications
from omega.policy.control_outcome import control_outcome_from_actions
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.policy.enforcement_state import EnforcementStateManager
from omega.rag.context_builder import ContextBuilder
from omega.telemetry.events import (
    build_enforcement_step_event,
    build_evidence_debug_event,
    build_off_event,
    build_policy_decision_event,
    build_step_event,
    build_tool_gateway_step_event,
)
from omega.telemetry.ids import build_decision_id, build_trace_id_runtime
from omega.telemetry.incident_artifact import build_incident_artifact, should_emit_incident_artifact
from omega.structured_logging import build_structured_emitter_from_config, engine_version
from omega.tools.adapters import ToolExecution, build_default_tool_registry

LOGGER = logging.getLogger(__name__)


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
        self.state = OmegaState(session_id=f"sess-local-{uuid.uuid4().hex[:10]}", m=np.zeros(4, dtype=float), step=0)
        self.enforcement = EnforcementStateManager.from_config(config)
        self.cross_session = CrossSessionStateManager.from_config(config)
        if hasattr(self.tool_registry, "list_tools"):
            self.tool_gateway.ensure_tool_coverage(list(self.tool_registry.list_tools()))
        self._default_actor_id: Optional[str] = None
        self._warned_actor_fallback: bool = False
        tuning_cfg = ((self.config.get("off_policy", {}) or {}).get("stateful_support_tuning", {}) or {})
        self._support_tuning_enabled = bool(tuning_cfg.get("enabled", False))
        self._support_tuning_cfg = dict(tuning_cfg) if isinstance(tuning_cfg, dict) else {}
        self._support_combo_streak = 0
        self._support_continuity_hits = 0
        self._support_sq_streak = 0
        self.guard_mode = resolve_guard_mode(self.config)
        self.monitor_collector = build_monitor_collector_from_config(
            config=self.config,
            force_enable=(self.guard_mode == GuardMode.MONITOR),
        )
        self.notification_dispatcher = build_dispatcher_from_config(config=self.config)
        self.structured_emitter = build_structured_emitter_from_config(config=self.config, logger_name="omega.runtime")
        profile_name = str(((self.config.get("profiles", {}) or {}).get("env", "") or (self.config.get("runtime", {}) or {}).get("mode", "custom")))
        self.startup_summary = run_startup_notifications(
            config=self.config,
            profile=profile_name,
            surface="runtime",
            projector=self.projector,
            dispatcher=self.notification_dispatcher,
        )

    def close(self) -> None:
        if self.notification_dispatcher is not None:
            self.notification_dispatcher.close()

    def reset_state(self, session_id: Optional[str] = None, actor_id: Optional[str] = None) -> None:
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

        fallback_enabled = bool(getattr(self.cross_session, "fallback_actor_to_session", True))
        if fallback_enabled:
            if not self._warned_actor_fallback:
                LOGGER.warning("actor_id is missing; fallback to session_id for cross-session state")
                self._warned_actor_fallback = True
            return self.state.session_id
        raise ValueError("actor_id is required when cross_session.fallback_actor_to_session=false")

    @staticmethod
    def _compose_effective_actions(policy_actions: List[OffAction], active_actions: List[OffAction]) -> List[OffAction]:
        out: List[OffAction] = []
        out.extend(action for action in policy_actions if action.type in {"SOFT_BLOCK", "HUMAN_ESCALATE"})
        out.extend(action for action in active_actions if action.type == "TOOL_FREEZE")
        out.extend(action for action in active_actions if action.type == "SOURCE_QUARANTINE")
        out.extend(
            action
            for action in policy_actions
            if action.type not in {"SOFT_BLOCK", "HUMAN_ESCALATE", "TOOL_FREEZE", "SOURCE_QUARANTINE"}
        )
        return out

    @staticmethod
    def _extract_tool_requests_from_text(text: str, session_id: str, step: int) -> List[ToolRequest]:
        pattern = re.compile(r"CALL_TOOL\s*\(\s*[\"']([a-zA-Z0-9_:-]+)[\"']\s*(?:,\s*(.*?))?\)", re.DOTALL)
        requests: List[ToolRequest] = []
        for idx, match in enumerate(pattern.finditer(text), start=1):
            tool_name = match.group(1)
            raw_args = (match.group(2) or "").strip()
            requests.append(
                ToolRequest(
                    tool_name=tool_name,
                    args={"raw_args": raw_args, "intent_id": idx, "request_origin": "inferred"},
                    session_id=session_id,
                    step=step,
                )
            )
        return requests

    @staticmethod
    def _marker_hits(text_norm: str, markers: List[str]) -> List[str]:
        return sorted({str(m).strip().lower() for m in markers if str(m).strip() and str(m).strip().lower() in text_norm})

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
        fallback_active = not semantic_active
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
            tenant_id="",
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
                    "trust": str(getattr(item, "trust", "untrusted")) if item is not None else "untrusted",
                    "contribution": float(getattr(contrib, "c", 0.0)),
                }
            )
        rows.sort(key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", ""))))
        return rows[:8]

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
        if not isinstance(sig_cfg, dict) or not bool(sig_cfg.get("enabled", False)) or not carryover_applied:
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

    def _apply_stateful_support_tuning(self, *, user_query: str, step_result: Any) -> Dict[str, Any]:
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
        continuity_markers = list(cfg.get("continuity_markers", [
            "continuity",
            "earlier case",
            "prior case",
            "reference case",
            "legacy service path",
            "same service footprint",
        ]))
        action_markers = list(cfg.get("action_markers", [
            "please send",
            "please apply",
            "restore",
            "resend",
            "switch",
            "update",
            "service credit",
            "archived statements",
        ]))
        review_bypass_markers = list(cfg.get("review_bypass_markers", [
            "queue can catch up",
            "review queue can catch up",
            "before month-end",
            "close the books today",
            "pending review",
        ]))
        continuity_hit = self._contains_marker(text_norm, continuity_markers) and self._contains_marker(text_norm, action_markers)
        review_bypass_hit = self._contains_marker(text_norm, review_bypass_markers) and self._contains_marker(text_norm, action_markers)
        out["continuity_hit"] = bool(continuity_hit)
        out["review_bypass_hit"] = bool(review_bypass_hit)

        if continuity_hit or review_bypass_hit:
            self._support_continuity_hits += 1
        else:
            self._support_continuity_hits = max(0, int(self._support_continuity_hits) - 1)

        walls = list((self.config.get("omega", {}) or {}).get("walls", []))
        idx_by_wall = {str(w): i for i, w in enumerate(walls)}
        participating = {
            str(walls[i])
            for i in range(min(len(walls), len(step_result.p), len(step_result.m_next)))
            if float(step_result.p[i]) > 0.0 or float(step_result.m_next[i]) > 0.0
        }
        combo_walls = list(cfg.get("combo_walls", ["override_instructions", "tool_or_action_abuse"]))
        combo_hit = all(str(w) in participating for w in combo_walls)
        out["combo_hit"] = bool(combo_hit)
        if combo_hit:
            self._support_combo_streak += 1
        else:
            self._support_combo_streak = 0

        min_hits_for_boost = max(1, int(cfg.get("min_hits_for_boost", 2)))
        if self._support_continuity_hits >= min_hits_for_boost:
            boost_walls = list(cfg.get("boost_walls", ["policy_evasion", "tool_or_action_abuse"]))
            boost_delta = float(cfg.get("boost_m_next_delta", 0.16))
            for wall in boost_walls:
                idx = idx_by_wall.get(str(wall))
                if idx is None:
                    continue
                step_result.m_next[idx] = float(step_result.m_next[idx]) + float(boost_delta)
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
        reason_wall = bool(float(np.max(step_result.m_next)) >= float(self.omega_core.params.off_Theta))
        reason_sum = bool(float(np.sum(step_result.m_next)) >= float(self.omega_core.params.off_Sigma))
        reason_multi = bool(
            int(np.count_nonzero(np.asarray(step_result.m_next, dtype=float) >= float(self.omega_core.params.off_theta)))
            >= int(self.omega_core.params.off_N)
        )
        step_result.reasons = OmegaOffReasons(
            reason_spike=bool(reason_spike),
            reason_wall=bool(reason_wall),
            reason_sum=bool(reason_sum),
            reason_multi=bool(reason_multi),
        )
        step_result.off = bool(reason_spike or reason_wall or reason_sum or reason_multi)
        out["dynamic_tau"] = float(tau)
        out["combo_streak"] = int(self._support_combo_streak)
        out["source_quarantine_streak"] = int(self._support_sq_streak)
        return out

    def run_step(
        self,
        user_query: str,
        packet_items: List[ContentItem],
        tool_requests: Optional[List[ToolRequest]] = None,
        actor_id: Optional[str] = None,
        config_refs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        enforcement_mode = str(self.config.get("off_policy", {}).get("enforcement_mode", "ENFORCE")).upper()
        tools_execution_mode = str(self.config.get("tools", {}).get("execution_mode", "ENFORCE")).upper()
        guard_mode = self.guard_mode
        monitor_enabled = bool(guard_mode == GuardMode.MONITOR)
        resolved_actor_id = self._resolve_actor_id(actor_id)

        cross_hydrated = self.cross_session.hydrate_actor_state(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
        )
        self.state.m = np.maximum(self.state.m, cross_hydrated.carried_scars_after_decay)
        carryover_signal = self._apply_cross_session_carryover_signal(
            user_query=user_query,
            carryover_applied=bool(cross_hydrated.carryover_applied),
        )

        projections = [self.projector.project(item) for item in packet_items]
        step_result = self.omega_core.step(self.state, packet_items, projections)
        support_tuning = self._apply_stateful_support_tuning(user_query=user_query, step_result=step_result)
        policy_decision = self.off_policy.select_actions(step_result, packet_items)

        self.cross_session.record_step(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
            step_result=step_result,
            policy_actions=policy_decision.actions,
            packet_items=packet_items,
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
            cross_block["carryover_signal_hit"] = bool(carryover_signal.get("hit", False))
            cross_block["carryover_signal_walls_applied"] = dict(carryover_signal.get("walls_applied", {}))
            cross_block["carryover_signal_order_hits"] = list(carryover_signal.get("order_hits", []))
            cross_block["carryover_signal_action_hits"] = list(carryover_signal.get("action_hits", []))
            cross_block["stateful_support_tuning"] = dict(support_tuning)

        effective_actions = self._compose_effective_actions(policy_decision.actions, cross_active_actions)
        intended_decision = OffDecision(
            off=policy_decision.off,
            severity=policy_decision.severity,
            actions=effective_actions,
            control_outcome=control_outcome_from_actions(effective_actions),
        )

        if monitor_enabled:
            decision = OffDecision(
                off=policy_decision.off,
                severity=policy_decision.severity,
                actions=[],
                control_outcome="ALLOW",
            )
            enforcement_actions = []
            tools_execution_mode = "DRY_RUN"
        elif enforcement_mode == "ENFORCE":
            decision = OffDecision(
                off=policy_decision.off,
                severity=policy_decision.severity,
                actions=effective_actions,
                control_outcome=control_outcome_from_actions(effective_actions),
            )
            enforcement_actions = list(decision.actions)
        else:
            decision = policy_decision
            enforcement_actions = []

        if any(str(a.type) == "SOURCE_QUARANTINE" for a in enforcement_actions):
            self._support_sq_streak += 1
        else:
            self._support_sq_streak = 0

        trace_id = build_trace_id_runtime(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            doc_ids=sorted({str(item.doc_id) for item in packet_items}),
        )
        intended_action_types = sorted({str(a.type) for a in list(intended_decision.actions)})
        action_types = sorted({str(a.type) for a in list(decision.actions)})
        intended_action = str(intended_decision.control_outcome)
        actual_action = str(decision.control_outcome)
        decision_id = build_decision_id(
            trace_id=trace_id,
            control_outcome=str(intended_action if monitor_enabled else decision.control_outcome),
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
            blocked.update(item.doc_id for item in packet_items if item.source_id in set(action.source_ids))
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
                intended_blocked.update(item.doc_id for item in packet_items if item.source_id in set(action.source_ids))

        allowed_items = [item for item in packet_items if item.doc_id not in blocked]
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
        latest_approval = self.notification_dispatcher.latest_approval_for_session(
            tenant_id="runtime",
            session_id=str(self.state.session_id),
        )
        latest_approval_status = str(latest_approval.status) if latest_approval is not None else "none"
        auto_human_approved = latest_approval_status == "approved"
        merged_requests: List[tuple[ToolRequest, str]] = []
        for request in list(tool_requests or []):
            if "request_origin" not in request.args:
                request.args["request_origin"] = "explicit"
            if auto_human_approved and "human_approved" not in request.args:
                request.args["human_approved"] = True
            merged_requests.append((request, "explicit"))
        for request in inferred_requests:
            if auto_human_approved and "human_approved" not in request.args:
                request.args["human_approved"] = True
            merged_requests.append((request, "inferred"))

        tool_decisions = []
        tool_executions: List[ToolExecution] = []
        tool_gateway_events: List[Dict[str, Any]] = []
        off_state = bool(self.tool_gateway.is_off_state(enforcement_actions))
        freeze_active = bool(self.tool_gateway.find_freeze(enforcement_actions) is not None)
        actor_hash = str((cross_snapshot.get("cross_session") or {}).get("actor_hash", "n/a"))
        source_ids_seen = sorted({item.source_id for item in allowed_items})[:16]

        for request, request_origin in merged_requests:
            decision_out = self.tool_gateway.enforce(request, enforcement_actions)
            tool_decisions.append(decision_out)
            capability_obj = self.tool_gateway.capability_for(request.tool_name)
            capability = {
                "mode": capability_obj.mode if capability_obj is not None else "unknown",
                "requires_human_approval": capability_obj.requires_human_approval if capability_obj is not None else False,
            }
            adapter_present = bool(self.tool_registry.has(request.tool_name))
            human_approved = bool(request.args.get("human_approved", False))
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
                        "tool_output_dir": self.config.get("tools", {}).get("output_dir", "artifacts/tools"),
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

        step_event = build_step_event(step_result, trace_id=trace_id, decision_id=decision_id)
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
                capture_text=self.config.get("logging", {}).get("capture_text", "NEVER"),
                max_text_chars=int(self.config.get("logging", {}).get("max_text_chars", 800)),
                trace_id=trace_id,
                decision_id=decision_id,
            )

        reason_flags = [key for key, value in step_result.reasons.__dict__.items() if bool(value)]
        walls_triggered = [
            str(self.config["omega"]["walls"][idx])
            for idx, value in enumerate(step_result.p)
            if float(value) > 0.0 or float(step_result.m_next[idx]) > 0.0
        ]
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
                for name, hits in sorted(signal_hits.items(), key=lambda row: (-int(row[1]), str(row[0])))[:20]
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
        if should_emit_incident_artifact(config=self.config, control_outcome=decision.control_outcome):
            top_docs_lookup = {item.doc_id: item for item in packet_items}
            top_docs = [
                {
                    "doc_id": doc_id,
                    "source_id": top_docs_lookup[doc_id].source_id,
                    "source_type": top_docs_lookup[doc_id].source_type,
                    "trust": top_docs_lookup[doc_id].trust,
                    "text": top_docs_lookup[doc_id].text,
                }
                for doc_id in list(step_result.top_docs)
                if doc_id in top_docs_lookup
            ]
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
                reason_flags=[key for key, value in step_result.reasons.__dict__.items() if bool(value)],
                contributing_signals={
                    "max_p": float(np.max(step_result.p)) if len(step_result.p) else 0.0,
                    "sum_m_next": float(np.sum(step_result.m_next)) if len(step_result.m_next) else 0.0,
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
            incident_artifact_id = str(incident_artifact.get("incident_artifact_id", ""))

        max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
        sum_m = float(np.sum(step_result.m_next)) if len(step_result.m_next) else 0.0
        sum_ratio = min(1.0, sum_m / max(float(self.omega_core.params.off_Sigma), 1e-6))
        severity_score = {"L1": 0.0, "L2": 0.5, "L3": 1.0}.get(str(decision.severity), 0.0)
        risk_score = float(max(0.0, min(1.0, 0.60 * max_p + 0.30 * sum_ratio + 0.10 * severity_score)))
        monitor_attribution = self._monitor_attribution(step_result=step_result, items=packet_items)
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
            item_text_by_doc={str(item.doc_id): str(item.text) for item in packet_items},
            max_fragments=4,
            max_chars=240,
        )
        prevented_tools = sorted(
            {
                str(name)
                for name in (
                    [request.tool_name for request, _ in merged_requests] if "TOOL_FREEZE" in intended_action_types else []
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
                input_length=sum(len(str(item.text or "")) for item in list(packet_items)),
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
        self.notification_dispatcher.emit_risk_event(risk_event)
        approval_required = ("HUMAN_ESCALATE" in action_types) or ("REQUIRE_APPROVAL" in action_types)
        approval_id: Optional[str] = None
        if approval_required:
            timeout_sec = int(
                ((self.config.get("notifications", {}) or {}).get("approvals", {}) or {}).get("timeout_sec", 900)
            )
            approval = self.notification_dispatcher.create_action_request(
                risk_event=risk_event,
                required_action="HUMAN_ESCALATE" if "HUMAN_ESCALATE" in action_types else "REQUIRE_APPROVAL",
                timeout_sec=max(10, timeout_sec),
            )
            approval_id = str(approval.approval_id)
            latest_approval_status = str(approval.status)
        elif latest_approval is not None:
            approval_id = str(latest_approval.approval_id)

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
            "monitor": monitor_payload,
            "notification_metrics": self.notification_dispatcher.metrics_snapshot(),
            "monitoring_metrics": self.monitor_collector.health_snapshot(),
        }
