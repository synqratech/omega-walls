"""End-to-end Omega harness with pluggable LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaState, ToolRequest
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.policy.enforcement_state import EnforcementStateManager
from omega.rag.context_builder import ContextBuilder
from omega.telemetry.events import (
    build_enforcement_step_event,
    build_off_event,
    build_step_event,
    build_tool_gateway_step_event,
)
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

    def reset_state(self, session_id: Optional[str] = None, actor_id: Optional[str] = None) -> None:
        self.state.m = np.zeros_like(self.state.m)
        self.state.step = 0
        self.enforcement.reset()
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
        resolved_actor_id = self._resolve_actor_id(actor_id)

        cross_hydrated = self.cross_session.hydrate_actor_state(
            actor_id=resolved_actor_id,
            session_id=self.state.session_id,
        )
        self.state.m = np.maximum(self.state.m, cross_hydrated.carried_scars_after_decay)

        projections = [self.projector.project(item) for item in packet_items]
        step_result = self.omega_core.step(self.state, packet_items, projections)
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

        if enforcement_mode == "ENFORCE":
            decision = OffDecision(
                off=policy_decision.off,
                severity=policy_decision.severity,
                actions=self._compose_effective_actions(policy_decision.actions, cross_active_actions),
            )
            enforcement_actions = list(decision.actions)
        else:
            decision = policy_decision
            enforcement_actions = []

        blocked = set()
        for action in enforcement_actions:
            if action.type == "SOFT_BLOCK" and action.doc_ids:
                blocked.update(action.doc_ids)
        for action in enforcement_actions:
            if action.type != "SOURCE_QUARANTINE" or not action.source_ids:
                continue
            blocked.update(item.doc_id for item in packet_items if item.source_id in set(action.source_ids))

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
        merged_requests: List[tuple[ToolRequest, str]] = []
        for request in list(tool_requests or []):
            if "request_origin" not in request.args:
                request.args["request_origin"] = "explicit"
            merged_requests.append((request, "explicit"))
        for request in inferred_requests:
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
                        },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
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
                        },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
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
                        },
                        capability=capability,
                        human_approved=human_approved,
                        executed=executed,
                        adapter_present=adapter_present,
                        execution_mode=tools_execution_mode,
                        actor_hash=actor_hash,
                        source_ids_seen=source_ids_seen,
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
                    },
                    capability=capability,
                    human_approved=human_approved,
                    executed=executed,
                    adapter_present=adapter_present,
                    execution_mode=tools_execution_mode,
                    actor_hash=actor_hash,
                    source_ids_seen=source_ids_seen,
                )
            )

        step_event = build_step_event(step_result)
        enforcement_event = build_enforcement_step_event(
            session_id=step_result.session_id,
            step=step_result.step,
            enforcement_snapshot=cross_snapshot,
            active_actions=enforcement_actions,
            cross_session=cross_snapshot.get("cross_session"),
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
            )

        return {
            "step_result": step_result,
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
            "off_event": off_event,
        }
