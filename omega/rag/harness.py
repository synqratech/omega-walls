"""End-to-end Omega harness with pluggable LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaState, ToolRequest
from omega.policy.enforcement_state import EnforcementStateManager
from omega.rag.context_builder import ContextBuilder
from omega.telemetry.events import build_enforcement_step_event, build_off_event, build_step_event
from omega.tools.adapters import ToolExecution, build_default_tool_registry


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
        self.state = OmegaState(session_id="sess-local", m=np.zeros(4, dtype=float), step=0)
        self.enforcement = EnforcementStateManager.from_config(config)

    def reset_state(self, session_id: Optional[str] = None) -> None:
        self.state.m = np.zeros_like(self.state.m)
        self.state.step = 0
        self.enforcement.reset()
        if session_id is not None:
            self.state.session_id = session_id

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
                    args={"raw_args": raw_args, "intent_id": idx},
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
        config_refs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        enforcement_mode = str(self.config.get("off_policy", {}).get("enforcement_mode", "ENFORCE")).upper()
        tools_execution_mode = str(self.config.get("tools", {}).get("execution_mode", "ENFORCE")).upper()

        projections = [self.projector.project(item) for item in packet_items]
        step_result = self.omega_core.step(self.state, packet_items, projections)
        policy_decision = self.off_policy.select_actions(step_result, packet_items)

        if enforcement_mode == "ENFORCE":
            self.enforcement.record_policy_actions(policy_decision.actions, step_result.step)
            active_actions = self.enforcement.active_actions(step_result.step)
            decision = OffDecision(
                off=policy_decision.off,
                severity=policy_decision.severity,
                actions=self._compose_effective_actions(policy_decision.actions, active_actions),
            )
            enforcement_actions = list(decision.actions)
        else:
            active_actions = []
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
        merged_requests = list(tool_requests or []) + inferred_requests

        tool_decisions = []
        tool_executions: List[ToolExecution] = []
        for request in merged_requests:
            decision_out = self.tool_gateway.enforce(request, enforcement_actions)
            tool_decisions.append(decision_out)

            if not decision_out.allowed:
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=False,
                        executed=False,
                        reason=decision_out.reason,
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
                continue

            if not self.tool_registry.has(request.tool_name):
                tool_executions.append(
                    ToolExecution(
                        tool_name=request.tool_name,
                        allowed=True,
                        executed=False,
                        reason="NO_ADAPTER",
                        error=f"No adapter registered for {request.tool_name}",
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

        step_event = build_step_event(step_result)
        enforcement_event = build_enforcement_step_event(
            session_id=step_result.session_id,
            step=step_result.step,
            enforcement_snapshot=self.enforcement.snapshot(step_result.step),
            active_actions=enforcement_actions,
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
            "step_event": step_event,
            "enforcement_event": enforcement_event,
            "off_event": off_event,
        }
