"""CrewAI guard wrapper backed by Omega adapter runtime."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from omega.adapters import (
    AdapterDecision,
    AdapterSessionContext,
    MemoryWriteDecision,
    OmegaAdapterRuntime,
    OmegaBlockedError,
    OmegaToolBlockedError,
)


class OmegaCrewAIGuard:
    """Framework guard for CrewAI execution hooks and tool flows."""

    _SESSION_KEYS = ("thread_id", "conversation_id", "session_id", "run_id", "kickoff_id")
    _ACTOR_KEYS = ("actor_id", "user_id", "customer_id", "principal_id", "agent_id")

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        projector_mode: Optional[str] = "hybrid_api",
        api_model: Optional[str] = "gpt-5.4-mini",
        session_id_getter: Optional[Callable[..., Optional[str]]] = None,
        actor_id_getter: Optional[Callable[..., Optional[str]]] = None,
        block_contract_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        security_metadata_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_chars: int = 8000,
        cli_overrides: Optional[Mapping[str, Any]] = None,
        config_dir: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        runtime: Optional[OmegaAdapterRuntime] = None,
    ) -> None:
        self._session_id_getter = session_id_getter
        self._actor_id_getter = actor_id_getter
        self._block_contract_hook = block_contract_hook
        self._security_metadata_hook = security_metadata_hook
        self._max_chars = max(256, int(max_chars))
        self._last_block_contract: Optional[Dict[str, Any]] = None
        self._last_security_metadata: Optional[Dict[str, Any]] = None
        self._runtime = runtime or OmegaAdapterRuntime(
            profile=profile,
            projector_mode=projector_mode,
            api_model=api_model,
            cli_overrides=cli_overrides,
            config_dir=config_dir,
            env=env,
            max_chars=max_chars,
        )
        self._hooks_module: Any = None
        self._hooks_registered = False
        self._before_llm_hook_fn = self.before_llm_hook
        self._before_tool_hook_fn = self.before_tool_hook

    def register_global_hooks(self) -> None:
        hooks = self._load_hooks_module()
        hooks.register_before_llm_call_hook(self._before_llm_hook_fn)
        hooks.register_before_tool_call_hook(self._before_tool_hook_fn)
        self._hooks_registered = True

    def unregister_global_hooks(self) -> None:
        if not self._hooks_registered:
            return
        hooks = self._load_hooks_module()
        hooks.unregister_before_llm_call_hook(self._before_llm_hook_fn)
        hooks.unregister_before_tool_call_hook(self._before_tool_hook_fn)
        self._hooks_registered = False

    @contextmanager
    def install_global_hooks(self) -> Iterator["OmegaCrewAIGuard"]:
        self.register_global_hooks()
        try:
            yield self
        finally:
            self.unregister_global_hooks()

    def before_llm_hook(self, context: Any) -> None:
        text = self._normalize_messages_text(messages=getattr(context, "messages", None))
        ctx = self._build_session_context(context=context, kwargs={})
        decision = self._runtime.check_model_input(text, ctx)
        self._emit_security_metadata(decision=decision, phase="model_input")
        if self._should_block_decision(decision):
            self._emit_block_contract(self._build_block_contract_from_decision(decision))
            raise OmegaBlockedError(
                f"Omega blocked CrewAI LLM call (control_outcome={decision.control_outcome}, off={decision.off})",
                decision=decision,
            )

    def before_tool_hook(self, context: Any) -> None:
        tool_name = str(getattr(context, "tool_name", "") or "unknown_tool")
        tool_input = getattr(context, "tool_input", None)
        tool_input_dict = dict(tool_input) if isinstance(tool_input, Mapping) else {"raw": str(tool_input)}
        ctx = self._build_session_context(context=context, kwargs={})
        gate = self._runtime.check_tool_call(tool_name=tool_name, tool_args=tool_input_dict, ctx=ctx)
        self._emit_security_metadata(decision=gate.decision_ref, phase="tool_precheck")
        if not gate.allowed:
            self._emit_block_contract(self._build_block_contract_from_tool_gate(gate))
            raise OmegaToolBlockedError(
                f"Omega blocked CrewAI tool call '{tool_name}' (reason={gate.reason}, mode={gate.mode})",
                gate_decision=gate,
            )

    def wrap_tool(self, tool_name: str, tool_callable: Callable[..., Any]) -> Callable[..., Any]:
        if not str(tool_name or "").strip():
            raise ValueError("tool_name is required")
        if not callable(tool_callable):
            raise ValueError("tool_callable must be callable")

        if inspect.iscoroutinefunction(tool_callable):

            async def _wrapped_async(*args: Any, **kwargs: Any) -> Any:
                gate = self._gate_tool_call(tool_name=str(tool_name), args=args, kwargs=kwargs, runtime_context=None)
                self._emit_security_metadata(decision=gate.decision_ref, phase="tool_precheck")
                if not gate.allowed:
                    self._emit_block_contract(self._build_block_contract_from_tool_gate(gate))
                    raise OmegaToolBlockedError(
                        f"Omega blocked tool call '{tool_name}' (reason={gate.reason}, mode={gate.mode})",
                        gate_decision=gate,
                    )
                return await tool_callable(*args, **self._without_context_kwargs(kwargs))

            return _wrapped_async

        def _wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            gate = self._gate_tool_call(tool_name=str(tool_name), args=args, kwargs=kwargs, runtime_context=None)
            self._emit_security_metadata(decision=gate.decision_ref, phase="tool_precheck")
            if not gate.allowed:
                self._emit_block_contract(self._build_block_contract_from_tool_gate(gate))
                raise OmegaToolBlockedError(
                    f"Omega blocked tool call '{tool_name}' (reason={gate.reason}, mode={gate.mode})",
                    gate_decision=gate,
                )
            return tool_callable(*args, **self._without_context_kwargs(kwargs))

        return _wrapped_sync

    def check_memory_write(
        self,
        *,
        memory_text: str,
        source_id: str,
        source_type: str = "other",
        source_trust: str = "untrusted",
        source_tags: Optional[Mapping[str, Any]] = None,
        context: Any = None,
        **context_kwargs: Any,
    ) -> MemoryWriteDecision:
        kwargs = self._extract_context_kwargs(dict(context_kwargs), pop=False)
        ctx = self._build_session_context(context=context, kwargs=kwargs)
        return self._runtime.check_memory_write(
            memory_text=memory_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            source_tags=source_tags,
            ctx=ctx,
        )

    def _load_hooks_module(self) -> Any:
        if self._hooks_module is not None:
            return self._hooks_module
        try:
            module = importlib.import_module("crewai.hooks")
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CrewAI integration requires crewai hooks support. Install with: pip install -e .[integrations]"
            ) from exc
        for fn_name in (
            "register_before_llm_call_hook",
            "register_before_tool_call_hook",
            "unregister_before_llm_call_hook",
            "unregister_before_tool_call_hook",
        ):
            if not callable(getattr(module, fn_name, None)):
                raise RuntimeError(f"CrewAI hooks API missing '{fn_name}'")
        self._hooks_module = module
        return module

    def _gate_tool_call(
        self,
        *,
        tool_name: str,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ):
        query_kwargs = self._extract_context_kwargs(kwargs, pop=False)
        ctx = self._build_session_context(context=runtime_context, kwargs=query_kwargs)
        tool_args = {
            "args": [str(x) for x in list(args)],
            "kwargs": {str(k): str(v) for k, v in dict(kwargs).items()},
        }
        return self._runtime.check_tool_call(tool_name=str(tool_name), tool_args=tool_args, ctx=ctx)

    def get_last_block_contract(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_block_contract) if isinstance(self._last_block_contract, dict) else None

    def get_last_security_metadata(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_security_metadata) if isinstance(self._last_security_metadata, dict) else None

    def _emit_block_contract(self, payload: Dict[str, Any]) -> None:
        self._last_block_contract = dict(payload)
        if callable(self._block_contract_hook):
            self._block_contract_hook(dict(payload))

    def _emit_security_metadata(
        self,
        *,
        decision: AdapterDecision,
        phase: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        runtime_builder = getattr(self._runtime, "build_security_metadata", None)
        if callable(runtime_builder):
            payload = runtime_builder(decision, phase=phase, extra=extra)
        else:
            payload = {
                "phase": str(phase),
                "mode": "deny" if bool(decision.off) else "allow",
                "risk": getattr(decision, "risk_score", None),
                "action": str(getattr(decision, "control_outcome", "ALLOW")),
                "trace_id": str(getattr(decision, "trace_id", "") or ""),
                "decision_id": str(getattr(decision, "decision_id", "") or ""),
                "llm_fallback_active": None,
                "fallback_level": None,
            }
            if isinstance(extra, Mapping):
                payload.update(dict(extra))
        self._last_security_metadata = dict(payload)
        if callable(self._security_metadata_hook):
            self._security_metadata_hook(dict(payload))

    def _build_block_contract_from_decision(self, decision: AdapterDecision) -> Dict[str, Any]:
        runtime_builder = getattr(self._runtime, "build_block_contract_from_decision", None)
        if callable(runtime_builder):
            return dict(runtime_builder(decision))
        reason = str(decision.control_outcome or "BLOCK")
        if decision.reason_codes:
            reason = str(decision.reason_codes[0])
        return {
            "action": str(decision.control_outcome or "BLOCK"),
            "reason": reason,
            "policy_id": None,
            "fallback_hint": None,
            "incident_id": None,
            "trace_id": str(decision.trace_id or ""),
            "decision_id": str(decision.decision_id or ""),
        }

    def _build_block_contract_from_tool_gate(self, gate_decision: Any) -> Dict[str, Any]:
        runtime_builder = getattr(self._runtime, "build_block_contract_from_tool_gate", None)
        if callable(runtime_builder):
            return dict(runtime_builder(gate_decision))
        decision = gate_decision.decision_ref
        resolve_action = getattr(self._runtime, "resolve_tool_block_action", None)
        if callable(resolve_action):
            action = str(resolve_action(gate_decision))
        else:
            outcome = str(getattr(decision, "control_outcome", "") or "").strip().upper()
            action = (outcome if outcome and outcome not in {"ALLOW", "WARN"} else "TOOL_FREEZE")
        return {
            "action": str(action),
            "reason": str(getattr(gate_decision, "reason", "TOOL_BLOCKED")),
            "policy_id": None,
            "fallback_hint": None,
            "incident_id": None,
            "trace_id": str(getattr(decision, "trace_id", "") or ""),
            "decision_id": str(getattr(decision, "decision_id", "") or ""),
        }

    @staticmethod
    def _call_optional_getter(
        getter: Optional[Callable[..., Optional[str]]],
        *,
        context: Any,
        kwargs: Dict[str, Any],
    ) -> Optional[str]:
        if getter is None:
            return None
        for args in (
            (context, kwargs),
            (context,),
        ):
            try:
                value = getter(*args)
                if value is not None and str(value).strip():
                    return str(value).strip()
                return None
            except TypeError:
                continue
        return None

    def _build_session_context(self, *, context: Any, kwargs: Dict[str, Any]) -> AdapterSessionContext:
        session_id = self._call_optional_getter(
            self._session_id_getter,
            context=context,
            kwargs=kwargs,
        )
        actor_id = self._call_optional_getter(
            self._actor_id_getter,
            context=context,
            kwargs=kwargs,
        )

        if session_id is None:
            session_id = (
                self._lookup_identifier(kwargs, self._SESSION_KEYS)
                or self._lookup_identifier(context, self._SESSION_KEYS)
                or "omega-cr-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(kwargs, self._ACTOR_KEYS)
                or self._lookup_identifier(context, self._ACTOR_KEYS)
                or self._lookup_identifier(getattr(context, "agent", None), ("id", "name", "role"))
                or session_id
            )
        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "crewai"},
        )

    @classmethod
    def _lookup_identifier(cls, obj: Any, keys: tuple[str, ...]) -> Optional[str]:
        if obj is None:
            return None

        if isinstance(obj, Mapping):
            for key in keys:
                value = obj.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
            for nested_key in ("context", "config", "metadata", "runtime", "agent", "task", "crew"):
                nested = obj.get(nested_key)
                if nested is obj:
                    continue
                nested_value = cls._lookup_identifier(nested, keys)
                if nested_value:
                    return nested_value
            return None

        for key in keys:
            value = getattr(obj, key, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        for nested_attr in ("context", "config", "metadata", "runtime", "agent", "task", "crew"):
            nested = getattr(obj, nested_attr, None)
            if nested is obj:
                continue
            nested_value = cls._lookup_identifier(nested, keys)
            if nested_value:
                return nested_value
        return None

    def _normalize_messages_text(self, *, messages: Any) -> str:
        if not isinstance(messages, list):
            raw = self._normalize_content(messages)
            return raw[: self._max_chars]
        parts: List[str] = []
        for msg in messages:
            parts.append(self._normalize_message(msg))
        return "\n".join(part for part in parts if part).strip()[: self._max_chars]

    @classmethod
    def _normalize_message(cls, msg: Any) -> str:
        role = None
        content = None
        if isinstance(msg, Mapping):
            role = msg.get("role") or msg.get("type") or msg.get("name")
            content = msg.get("content")
            if content is None:
                content = msg.get("text")
        else:
            role = getattr(msg, "role", None) or getattr(msg, "type", None) or getattr(msg, "name", None)
            content = getattr(msg, "content", None)
            if content is None:
                content = getattr(msg, "text", None)
        role_text = str(role) if role is not None else "message"
        content_text = cls._normalize_content(content)
        if content_text:
            return f"{role_text}: {content_text}"
        return role_text

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return " ".join(content.split())
        if isinstance(content, Mapping):
            if isinstance(content.get("text"), str):
                return " ".join(str(content["text"]).split())
            if isinstance(content.get("content"), str):
                return " ".join(str(content["content"]).split())
            chunks: List[str] = []
            for key in ("payload", "parts", "messages"):
                value = content.get(key)
                if value is None:
                    continue
                chunk = cls._normalize_content(value)
                if chunk:
                    chunks.append(chunk)
            if chunks:
                return " ".join(chunks)
            return " ".join(str(content).split())
        if isinstance(content, list):
            chunks = []
            for part in content:
                chunk = cls._normalize_content(part)
                if chunk:
                    chunks.append(chunk)
            return " ".join(chunks)
        return " ".join(str(content).split())

    @staticmethod
    def _extract_context_kwargs(kwargs: Dict[str, Any], *, pop: bool) -> Dict[str, Any]:
        keys = {
            "thread_id",
            "conversation_id",
            "session_id",
            "actor_id",
            "user_id",
            "customer_id",
            "principal_id",
            "run_id",
            "kickoff_id",
        }
        out: Dict[str, Any] = {}
        for key in list(kwargs.keys()):
            if key not in keys:
                continue
            out[key] = kwargs[key]
            if pop:
                del kwargs[key]
        return out

    @classmethod
    def _without_context_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        context_keys = set(cls._SESSION_KEYS) | set(cls._ACTOR_KEYS)
        return {k: v for k, v in kwargs.items() if k not in context_keys}

    @staticmethod
    def _should_block_decision(decision: AdapterDecision) -> bool:
        if decision.off:
            return True
        if str(decision.control_outcome).upper() in {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}:
            return True
        block_actions = {"SOFT_BLOCK", "TOOL_FREEZE", "SOURCE_QUARANTINE", "HUMAN_ESCALATE"}
        action_types = {str(action.type).upper() for action in decision.actions}
        return bool(action_types & block_actions)
