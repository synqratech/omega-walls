"""LangChain middleware guard backed by Omega adapter runtime."""

from __future__ import annotations

import importlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from omega.adapters import (
    AdapterDecision,
    AdapterSessionContext,
    MemoryWriteDecision,
    OmegaAdapterRuntime,
    OmegaBlockedError,
    OmegaToolBlockedError,
)


class OmegaLangChainGuard:
    """Framework guard for LangChain middleware integration."""

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        projector_mode: Optional[str] = "hybrid_api",
        api_model: Optional[str] = "gpt-5.4-mini",
        session_id_getter: Optional[Callable[..., Optional[str]]] = None,
        actor_id_getter: Optional[Callable[..., Optional[str]]] = None,
        max_chars: int = 8000,
        cli_overrides: Optional[Mapping[str, Any]] = None,
        config_dir: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        runtime: Optional[OmegaAdapterRuntime] = None,
    ) -> None:
        self._session_id_getter = session_id_getter
        self._actor_id_getter = actor_id_getter
        self._max_chars = max(256, int(max_chars))
        self._runtime = runtime or OmegaAdapterRuntime(
            profile=profile,
            projector_mode=projector_mode,
            api_model=api_model,
            cli_overrides=cli_overrides,
            config_dir=config_dir,
            env=env,
            max_chars=max_chars,
        )

    def middleware(self) -> List[Any]:
        try:
            middleware_mod = importlib.import_module("langchain.agents.middleware")
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise ImportError(
                "LangChain middleware integration requires langchain>=1.0 with middleware support. "
                "Install with: pip install -e .[integrations]"
            ) from exc

        before_model = getattr(middleware_mod, "before_model", None)
        wrap_tool_call = getattr(middleware_mod, "wrap_tool_call", None)
        if not callable(before_model) or not callable(wrap_tool_call):
            raise RuntimeError("LangChain middleware decorators before_model/wrap_tool_call are not available")

        @before_model
        def omega_before_model(state: Any, runtime: Any) -> None:
            self._before_model_impl(state=state, runtime=runtime)
            return None

        @wrap_tool_call
        def omega_wrap_tool_call(request: Any, handler: Callable[..., Any]) -> Any:
            return self._wrap_tool_call_impl(request=request, handler=handler)

        return [omega_before_model, omega_wrap_tool_call]

    def _before_model_impl(self, *, state: Any, runtime: Any) -> None:
        messages_text = self._extract_messages_text_from_state(state)
        ctx = self._build_session_context(state=state, runtime=runtime)
        decision = self._runtime.check_model_input(messages_text, ctx)
        if self._should_block_decision(decision):
            raise OmegaBlockedError(
                f"Omega blocked model input (control_outcome={decision.control_outcome}, off={decision.off})",
                decision=decision,
            )

    def _wrap_tool_call_impl(self, *, request: Any, handler: Callable[..., Any]) -> Any:
        tool_name, tool_args = self._extract_tool_call(request)
        state = getattr(request, "state", None)
        runtime = getattr(request, "runtime", None)
        ctx = self._build_session_context(state=state, runtime=runtime)
        gate_decision = self._runtime.check_tool_call(tool_name=tool_name, tool_args=tool_args, ctx=ctx)
        if not gate_decision.allowed:
            raise OmegaToolBlockedError(
                f"Omega blocked tool call '{tool_name}' (reason={gate_decision.reason}, mode={gate_decision.mode})",
                gate_decision=gate_decision,
            )
        return handler(request)

    def check_memory_write(
        self,
        *,
        memory_text: str,
        source_id: str,
        source_type: str = "other",
        source_trust: str = "untrusted",
        source_tags: Optional[Mapping[str, Any]] = None,
        state: Any = None,
        runtime: Any = None,
        **context: Any,
    ) -> MemoryWriteDecision:
        state_for_ctx = state
        if isinstance(state_for_ctx, Mapping):
            merged_state = dict(state_for_ctx)
            merged_state.update(context)
            state_for_ctx = merged_state
        elif context:
            state_for_ctx = dict(context)
        ctx = self._build_session_context(state=state_for_ctx, runtime=runtime)
        return self._runtime.check_memory_write(
            memory_text=memory_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            source_tags=source_tags,
            ctx=ctx,
        )

    @staticmethod
    def _call_optional_getter(
        getter: Optional[Callable[..., Optional[str]]],
        *,
        state: Any,
        runtime: Any,
    ) -> Optional[str]:
        if getter is None:
            return None
        try:
            value = getter(state, runtime)
        except TypeError:
            value = getter(state)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _build_session_context(self, *, state: Any, runtime: Any) -> AdapterSessionContext:
        session_id = self._call_optional_getter(self._session_id_getter, state=state, runtime=runtime)
        actor_id = self._call_optional_getter(self._actor_id_getter, state=state, runtime=runtime)

        if session_id is None:
            session_id = (
                self._lookup_identifier(state, ("thread_id", "conversation_id", "session_id"))
                or self._lookup_identifier(runtime, ("thread_id", "conversation_id", "session_id"))
                or "omega-lc-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(state, ("actor_id", "user_id", "customer_id", "principal_id"))
                or self._lookup_identifier(runtime, ("actor_id", "user_id", "customer_id", "principal_id"))
                or session_id
            )
        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "langchain"},
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
            for nested_key in ("context", "config", "metadata", "runtime"):
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
        for nested_attr in ("context", "config", "metadata", "runtime"):
            nested = getattr(obj, nested_attr, None)
            if nested is obj:
                continue
            nested_value = cls._lookup_identifier(nested, keys)
            if nested_value:
                return nested_value
        return None

    def _extract_messages_text_from_state(self, state: Any) -> str:
        messages = None
        if isinstance(state, Mapping):
            messages = state.get("messages")
        if messages is None:
            messages = getattr(state, "messages", None)
        if not isinstance(messages, list):
            raw = self._normalize_text(state)
            return raw[: self._max_chars]

        parts: List[str] = []
        for msg in messages:
            parts.append(self._normalize_message(msg))
        text = "\n".join(part for part in parts if part).strip()
        return text[: self._max_chars]

    @classmethod
    def _normalize_message(cls, msg: Any) -> str:
        role = None
        content = None
        if isinstance(msg, Mapping):
            role = msg.get("role") or msg.get("type") or msg.get("name")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            content = getattr(msg, "content", None)

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
            return " ".join(json.dumps(content, ensure_ascii=True, default=str).split())
        if isinstance(content, list):
            chunks: List[str] = []
            for part in content:
                chunk = cls._normalize_content(part)
                if chunk:
                    chunks.append(chunk)
            return " ".join(chunks)
        return " ".join(str(content).split())

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value).split())

    @staticmethod
    def _extract_tool_call(request: Any) -> tuple[str, Dict[str, Any]]:
        tool_call = getattr(request, "tool_call", None)
        if tool_call is None and isinstance(request, Mapping):
            tool_call = request.get("tool_call")
        if tool_call is None:
            return "unknown_tool", {}

        if isinstance(tool_call, Mapping):
            name = tool_call.get("name") or tool_call.get("tool_name") or "unknown_tool"
            args = tool_call.get("args") or tool_call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"raw_args": args}
            return str(name), dict(args)

        name = getattr(tool_call, "name", None) or getattr(tool_call, "tool_name", None) or "unknown_tool"
        args = getattr(tool_call, "args", None) or getattr(tool_call, "arguments", None) or {}
        if not isinstance(args, dict):
            args = {"raw_args": args}
        return str(name), dict(args)

    @staticmethod
    def _should_block_decision(decision: AdapterDecision) -> bool:
        if decision.off:
            return True
        if str(decision.control_outcome).upper() in {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}:
            return True
        block_actions = {"SOFT_BLOCK", "TOOL_FREEZE", "SOURCE_QUARANTINE", "HUMAN_ESCALATE"}
        action_types = {str(action.type).upper() for action in decision.actions}
        return bool(action_types & block_actions)
