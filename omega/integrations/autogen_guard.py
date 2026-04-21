"""AutoGen AgentChat guard wrapper backed by Omega adapter runtime."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional

from omega.adapters import (
    AdapterDecision,
    AdapterSessionContext,
    MemoryWriteDecision,
    OmegaAdapterRuntime,
    OmegaBlockedError,
    OmegaToolBlockedError,
)


class OmegaAutoGenGuard:
    """Framework guard for AutoGen AgentChat message and tool flows."""

    _SESSION_KEYS = ("thread_id", "conversation_id", "session_id")
    _ACTOR_KEYS = ("actor_id", "user_id", "customer_id", "principal_id")

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

    def wrap_agent(self, agent: Any) -> Any:
        if not hasattr(agent, "on_messages") and not hasattr(agent, "on_messages_stream"):
            raise ValueError("AutoGen agent must provide on_messages(...) or on_messages_stream(...)")
        guard = self

        class _WrappedAutoGenAgent:
            def __init__(self, inner: Any):
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            async def on_messages(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "on_messages"):
                    raise AttributeError("Wrapped agent does not support on_messages(...)")
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_messages(messages=messages, kwargs=context_kwargs, runtime_context=self._inner)
                result = self._inner.on_messages(messages, *args, **kwargs_copy)
                if inspect.isawaitable(result):
                    return await result
                return result

            async def on_messages_stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "on_messages_stream"):
                    raise AttributeError("Wrapped agent does not support on_messages_stream(...)")
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_messages(messages=messages, kwargs=context_kwargs, runtime_context=self._inner)
                result = self._inner.on_messages_stream(messages, *args, **kwargs_copy)
                if inspect.isawaitable(result):
                    return await result
                return result

        return _WrappedAutoGenAgent(agent)

    def wrap_tool(self, tool_name: str, tool_callable: Callable[..., Any]) -> Callable[..., Any]:
        if not str(tool_name or "").strip():
            raise ValueError("tool_name is required")
        if not callable(tool_callable):
            raise ValueError("tool_callable must be callable")

        if inspect.iscoroutinefunction(tool_callable):

            async def _wrapped_async(*args: Any, **kwargs: Any) -> Any:
                gate = self._gate_tool_call(tool_name=str(tool_name), args=args, kwargs=kwargs, runtime_context=None)
                if not gate.allowed:
                    raise OmegaToolBlockedError(
                        f"Omega blocked tool call '{tool_name}' (reason={gate.reason}, mode={gate.mode})",
                        gate_decision=gate,
                    )
                return await tool_callable(*args, **self._without_context_kwargs(kwargs))

            return _wrapped_async

        def _wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            gate = self._gate_tool_call(tool_name=str(tool_name), args=args, kwargs=kwargs, runtime_context=None)
            if not gate.allowed:
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
        messages: Any = None,
        runtime_context: Any = None,
        **context: Any,
    ) -> MemoryWriteDecision:
        kwargs = self._extract_context_kwargs(dict(context), pop=False)
        ctx = self._build_session_context(messages=messages, kwargs=kwargs, runtime_context=runtime_context)
        return self._runtime.check_memory_write(
            memory_text=memory_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            source_tags=source_tags,
            ctx=ctx,
        )

    def _guard_messages(self, *, messages: Any, kwargs: Dict[str, Any], runtime_context: Any) -> AdapterDecision:
        text = self._normalize_messages_text(messages=messages)
        ctx = self._build_session_context(messages=messages, kwargs=kwargs, runtime_context=runtime_context)
        decision = self._runtime.check_model_input(text, ctx)
        if self._should_block_decision(decision):
            raise OmegaBlockedError(
                f"Omega blocked autogen messages (control_outcome={decision.control_outcome}, off={decision.off})",
                decision=decision,
            )
        return decision

    def _gate_tool_call(
        self,
        *,
        tool_name: str,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ):
        query_kwargs = self._extract_context_kwargs(kwargs, pop=False)
        ctx = self._build_session_context(messages=[], kwargs=query_kwargs, runtime_context=runtime_context)
        tool_args = {
            "args": [str(x) for x in list(args)],
            "kwargs": {str(k): str(v) for k, v in dict(kwargs).items()},
        }
        return self._runtime.check_tool_call(tool_name=str(tool_name), tool_args=tool_args, ctx=ctx)

    @staticmethod
    def _call_optional_getter(
        getter: Optional[Callable[..., Optional[str]]],
        *,
        messages: Any,
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> Optional[str]:
        if getter is None:
            return None
        for args in (
            (messages, kwargs, runtime_context),
            (messages, kwargs),
            (messages,),
        ):
            try:
                value = getter(*args)
                if value is not None and str(value).strip():
                    return str(value).strip()
                return None
            except TypeError:
                continue
        return None

    def _build_session_context(
        self,
        *,
        messages: Any,
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> AdapterSessionContext:
        session_id = self._call_optional_getter(
            self._session_id_getter,
            messages=messages,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )
        actor_id = self._call_optional_getter(
            self._actor_id_getter,
            messages=messages,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )

        if session_id is None:
            session_id = (
                self._lookup_identifier(kwargs, self._SESSION_KEYS)
                or self._lookup_identifier_in_messages(messages, self._SESSION_KEYS)
                or self._lookup_identifier(runtime_context, self._SESSION_KEYS)
                or "omega-ag-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(kwargs, self._ACTOR_KEYS)
                or self._lookup_identifier_in_messages(messages, self._ACTOR_KEYS)
                or self._lookup_identifier(runtime_context, self._ACTOR_KEYS)
                or session_id
            )
        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "autogen"},
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

    @classmethod
    def _lookup_identifier_in_messages(cls, messages: Any, keys: tuple[str, ...]) -> Optional[str]:
        if not isinstance(messages, list):
            return cls._lookup_identifier(messages, keys)
        for msg in messages:
            hit = cls._lookup_identifier(msg, keys)
            if hit:
                return hit
        return None

    def _normalize_messages_text(self, *, messages: Any) -> str:
        if not isinstance(messages, list):
            return self._normalize_content(messages)[: self._max_chars]
        parts: List[str] = []
        for message in messages:
            parts.append(self._normalize_message(message))
        return "\n".join(part for part in parts if part).strip()[: self._max_chars]

    @classmethod
    def _normalize_message(cls, message: Any) -> str:
        role = None
        content = None
        if isinstance(message, Mapping):
            role = message.get("role") or message.get("source") or message.get("type") or message.get("name")
            content = message.get("content")
            if content is None:
                content = message.get("text")
        else:
            role = (
                getattr(message, "role", None)
                or getattr(message, "source", None)
                or getattr(message, "type", None)
                or getattr(message, "name", None)
            )
            content = getattr(message, "content", None)
            if content is None:
                content = getattr(message, "text", None)

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
