"""LangGraph guard wrapper backed by Omega adapter runtime."""

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


class OmegaLangGraphGuard:
    """Framework guard for LangGraph graph/runtime and tool flows."""

    _SESSION_KEYS = ("thread_id", "conversation_id", "session_id", "run_id")
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

    def wrap_graph(self, graph: Any) -> Any:
        guard = self

        class _WrappedGraph:
            def __init__(self, inner: Any):
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def invoke(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_graph_input(payload=payload, config=config, kwargs=context_kwargs, runtime_context=self._inner)
                if config is not None:
                    return self._inner.invoke(payload, config=config, **kwargs_copy)
                return self._inner.invoke(payload, **kwargs_copy)

            async def ainvoke(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "ainvoke"):
                    raise AttributeError("Wrapped graph does not support ainvoke(...)")
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_graph_input(payload=payload, config=config, kwargs=context_kwargs, runtime_context=self._inner)
                if config is not None:
                    return await self._inner.ainvoke(payload, config=config, **kwargs_copy)
                return await self._inner.ainvoke(payload, **kwargs_copy)

            def stream(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "stream"):
                    raise AttributeError("Wrapped graph does not support stream(...)")
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_graph_input(payload=payload, config=config, kwargs=context_kwargs, runtime_context=self._inner)
                if config is not None:
                    return self._inner.stream(payload, config=config, **kwargs_copy)
                return self._inner.stream(payload, **kwargs_copy)

            def astream(self, payload: Any, config: Any = None, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "astream"):
                    raise AttributeError("Wrapped graph does not support astream(...)")
                kwargs_copy = dict(kwargs)
                context_kwargs = guard._extract_context_kwargs(kwargs_copy, pop=True)
                guard._guard_graph_input(payload=payload, config=config, kwargs=context_kwargs, runtime_context=self._inner)
                if config is not None:
                    return self._inner.astream(payload, config=config, **kwargs_copy)
                return self._inner.astream(payload, **kwargs_copy)

        return _WrappedGraph(graph)

    def build_guard_node(self) -> Callable[..., Any]:
        def _guard_node(state: Any, config: Any = None, **kwargs: Any) -> Any:
            kwargs_copy = dict(kwargs)
            context_kwargs = self._extract_context_kwargs(kwargs_copy, pop=True)
            self._guard_graph_input(payload=state, config=config, kwargs=context_kwargs, runtime_context=None)
            return state

        return _guard_node

    def build_async_guard_node(self) -> Callable[..., Any]:
        async def _guard_node(state: Any, config: Any = None, **kwargs: Any) -> Any:
            kwargs_copy = dict(kwargs)
            context_kwargs = self._extract_context_kwargs(kwargs_copy, pop=True)
            self._guard_graph_input(payload=state, config=config, kwargs=context_kwargs, runtime_context=None)
            return state

        return _guard_node

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
        payload: Any = None,
        config: Any = None,
        runtime_context: Any = None,
        **context: Any,
    ) -> MemoryWriteDecision:
        context_kwargs = self._extract_context_kwargs(dict(context), pop=False)
        ctx = self._build_session_context(
            payload=payload,
            config=config,
            kwargs=context_kwargs,
            runtime_context=runtime_context,
        )
        return self._runtime.check_memory_write(
            memory_text=memory_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            source_tags=source_tags,
            ctx=ctx,
        )

    def _guard_graph_input(
        self,
        *,
        payload: Any,
        config: Any,
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> AdapterDecision:
        text = self._normalize_payload_text(payload=payload, config=config, kwargs=kwargs)
        ctx = self._build_session_context(payload=payload, config=config, kwargs=kwargs, runtime_context=runtime_context)
        decision = self._runtime.check_model_input(text, ctx)
        if self._should_block_decision(decision):
            raise OmegaBlockedError(
                f"Omega blocked langgraph input (control_outcome={decision.control_outcome}, off={decision.off})",
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
        ctx = self._build_session_context(
            payload={"tool_name": str(tool_name)},
            config=None,
            kwargs=query_kwargs,
            runtime_context=runtime_context,
        )
        tool_args = {
            "args": [str(x) for x in list(args)],
            "kwargs": {str(k): str(v) for k, v in dict(kwargs).items()},
        }
        return self._runtime.check_tool_call(tool_name=str(tool_name), tool_args=tool_args, ctx=ctx)

    @staticmethod
    def _call_optional_getter(
        getter: Optional[Callable[..., Optional[str]]],
        *,
        payload: Any,
        config: Any,
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> Optional[str]:
        if getter is None:
            return None
        for args in (
            (payload, config, kwargs, runtime_context),
            (payload, config, kwargs),
            (payload, config),
            (payload,),
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
        payload: Any,
        config: Any,
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> AdapterSessionContext:
        session_id = self._call_optional_getter(
            self._session_id_getter,
            payload=payload,
            config=config,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )
        actor_id = self._call_optional_getter(
            self._actor_id_getter,
            payload=payload,
            config=config,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )

        if session_id is None:
            session_id = (
                self._lookup_identifier(kwargs, self._SESSION_KEYS)
                or self._lookup_identifier(payload, self._SESSION_KEYS)
                or self._lookup_identifier(config, self._SESSION_KEYS)
                or self._lookup_identifier(runtime_context, self._SESSION_KEYS)
                or "omega-lg-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(kwargs, self._ACTOR_KEYS)
                or self._lookup_identifier(payload, self._ACTOR_KEYS)
                or self._lookup_identifier(config, self._ACTOR_KEYS)
                or self._lookup_identifier(runtime_context, self._ACTOR_KEYS)
                or session_id
            )
        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "langgraph"},
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
            for nested_key in ("configurable", "metadata", "context", "config", "runtime", "state"):
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
        for nested_attr in ("configurable", "metadata", "context", "config", "runtime", "state"):
            nested = getattr(obj, nested_attr, None)
            if nested is obj:
                continue
            nested_value = cls._lookup_identifier(nested, keys)
            if nested_value:
                return nested_value
        return None

    def _normalize_payload_text(self, *, payload: Any, config: Any, kwargs: Dict[str, Any]) -> str:
        text = self._extract_query_text(payload)
        if not text:
            text = self._extract_query_text(kwargs)
        if not text:
            text = self._extract_query_text(config)
        return " ".join(str(text or "").split())[: self._max_chars]

    @classmethod
    def _extract_query_text(cls, obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Mapping):
            messages = obj.get("messages")
            if isinstance(messages, list):
                parts = [cls._normalize_message(msg) for msg in messages]
                msg_text = "\n".join(part for part in parts if part).strip()
                if msg_text:
                    return msg_text
            for key in ("query_str", "query", "text", "message", "input"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return str(obj)

        for attr in ("messages", "query_str", "query", "text", "message", "input"):
            value = getattr(obj, attr, None)
            if attr == "messages" and isinstance(value, list):
                parts = [cls._normalize_message(msg) for msg in value]
                msg_text = "\n".join(part for part in parts if part).strip()
                if msg_text:
                    return msg_text
            if isinstance(value, str) and value.strip():
                return value
        return str(obj)

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
            chunks: List[str] = []
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
            "run_id",
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
