"""LlamaIndex guard wrapper backed by Omega adapter runtime."""

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


class OmegaLlamaIndexGuard:
    """Framework guard for LlamaIndex query and tool flows."""

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

    def wrap_query_engine(self, query_engine: Any) -> Any:
        guard = self
        wrapped = query_engine

        class _WrappedQueryEngine:
            def __init__(self, inner: Any):
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def query(self, query_obj: Any, **kwargs: Any) -> Any:
                query_kwargs = guard._extract_context_kwargs(kwargs, pop=True)
                guard._guard_query(
                    query_obj=query_obj,
                    query_kwargs=query_kwargs,
                    runtime_context=self._inner,
                )
                if kwargs:
                    return self._inner.query(query_obj, **kwargs)
                return self._inner.query(query_obj)

            async def aquery(self, query_obj: Any, **kwargs: Any) -> Any:
                if not hasattr(self._inner, "aquery"):
                    raise AttributeError("Wrapped query engine does not support aquery(...)")
                query_kwargs = guard._extract_context_kwargs(kwargs, pop=True)
                guard._guard_query(
                    query_obj=query_obj,
                    query_kwargs=query_kwargs,
                    runtime_context=self._inner,
                )
                if kwargs:
                    return await self._inner.aquery(query_obj, **kwargs)
                return await self._inner.aquery(query_obj)

        return _WrappedQueryEngine(wrapped)

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
        query_obj: Any = None,
        runtime_context: Any = None,
        **context: Any,
    ) -> MemoryWriteDecision:
        query_kwargs = self._extract_context_kwargs(dict(context), pop=False)
        ctx = self._build_session_context(
            query_obj=query_obj,
            query_kwargs=query_kwargs,
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

    def _guard_query(self, *, query_obj: Any, query_kwargs: Dict[str, Any], runtime_context: Any) -> None:
        text = self._normalize_query_text(query_obj=query_obj, query_kwargs=query_kwargs)
        ctx = self._build_session_context(
            query_obj=query_obj,
            query_kwargs=query_kwargs,
            runtime_context=runtime_context,
        )
        decision = self._runtime.check_model_input(text, ctx)
        if self._should_block_decision(decision):
            raise OmegaBlockedError(
                f"Omega blocked llamaindex query (control_outcome={decision.control_outcome}, off={decision.off})",
                decision=decision,
            )

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
            query_obj={"tool_name": tool_name},
            query_kwargs=query_kwargs,
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
        query_obj: Any,
        query_kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> Optional[str]:
        if getter is None:
            return None
        for args in (
            (query_obj, query_kwargs, runtime_context),
            (query_obj, query_kwargs),
            (query_obj,),
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
        query_obj: Any,
        query_kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> AdapterSessionContext:
        session_id = self._call_optional_getter(
            self._session_id_getter,
            query_obj=query_obj,
            query_kwargs=query_kwargs,
            runtime_context=runtime_context,
        )
        actor_id = self._call_optional_getter(
            self._actor_id_getter,
            query_obj=query_obj,
            query_kwargs=query_kwargs,
            runtime_context=runtime_context,
        )

        if session_id is None:
            session_id = (
                self._lookup_identifier(query_kwargs, self._SESSION_KEYS)
                or self._lookup_identifier(query_obj, self._SESSION_KEYS)
                or self._lookup_identifier(runtime_context, self._SESSION_KEYS)
                or "omega-li-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(query_kwargs, self._ACTOR_KEYS)
                or self._lookup_identifier(query_obj, self._ACTOR_KEYS)
                or self._lookup_identifier(runtime_context, self._ACTOR_KEYS)
                or session_id
            )

        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "llamaindex"},
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

    def _normalize_query_text(self, *, query_obj: Any, query_kwargs: Dict[str, Any]) -> str:
        text = self._extract_query_text(query_obj)
        if not text:
            text = (
                self._extract_query_text(query_kwargs.get("query"))
                or self._extract_query_text(query_kwargs.get("query_str"))
                or self._extract_query_text(query_kwargs.get("text"))
            )
        return " ".join(str(text or "").split())[: self._max_chars]

    @classmethod
    def _extract_query_text(cls, obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Mapping):
            for key in ("query_str", "query", "text", "message", "input"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return str(obj)

        for attr in ("query_str", "query", "text", "message", "input"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return str(obj)

    @staticmethod
    def _should_block_decision(decision: AdapterDecision) -> bool:
        if decision.off:
            return True
        if str(decision.control_outcome).upper() in {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}:
            return True
        block_actions = {"SOFT_BLOCK", "TOOL_FREEZE", "SOURCE_QUARANTINE", "HUMAN_ESCALATE"}
        action_types = {str(action.type).upper() for action in decision.actions}
        return bool(action_types & block_actions)
