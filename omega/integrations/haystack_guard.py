"""Haystack guard wrapper backed by Omega adapter runtime."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, Optional

from omega.adapters import (
    AdapterDecision,
    AdapterSessionContext,
    MemoryWriteDecision,
    OmegaAdapterRuntime,
    OmegaBlockedError,
    OmegaToolBlockedError,
)

try:  # pragma: no cover - optional dependency
    from haystack import component as _haystack_component
except Exception:  # pragma: no cover - optional dependency
    _haystack_component = None


def _component_decorator(cls):
    if _haystack_component is None:
        return cls
    return _haystack_component(cls)


def _output_types(**kwargs):
    def _decorator(fn):
        if _haystack_component is None:
            return fn
        return _haystack_component.output_types(**kwargs)(fn)

    return _decorator


class OmegaHaystackGuard:
    """Framework guard for Haystack pipeline and tool flows."""

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

    def build_guard_component(self) -> "OmegaGuardComponent":
        return OmegaGuardComponent(guard=self)

    def wrap_pipeline(self, pipeline: Any, *, component_name: str = "omega_guard_component") -> Any:
        if not hasattr(pipeline, "add_component"):
            raise ValueError("Haystack pipeline must provide add_component(name, component)")
        pipeline.add_component(component_name, self.build_guard_component())
        return pipeline

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
        payload: Optional[Dict[str, Any]] = None,
        runtime_context: Any = None,
        **context: Any,
    ) -> MemoryWriteDecision:
        kwargs = self._extract_context_kwargs(dict(context), pop=False)
        ctx = self._build_session_context(payload=payload, kwargs=kwargs, runtime_context=runtime_context)
        return self._runtime.check_memory_write(
            memory_text=memory_text,
            source_id=source_id,
            source_type=source_type,
            source_trust=source_trust,
            source_tags=source_tags,
            ctx=ctx,
        )

    def _guard_input(self, *, payload: Optional[Dict[str, Any]], kwargs: Dict[str, Any], runtime_context: Any) -> AdapterDecision:
        text = self._normalize_input_text(payload=payload, kwargs=kwargs)
        ctx = self._build_session_context(payload=payload, kwargs=kwargs, runtime_context=runtime_context)
        decision = self._runtime.check_model_input(text, ctx)
        if self._should_block_decision(decision):
            raise OmegaBlockedError(
                f"Omega blocked haystack input (control_outcome={decision.control_outcome}, off={decision.off})",
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
        ctx = self._build_session_context(payload=None, kwargs=query_kwargs, runtime_context=runtime_context)
        tool_args = {
            "args": [str(x) for x in list(args)],
            "kwargs": {str(k): str(v) for k, v in dict(kwargs).items()},
        }
        return self._runtime.check_tool_call(tool_name=str(tool_name), tool_args=tool_args, ctx=ctx)

    @staticmethod
    def _call_optional_getter(
        getter: Optional[Callable[..., Optional[str]]],
        *,
        payload: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> Optional[str]:
        if getter is None:
            return None
        for args in (
            (payload, kwargs, runtime_context),
            (payload, kwargs),
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
        payload: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        runtime_context: Any,
    ) -> AdapterSessionContext:
        session_id = self._call_optional_getter(
            self._session_id_getter,
            payload=payload,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )
        actor_id = self._call_optional_getter(
            self._actor_id_getter,
            payload=payload,
            kwargs=kwargs,
            runtime_context=runtime_context,
        )

        if session_id is None:
            session_id = (
                self._lookup_identifier(kwargs, self._SESSION_KEYS)
                or self._lookup_identifier(payload, self._SESSION_KEYS)
                or self._lookup_identifier(runtime_context, self._SESSION_KEYS)
                or "omega-hs-default"
            )
        if actor_id is None:
            actor_id = (
                self._lookup_identifier(kwargs, self._ACTOR_KEYS)
                or self._lookup_identifier(payload, self._ACTOR_KEYS)
                or self._lookup_identifier(runtime_context, self._ACTOR_KEYS)
                or session_id
            )

        return AdapterSessionContext(
            session_id=session_id,
            actor_id=actor_id,
            metadata={"framework": "haystack"},
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

    def _normalize_input_text(self, *, payload: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> str:
        text = ""
        for key in ("text", "query", "input_text", "message"):
            if isinstance(kwargs.get(key), str) and kwargs.get(key, "").strip():
                text = str(kwargs[key])
                break
        if not text and isinstance(payload, Mapping):
            for key in ("text", "query", "input_text", "message"):
                if isinstance(payload.get(key), str) and payload.get(key, "").strip():
                    text = str(payload[key])
                    break
        if not text:
            text = str(payload if payload is not None else kwargs)
        return " ".join(str(text).split())[: self._max_chars]

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


@_component_decorator
class OmegaGuardComponent:
    """Haystack pipeline component for model/input guard step."""

    def __init__(self, *, guard: OmegaHaystackGuard):
        self.guard = guard

    @_output_types(payload=dict, omega_decision=dict)
    def run(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        payload_copy = dict(payload or {})
        kwargs_copy = dict(kwargs)
        decision = self.guard._guard_input(payload=payload_copy, kwargs=kwargs_copy, runtime_context=self)  # noqa: SLF001

        passthrough = dict(payload_copy)
        passthrough.update({k: v for k, v in kwargs_copy.items() if k not in passthrough})
        return {
            "payload": passthrough,
            "omega_decision": {
                "session_id": decision.session_id,
                "step": decision.step,
                "off": decision.off,
                "control_outcome": decision.control_outcome,
                "reason_codes": list(decision.reason_codes),
                "trace_id": decision.trace_id,
                "decision_id": decision.decision_id,
            },
        }
