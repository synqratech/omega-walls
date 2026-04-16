"""Public SDK facade for frictionless Omega Walls integration."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import uuid

import numpy as np

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.errors import (
    OmegaAPIError,
    OmegaConfigError,
    OmegaInitializationError,
    OmegaMissingDependencyError,
    OmegaRuntimeError,
    OmegaSDKError,
)
from omega.interfaces.contracts_v1 import ContentItem, OffAction, OmegaState
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.sdk_types import DetectionResult, GuardAction, GuardDecision, OmegaDetectionResult


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _reason_codes(reasons: Any) -> List[str]:
    out: List[str] = []
    if bool(getattr(reasons, "reason_spike", False)):
        out.append("reason_spike")
    if bool(getattr(reasons, "reason_wall", False)):
        out.append("reason_wall")
    if bool(getattr(reasons, "reason_sum", False)):
        out.append("reason_sum")
    if bool(getattr(reasons, "reason_multi", False)):
        out.append("reason_multi")
    return out


def _to_missing_dependency_error(exc: ModuleNotFoundError) -> OmegaMissingDependencyError:
    dep = str(getattr(exc, "name", "") or "unknown")
    return OmegaMissingDependencyError(
        dep,
        hint="Install required dependencies (for example: pip install omega-walls or omega-walls[api]).",
    )


def _is_api_error_message(message: str) -> bool:
    text = str(message or "").lower()
    markers = (
        "missing_api_key",
        "missing_env:",
        "api_call_failed:",
        "url_error:",
        "api_adapter",
        "schema_error:",
    )
    return any(m in text for m in markers)


def _map_exception(*, exc: Exception, phase: str) -> OmegaSDKError:
    if isinstance(exc, OmegaSDKError):
        return exc
    if isinstance(exc, ModuleNotFoundError):
        return _to_missing_dependency_error(exc)

    msg = str(exc)
    if phase == "config":
        return OmegaConfigError(f"Failed to load/validate config: {msg}")

    # API adapter may raise APIRequestError with code/body fields without exposing class publicly.
    if hasattr(exc, "code") and hasattr(exc, "body"):
        code = getattr(exc, "code", None)
        body = getattr(exc, "body", None)
        return OmegaAPIError(f"API request failed (code={code})", code=code, details={"body": body})
    if _is_api_error_message(msg):
        return OmegaAPIError(msg)

    if phase == "init":
        return OmegaInitializationError(f"Failed to initialize OmegaWalls: {msg}")
    return OmegaRuntimeError(f"Detection runtime failed: {msg}")


class OmegaWalls:
    """Stable public facade for text-level detection in 5-10 lines of code."""

    def __init__(
        self,
        *,
        profile: str = "quickstart",
        config_dir: Optional[str] = None,
        projector_mode: Optional[str] = None,
        api_model: Optional[str] = None,
        cli_overrides: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
        default_trust: str = "untrusted",
        default_source_type: str = "other",
        default_session_id: str = "omega-sdk-default",
    ) -> None:
        effective_overrides: Dict[str, Any] = {}
        if isinstance(cli_overrides, Mapping):
            effective_overrides = dict(cli_overrides)

        projector_override: Dict[str, Any] = {}
        if projector_mode is not None:
            projector_override.setdefault("projector", {})["mode"] = str(projector_mode)
        if api_model is not None:
            projector_override.setdefault("projector", {}).setdefault("api_perception", {})["model"] = str(api_model)
        if projector_override:
            effective_overrides = _deep_merge(effective_overrides, projector_override)

        try:
            snapshot = load_resolved_config(
                config_dir=str(config_dir) if config_dir else None,
                profile=str(profile),
                cli_overrides=effective_overrides or None,
                env=dict(env) if env is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="config") from exc

        cfg = snapshot.resolved
        try:
            params = omega_params_from_config(cfg)
            self._snapshot = snapshot
            self._config = cfg
            self._projector = build_projector(cfg)
            self._core = OmegaCoreV1(params)
            self._off_policy = OffPolicyV1(cfg)
            self._states: Dict[str, OmegaState] = {}
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="init") from exc

        self.profile = str(profile)
        self.walls = list(params.walls)
        self.default_trust = str(default_trust or "untrusted")
        self.default_source_type = str(default_source_type or "other")
        self.default_session_id = str(default_session_id or "omega-sdk-default")

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def reset_session(self, session_id: Optional[str] = None) -> None:
        sid = str(session_id or self.default_session_id)
        self._states.pop(sid, None)

    def reset_all_sessions(self) -> None:
        self._states.clear()

    def analyze_text(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        source_id: str = "sdk:text",
        source_type: Optional[str] = None,
        trust: Optional[str] = None,
        doc_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        reset_session: bool = False,
    ) -> DetectionResult:
        sid = str(session_id or self.default_session_id)
        if bool(reset_session) or sid not in self._states:
            self._states[sid] = OmegaState(
                session_id=sid,
                m=np.zeros(len(self.walls), dtype=float),
                step=0,
            )
        state = self._states[sid]

        item = ContentItem(
            doc_id=str(doc_id or f"sdk-{uuid.uuid4().hex[:12]}"),
            source_id=str(source_id),
            source_type=str(source_type or self.default_source_type),
            trust=str(trust or self.default_trust),
            text=str(text or ""),
            meta=dict(meta or {}),
        )
        try:
            projection = self._projector.project(item)
            step_result = self._core.step(state=state, items=[item], projections=[projection])
            decision = self._off_policy.select_actions(step_result=step_result, items=[item])
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc=exc, phase="runtime") from exc

        wall_scores = {
            wall: float(step_result.p[idx])
            for idx, wall in enumerate(self.walls)
        }
        memory_scores = {
            wall: float(step_result.m_next[idx])
            for idx, wall in enumerate(self.walls)
        }
        walls_triggered = sorted(
            wall
            for idx, wall in enumerate(self.walls)
            if float(step_result.p[idx]) > 0.0 or float(step_result.m_next[idx]) > 0.0
        )
        actions = [self._action_to_guard_action(action) for action in list(decision.actions)]
        guard_decision = GuardDecision(
            off=bool(step_result.off),
            severity=str(decision.severity),
            control_outcome=str(decision.control_outcome),
            reason_codes=_reason_codes(step_result.reasons),
            top_docs=[str(x) for x in list(step_result.top_docs)],
            actions=actions,
        )

        return DetectionResult(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            walls_triggered=walls_triggered,
            wall_scores=wall_scores,
            memory_scores=memory_scores,
            decision=guard_decision,
        )

    def _action_to_guard_action(self, action: OffAction) -> GuardAction:
        return GuardAction(
            type=str(action.type),
            target=str(action.target),
            doc_ids=list(action.doc_ids) if action.doc_ids else None,
            source_ids=list(action.source_ids) if action.source_ids else None,
            tool_mode=str(action.tool_mode) if action.tool_mode else None,
            allowlist=list(action.allowlist) if action.allowlist else None,
            horizon_steps=int(action.horizon_steps) if action.horizon_steps is not None else None,
            incident_packet=dict(action.incident_packet) if action.incident_packet else None,
        )
