from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.rag.harness import MockLLM, OmegaRAGHarness
from omega.tools.tool_gateway import ToolGatewayV1
from scripts.session_pack_leakage import load_dual_session_pack


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(x) for x in values)
    if len(vals) == 1:
        return vals[0]
    # Linear interpolation percentile equivalent to common default behavior.
    rank = 0.95 * (len(vals) - 1)
    low = int(rank)
    high = min(low + 1, len(vals) - 1)
    frac = rank - low
    return float(vals[low] * (1.0 - frac) + vals[high] * frac)


def _norm_label(value: Any, *, default: str) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "true", "attack"}:
        return "attack"
    if raw in {"0", "false", "benign"}:
        return "benign"
    return default


def infer_eval_slice_from_source_ref(source_ref: str) -> str:
    ref = str(source_ref or "")
    if "EIA_wo_EI.jsonl" in ref or "VWA_adv_wo_EI.jsonl" in ref:
        return "context_required"
    return "text_intrinsic"


@dataclass(frozen=True)
class SessionTurnRow:
    session_id: str
    turn_id: int
    text: str
    source_id: str
    source_type: str
    label_turn: str
    label_session: str
    family: str
    source_ref: str
    actor_id: str
    bucket: str
    eval_slice: str


@dataclass(frozen=True)
class RuntimeTurnPayload:
    session_id: str
    turn_id: int
    text: str
    source_id: str
    source_type: str


@dataclass(frozen=True)
class RuntimePacketPayload:
    user_query: str
    packet_items: List[ContentItem]
    provenance_mode: str
    segment_count: int
    trusted_user_segments: int
    untrusted_segments: int


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    actor_id: str
    bucket: str
    family: str
    label_session: str
    eval_slice: str
    turns: List[SessionTurnRow]


@dataclass(frozen=True)
class SessionOutcome:
    session_id: str
    actor_id: str
    bucket: str
    family: str
    label_session: str
    eval_slice: str
    turn_count: int
    detected_off: bool
    first_off_turn: Optional[int]
    late_detect: bool
    max_turn_p: float
    off_reasons: Dict[str, int]


class SessionRunner(Protocol):
    def reset(self, *, session_id: str, actor_id: str) -> None:
        ...

    def run_turn(self, *, session_id: str, actor_id: str, turn: RuntimeTurnPayload) -> Dict[str, Any]:
        ...


def build_runtime_turn_payload(turn: SessionTurnRow) -> RuntimeTurnPayload:
    return RuntimeTurnPayload(
        session_id=str(turn.session_id),
        turn_id=int(turn.turn_id),
        text=str(turn.text),
        source_id=str(turn.source_id),
        source_type=str(turn.source_type),
    )


def _trust_for_source_type(source_type: str) -> str:
    st = str(source_type).strip().lower()
    if st in {"trusted", "internal_trusted", "policy"}:
        return "trusted"
    if st in {"semi", "semi_trusted", "ticket", "chat"}:
        return "semi"
    return "untrusted"


def _split_segmented_turn_text(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if "\n\n" not in raw:
        return raw, ""
    first, rest = raw.split("\n\n", 1)
    return first.strip(), rest.strip()


def build_runtime_packet_payload(
    turn: RuntimeTurnPayload,
    *,
    provenance_mode: str = "blob",
) -> RuntimePacketPayload:
    mode = str(provenance_mode or "blob").strip().lower()
    if mode not in {"blob", "segmented"}:
        raise ValueError("provenance_mode must be blob or segmented")

    source_type = str(turn.source_type or "external_untrusted").strip() or "external_untrusted"
    source_id = str(turn.source_id or "").strip()
    if mode == "blob":
        item = ContentItem(
            doc_id=f"{turn.session_id}:turn:{int(turn.turn_id):03d}",
            source_id=source_id,
            source_type=source_type,
            trust=_trust_for_source_type(source_type),
            text=str(turn.text),
        )
        return RuntimePacketPayload(
            user_query=str(turn.text),
            packet_items=[item],
            provenance_mode="blob",
            segment_count=1,
            trusted_user_segments=0,
            untrusted_segments=1,
        )

    user_text, evidence_text = _split_segmented_turn_text(str(turn.text))
    items: List[ContentItem] = []
    if user_text:
        items.append(
            ContentItem(
                doc_id=f"{turn.session_id}:turn:{int(turn.turn_id):03d}:user",
                source_id=f"{source_id}:user",
                source_type="user",
                trust="trusted_user",
                text=user_text,
                origin="user",
                meta={"provenance_mode": "segmented", "segment_role": "trusted_user"},
            )
        )
    if evidence_text:
        items.append(
            ContentItem(
                doc_id=f"{turn.session_id}:turn:{int(turn.turn_id):03d}:evidence",
                source_id=f"{source_id}:evidence",
                source_type=source_type,
                trust=_trust_for_source_type(source_type),
                text=evidence_text,
                origin="retrieval",
                meta={"provenance_mode": "segmented", "segment_role": "untrusted_evidence"},
            )
        )
    if not items:
        items.append(
            ContentItem(
                doc_id=f"{turn.session_id}:turn:{int(turn.turn_id):03d}:empty",
                source_id=source_id,
                source_type=source_type,
                trust=_trust_for_source_type(source_type),
                text=str(turn.text),
                meta={"provenance_mode": "segmented", "segment_role": "fallback_blob"},
            )
        )
    return RuntimePacketPayload(
        user_query=user_text or str(turn.text),
        packet_items=items,
        provenance_mode="segmented",
        segment_count=len(items),
        trusted_user_segments=sum(1 for item in items if str(item.trust) == "trusted_user"),
        untrusted_segments=sum(1 for item in items if str(item.trust) != "trusted_user"),
    )


class OmegaHarnessRunner:
    def __init__(
        self,
        *,
        profile: str,
        mode: str,
        seed: int,
        state_db_path: Path,
        strict_projector: bool = False,
        require_api_adapter: bool = False,
        api_model: Optional[str] = None,
        api_provider: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_timeout_sec: Optional[float] = None,
        api_retries: Optional[int] = None,
        api_cache_path: Optional[str] = None,
        api_error_log_path: Optional[str] = None,
        blind_eval: bool = False,
        enable_stateful_support_tuning: bool = False,
        enable_effects_shadow: bool = False,
        provenance_mode: str = "blob",
    ) -> None:
        random.seed(int(seed))
        mode_norm = str(mode).strip().lower()
        provenance_mode_norm = str(provenance_mode or "blob").strip().lower()
        if provenance_mode_norm not in {"blob", "segmented"}:
            raise ValueError("provenance_mode must be blob or segmented")
        projector_override: Dict[str, Any] = {"mode": mode_norm}
        if mode_norm == "hybrid_api":
            api_perception: Dict[str, Any] = {
                "enabled": "true",
                **({"strict": True} if bool(strict_projector) else {}),
                **({"model": str(api_model)} if api_model else {}),
                **({"provider": str(api_provider)} if api_provider else {}),
                **({"api_key_env": str(api_key_env)} if api_key_env else {}),
                **({"base_url": str(api_base_url)} if api_base_url else {}),
                **({"timeout_sec": float(api_timeout_sec)} if api_timeout_sec is not None else {}),
                **({"max_retries": int(api_retries)} if api_retries is not None else {}),
                **({"cache_path": str(api_cache_path)} if api_cache_path else {}),
                **({"error_log_path": str(api_error_log_path)} if api_error_log_path else {}),
            }
            projector_override = {
                "mode": "hybrid_api",
                "fallback_to_pi0": not bool(strict_projector),
                "api_perception": api_perception,
            }
        elif bool(strict_projector):
            projector_override = {
                "mode": mode_norm,
                "fallback_to_pi0": False,
                **({"pitheta": {"enabled": "true"}} if mode_norm in {"pitheta", "hybrid"} else {}),
            }

        cli_overrides: Dict[str, Any] = {"projector": projector_override}
        if bool(enable_stateful_support_tuning):
            cli_overrides["off_policy"] = {"stateful_support_tuning": {"enabled": True}}
        if bool(enable_effects_shadow):
            cli_overrides["effects"] = {"enabled": True, "mode": "shadow"}
        snapshot = load_resolved_config(profile=profile, cli_overrides=cli_overrides)
        cfg = deepcopy(snapshot.resolved)
        cfg.setdefault("off_policy", {}).setdefault("cross_session", {})
        cfg["off_policy"]["cross_session"]["sqlite_path"] = str(state_db_path.as_posix())
        state_db_path.parent.mkdir(parents=True, exist_ok=True)
        projector = build_projector(cfg)
        if mode_norm == "hybrid_api" and bool(require_api_adapter):
            ensure_api = getattr(projector, "ensure_api_adapter_active", None)
            if not callable(ensure_api):
                raise RuntimeError("hybrid_api adapter verification is not supported by the current projector")
            if not bool(ensure_api()):
                status_fn = getattr(projector, "api_perception_status", None)
                status = status_fn() if callable(status_fn) else {}
                err = (
                    status.get("api_adapter_error", "unknown")
                    if isinstance(status, Mapping)
                    else "unknown"
                )
                raise RuntimeError(f"hybrid_api adapter inactive: {err}")

        self._mode = mode_norm
        self._strict_projector = bool(strict_projector)
        self._require_api_adapter = bool(require_api_adapter)
        self._blind_eval = bool(blind_eval)
        self._provenance_mode = provenance_mode_norm

        self._harness = OmegaRAGHarness(
            projector=projector,
            omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
            off_policy=OffPolicyV1(cfg),
            tool_gateway=ToolGatewayV1(cfg),
            config=cfg,
            llm_backend=MockLLM(),
        )

    def projector_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "mode": str(self._mode),
            "strict_projector": bool(self._strict_projector),
            "require_api_adapter": bool(self._require_api_adapter),
            "blind_eval": bool(self._blind_eval),
        }
        projector = getattr(self._harness, "projector", None)
        semantic_status_fn = getattr(projector, "semantic_status", None)
        semantic_status = semantic_status_fn() if callable(semantic_status_fn) else {}
        if isinstance(semantic_status, Mapping):
            status["semantic"] = dict(semantic_status)
            status["semantic_active"] = bool(semantic_status.get("active", False))
        if str(self._mode) == "hybrid_api":
            api_status_fn = getattr(projector, "api_perception_status", None)
            api_status = api_status_fn() if callable(api_status_fn) else {}
            if isinstance(api_status, Mapping):
                status["api_perception"] = dict(api_status)
        return status

    def reset(self, *, session_id: str, actor_id: str) -> None:
        self._harness.reset_state(session_id=session_id, actor_id=actor_id)

    def run_turn(self, *, session_id: str, actor_id: str, turn: RuntimeTurnPayload) -> Dict[str, Any]:
        runtime_packet = build_runtime_packet_payload(
            turn,
            provenance_mode=getattr(self, "_provenance_mode", "blob"),
        )
        out = self._harness.run_step(
            user_query=runtime_packet.user_query,
            packet_items=runtime_packet.packet_items,
            actor_id=actor_id,
        )
        step_result = out["step_result"]
        p_vec = getattr(step_result, "p", None)
        if p_vec is None:
            max_p = 0.0
        else:
            try:
                p_len = len(p_vec)  # type: ignore[arg-type]
            except TypeError:
                p_len = 1
            if p_len == 0:
                max_p = 0.0
            else:
                try:
                    max_p = float(max(float(x) for x in p_vec))  # type: ignore[union-attr]
                except TypeError:
                    max_p = float(p_vec)
        return {
            "off": bool(step_result.off),
            "max_p": max_p,
            "effect_wall_candidate": (
                dict(out.get("effect_wall_candidate", {}))
                if isinstance(out.get("effect_wall_candidate"), Mapping)
                else None
            ),
            "effect_policy_gate": (
                dict(out.get("effect_policy_gate", {}))
                if isinstance(out.get("effect_policy_gate"), Mapping)
                else None
            ),
            "effect_forecast": (
                dict(out.get("effect_forecast", {}))
                if isinstance(out.get("effect_forecast"), Mapping)
                else None
            ),
            "effect_forecast_status": str(out.get("effect_forecast_status", "disabled")),
            "effect_policy_gate_status": str(out.get("effect_policy_gate_status", "disabled")),
            "named_skill_invocation": (
                dict(out.get("named_skill_invocation", {}))
                if isinstance(out.get("named_skill_invocation"), Mapping)
                else None
            ),
            "skill_provenance_assessment": (
                dict(out.get("skill_provenance_assessment", {}))
                if isinstance(out.get("skill_provenance_assessment"), Mapping)
                else None
            ),
            "skillbox_status": str(out.get("skillbox_status", "disabled")),
            "skillbox_verification": (
                dict(out.get("skillbox_verification", {}))
                if isinstance(out.get("skillbox_verification"), Mapping)
                else None
            ),
            "skillbox_ledger_hit": bool(out.get("skillbox_ledger_hit", False)),
            "skillbox_content_sha256": out.get("skillbox_content_sha256"),
            "skillbox_capabilities": [
                str(x) for x in list(out.get("skillbox_capabilities", []) or [])
            ],
            "skillbox_gate_decision": str(out.get("skillbox_gate_decision", "disabled")),
            "effect_text_analysis": (
                dict(out.get("effect_text_analysis", {}))
                if isinstance(out.get("effect_text_analysis"), Mapping)
                else None
            ),
            "provenance": {
                "mode": str(runtime_packet.provenance_mode),
                "segment_count": int(runtime_packet.segment_count),
                "trusted_user_segments": int(runtime_packet.trusted_user_segments),
                "untrusted_segments": int(runtime_packet.untrusted_segments),
            },
            "off_reasons": {
                "reason_spike": int(bool(step_result.reasons.reason_spike)),
                "reason_wall": int(bool(step_result.reasons.reason_wall)),
                "reason_sum": int(bool(step_result.reasons.reason_sum)),
                "reason_multi": int(bool(step_result.reasons.reason_multi)),
            },
        }


def load_pack_rows(
    path: Path,
    *,
    labels_path: Optional[Path] = None,
    allow_legacy_runtime_leakage: bool = False,
) -> List[SessionTurnRow]:
    rows: List[SessionTurnRow] = []
    loaded = load_dual_session_pack(
        runtime_pack_path=path,
        labels_pack_path=labels_path,
        allow_legacy_runtime_leakage=bool(allow_legacy_runtime_leakage),
    )
    for raw in loaded:
        session_id = str(raw.get("session_id", "")).strip()
        source_id = str(raw.get("source_id", "")).strip()
        if not source_id and allow_legacy_runtime_leakage:
            source_ref_fallback = str(raw.get("source_ref", "")).strip()
            source_id = source_ref_fallback or f"legacy:{session_id}:{raw.get('turn_id', 0)}"
        text = str(raw.get("text", "")).strip()
        if not session_id or not source_id or not text:
            continue
        turn_id = int(raw.get("turn_id", 0))
        if turn_id <= 0:
            continue
        label_session = _norm_label(raw.get("label_session"), default="benign")
        label_turn = _norm_label(raw.get("label_turn"), default=label_session)
        rows.append(
            SessionTurnRow(
                session_id=session_id,
                turn_id=turn_id,
                text=text,
                source_id=source_id,
                source_type=str(raw.get("source_type", "external_untrusted")).strip() or "external_untrusted",
                label_turn=label_turn,
                label_session=label_session,
                family=str(raw.get("family", "unknown")).strip() or "unknown",
                source_ref=str(raw.get("source_ref", "unknown")).strip() or "unknown",
                actor_id=str(raw.get("actor_id", session_id)).strip() or session_id,
                bucket=str(raw.get("bucket", "core")).strip() or "core",
                eval_slice=str(raw.get("eval_slice", "")).strip() or infer_eval_slice_from_source_ref(str(raw.get("source_ref", ""))),
            )
        )
    return rows


def group_sessions(rows: Sequence[SessionTurnRow]) -> List[SessionSpec]:
    grouped: Dict[str, List[SessionTurnRow]] = defaultdict(list)
    for row in rows:
        grouped[row.session_id].append(row)
    sessions: List[SessionSpec] = []
    for sid in sorted(grouped.keys()):
        turns = sorted(grouped[sid], key=lambda r: int(r.turn_id))
        head = turns[0]
        for row in turns[1:]:
            if row.label_session != head.label_session:
                raise ValueError(f"inconsistent label_session for session_id={sid}")
            if row.family != head.family:
                raise ValueError(f"inconsistent family for session_id={sid}")
            if row.actor_id != head.actor_id:
                raise ValueError(f"inconsistent actor_id for session_id={sid}")
            if row.bucket != head.bucket:
                raise ValueError(f"inconsistent bucket for session_id={sid}")
        session_eval_slice = "text_intrinsic"
        attack_rows = [r for r in turns if r.label_turn == "attack"]
        scope_rows = attack_rows if attack_rows else turns
        if any(r.eval_slice == "context_required" for r in scope_rows):
            session_eval_slice = "context_required"
        sessions.append(
            SessionSpec(
                session_id=sid,
                actor_id=head.actor_id,
                bucket=head.bucket,
                family=head.family,
                label_session=head.label_session,
                eval_slice=session_eval_slice,
                turns=turns,
            )
        )
    return sessions


def _is_late_detect(*, first_off_turn: int, turn_count: int) -> bool:
    if turn_count <= 0:
        return False
    return (float(first_off_turn) / float(turn_count)) > 0.70


def evaluate_sessions(
    *,
    sessions: Sequence[SessionSpec],
    runner: SessionRunner,
) -> tuple[List[SessionOutcome], List[Dict[str, Any]]]:
    outcomes: List[SessionOutcome] = []
    trace_rows: List[Dict[str, Any]] = []

    for session in sessions:
        runner.reset(session_id=session.session_id, actor_id=session.actor_id)
        first_off_turn: Optional[int] = None
        max_turn_p = 0.0
        off_reason_counts: Dict[str, int] = Counter()
        for turn in session.turns:
            runtime_turn = build_runtime_turn_payload(turn)
            result = runner.run_turn(
                session_id=session.session_id,
                actor_id=session.actor_id,
                turn=runtime_turn,
            )
            off = bool(result.get("off", False))
            max_p = float(result.get("max_p", 0.0))
            max_turn_p = max(max_turn_p, max_p)
            for key, val in (result.get("off_reasons", {}) or {}).items():
                off_reason_counts[str(key)] += int(val)
            if off and first_off_turn is None:
                first_off_turn = int(turn.turn_id)

            trace_rows.append(
                {
                    "session_id": session.session_id,
                    "actor_id": session.actor_id,
                    "bucket": session.bucket,
                    "family": session.family,
                    "eval_slice": session.eval_slice,
                    "turn_id": int(turn.turn_id),
                    "label_turn": turn.label_turn,
                    "label_session": session.label_session,
                    "off": off,
                    "max_p": max_p,
                    "effect_wall_candidate": (
                        dict(result.get("effect_wall_candidate", {}))
                        if isinstance(result.get("effect_wall_candidate"), Mapping)
                        else None
                    ),
                    "effect_policy_gate": (
                        dict(result.get("effect_policy_gate", {}))
                        if isinstance(result.get("effect_policy_gate"), Mapping)
                        else None
                    ),
                    "effect_forecast": (
                        dict(result.get("effect_forecast", {}))
                        if isinstance(result.get("effect_forecast"), Mapping)
                        else None
                    ),
                    "effect_forecast_status": str(
                        result.get("effect_forecast_status", "disabled")
                    ),
                    "effect_policy_gate_status": str(
                        result.get("effect_policy_gate_status", "disabled")
                    ),
                    "named_skill_invocation": (
                        dict(result.get("named_skill_invocation", {}))
                        if isinstance(result.get("named_skill_invocation"), Mapping)
                        else None
                    ),
                    "skill_provenance_assessment": (
                        dict(result.get("skill_provenance_assessment", {}))
                        if isinstance(result.get("skill_provenance_assessment"), Mapping)
                        else None
                    ),
                    "skillbox_status": str(result.get("skillbox_status", "disabled")),
                    "skillbox_verification": (
                        dict(result.get("skillbox_verification", {}))
                        if isinstance(result.get("skillbox_verification"), Mapping)
                        else None
                    ),
                    "skillbox_ledger_hit": bool(result.get("skillbox_ledger_hit", False)),
                    "skillbox_content_sha256": result.get("skillbox_content_sha256"),
                    "skillbox_capabilities": [
                        str(x) for x in list(result.get("skillbox_capabilities", []) or [])
                    ],
                    "skillbox_gate_decision": str(result.get("skillbox_gate_decision", "disabled")),
                    "effect_text_analysis": (
                        dict(result.get("effect_text_analysis", {}))
                        if isinstance(result.get("effect_text_analysis"), Mapping)
                        else None
                    ),
                    "provenance": (
                        dict(result.get("provenance", {}))
                        if isinstance(result.get("provenance"), Mapping)
                        else None
                    ),
                }
            )

        detected = first_off_turn is not None
        late_detect = _is_late_detect(first_off_turn=int(first_off_turn), turn_count=len(session.turns)) if detected else False
        outcomes.append(
            SessionOutcome(
                session_id=session.session_id,
                actor_id=session.actor_id,
                bucket=session.bucket,
                family=session.family,
                label_session=session.label_session,
                eval_slice=session.eval_slice,
                turn_count=len(session.turns),
                detected_off=detected,
                first_off_turn=int(first_off_turn) if first_off_turn is not None else None,
                late_detect=late_detect,
                max_turn_p=max_turn_p,
                off_reasons=dict(sorted(off_reason_counts.items())),
            )
        )

    return outcomes, trace_rows


def summarize_outcomes(outcomes: Sequence[SessionOutcome]) -> Dict[str, Any]:
    attacks = [x for x in outcomes if x.label_session == "attack"]
    benign = [x for x in outcomes if x.label_session == "benign"]

    tp = sum(1 for x in attacks if x.detected_off)
    fn = sum(1 for x in attacks if not x.detected_off)
    fp = sum(1 for x in benign if x.detected_off)
    tn = sum(1 for x in benign if not x.detected_off)

    t_off_values = [int(x.first_off_turn) for x in attacks if x.first_off_turn is not None]
    late_detect_total = sum(1 for x in attacks if x.late_detect)
    detected_attack_total = sum(1 for x in attacks if x.detected_off)

    hist: Counter[str] = Counter()
    for x in attacks:
        if x.first_off_turn is None:
            hist["never"] += 1
        else:
            hist[str(int(x.first_off_turn))] += 1

    family_attack_total: Dict[str, int] = Counter()
    family_attack_missed: Dict[str, int] = Counter()
    family_attack_detected: Dict[str, int] = Counter()
    for x in attacks:
        family_attack_total[x.family] += 1
        if x.detected_off:
            family_attack_detected[x.family] += 1
        else:
            family_attack_missed[x.family] += 1

    never_detected_rate_by_family: Dict[str, Dict[str, Any]] = {}
    for fam in sorted(family_attack_total.keys()):
        total = int(family_attack_total[fam])
        missed = int(family_attack_missed.get(fam, 0))
        detected = int(family_attack_detected.get(fam, 0))
        never_detected_rate_by_family[fam] = {
            "attack_total": total,
            "detected": detected,
            "never_detected": missed,
            "never_detected_rate": _safe_div(missed, total),
            "attack_off_rate": _safe_div(detected, total),
        }

    return {
        "sessions_total": int(len(outcomes)),
        "attack_sessions": int(len(attacks)),
        "benign_sessions": int(len(benign)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "session_attack_off_rate": _safe_div(tp, tp + fn),
        "session_benign_off_rate": _safe_div(fp, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "time_to_off": {
            "count_detected": int(len(t_off_values)),
            "median": float(median(t_off_values)) if t_off_values else 0.0,
            "p95": _p95(t_off_values),
        },
        "late_detect_rate": _safe_div(late_detect_total, len(attacks)),
        "late_detect_rate_detected_only": _safe_div(late_detect_total, detected_attack_total),
        "first_off_turn_histogram": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "never", kv[0]))),
        "never_detected_rate_by_family": never_detected_rate_by_family,
    }


def summarize_effect_shadow_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    candidate_turns = 0
    policy_gate_passed_turns = 0
    provider_failure_turns = 0
    benign_turns = 0
    benign_candidate_turns = 0
    benign_policy_gate_passed_turns = 0
    status_counts: Counter[str] = Counter()
    policy_gate_status_counts: Counter[str] = Counter()
    candidate_by_family: Counter[str] = Counter()
    candidate_by_label: Counter[str] = Counter()
    policy_gate_passed_by_family: Counter[str] = Counter()
    policy_gate_passed_by_label: Counter[str] = Counter()
    confidence_histogram: Counter[str] = Counter()
    named_skill_invocation_turns = 0
    named_skill_invocation_by_label: Counter[str] = Counter()
    skipped_due_to_missing_effect_text = 0
    no_effect_but_named_skill_invocation = 0
    source_mismatch_simulated_blocks = 0
    skillbox_checked_count = 0
    skillbox_verified_count = 0
    skillbox_unknown_count = 0
    skillbox_source_mismatch_count = 0
    skillbox_hash_mismatch_count = 0
    skillbox_tampered_count = 0
    skillbox_dangerous_capability_unapproved_count = 0
    attack_turns = 0
    benign_source_mismatch_simulated_blocks = 0
    attack_skillbox_block_sessions = set()
    attack_sessions_with_source_mismatch = set()
    attack_sessions_total = set()
    benign_skillbox_block_sessions = set()
    benign_sessions_with_source_mismatch = set()
    benign_sessions_total = set()

    for row in rows:
        total += 1
        label = str(row.get("label_session", "unknown"))
        session_id = str(row.get("session_id", ""))
        if label == "attack":
            attack_turns += 1
            if session_id:
                attack_sessions_total.add(session_id)
        elif label == "benign" and session_id:
            benign_sessions_total.add(session_id)
        status = str(row.get("effect_forecast_status", "disabled"))
        status_counts[status] += 1
        named_skill = row.get("named_skill_invocation")
        if isinstance(named_skill, Mapping) and bool(named_skill.get("detected", False)):
            named_skill_invocation_turns += 1
            named_skill_invocation_by_label[label] += 1
            if status == "no_effect":
                no_effect_but_named_skill_invocation += 1
        effect_text = row.get("effect_text_analysis")
        if (
            status == "skipped"
            and isinstance(effect_text, Mapping)
            and bool(effect_text.get("missing_effect_text", False))
        ):
            skipped_due_to_missing_effect_text += 1
        gate = row.get("effect_policy_gate")
        gate_status = str(row.get("effect_policy_gate_status", "disabled"))
        if isinstance(gate, Mapping):
            gate_status = str(gate.get("status", gate_status))
        policy_gate_status_counts[gate_status] += 1
        if label == "benign":
            benign_turns += 1
        if status in {"provider_error", "provider_unavailable", "invalid_response"}:
            provider_failure_turns += 1
        candidate = row.get("effect_wall_candidate")
        if isinstance(candidate, Mapping):
            candidate_turns += 1
            candidate_by_family[str(row.get("family", "unknown"))] += 1
            candidate_by_label[label] += 1
            if label == "benign":
                benign_candidate_turns += 1
            conf = float(candidate.get("confidence", 0.0) or 0.0)
            if conf >= 0.90:
                confidence_histogram["0.90-1.00"] += 1
            elif conf >= 0.80:
                confidence_histogram["0.80-0.89"] += 1
            else:
                confidence_histogram["0.70-0.79"] += 1
        if isinstance(gate, Mapping) and bool(gate.get("would_enforce", False)):
            policy_gate_passed_turns += 1
            policy_gate_passed_by_family[str(row.get("family", "unknown"))] += 1
            policy_gate_passed_by_label[label] += 1
            if label == "benign":
                benign_policy_gate_passed_turns += 1
        provenance = row.get("skill_provenance_assessment")
        if isinstance(provenance, Mapping) and bool(provenance.get("simulated_block", False)):
            source_mismatch_simulated_blocks += 1
            if label == "benign":
                benign_source_mismatch_simulated_blocks += 1
                if session_id:
                    benign_sessions_with_source_mismatch.add(session_id)
            elif label == "attack" and session_id:
                attack_sessions_with_source_mismatch.add(session_id)
        skillbox_verification = row.get("skillbox_verification")
        if isinstance(skillbox_verification, Mapping):
            skillbox_checked_count += 1
            verification_status = str(skillbox_verification.get("verification_status", "unknown"))
            if verification_status == "verified":
                skillbox_verified_count += 1
            elif verification_status == "unknown":
                skillbox_unknown_count += 1
            elif verification_status == "source_mismatch":
                skillbox_source_mismatch_count += 1
            elif verification_status == "hash_mismatch":
                skillbox_hash_mismatch_count += 1
            elif verification_status == "tampered":
                skillbox_tampered_count += 1
            elif verification_status == "dangerous_capability_unapproved":
                skillbox_dangerous_capability_unapproved_count += 1
            if bool(skillbox_verification.get("simulated_block", False)) and session_id:
                if label == "attack":
                    attack_skillbox_block_sessions.add(session_id)
                elif label == "benign":
                    benign_skillbox_block_sessions.add(session_id)

    return {
        "turns_total": int(total),
        "candidate_turns": int(candidate_turns),
        "policy_gate_passed_turns": int(policy_gate_passed_turns),
        "provider_failure_turns": int(provider_failure_turns),
        "benign_turns": int(benign_turns),
        "benign_candidate_turns": int(benign_candidate_turns),
        "benign_policy_gate_passed_turns": int(benign_policy_gate_passed_turns),
        "candidate_turn_rate": _safe_div(candidate_turns, total),
        "benign_candidate_turn_rate": _safe_div(benign_candidate_turns, benign_turns),
        "policy_gate_passed_turn_rate": _safe_div(policy_gate_passed_turns, total),
        "benign_policy_gate_passed_turn_rate": _safe_div(
            benign_policy_gate_passed_turns, benign_turns
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "policy_gate_status_counts": dict(sorted(policy_gate_status_counts.items())),
        "candidate_by_family": dict(sorted(candidate_by_family.items())),
        "candidate_by_label": dict(sorted(candidate_by_label.items())),
        "policy_gate_passed_by_family": dict(sorted(policy_gate_passed_by_family.items())),
        "policy_gate_passed_by_label": dict(sorted(policy_gate_passed_by_label.items())),
        "named_skill_invocation_turns": int(named_skill_invocation_turns),
        "named_skill_invocation_by_label": dict(sorted(named_skill_invocation_by_label.items())),
        "skipped_due_to_missing_effect_text": int(skipped_due_to_missing_effect_text),
        "no_effect_but_named_skill_invocation": int(no_effect_but_named_skill_invocation),
        "source_mismatch_simulated_blocks": int(source_mismatch_simulated_blocks),
        "skillbox_checked_count": int(skillbox_checked_count),
        "skillbox_verified_count": int(skillbox_verified_count),
        "skillbox_unknown_count": int(skillbox_unknown_count),
        "skillbox_source_mismatch_count": int(skillbox_source_mismatch_count),
        "skillbox_hash_mismatch_count": int(skillbox_hash_mismatch_count),
        "skillbox_tampered_count": int(skillbox_tampered_count),
        "skillbox_dangerous_capability_unapproved_count": int(
            skillbox_dangerous_capability_unapproved_count
        ),
        "simulated_skillbox_block_sessions": int(len(attack_skillbox_block_sessions)),
        "simulated_skillbox_attack_recall": _safe_div(
            len(attack_skillbox_block_sessions), len(attack_sessions_total)
        ),
        "simulated_skillbox_benign_fp_rate": _safe_div(
            len(benign_skillbox_block_sessions), len(benign_sessions_total)
        ),
        "simulate_source_mismatch_attack_turn_rate": _safe_div(
            source_mismatch_simulated_blocks - benign_source_mismatch_simulated_blocks,
            attack_turns,
        ),
        "simulate_source_mismatch_session_recall": _safe_div(
            len(attack_sessions_with_source_mismatch), len(attack_sessions_total)
        ),
        "simulate_source_mismatch_fp_rate": _safe_div(
            benign_source_mismatch_simulated_blocks, benign_turns
        ),
        "simulate_source_mismatch_session_fp_rate": _safe_div(
            len(benign_sessions_with_source_mismatch), len(benign_sessions_total)
        ),
        "confidence_histogram": {
            "0.70-0.79": int(confidence_histogram.get("0.70-0.79", 0)),
            "0.80-0.89": int(confidence_histogram.get("0.80-0.89", 0)),
            "0.90-1.00": int(confidence_histogram.get("0.90-1.00", 0)),
        },
    }


def evaluate_pack_with_runner(
    *,
    rows: Sequence[SessionTurnRow],
    core_runner: SessionRunner,
    cross_runner: SessionRunner,
) -> Dict[str, Any]:
    sessions = group_sessions(rows)
    core_sessions = [s for s in sessions if s.bucket != "cross_session"]
    cross_sessions = [s for s in sessions if s.bucket == "cross_session"]

    core_outcomes, core_traces = evaluate_sessions(sessions=core_sessions, runner=core_runner)
    cross_outcomes, cross_traces = evaluate_sessions(sessions=cross_sessions, runner=cross_runner)
    core_text_intrinsic = [x for x in core_outcomes if x.eval_slice == "text_intrinsic"]
    core_context_required = [x for x in core_outcomes if x.eval_slice == "context_required"]

    misses_by_family: Dict[str, List[str]] = defaultdict(list)
    for out in core_outcomes:
        if out.label_session == "attack" and not out.detected_off:
            key = f"{out.family}::{out.eval_slice}"
            misses_by_family[key].append(out.session_id)
    for out in cross_outcomes:
        if out.label_session == "attack" and not out.detected_off:
            misses_by_family[f"{out.family}::cross_session"].append(out.session_id)

    return {
        "core": {
            "outcomes": core_outcomes,
            "summary_all": summarize_outcomes(core_outcomes),
            "summary_text_intrinsic": summarize_outcomes(core_text_intrinsic),
            "summary_context_required": summarize_outcomes(core_context_required),
        },
        "cross_session": {
            "outcomes": cross_outcomes,
            "summary": summarize_outcomes(cross_outcomes),
        },
        "trace_rows": core_traces + cross_traces,
        "misses_by_family": {k: sorted(v) for k, v in sorted(misses_by_family.items())},
        "effect_shadow": {
            "all": summarize_effect_shadow_rows(core_traces + cross_traces),
            "core": summarize_effect_shadow_rows(core_traces),
            "cross_session": summarize_effect_shadow_rows(cross_traces),
        },
        "provenance": {
            "all": summarize_provenance_rows(core_traces + cross_traces),
            "core": summarize_provenance_rows(core_traces),
            "cross_session": summarize_provenance_rows(cross_traces),
        },
    }


def summarize_provenance_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    mode_counts: Counter[str] = Counter()
    total_segments = 0
    trusted_user_segments = 0
    untrusted_segments = 0
    for row in rows:
        total += 1
        prov = row.get("provenance")
        if not isinstance(prov, Mapping):
            mode_counts["unknown"] += 1
            continue
        mode_counts[str(prov.get("mode", "unknown"))] += 1
        total_segments += int(prov.get("segment_count", 0) or 0)
        trusted_user_segments += int(prov.get("trusted_user_segments", 0) or 0)
        untrusted_segments += int(prov.get("untrusted_segments", 0) or 0)
    return {
        "turns_total": int(total),
        "mode_counts": dict(sorted(mode_counts.items())),
        "segment_count": int(total_segments),
        "trusted_user_segments": int(trusted_user_segments),
        "untrusted_segments": int(untrusted_segments),
        "avg_segments_per_turn": _safe_div(total_segments, total),
    }


def _latest_report_json(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted(
        [p / "report.json" for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    core_cur = (
        (current.get("summary_core_text_intrinsic", {}) or {})
        if isinstance(current.get("summary_core_text_intrinsic"), Mapping)
        else {}
    )
    core_base = (
        (baseline.get("summary_core_text_intrinsic", {}) or {})
        if isinstance(baseline.get("summary_core_text_intrinsic"), Mapping)
        else {}
    )
    # Backward compatibility with older report names.
    if not core_cur and isinstance(current.get("summary_core"), Mapping):
        core_cur = dict(current.get("summary_core", {}) or {})
    if not core_base and isinstance(baseline.get("summary_core"), Mapping):
        core_base = dict(baseline.get("summary_core", {}) or {})
    cross_cur = (current.get("cross_session", {}) or {}) if isinstance(current.get("cross_session"), Mapping) else {}
    cross_base = (baseline.get("cross_session", {}) or {}) if isinstance(baseline.get("cross_session"), Mapping) else {}

    keys = ("session_attack_off_rate", "session_benign_off_rate", "precision", "recall", "late_detect_rate")
    core_delta = {k: float(core_cur.get(k, 0.0)) - float(core_base.get(k, 0.0)) for k in keys}
    cross_delta = {k: float(cross_cur.get(k, 0.0)) - float(cross_base.get(k, 0.0)) for k in keys}

    cur_tto = (core_cur.get("time_to_off", {}) or {}) if isinstance(core_cur.get("time_to_off"), Mapping) else {}
    base_tto = (core_base.get("time_to_off", {}) or {}) if isinstance(core_base.get("time_to_off"), Mapping) else {}
    core_delta["time_to_off_median"] = float(cur_tto.get("median", 0.0)) - float(base_tto.get("median", 0.0))
    core_delta["time_to_off_p95"] = float(cur_tto.get("p95", 0.0)) - float(base_tto.get("p95", 0.0))

    return {"summary_core_text_intrinsic_delta": core_delta, "cross_session_delta": cross_delta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate session-based prompt-injection benchmark without per-turn reset.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--mode", choices=["pi0", "hybrid", "hybrid_api"], default="pi0")
    parser.add_argument("--pack", default="tests/data/session_benchmark/runtime/session_pack.jsonl")
    parser.add_argument("--labels-pack", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifacts-root", default="artifacts/session_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--strict-projector", action="store_true")
    parser.add_argument("--allow-api-fallback", action="store_true")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-provider", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-timeout-sec", type=float, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--api-cache-path", default=None)
    parser.add_argument("--api-error-log-path", default=None)
    parser.add_argument("--blind-eval", action="store_true")
    parser.add_argument("--enable-effects-shadow", action="store_true")
    parser.add_argument("--provenance-mode", choices=["blob", "segmented"], default="blob")
    parser.add_argument("--allow-legacy-runtime-leakage", action="store_true")
    args = parser.parse_args()

    rows = load_pack_rows(
        (ROOT / str(args.pack)).resolve(),
        labels_path=((ROOT / str(args.labels_pack)).resolve() if args.labels_pack else None),
        allow_legacy_runtime_leakage=bool(args.allow_legacy_runtime_leakage),
    )
    if not rows:
        payload = {
            "status": "dataset_not_ready",
            "pack": str((ROOT / str(args.pack)).resolve()),
            "reason": "no rows loaded",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"session_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    core_runner = OmegaHarnessRunner(
        profile=str(args.profile),
        mode=str(args.mode),
        seed=int(args.seed),
        state_db_path=(out_dir / "cross_session_core.db"),
        strict_projector=bool(args.strict_projector),
        require_api_adapter=(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
        api_model=(str(args.api_model) if args.api_model else None),
        api_provider=(str(args.api_provider) if args.api_provider else None),
        api_key_env=(str(args.api_key_env) if args.api_key_env else None),
        api_base_url=(str(args.api_base_url) if args.api_base_url else None),
        api_timeout_sec=(float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
        api_retries=(int(args.api_retries) if args.api_retries is not None else None),
        api_cache_path=(str(args.api_cache_path) if args.api_cache_path else None),
        api_error_log_path=(str(args.api_error_log_path) if args.api_error_log_path else None),
        blind_eval=bool(args.blind_eval),
        enable_effects_shadow=bool(args.enable_effects_shadow),
        provenance_mode=str(args.provenance_mode),
    )
    cross_runner = OmegaHarnessRunner(
        profile=str(args.profile),
        mode=str(args.mode),
        seed=int(args.seed),
        state_db_path=(out_dir / "cross_session_slice.db"),
        strict_projector=bool(args.strict_projector),
        require_api_adapter=(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
        api_model=(str(args.api_model) if args.api_model else None),
        api_provider=(str(args.api_provider) if args.api_provider else None),
        api_key_env=(str(args.api_key_env) if args.api_key_env else None),
        api_base_url=(str(args.api_base_url) if args.api_base_url else None),
        api_timeout_sec=(float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
        api_retries=(int(args.api_retries) if args.api_retries is not None else None),
        api_cache_path=(str(args.api_cache_path) if args.api_cache_path else None),
        api_error_log_path=(str(args.api_error_log_path) if args.api_error_log_path else None),
        blind_eval=bool(args.blind_eval),
        enable_effects_shadow=bool(args.enable_effects_shadow),
        provenance_mode=str(args.provenance_mode),
    )
    result = evaluate_pack_with_runner(rows=rows, core_runner=core_runner, cross_runner=cross_runner)

    rows_jsonl = out_dir / "rows.jsonl"
    with rows_jsonl.open("w", encoding="utf-8") as fh:
        for row in result["trace_rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    misses_path = out_dir / "misses_by_family.json"
    misses_path.write_text(json.dumps(result["misses_by_family"], ensure_ascii=False, indent=2), encoding="utf-8")

    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "profile": str(args.profile),
        "mode": str(args.mode),
        "blind_eval": bool(args.blind_eval),
        "enable_effects_shadow": bool(args.enable_effects_shadow),
        "provenance_mode": str(args.provenance_mode),
        "seed": int(args.seed),
        "pack": str((ROOT / str(args.pack)).resolve()),
        "summary_core_text_intrinsic": result["core"]["summary_text_intrinsic"],
        "summary_context_required": result["core"]["summary_context_required"],
        "summary_all": result["core"]["summary_all"],
        # Backward compatibility key.
        "summary_core": result["core"]["summary_text_intrinsic"],
        "cross_session": result["cross_session"]["summary"],
        "effect_shadow": result["effect_shadow"],
        "provenance": result["provenance"],
        "projector": {
            "strict_projector": bool(args.strict_projector),
            "require_api_adapter": bool(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
            "api_model": str(args.api_model) if args.api_model else None,
            "api_provider": str(args.api_provider) if args.api_provider else None,
            "api_key_env": str(args.api_key_env) if args.api_key_env else None,
            "api_base_url": str(args.api_base_url) if args.api_base_url else None,
            "core_runtime": core_runner.projector_status(),
            "cross_runtime": cross_runner.projector_status(),
        },
        "artifacts": {
            "report_json": str((out_dir / "report.json").resolve()),
            "rows_jsonl": str(rows_jsonl.resolve()),
            "misses_by_family_json": str(misses_path.resolve()),
        },
    }

    baseline_compare = None
    baseline_path = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if isinstance(baseline, dict):
            baseline_compare = {"path": str(baseline_path), **_baseline_compare(report, baseline)}
    report["baseline_compare"] = baseline_compare

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
