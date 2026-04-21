from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from omega.adapters import (
    AdapterDecision,
    AdapterSessionContext,
    MemoryWriteDecision,
    OmegaAdapterRuntime,
)
from omega.integrations import (
    OmegaAutoGenGuard,
    OmegaCrewAIGuard,
    OmegaHaystackGuard,
    OmegaLangChainGuard,
    OmegaLangGraphGuard,
    OmegaLlamaIndexGuard,
)
from omega.interfaces.contracts_v1 import OffAction


def _decision(
    *,
    off: bool = False,
    control_outcome: str = "ALLOW",
    actions: List[OffAction] | None = None,
    reason_codes: List[str] | None = None,
    risk_score: float | None = None,
) -> AdapterDecision:
    return AdapterDecision(
        session_id="sess-test",
        step=1,
        off=off,
        control_outcome=control_outcome,
        actions=list(actions or []),
        reason_codes=list(reason_codes or []),
        trace_id="trace-test",
        decision_id="decision-test",
        risk_score=risk_score,
    )


def test_runtime_memory_hygiene_allow_quarantine_deny(monkeypatch: Any) -> None:
    runtime = OmegaAdapterRuntime(profile="dev", projector_mode="pi0")
    ctx = AdapterSessionContext(session_id="sess-mem", actor_id="actor-mem")

    monkeypatch.setattr(runtime, "check_model_input", lambda messages_text, ctx: _decision(risk_score=0.15))
    allow = runtime.check_memory_write(
        memory_text="benign preference note",
        source_id="doc-trusted-1",
        source_type="policy_doc",
        source_trust="trusted",
        ctx=ctx,
    )
    assert allow.allowed is True
    assert allow.mode == "allow"
    assert allow.source_trust == "trusted"

    monkeypatch.setattr(
        runtime,
        "check_model_input",
        lambda messages_text, ctx: _decision(reason_codes=["reason_spike"], risk_score=0.76),
    )
    quarantined = runtime.check_memory_write(
        memory_text="suspicious external instruction",
        source_id="doc-web-2",
        source_type="web",
        source_trust="untrusted",
        ctx=ctx,
    )
    assert quarantined.allowed is False
    assert quarantined.mode == "quarantine"
    assert quarantined.reason == "UNTRUSTED_OR_QUARANTINED_SOURCE"

    monkeypatch.setattr(
        runtime,
        "check_model_input",
        lambda messages_text, ctx: _decision(
            control_outcome="TOOL_FREEZE",
            actions=[OffAction(type="TOOL_FREEZE", target="tools")],
            reason_codes=["reason_multi"],
            risk_score=0.91,
        ),
    )
    denied = runtime.check_memory_write(
        memory_text="override and persist tool abuse chain",
        source_id="doc-mail-3",
        source_type="email",
        source_trust="mixed",
        ctx=ctx,
    )
    assert denied.allowed is False
    assert denied.mode == "deny"
    assert denied.reason == "BLOCKING_POLICY_SIGNAL"


@dataclass
class _FakeMemoryRuntime:
    calls: List[Tuple[str, str, str, AdapterSessionContext, Dict[str, Any]]]

    def __init__(self) -> None:
        self.calls = []

    def check_memory_write(
        self,
        *,
        memory_text: str,
        source_id: str,
        source_type: str = "other",
        source_trust: str = "untrusted",
        ctx: AdapterSessionContext,
        source_tags: Dict[str, Any] | None = None,
    ) -> MemoryWriteDecision:
        self.calls.append((memory_text, source_id, source_trust, ctx, dict(source_tags or {})))
        decision_ref = _decision()
        trust_norm = str(source_trust).strip().lower()
        if trust_norm != "trusted":
            return MemoryWriteDecision(
                allowed=False,
                mode="quarantine",
                reason="UNTRUSTED_OR_QUARANTINED_SOURCE",
                source_id=source_id,
                source_type=source_type,
                source_trust=trust_norm,
                tags={"source_id": source_id, "source_type": source_type, "source_trust": trust_norm},
                decision_ref=decision_ref,
            )
        return MemoryWriteDecision(
            allowed=True,
            mode="allow",
            reason="ALLOW_WRITE",
            source_id=source_id,
            source_type=source_type,
            source_trust=trust_norm,
            tags={"source_id": source_id, "source_type": source_type, "source_trust": trust_norm},
            decision_ref=decision_ref,
        )


def test_memory_hygiene_contract_all_adapters() -> None:
    runtime = _FakeMemoryRuntime()

    lc = OmegaLangChainGuard(runtime=runtime)
    out_lc = lc.check_memory_write(
        memory_text="store safe user preference",
        source_id="src-lc",
        source_trust="trusted",
        state={"thread_id": "sess-lc", "user_id": "actor-lc"},
    )
    assert out_lc.allowed is True

    lg = OmegaLangGraphGuard(runtime=runtime)
    out_lg = lg.check_memory_write(
        memory_text="store risky web snippet",
        source_id="src-lg",
        source_type="web",
        source_trust="untrusted",
        thread_id="sess-lg",
        user_id="actor-lg",
    )
    assert out_lg.mode == "quarantine"

    li = OmegaLlamaIndexGuard(runtime=runtime)
    out_li = li.check_memory_write(
        memory_text="persist indexed citation",
        source_id="src-li",
        source_type="knowledge_base",
        source_trust="trusted",
        thread_id="sess-li",
        user_id="actor-li",
    )
    assert out_li.allowed is True

    hs = OmegaHaystackGuard(runtime=runtime)
    out_hs = hs.check_memory_write(
        memory_text="persist untrusted extracted chunk",
        source_id="src-hs",
        source_type="web",
        source_trust="untrusted",
        thread_id="sess-hs",
        user_id="actor-hs",
    )
    assert out_hs.mode == "quarantine"

    ag = OmegaAutoGenGuard(runtime=runtime)
    out_ag = ag.check_memory_write(
        memory_text="persist agent summary",
        source_id="src-ag",
        source_type="conversation",
        source_trust="trusted",
        thread_id="sess-ag",
        user_id="actor-ag",
    )
    assert out_ag.allowed is True

    cr = OmegaCrewAIGuard(runtime=runtime)
    out_cr = cr.check_memory_write(
        memory_text="persist external instruction",
        source_id="src-cr",
        source_type="email",
        source_trust="untrusted",
        thread_id="sess-cr",
        user_id="actor-cr",
    )
    assert out_cr.mode == "quarantine"

    assert len(runtime.calls) == 6
    seen = {(call[1], call[3].session_id, call[3].actor_id) for call in runtime.calls}
    assert ("src-lc", "sess-lc", "actor-lc") in seen
    assert ("src-lg", "sess-lg", "actor-lg") in seen
    assert ("src-li", "sess-li", "actor-li") in seen
    assert ("src-hs", "sess-hs", "actor-hs") in seen
    assert ("src-ag", "sess-ag", "actor-ag") in seen
    assert ("src-cr", "sess-cr", "actor-cr") in seen
