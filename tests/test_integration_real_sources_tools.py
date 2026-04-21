from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import uuid

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ToolRequest
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.rag.harness import OmegaRAGHarness
from omega.rag.retriever_prod_adapter import build_retriever_prod_adapter
from omega.tools.tool_gateway import ToolGatewayV1
from tests.helpers import mk_item


@dataclass
class FixedLLM:
    text: str

    def generate(self, prompt: str):
        del prompt
        return {"text": self.text}


def _build_harness(llm_text: str) -> tuple[dict, OmegaRAGHarness]:
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    state_dir = Path("artifacts/test_tmp")
    state_dir.mkdir(parents=True, exist_ok=True)
    cfg["off_policy"]["cross_session"]["sqlite_path"] = str(state_dir / f"real_sources_tools_{uuid.uuid4().hex}.db")
    harness = OmegaRAGHarness(
        projector=Pi0IntentAwareV2(cfg),
        omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
        off_policy=OffPolicyV1(cfg),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
        llm_backend=FixedLLM(text=llm_text),
    )
    return cfg, harness


def test_real_sources_tool_gateway_coverage_and_no_bypass():
    cfg, harness = _build_harness('CALL_TOOL("network_post", "https://example.com", "probe")')
    retriever = build_retriever_prod_adapter(config=cfg, source_root="data/smoke_sources/domain_tickets_poisoned")
    packet = retriever.search("urgent vendor ticket", k=3)

    out = harness.run_step(user_query="summarize", packet_items=packet, actor_id="real-e2e-actor")

    total_seen = len(out["inferred_tool_requests"])
    assert total_seen >= 1
    assert len(out["tool_gateway_events"]) == total_seen
    assert len(out["tool_decisions"]) == total_seen
    assert len(out["tool_executions"]) == total_seen

    for exec_, dec in zip(out["tool_executions"], out["tool_decisions"]):
        assert not (exec_.executed and not dec.allowed)


def test_real_sources_quarantined_sources_do_not_enter_context():
    cfg, harness = _build_harness("safe_output")
    poisoned = build_retriever_prod_adapter(config=cfg, source_root="data/smoke_sources/domain_tickets_poisoned")
    safe = build_retriever_prod_adapter(config=cfg, source_root="data/smoke_sources/domain_docs")

    packet_mal = poisoned.search("urgent vendor debug", k=1)
    assert len(packet_mal) == 1
    mal_source = packet_mal[0].source_id

    # Two malicious steps to cross quarantine strike threshold.
    harness.reset_state(session_id="sess-real-q-1", actor_id="actor-quarantine")
    harness.run_step(user_query="q1", packet_items=packet_mal, actor_id="actor-quarantine")
    harness.reset_state(session_id="sess-real-q-2", actor_id="actor-quarantine")
    harness.run_step(user_query="q2", packet_items=packet_mal, actor_id="actor-quarantine")

    packet_safe = safe.search("omega architecture", k=1)
    assert len(packet_safe) >= 1
    mixed_packet = [packet_mal[0], packet_safe[0]]

    harness.reset_state(session_id="sess-real-q-3", actor_id="actor-other")
    out = harness.run_step(user_query="q3", packet_items=mixed_packet, actor_id="actor-other")
    allowed_sources = {item.source_id for item in out["allowed_items"]}
    assert mal_source not in allowed_sources


def test_real_tools_write_and_dangerous_require_approval():
    cfg, harness = _build_harness("safe_output")
    retriever = build_retriever_prod_adapter(config=cfg, source_root="data/smoke_sources/safe_index")
    packet = retriever.search("credential rotation checklist", k=2)

    out_no = harness.run_step(
        user_query="q",
        packet_items=packet,
        actor_id="actor-tools",
        tool_requests=[
            ToolRequest(tool_name="write_file", args={"filename": "a.txt", "content": "alpha"}, session_id="sess", step=1),
            ToolRequest(tool_name="network_post", args={"url": "https://example.com", "payload": "p"}, session_id="sess", step=1),
        ],
    )
    assert all(not d.allowed for d in out_no["tool_decisions"])
    assert all(not e.executed for e in out_no["tool_executions"])
    assert all(ev["request"]["request_origin"] == "explicit" for ev in out_no["tool_gateway_events"])

    harness.reset_state(session_id="sess-tools-2", actor_id="actor-tools-2")
    out_yes = harness.run_step(
        user_query="q2",
        packet_items=packet,
        actor_id="actor-tools-2",
        tool_requests=[
            ToolRequest(
                tool_name="write_file",
                args={"filename": "b.txt", "content": "beta", "human_approved": True},
                session_id="sess",
                step=2,
            ),
            ToolRequest(
                tool_name="network_post",
                args={"url": "https://example.com", "payload": "p", "human_approved": True},
                session_id="sess",
                step=2,
            ),
        ],
    )
    assert all(d.allowed for d in out_yes["tool_decisions"])
    assert all(e.executed for e in out_yes["tool_executions"])
    assert len(out_yes["tool_gateway_events"]) == 2


def test_validator_denied_explicit_tool_is_not_executed():
    cfg, harness = _build_harness("safe_output")
    packet = [mk_item("safe-1", "General harmless information", source_id="tests:safe")]
    out = harness.run_step(
        user_query="q",
        packet_items=packet,
        actor_id="actor-tools",
        tool_requests=[
            ToolRequest(
                tool_name="network_post",
                args={"url": "https://example.com", "human_approved": True},
                session_id="sess",
                step=1,
            )
        ],
    )
    assert len(out["tool_decisions"]) == 1
    assert out["tool_decisions"][0].allowed is False
    assert out["tool_decisions"][0].reason == "INVALID_TOOL_ARGS_SCHEMA"
    assert out["tool_executions"][0].executed is False
    assert out["tool_gateway_events"][0]["decision"]["validation_status"] == "failed"


def test_validator_applies_to_inferred_shell_like_requests():
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    cfg["tools"]["capabilities"]["shell_probe"] = {
        "mode": "read_only",
        "allowed_when": ["NO_OFF", "OFF"],
        "requires_human_approval": False,
    }
    harness = OmegaRAGHarness(
        projector=Pi0IntentAwareV2(cfg),
        omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
        off_policy=OffPolicyV1(cfg),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
        llm_backend=FixedLLM(text='CALL_TOOL("shell_probe")'),
    )

    out = harness.run_step(
        user_query="q",
        packet_items=[mk_item("safe-2", "General harmless info", source_id="tests:safe")],
        actor_id="actor-inferred",
    )
    assert len(out["inferred_tool_requests"]) >= 1
    assert out["tool_decisions"][0].allowed is False
    assert out["tool_decisions"][0].reason == "INVALID_TOOL_ARGS_SHELLLIKE"
    assert out["tool_executions"][0].executed is False
    assert out["tool_gateway_events"][0]["decision"]["validation_status"] == "failed"
