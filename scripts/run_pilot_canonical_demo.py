from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import config_refs_from_snapshot, load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ToolRequest
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.rag.harness import MockLLM, OmegaRAGHarness
from omega.rag.retriever_prod_adapter import build_retriever_prod_adapter
from omega.tools.tool_gateway import ToolGatewayV1


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _packet_with_marker(retriever: object, query: str, k: int, marker: str) -> list[object]:
    packet = list(retriever.search(query, k=k))
    marker_l = str(marker).strip().lower()
    if any(marker_l in f"{item.doc_id} {item.source_id} {item.text}".lower() for item in packet):
        return packet
    provider = getattr(retriever, "provider", None)
    corpus = list(getattr(getattr(provider, "retriever", None), "corpus", []))
    for item in corpus:
        hay = f"{item.doc_id} {item.source_id} {item.text}".lower()
        if marker_l in hay:
            packet = [item] + packet
            break
    seen = set()
    uniq = []
    for item in packet:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        uniq.append(item)
    return uniq[: max(1, k)]


def _emit_replay_input(profile: str, mode: str) -> bool:
    choice = str(mode).strip().lower()
    if choice == "on":
        return True
    if choice == "off":
        return False
    profile_l = str(profile).strip().lower()
    return ("pilot" in profile_l) or ("synthetic" in profile_l)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT).as_posix())
    except ValueError:
        return str(path.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic pilot canonical demo flow")
    parser.add_argument("--profile", default="pilot_canonical")
    parser.add_argument("--source-root", default="data/local_contour")
    parser.add_argument(
        "--query",
        default="DIRECT_INJECTION_MARKER Summarize this document",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-root", default="artifacts/pilot_demo")
    parser.add_argument("--emit-replay-input", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--replay-input-output", default=None)
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    harness = OmegaRAGHarness(
        projector=build_projector(cfg),
        omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
        off_policy=OffPolicyV1(cfg),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
        llm_backend=MockLLM(),
    )
    run_nonce = uuid.uuid4().hex[:8]
    session_id = f"sess-pilot-demo-{run_nonce}"
    actor_id = f"actor-pilot-demo-{run_nonce}"
    harness.reset_state(session_id=session_id, actor_id=actor_id)

    retriever = build_retriever_prod_adapter(config=cfg, source_root=args.source_root)
    packet = _packet_with_marker(retriever, args.query, args.top_k, marker="direct_injection_marker")
    if not packet:
        raise SystemExit(f"No retriever output from {args.source_root}; cannot run canonical demo.")

    tool_requests = [
        ToolRequest(
            tool_name="network_post",
            args={"payload": "demo_probe", "human_approved": False},
            session_id=session_id,
            step=1,
        )
    ]
    out = harness.run_step(
        user_query=args.query,
        packet_items=packet,
        tool_requests=tool_requests,
        actor_id=actor_id,
        config_refs=config_refs_from_snapshot(snapshot, code_commit="local"),
    )

    retrieved_doc_ids = [item.doc_id for item in packet]
    allowed_doc_ids = [item.doc_id for item in out["allowed_items"]]
    blocked_doc_ids = sorted(set(retrieved_doc_ids) - set(allowed_doc_ids))
    item_by_id = {item.doc_id: item for item in packet}

    context_text = str(out["context"])
    blocked_text_absent = True
    for doc_id in blocked_doc_ids:
        blocked_item = item_by_id.get(doc_id)
        if blocked_item is None:
            continue
        probe = str(blocked_item.text).strip()
        if probe and probe in context_text:
            blocked_text_absent = False
            break
    context_filtered_before_prompt = bool(blocked_doc_ids) and blocked_text_absent

    tool_stop_before_exec = False
    if out["tool_decisions"] and out["tool_executions"]:
        tool_stop_before_exec = any(
            (not decision.allowed) and (not execution.executed)
            for decision, execution in zip(out["tool_decisions"], out["tool_executions"])
        )
    incident_artifact = out.get("incident_artifact")
    artifact_context_filtered = bool(
        ((incident_artifact or {}).get("timeline", {}) or {}).get("context_admission", {}).get("filtered_before_prompt", False)
    )
    artifact_tool_prevented = bool(
        ((incident_artifact or {}).get("prevention", {}) or {}).get("tool_execution_prevented", False)
    )

    acceptance = {
        "risky_context_removed_before_prompt": bool(context_filtered_before_prompt and artifact_context_filtered),
        "tool_call_stopped_before_execution": bool(tool_stop_before_exec and artifact_tool_prevented),
        "single_evidence_artifact_present": bool(isinstance(incident_artifact, dict)),
    }
    acceptance["pass"] = bool(
        acceptance["risky_context_removed_before_prompt"]
        and acceptance["tool_call_stopped_before_execution"]
        and acceptance["single_evidence_artifact_present"]
    )

    report = {
        "event": "pilot_canonical_demo_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": args.profile,
        "canonical_path": "RAG service -> RetrieverProdAdapter -> OmegaRAGHarness -> ToolGateway",
        "input": {
            "source_root": args.source_root,
            "query": args.query,
            "top_k": args.top_k,
        },
        "retrieval": {
            "retrieved_doc_ids": retrieved_doc_ids,
            "allowed_doc_ids": allowed_doc_ids,
            "blocked_doc_ids": blocked_doc_ids,
        },
        "control": {
            "control_outcome": str(out["control_outcome"]),
            "actions": [a.__dict__ for a in out["decision"].actions],
            "tool_decisions": [asdict(x) for x in out["tool_decisions"]],
            "tool_executions": [asdict(x) for x in out["tool_executions"]],
        },
        "acceptance": acceptance,
        "evidence_artifact": incident_artifact,
    }

    run_id = f"pilot_demo_{_utc_stamp()}_{snapshot.resolved_sha256[:12]}"
    out_dir = ROOT / args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    replay_input_path: Path | None = None
    if _emit_replay_input(args.profile, args.emit_replay_input):
        replay_input = {
            "event": "omega_replay_input_v1",
            "schema_version": "1.0",
            "replay_id": f"pilot_replay_{run_id}",
            "canonical_path": "RAG service -> RetrieverProdAdapter -> OmegaRAGHarness -> ToolGateway",
            "execution": {"state_bootstrap": "fresh_state"},
            "turns": [
                {
                    "session_id": session_id,
                    "actor_id": actor_id,
                    "turn_index": 1,
                    "user_query": args.query,
                    "packet_items": [
                        {
                            "doc_id": str(item.doc_id),
                            "source_id": str(item.source_id),
                            "source_type": str(item.source_type),
                            "trust": str(item.trust),
                            "text": str(item.text),
                            "language": getattr(item, "language", None),
                            "meta": dict(getattr(item, "meta", {}) or {}),
                        }
                        for item in packet
                    ],
                    "tool_requests": [asdict(req) for req in tool_requests],
                }
            ],
            "refs": {
                "profile": args.profile,
                "resolved_config_sha256": str(snapshot.resolved_sha256),
                "policy_version": str((cfg.get("off_policy", {}) or {}).get("policy_version", "")),
            },
        }
        replay_input_path = Path(args.replay_input_output) if args.replay_input_output else out_dir / "replay_input.json"
        replay_input_path.parent.mkdir(parents=True, exist_ok=True)
        replay_input_path.write_text(json.dumps(replay_input, ensure_ascii=True, indent=2), encoding="utf-8")

    if replay_input_path is not None:
        report["replay_input"] = _safe_rel(replay_input_path)
        report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    latest_path = ROOT / args.output_root / "LATEST.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "report": _safe_rel(report_path),
                "acceptance_pass": bool(acceptance["pass"]),
                "replay_input": (_safe_rel(replay_input_path) if replay_input_path is not None else ""),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"run_id": run_id, "report": _safe_rel(report_path), "replay_input": (_safe_rel(replay_input_path) if replay_input_path is not None else ""), "acceptance": acceptance}, ensure_ascii=True, indent=2))
    return 0 if acceptance["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
