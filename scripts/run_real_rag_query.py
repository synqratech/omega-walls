from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import config_refs_from_snapshot, load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.rag.harness import OmegaRAGHarness
from omega.rag.llm_backends import LocalTransformersLLM
from omega.rag.retriever_adapters import SQLiteFTSRetrieverAdapter
from omega.tools.tool_gateway import ToolGatewayV1



def main() -> int:
    parser = argparse.ArgumentParser(description="Run real RAG+Omega query against filesystem sources")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    retriever = SQLiteFTSRetrieverAdapter.from_directory(args.source_dir, config=cfg)
    packet = retriever.search(args.query, k=args.top_k)
    if not packet:
        raise SystemExit(f"No readable/retrievable text files found in {args.source_dir}")

    harness = OmegaRAGHarness(
        projector=Pi0IntentAwareV2(cfg),
        omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
        off_policy=OffPolicyV1(cfg),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
        llm_backend=LocalTransformersLLM(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
    )

    out = harness.run_step(
        user_query=args.query,
        packet_items=packet,
        config_refs=config_refs_from_snapshot(snapshot, code_commit="local"),
    )

    report = {
        "query": args.query,
        "source_dir": args.source_dir,
        "retrieved": [
            {"doc_id": d.doc_id, "source_id": d.source_id, "source_type": d.source_type}
            for d in packet
        ],
        "off": out["step_result"].off,
        "reasons": out["step_result"].reasons.__dict__,
        "top_docs": out["step_result"].top_docs,
        "actions": [a.__dict__ for a in out["decision"].actions],
        "enforcement_event": out["enforcement_event"],
        "llm_response": out["llm_response"].get("text", ""),
        "inferred_tool_requests": [asdict(req) for req in out["inferred_tool_requests"]],
        "tool_decisions": [asdict(dec) for dec in out["tool_decisions"]],
        "tool_executions": [asdict(exec_) for exec_ in out["tool_executions"]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
