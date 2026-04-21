from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

ROOT = Path(__file__).resolve().parent.parent


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_stable(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": action.get("type"),
        "target": action.get("target"),
        "doc_ids": sorted(action.get("doc_ids") or []),
        "source_ids": sorted(action.get("source_ids") or []),
        "tool_mode": action.get("tool_mode"),
        "allowlist": sorted(action.get("allowlist") or []),
        "horizon_steps": action.get("horizon_steps"),
    }


def _normalize_tool_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "allowed": bool(decision.get("allowed", False)),
        "mode": decision.get("mode"),
        "reason": decision.get("reason"),
        "logged": bool(decision.get("logged", False)),
    }


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# ------------------------ legacy snapshot/artifact replay ---------------------

def extract_incident_snapshot(
    payload: Dict[str, Any],
    report_kind: str,
    scenario: str,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    if report_kind == "local_contour":
        scenarios = payload.get("scenarios", [])
        match = next((s for s in scenarios if isinstance(s, dict) and s.get("scenario") == scenario), None)
        if match is None:
            raise ValueError(f"Scenario not found in local_contour report: {scenario}")
        steps = [s for s in match.get("steps", []) if isinstance(s, dict)]
        if not steps:
            raise ValueError(f"Scenario has no steps: {scenario}")
        chosen: Dict[str, Any]
        if step is not None:
            chosen = next((s for s in steps if int(s.get("step", -1)) == int(step)), None)  # type: ignore[arg-type]
            if chosen is None:
                raise ValueError(f"Step {step} not found in scenario: {scenario}")
        else:
            chosen = next((s for s in steps if bool(s.get("off", False))), steps[-1])
        return {
            "report_kind": "local_contour",
            "scenario": scenario,
            "step": int(chosen.get("step", 0)),
            "off": bool(chosen.get("off", False)),
            "reasons": {k: bool(v) for k, v in dict(chosen.get("reasons", {})).items()},
            "actions": [_normalize_action(a) for a in (chosen.get("actions", []) or []) if isinstance(a, dict)],
            "top_docs": list(chosen.get("top_docs", []) or []),
            "tool_decisions": [
                _normalize_tool_decision(d) for d in (chosen.get("tool_decisions", []) or []) if isinstance(d, dict)
            ],
        }

    if report_kind == "smoke_real_rag":
        reports = payload.get("reports", [])
        match = next((s for s in reports if isinstance(s, dict) and s.get("scenario") == scenario), None)
        if match is None:
            raise ValueError(f"Scenario not found in smoke report: {scenario}")
        return {
            "report_kind": "smoke_real_rag",
            "scenario": scenario,
            "step": 1,
            "off": bool(match.get("off", False)),
            "reasons": {k: bool(v) for k, v in dict(match.get("reasons", {})).items()},
            "actions": [_normalize_action(a) for a in (match.get("actions", []) or []) if isinstance(a, dict)],
            "top_docs": list(match.get("top_docs", []) or []),
            "tool_decisions": [
                _normalize_tool_decision(d) for d in (match.get("tool_decisions", []) or []) if isinstance(d, dict)
            ],
        }

    raise ValueError(f"Unsupported report_kind: {report_kind}")


def _infer_report_kind(payload: Dict[str, Any]) -> str:
    event_name = str(payload.get("event", "")).strip()
    if event_name == "omega_incident_artifact_v1":
        return "incident_artifact"
    if isinstance(payload.get("incident_artifact"), dict):
        return "incident_artifact"
    if isinstance(payload.get("evidence_artifact"), dict):
        evidence = payload.get("evidence_artifact", {})
        if isinstance(evidence, dict) and str(evidence.get("event", "")).strip() == "omega_incident_artifact_v1":
            return "incident_artifact"
    if isinstance(payload.get("scenarios"), list):
        return "local_contour"
    if isinstance(payload.get("reports"), list):
        return "smoke_real_rag"
    raise ValueError("Unable to infer report type; use --report-kind explicitly")


def compare_snapshots(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    components = {
        "off": actual.get("off") == expected.get("off"),
        "reasons": actual.get("reasons") == expected.get("reasons"),
        "actions": actual.get("actions") == expected.get("actions"),
        "top_docs": actual.get("top_docs") == expected.get("top_docs"),
        "tool_decisions": actual.get("tool_decisions") == expected.get("tool_decisions"),
    }
    return {
        "match": all(components.values()),
        "components": components,
    }


def extract_incident_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    if str(payload.get("event", "")).strip() == "omega_incident_artifact_v1":
        return payload
    artifact = payload.get("incident_artifact")
    if isinstance(artifact, dict) and str(artifact.get("event", "")).strip() == "omega_incident_artifact_v1":
        return artifact
    evidence = payload.get("evidence_artifact")
    if isinstance(evidence, dict) and str(evidence.get("event", "")).strip() == "omega_incident_artifact_v1":
        return evidence
    raise ValueError("Incident artifact payload must contain omega_incident_artifact_v1")


def normalize_incident_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    context = payload.get("context", {}) if isinstance(payload.get("context", {}), dict) else {}
    decision = payload.get("decision", {}) if isinstance(payload.get("decision", {}), dict) else {}
    reasons = payload.get("reasons", {}) if isinstance(payload.get("reasons", {}), dict) else {}
    sources = payload.get("sources", {}) if isinstance(payload.get("sources", {}), dict) else {}
    prevention = payload.get("prevention", {}) if isinstance(payload.get("prevention", {}), dict) else {}
    incident_30s = payload.get("incident_30s", {}) if isinstance(payload.get("incident_30s", {}), dict) else {}
    top_docs_raw = sources.get("top_docs", []) if isinstance(sources.get("top_docs", []), list) else []
    top_docs = []
    for item in top_docs_raw:
        if not isinstance(item, dict):
            continue
        top_docs.append(
            {
                "doc_id": str(item.get("doc_id", "")),
                "source_id": str(item.get("source_id", "")),
                "source_type": str(item.get("source_type", "")),
                "trust": str(item.get("trust", "")),
                "text_sha256": str(item.get("text_sha256", "")),
            }
        )
    top_docs = sorted(top_docs, key=lambda row: (row["doc_id"], row["source_id"]))
    actions = [
        _normalize_action(item)
        for item in (decision.get("actions", []) if isinstance(decision.get("actions", []), list) else [])
        if isinstance(item, dict)
    ]
    actions = sorted(actions, key=lambda row: (str(row.get("type", "")), str(row.get("target", ""))))
    prevented_tools_raw = prevention.get("prevented_tools", []) if isinstance(prevention.get("prevented_tools", []), list) else []
    prevented_tools = []
    for item in prevented_tools_raw:
        if not isinstance(item, dict):
            continue
        prevented_tools.append(
            {
                "tool_name": str(item.get("tool_name", "")),
                "reason": str(item.get("reason", "")),
                "request_origin": str(item.get("request_origin", "")),
            }
        )
    prevented_tools = sorted(
        prevented_tools,
        key=lambda row: (row["tool_name"], row["reason"], row["request_origin"]),
    )
    return {
        "event": "omega_incident_artifact_v1",
        "schema_version": str(payload.get("schema_version", "")),
        "incident_artifact_id": str(payload.get("incident_artifact_id", "")),
        "trace_id": _str_or_empty(payload.get("trace_id", "")),
        "decision_id": _str_or_empty(payload.get("decision_id", "")),
        "surface": str(payload.get("surface", "")),
        "context": {
            "session_id": _str_or_empty(context.get("session_id", "")),
            "step": int(context.get("step", 0)),
            "request_id": _str_or_empty(context.get("request_id", "")),
            "policy_version": _str_or_empty(context.get("policy_version", "")),
            "profile": _str_or_empty(context.get("profile", "")),
            "runtime_mode": _str_or_empty(context.get("runtime_mode", "")),
        },
        "decision": {
            "control_outcome": _str_or_empty(decision.get("control_outcome", "")),
            "off": bool(decision.get("off", False)),
            "severity": _str_or_empty(decision.get("severity", "")),
            "verdict": _str_or_empty(decision.get("verdict", "")),
            "action_types": sorted(str(x) for x in list(decision.get("action_types", []) or [])),
            "actions": actions,
        },
        "reasons": {
            "reason_flags": sorted(str(x) for x in list(reasons.get("reason_flags", []) or [])),
            "contributing_signals": dict(reasons.get("contributing_signals", {}) or {}),
        },
        "sources": {
            "top_docs": top_docs,
            "blocked_doc_ids": sorted(str(x) for x in list(sources.get("blocked_doc_ids", []) or [])),
            "quarantined_source_ids": sorted(str(x) for x in list(sources.get("quarantined_source_ids", []) or [])),
        },
        "prevention": {
            "context_prevented": bool(prevention.get("context_prevented", False)),
            "tool_execution_prevented": bool(prevention.get("tool_execution_prevented", False)),
            "prevented_tools": prevented_tools,
            "prevented_docs": sorted(str(x) for x in list(prevention.get("prevented_docs", []) or [])),
        },
        "incident_30s": dict(incident_30s),
    }


def compare_incident_artifacts(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    components = {
        "ids": {
            "trace_id": actual.get("trace_id") == expected.get("trace_id"),
            "decision_id": actual.get("decision_id") == expected.get("decision_id"),
        },
        "context": actual.get("context") == expected.get("context"),
        "decision": actual.get("decision") == expected.get("decision"),
        "reasons": actual.get("reasons") == expected.get("reasons"),
        "sources": actual.get("sources") == expected.get("sources"),
        "prevention": actual.get("prevention") == expected.get("prevention"),
    }
    return {
        "match": bool(
            components["ids"]["trace_id"]
            and components["ids"]["decision_id"]
            and components["context"]
            and components["decision"]
            and components["reasons"]
            and components["sources"]
            and components["prevention"]
        ),
        "components": components,
    }


def triage_from_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    incident_30s = payload.get("incident_30s", {}) if isinstance(payload.get("incident_30s", {}), dict) else {}
    if incident_30s:
        return {
            "why": dict(incident_30s.get("why", {}) or {}),
            "when": dict(incident_30s.get("when", {}) or {}),
            "what_prevented": dict(incident_30s.get("what_prevented", {}) or {}),
        }
    decision = payload.get("decision", {}) if isinstance(payload.get("decision", {}), dict) else {}
    reasons = payload.get("reasons", {}) if isinstance(payload.get("reasons", {}), dict) else {}
    prevention = payload.get("prevention", {}) if isinstance(payload.get("prevention", {}), dict) else {}
    context = payload.get("context", {}) if isinstance(payload.get("context", {}), dict) else {}
    return {
        "why": {
            "control_outcome": decision.get("control_outcome"),
            "reason_flags": list(reasons.get("reason_flags", []) or []),
            "action_types": list(decision.get("action_types", []) or []),
        },
        "when": {
            "step": context.get("step"),
            "surface": payload.get("surface"),
            "timestamp": payload.get("timestamp"),
        },
        "what_prevented": {
            "context_prevented": bool(prevention.get("context_prevented", False)),
            "tool_execution_prevented": bool(prevention.get("tool_execution_prevented", False)),
            "prevented_tools": [row.get("tool_name", "") for row in list(prevention.get("prevented_tools", []) or [])],
            "prevented_docs": list(prevention.get("prevented_docs", []) or []),
        },
    }


# -------------------------------- replay v3 -----------------------------------

def parse_replay_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    event = str(payload.get("event", "")).strip()
    if event and event != "omega_replay_input_v1":
        raise ValueError("replay input event must be omega_replay_input_v1")

    turns_raw = payload.get("turns")
    if not isinstance(turns_raw, list):
        # single-turn backward-friendly shape
        if all(key in payload for key in ("session_id", "actor_id", "turn_index", "user_query", "packet_items")):
            turns_raw = [
                {
                    "session_id": payload.get("session_id"),
                    "actor_id": payload.get("actor_id"),
                    "turn_index": payload.get("turn_index"),
                    "user_query": payload.get("user_query"),
                    "packet_items": payload.get("packet_items"),
                    "tool_requests": payload.get("tool_requests", []),
                }
            ]
        else:
            raise ValueError("replay input must contain turns[]")

    normalized_turns: List[Dict[str, Any]] = []
    seen_turn_indexes: set[int] = set()
    for idx, row in enumerate(turns_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"turn[{idx}] must be object")
        turn_index = int(row.get("turn_index", idx))
        if turn_index <= 0:
            raise ValueError(f"turn[{idx}].turn_index must be >= 1")
        if turn_index in seen_turn_indexes:
            raise ValueError(f"turn[{idx}].turn_index must be unique")
        seen_turn_indexes.add(turn_index)
        session_id = str(row.get("session_id", payload.get("session_id", ""))).strip()
        if not session_id:
            raise ValueError(f"turn[{idx}].session_id is required")
        actor_id = str(row.get("actor_id", payload.get("actor_id", ""))).strip() or session_id
        user_query = str(row.get("user_query", "")).strip()
        if not user_query:
            raise ValueError(f"turn[{idx}].user_query is required")

        packet_items_raw = row.get("packet_items", [])
        if not isinstance(packet_items_raw, list) or not packet_items_raw:
            raise ValueError(f"turn[{idx}].packet_items must be non-empty list")
        packet_items: List[Dict[str, Any]] = []
        for p_idx, item in enumerate(packet_items_raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"turn[{idx}].packet_items[{p_idx}] must be object")
            if "text" not in item:
                raise ValueError(f"turn[{idx}].packet_items[{p_idx}].text is required")
            out = {
                "doc_id": str(item.get("doc_id", "")).strip(),
                "source_id": str(item.get("source_id", "")).strip(),
                "source_type": str(item.get("source_type", "")).strip(),
                "trust": str(item.get("trust", "")).strip(),
                "text": str(item.get("text", "")),
                "language": item.get("language"),
                "meta": dict(item.get("meta", {}) or {}),
            }
            if not all((out["doc_id"], out["source_id"], out["source_type"], out["trust"])):
                raise ValueError(f"turn[{idx}].packet_items[{p_idx}] missing required fields")
            packet_items.append(out)

        tool_requests_raw = row.get("tool_requests", [])
        if tool_requests_raw is None:
            tool_requests_raw = []
        if not isinstance(tool_requests_raw, list):
            raise ValueError(f"turn[{idx}].tool_requests must be list")
        tool_requests: List[Dict[str, Any]] = []
        for t_idx, req in enumerate(tool_requests_raw, start=1):
            if not isinstance(req, dict):
                raise ValueError(f"turn[{idx}].tool_requests[{t_idx}] must be object")
            tool_name = str(req.get("tool_name", "")).strip()
            if not tool_name:
                raise ValueError(f"turn[{idx}].tool_requests[{t_idx}].tool_name is required")
            args = req.get("args", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ValueError(f"turn[{idx}].tool_requests[{t_idx}].args must be object")
            tool_requests.append(
                {
                    "tool_name": tool_name,
                    "args": dict(args),
                    "session_id": str(req.get("session_id", session_id)).strip() or session_id,
                    "step": int(req.get("step", turn_index)),
                }
            )

        normalized_turns.append(
            {
                "turn_index": turn_index,
                "session_id": session_id,
                "actor_id": actor_id,
                "user_query": user_query,
                "packet_items": packet_items,
                "tool_requests": tool_requests,
                "meta": dict(row.get("meta", {}) or {}),
            }
        )

    normalized_turns.sort(key=lambda row: int(row["turn_index"]))

    execution = dict(payload.get("execution", {}) or {})
    state_bootstrap = str(execution.get("state_bootstrap", "fresh_state")).strip().lower()
    if state_bootstrap not in {"fresh_state", "reuse_state", "reset_actor_before_run"}:
        raise ValueError("execution.state_bootstrap must be fresh_state|reuse_state|reset_actor_before_run")

    return {
        "event": "omega_replay_input_v1",
        "schema_version": str(payload.get("schema_version", "1.0")),
        "replay_id": str(payload.get("replay_id", "")).strip() or f"replay_{uuid.uuid4().hex[:12]}",
        "canonical_path": str(
            payload.get(
                "canonical_path",
                "RAG service -> RetrieverProdAdapter -> OmegaRAGHarness -> ToolGateway",
            )
        ).strip(),
        "execution": {
            "state_bootstrap": state_bootstrap,
        },
        "turns": normalized_turns,
        "expected": dict(payload.get("expected", {}) or {}),
        "refs": dict(payload.get("refs", {}) or {}),
    }


def _normalize_replay_turn_output(turn: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    from dataclasses import asdict

    step_result = out["step_result"]
    reason_flags = [name for name, val in getattr(step_result, "reasons").__dict__.items() if bool(val)]
    actions = [_normalize_action(a.__dict__) for a in list(out["decision"].actions)]
    action_types = sorted({str(item.get("type", "")) for item in actions if str(item.get("type", "")).strip()})

    tool_decisions = []
    for row in list(out.get("tool_decisions", []) or []):
        d = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
        tool_decisions.append(_normalize_tool_decision(d))

    tool_executions = []
    for row in list(out.get("tool_executions", []) or []):
        e = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
        tool_executions.append(
            {
                "tool_name": str(e.get("tool_name", "")),
                "allowed": bool(e.get("allowed", False)),
                "executed": bool(e.get("executed", False)),
                "reason": str(e.get("reason", "")),
            }
        )

    incident_artifact = out.get("incident_artifact")
    incident_norm = normalize_incident_artifact(incident_artifact) if isinstance(incident_artifact, dict) else None
    blocked_doc_ids = []
    quarantined_source_ids = []
    prevented_tools = []
    if incident_norm is not None:
        blocked_doc_ids = list((incident_norm.get("sources", {}) or {}).get("blocked_doc_ids", []) or [])
        quarantined_source_ids = list((incident_norm.get("sources", {}) or {}).get("quarantined_source_ids", []) or [])
        prevented_tools = list((incident_norm.get("prevention", {}) or {}).get("prevented_tools", []) or [])
    incident_30s = dict((incident_norm or {}).get("incident_30s", {}) or {})

    return {
        "turn_index": int(turn["turn_index"]),
        "session_id": str(turn["session_id"]),
        "actor_id": str(turn["actor_id"]),
        "control_outcome": str(out.get("control_outcome", "ALLOW")),
        "off": bool(step_result.off),
        "severity": str(out["decision"].severity),
        "reason_flags": sorted(reason_flags),
        "action_types": action_types,
        "actions": actions,
        "top_docs": list(step_result.top_docs),
        "p": [float(x) for x in list(step_result.p)],
        "m_next": [float(x) for x in list(step_result.m_next)],
        "tool_decisions": tool_decisions,
        "tool_executions": tool_executions,
        "blocked_doc_ids": sorted(str(x) for x in blocked_doc_ids),
        "quarantined_source_ids": sorted(str(x) for x in quarantined_source_ids),
        "prevented_tools": prevented_tools,
        "incident_30s": incident_30s,
        "incident_artifact": incident_norm,
    }


def _resolve_policy_config(policy_ref: str) -> Tuple[str, Dict[str, Any], str]:
    from omega.config.loader import load_resolved_config

    candidate = Path(policy_ref)
    if candidate.exists() and candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Policy file must contain object: {policy_ref}")
        return str(candidate.as_posix()), payload, _sha256_text(_json_stable(payload))

    snapshot = load_resolved_config(profile=policy_ref)
    return str(policy_ref), dict(snapshot.resolved), str(snapshot.resolved_sha256)


def _with_replay_overrides(
    config: Dict[str, Any],
    *,
    run_id: str,
    label: str,
    isolate_state: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    tools_cfg = cfg.setdefault("tools", {})
    tools_cfg["execution_mode"] = "DRY_RUN"

    if not isolate_state:
        return cfg

    cs_cfg = (cfg.setdefault("off_policy", {}).setdefault("cross_session", {}))
    cs_cfg["sqlite_path"] = str((ROOT / "artifacts" / "replay" / "state" / f"{run_id}_{label}.db").as_posix())
    return cfg


def _clear_cross_session_state(config: Dict[str, Any]) -> None:
    cs_cfg = ((config.get("off_policy") or {}).get("cross_session") or {})
    sqlite_path = str(cs_cfg.get("sqlite_path", "")).strip()
    enabled = bool(cs_cfg.get("enabled", False))
    if not enabled or not sqlite_path:
        return
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=10.0) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT OR IGNORE INTO meta(key, value) VALUES ('global_step', 0);
            DELETE FROM actor_state;
            DELETE FROM actor_freeze;
            DELETE FROM source_quarantine;
            DELETE FROM session_actor;
            UPDATE meta SET value = 0 WHERE key = 'global_step';
            """
        )
        conn.commit()


def _run_replay_once(
    *,
    replay_input: Dict[str, Any],
    config: Dict[str, Any],
    policy_ref: str,
    resolved_sha256: str,
    state_bootstrap: str,
) -> Dict[str, Any]:
    from omega.interfaces.contracts_v1 import ContentItem, ToolRequest
    from omega.core.omega_core import OmegaCoreV1
    from omega.core.params import omega_params_from_config
    from omega.policy.off_policy_v1 import OffPolicyV1
    from omega.projector.factory import build_projector
    from omega.rag.harness import MockLLM, OmegaRAGHarness
    from omega.tools.tool_gateway import ToolGatewayV1

    harness = OmegaRAGHarness(
        projector=build_projector(config),
        omega_core=OmegaCoreV1(omega_params_from_config(config)),
        off_policy=OffPolicyV1(config),
        tool_gateway=ToolGatewayV1(config),
        config=config,
        llm_backend=MockLLM(),
    )

    turns = list(replay_input.get("turns", []) or [])
    if not turns:
        raise ValueError("Replay input turns is empty")

    if state_bootstrap == "fresh_state":
        _clear_cross_session_state(config)
    elif state_bootstrap == "reset_actor_before_run":
        actor_ids = sorted({str(t.get("actor_id", "")).strip() for t in turns if str(t.get("actor_id", "")).strip()})
        for actor_id in actor_ids:
            harness.cross_session.reset_actor(actor_id)

    current_session = ""
    current_actor = ""
    first = True
    normalized_turns: List[Dict[str, Any]] = []

    for turn in turns:
        session_id = str(turn["session_id"])
        actor_id = str(turn["actor_id"])

        if first:
            harness.reset_state(session_id=session_id, actor_id=actor_id)
            first = False
        elif session_id != current_session or actor_id != current_actor:
            harness.reset_state(session_id=session_id, actor_id=actor_id)

        packet_items = [
            ContentItem(
                doc_id=str(item["doc_id"]),
                source_id=str(item["source_id"]),
                source_type=str(item["source_type"]),
                trust=str(item["trust"]),
                text=str(item["text"]),
                language=item.get("language"),
                meta=dict(item.get("meta", {}) or {}),
            )
            for item in list(turn.get("packet_items", []) or [])
        ]

        tool_requests = [
            ToolRequest(
                tool_name=str(req["tool_name"]),
                args=dict(req.get("args", {}) or {}),
                session_id=str(req.get("session_id", session_id)),
                step=int(req.get("step", int(turn["turn_index"]))),
            )
            for req in list(turn.get("tool_requests", []) or [])
        ]

        out = harness.run_step(
            user_query=str(turn["user_query"]),
            packet_items=packet_items,
            tool_requests=tool_requests,
            actor_id=actor_id,
            config_refs={
                "policy_ref": policy_ref,
                "resolved_config_sha256": resolved_sha256,
            },
        )
        normalized_turns.append(_normalize_replay_turn_output(turn=turn, out=out))
        current_session = session_id
        current_actor = actor_id

    signature = _sha256_text(_json_stable(normalized_turns))
    return {
        "policy_ref": policy_ref,
        "policy_version": str((config.get("off_policy", {}) or {}).get("policy_version", "")),
        "resolved_config_sha256": resolved_sha256,
        "state_bootstrap": state_bootstrap,
        "turns": normalized_turns,
        "signature": signature,
    }


def compare_replay_results(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    actual_turns = list(actual.get("turns", []) or [])
    expected_turns = list(expected.get("turns", []) or [])
    components: Dict[str, Any] = {
        "turn_count": len(actual_turns) == len(expected_turns),
        "signature": str(actual.get("signature", "")) == str(expected.get("signature", "")),
        "turns": [],
    }

    max_len = max(len(actual_turns), len(expected_turns))
    turn_match = True
    for idx in range(max_len):
        a = actual_turns[idx] if idx < len(actual_turns) else None
        b = expected_turns[idx] if idx < len(expected_turns) else None
        if a is None or b is None:
            turn_match = False
            components["turns"].append({"turn_index": idx + 1, "match": False, "changed_components": ["missing_turn"]})
            continue
        changed: List[str] = []
        for key in (
            "control_outcome",
            "off",
            "severity",
            "reason_flags",
            "action_types",
            "actions",
            "top_docs",
            "tool_decisions",
            "tool_executions",
            "blocked_doc_ids",
            "quarantined_source_ids",
            "prevented_tools",
            "incident_30s",
            "incident_artifact",
        ):
            if a.get(key) != b.get(key):
                changed.append(key)
        is_match = len(changed) == 0
        turn_match = turn_match and is_match
        components["turns"].append(
            {
                "turn_index": int(a.get("turn_index", idx + 1)),
                "match": is_match,
                "changed_components": changed,
            }
        )

    components["turns_match"] = turn_match
    return {
        "match": bool(components["turn_count"] and turn_match),
        "components": components,
    }


def diff_policy_outputs(policy_a: Dict[str, Any], policy_b: Dict[str, Any]) -> Dict[str, Any]:
    turns_a = list(policy_a.get("turns", []) or [])
    turns_b = list(policy_b.get("turns", []) or [])
    max_len = max(len(turns_a), len(turns_b))
    turn_deltas: List[Dict[str, Any]] = []

    for idx in range(max_len):
        a = turns_a[idx] if idx < len(turns_a) else None
        b = turns_b[idx] if idx < len(turns_b) else None
        if a is None or b is None:
            turn_deltas.append(
                {
                    "turn_index": idx + 1,
                    "changed_components": ["missing_turn"],
                    "policy_a": a,
                    "policy_b": b,
                }
            )
            continue
        changed: List[str] = []
        for key in (
            "control_outcome",
            "off",
            "severity",
            "reason_flags",
            "action_types",
            "actions",
            "top_docs",
            "tool_decisions",
            "tool_executions",
            "blocked_doc_ids",
            "quarantined_source_ids",
            "prevented_tools",
            "incident_30s",
        ):
            if a.get(key) != b.get(key):
                changed.append(key)
        if changed:
            turn_deltas.append(
                {
                    "turn_index": int(a.get("turn_index", idx + 1)),
                    "changed_components": changed,
                    "policy_a": {
                        k: a.get(k)
                        for k in (
                            "control_outcome",
                            "off",
                            "action_types",
                            "blocked_doc_ids",
                            "quarantined_source_ids",
                            "prevented_tools",
                            "incident_30s",
                        )
                    },
                    "policy_b": {
                        k: b.get(k)
                        for k in (
                            "control_outcome",
                            "off",
                            "action_types",
                            "blocked_doc_ids",
                            "quarantined_source_ids",
                            "prevented_tools",
                            "incident_30s",
                        )
                    },
                }
            )

    no_delta = len(turn_deltas) == 0
    return {
        "no_decision_delta": bool(no_delta),
        "turn_deltas": turn_deltas,
    }


def _default_replay_dir() -> Path:
    out_dir = ROOT / "artifacts" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _compare_run_series(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if len(runs) <= 1:
        return {"match": True, "runs": rows}
    baseline = runs[0]
    all_match = True
    for idx in range(1, len(runs)):
        comp = compare_replay_results(baseline, runs[idx])
        rows.append({"baseline_run": 1, "run": idx + 1, "comparison": comp})
        all_match = all_match and bool(comp.get("match", False))
    return {"match": all_match, "runs": rows}


def _run_raw_replay_mode(args: argparse.Namespace) -> int:
    replay_source = Path(str(args.replay_input))
    replay_payload = json.loads(replay_source.read_text(encoding="utf-8"))
    replay_input = parse_replay_input(replay_payload)

    state_bootstrap = str(args.state_bootstrap or replay_input.get("execution", {}).get("state_bootstrap", "fresh_state")).strip().lower()
    if state_bootstrap not in {"fresh_state", "reuse_state", "reset_actor_before_run"}:
        raise ValueError("state_bootstrap must be fresh_state|reuse_state|reset_actor_before_run")

    replay_id = str(replay_input.get("replay_id", "")).strip() or f"replay_{uuid.uuid4().hex[:12]}"
    deterministic_runs = max(2, int(args.deterministic_runs))

    policy_ref_a = str(args.policy_a or args.profile)
    policy_ref_b = str(args.policy_b or "").strip() or None

    label_a, config_a_raw, resolved_sha_a = _resolve_policy_config(policy_ref_a)

    runs_a: List[Dict[str, Any]] = []
    reuse_state_backend = state_bootstrap == "reuse_state"
    for i in range(1, deterministic_runs + 1):
        cfg = _with_replay_overrides(
            config=config_a_raw,
            run_id=replay_id,
            label=("policy_a_shared" if reuse_state_backend else f"policy_a_run{i}"),
            isolate_state=True,
        )
        runs_a.append(
            _run_replay_once(
                replay_input=replay_input,
                config=cfg,
                policy_ref=label_a,
                resolved_sha256=resolved_sha_a,
                state_bootstrap=state_bootstrap,
            )
        )

    det_a = compare_replay_results(runs_a[0], runs_a[1])
    det_a_series = _compare_run_series(runs_a)

    if policy_ref_b is None:
        report = {
            "event": "omega_replay_result_v1",
            "schema_version": "1.0",
            "timestamp": _now_utc_iso(),
            "mode": "raw_replay",
            "replay_id": replay_id,
            "replay_input": str(replay_source.as_posix()),
            "policy": {
                "policy_ref": label_a,
                "policy_version": str(runs_a[0].get("policy_version", "")),
                "resolved_config_sha256": resolved_sha_a,
            },
            "execution": {
                "state_bootstrap": state_bootstrap,
                "deterministic_runs": deterministic_runs,
            },
            "replay_result": runs_a[0],
            "determinism": det_a,
            "determinism_runs": det_a_series,
        }
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = _default_replay_dir() / f"replay_result_{replay_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        if args.strict and not bool(det_a_series.get("match", False)):
            return 1
        return 0

    label_b, config_b_raw, resolved_sha_b = _resolve_policy_config(policy_ref_b)

    runs_b: List[Dict[str, Any]] = []
    for i in range(1, deterministic_runs + 1):
        cfg = _with_replay_overrides(
            config=config_b_raw,
            run_id=replay_id,
            label=("policy_b_shared" if reuse_state_backend else f"policy_b_run{i}"),
            isolate_state=True,
        )
        runs_b.append(
            _run_replay_once(
                replay_input=replay_input,
                config=cfg,
                policy_ref=label_b,
                resolved_sha256=resolved_sha_b,
                state_bootstrap=state_bootstrap,
            )
        )

    det_b = compare_replay_results(runs_b[0], runs_b[1])
    det_b_series = _compare_run_series(runs_b)
    policy_diff = diff_policy_outputs(runs_a[0], runs_b[0])

    report = {
        "event": "omega_policy_diff_v1",
        "schema_version": "1.0",
        "timestamp": _now_utc_iso(),
        "mode": "raw_policy_diff",
        "replay_id": replay_id,
        "replay_input": str(replay_source.as_posix()),
        "policy_a": {
            "policy_ref": label_a,
            "policy_version": str(runs_a[0].get("policy_version", "")),
            "resolved_config_sha256": resolved_sha_a,
        },
        "policy_b": {
            "policy_ref": label_b,
            "policy_version": str(runs_b[0].get("policy_version", "")),
            "resolved_config_sha256": resolved_sha_b,
        },
        "execution": {
            "state_bootstrap": state_bootstrap,
            "deterministic_runs": deterministic_runs,
        },
        "determinism": {
            "policy_a": det_a,
            "policy_b": det_b,
        },
        "determinism_runs": {
            "policy_a": det_a_series,
            "policy_b": det_b_series,
        },
        "policy_diff": policy_diff,
        "result_a": runs_a[0],
        "result_b": runs_b[0],
    }
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_replay_dir() / f"policy_diff_{replay_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if args.strict and (not bool(det_a_series.get("match", False)) or not bool(det_b_series.get("match", False))):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay incident from report and compare against expected snapshot/artifact")
    parser.add_argument("--incident-report", default=None, help="Path to local_contour_report.json or smoke output JSON")
    parser.add_argument("--incident-artifact", default=None, help="Path to unified incident artifact JSON")
    parser.add_argument("--replay-input", default=None, help="Path to omega_replay_input_v1 JSON")
    parser.add_argument(
        "--report-kind",
        choices=["auto", "local_contour", "smoke_real_rag", "incident_artifact"],
        default="auto",
    )
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--step", type=int, default=None, help="Step for local_contour scenario")
    parser.add_argument("--expected-snapshot", default=None, help="Path to expected replay snapshot JSON")
    parser.add_argument(
        "--expected-incident-artifact",
        default=None,
        help="Path to expected unified incident artifact JSON",
    )
    parser.add_argument("--write-snapshot", default=None, help="Optional path to write extracted snapshot")
    parser.add_argument("--write-incident-artifact", default=None, help="Optional path to write normalized incident artifact")
    parser.add_argument("--profile", default="dev", help="Policy profile for replay-input mode")
    parser.add_argument("--policy-a", default=None, help="Policy A: profile or resolved_config.json")
    parser.add_argument("--policy-b", default=None, help="Policy B: profile or resolved_config.json")
    parser.add_argument(
        "--state-bootstrap",
        choices=["fresh_state", "reuse_state", "reset_actor_before_run"],
        default=None,
        help="Replay state bootstrap mode (raw replay mode)",
    )
    parser.add_argument("--deterministic-runs", type=int, default=2, help="Number of replay runs per policy in raw mode")
    parser.add_argument("--output", default=None, help="Optional output report path")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if mismatch exists")
    args = parser.parse_args()

    if args.replay_input:
        return _run_raw_replay_mode(args)

    source_path: Optional[Path] = None
    payload: Optional[Dict[str, Any]] = None
    report_kind = args.report_kind
    if args.incident_artifact:
        source_path = Path(args.incident_artifact)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if report_kind == "auto":
            report_kind = "incident_artifact"
    elif args.incident_report:
        source_path = Path(args.incident_report)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if report_kind == "auto":
            report_kind = _infer_report_kind(payload)
    else:
        raise ValueError("Either --incident-report, --incident-artifact or --replay-input is required")

    if report_kind == "incident_artifact":
        artifact = extract_incident_artifact(payload or {})
        normalized = normalize_incident_artifact(artifact)
        step = int((normalized.get("context", {}) or {}).get("step", 0))
        artifact_id = str(normalized.get("incident_artifact_id", "")).strip() or "unknown"
        if args.write_incident_artifact:
            artifact_path = Path(args.write_incident_artifact)
        else:
            artifact_path = _default_replay_dir() / f"incident_artifact_{artifact_id}_{step}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2), encoding="utf-8")

        comparison: Dict[str, Any] = {"match": True, "components": {}}
        mode = "artifact_only"
        expected_artifact_payload: Optional[Dict[str, Any]] = None
        expected_source = args.expected_incident_artifact or args.expected_snapshot
        if expected_source:
            expected_artifact_raw = json.loads(Path(expected_source).read_text(encoding="utf-8"))
            expected_artifact_payload = normalize_incident_artifact(extract_incident_artifact(expected_artifact_raw))
            comparison = compare_incident_artifacts(normalized, expected_artifact_payload)
            mode = "artifact_compare"

        triage = triage_from_artifact(artifact)
        report = {
            "event": "incident_replay_v2",
            "timestamp": _now_utc_iso(),
            "mode": mode,
            "report_kind": "incident_artifact",
            "incident_source": str(source_path.as_posix()) if source_path is not None else "",
            "incident_artifact_id": artifact_id,
            "step": step,
            "triage": triage,
            "actual_incident_artifact": normalized,
            "expected_incident_artifact": expected_artifact_payload,
            "comparison": comparison,
            "artifact_file": str(artifact_path.as_posix()),
        }
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = _default_replay_dir() / f"replay_report_incident_{artifact_id}_{step}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        if args.strict and mode == "artifact_compare" and not comparison["match"]:
            return 1
        return 0

    if not str(args.scenario or "").strip():
        raise ValueError("--scenario is required for local_contour/smoke_real_rag replay")
    actual_snapshot = extract_incident_snapshot(payload or {}, report_kind=report_kind, scenario=str(args.scenario), step=args.step)

    if args.write_snapshot:
        out_snapshot = Path(args.write_snapshot)
    else:
        out_snapshot = _default_replay_dir() / f"snapshot_{args.scenario}_{actual_snapshot['step']}.json"
    out_snapshot.parent.mkdir(parents=True, exist_ok=True)
    out_snapshot.write_text(json.dumps(actual_snapshot, ensure_ascii=True, indent=2), encoding="utf-8")

    comparison = {"match": True, "components": {}}
    mode = "snapshot_only"
    expected_snapshot_payload: Dict[str, Any] | None = None
    if args.expected_snapshot:
        expected_path = Path(args.expected_snapshot)
        expected_snapshot_payload = json.loads(expected_path.read_text(encoding="utf-8"))
        comparison = compare_snapshots(actual_snapshot, expected_snapshot_payload)
        mode = "compare"

    report = {
        "event": "incident_replay_v1",
        "timestamp": _now_utc_iso(),
        "mode": mode,
        "report_kind": report_kind,
        "incident_report": str(source_path.as_posix()) if source_path is not None else "",
        "scenario": args.scenario,
        "step": actual_snapshot["step"],
        "actual_snapshot": actual_snapshot,
        "expected_snapshot": expected_snapshot_payload,
        "comparison": comparison,
        "snapshot_file": str(out_snapshot.as_posix()),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_replay_dir() / f"replay_report_{args.scenario}_{actual_snapshot['step']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if args.strict and mode == "compare" and not comparison["match"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
