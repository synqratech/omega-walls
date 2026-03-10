from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay incident from report and compare against expected snapshot")
    parser.add_argument("--incident-report", required=True, help="Path to local_contour_report.json or smoke output JSON")
    parser.add_argument(
        "--report-kind",
        choices=["auto", "local_contour", "smoke_real_rag"],
        default="auto",
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--step", type=int, default=None, help="Step for local_contour scenario")
    parser.add_argument("--expected-snapshot", default=None, help="Path to expected replay snapshot JSON")
    parser.add_argument("--write-snapshot", default=None, help="Optional path to write extracted snapshot")
    parser.add_argument("--output", default=None, help="Optional output report path")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if mismatch exists")
    args = parser.parse_args()

    incident_path = Path(args.incident_report)
    payload = json.loads(incident_path.read_text(encoding="utf-8"))
    report_kind = args.report_kind
    if report_kind == "auto":
        report_kind = _infer_report_kind(payload)

    actual_snapshot = extract_incident_snapshot(payload, report_kind=report_kind, scenario=args.scenario, step=args.step)

    if args.write_snapshot:
        out_snapshot = Path(args.write_snapshot)
    else:
        out_dir = ROOT / "artifacts" / "replay"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_snapshot = out_dir / f"snapshot_{args.scenario}_{actual_snapshot['step']}.json"
    out_snapshot.parent.mkdir(parents=True, exist_ok=True)
    out_snapshot.write_text(json.dumps(actual_snapshot, ensure_ascii=True, indent=2), encoding="utf-8")

    comparison: Dict[str, Any] = {"match": True, "components": {}}
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
        "incident_report": str(incident_path.as_posix()),
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
        out_dir = ROOT / "artifacts" / "replay"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"replay_report_{args.scenario}_{actual_snapshot['step']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if args.strict and mode == "compare" and not comparison["match"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
