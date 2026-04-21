from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_week_token(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}_{iso.week:02d}"


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _resolve_baseline_path(
    *,
    explicit_path: Optional[Path],
    weekly_dir: Path,
    exclude_path: Optional[Path],
) -> Optional[Path]:
    if explicit_path:
        return explicit_path if explicit_path.exists() else None
    if not weekly_dir.exists():
        return None
    candidates = sorted(
        weekly_dir.glob("weekly_security_report_*.json"),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    for path in candidates:
        if exclude_path and path.resolve() == exclude_path.resolve():
            continue
        return path
    return None


def _load_baseline_metrics(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"status": "missing"}

    payload = _load_json(path)
    kpi = payload.get("kpi", {}) if isinstance(payload.get("kpi"), Mapping) else {}
    if "attack_off_rate" in kpi and "benign_off_rate" in kpi:
        return {
            "status": "provided",
            "source": str(path),
            "attack_off_rate": float(kpi.get("attack_off_rate", 0.0) or 0.0),
            "benign_off_rate": float(kpi.get("benign_off_rate", 0.0) or 0.0),
        }

    if "attack_off_rate" in payload and "benign_off_rate" in payload:
        return {
            "status": "provided",
            "source": str(path),
            "attack_off_rate": float(payload.get("attack_off_rate", 0.0) or 0.0),
            "benign_off_rate": float(payload.get("benign_off_rate", 0.0) or 0.0),
        }

    summary_all = payload.get("summary_all", {}) if isinstance(payload.get("summary_all"), Mapping) else {}
    if "session_attack_off_rate" in summary_all and "session_benign_off_rate" in summary_all:
        return {
            "status": "provided",
            "source": str(path),
            "attack_off_rate": float(summary_all.get("session_attack_off_rate", 0.0) or 0.0),
            "benign_off_rate": float(summary_all.get("session_benign_off_rate", 0.0) or 0.0),
        }

    return {"status": "missing"}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def build_weekly_summary(
    *,
    dashboard_feed: Mapping[str, Any],
    incident_backlog: Mapping[str, Any],
    baseline: Mapping[str, Any],
    week_token: str,
    dashboard_feed_path: str,
    incident_backlog_path: str,
) -> Dict[str, Any]:
    kpi = dashboard_feed.get("kpi", {}) if isinstance(dashboard_feed.get("kpi"), Mapping) else {}
    volume = kpi.get("attack_volume", {}) if isinstance(kpi.get("attack_volume"), Mapping) else {}
    family_split = _as_list(kpi.get("family_split"))
    confusion = kpi.get("confusion", {}) if isinstance(kpi.get("confusion"), Mapping) else {}
    mismatch = (
        dashboard_feed.get("mismatch_clusters", {})
        if isinstance(dashboard_feed.get("mismatch_clusters"), Mapping)
        else {}
    )
    alerts = _as_list(dashboard_feed.get("alerts"))

    attack_off_rate = float(kpi.get("attack_off_rate", 0.0) or 0.0)
    benign_off_rate = float(kpi.get("benign_off_rate", 0.0) or 0.0)

    incidents = _as_list(incident_backlog.get("incidents"))
    sev1 = [x for x in incidents if isinstance(x, Mapping) and str(x.get("severity", "")) == "SEV-1"]
    sev2 = [x for x in incidents if isinstance(x, Mapping) and str(x.get("severity", "")) == "SEV-2"]
    sev3 = [x for x in incidents if isinstance(x, Mapping) and str(x.get("severity", "")) == "SEV-3"]

    baseline_status = str(baseline.get("status", "missing"))
    base_attack = float(baseline.get("attack_off_rate", 0.0) or 0.0) if baseline_status == "provided" else None
    base_benign = float(baseline.get("benign_off_rate", 0.0) or 0.0) if baseline_status == "provided" else None

    if base_attack is not None and base_benign is not None:
        attack_delta_pp = (attack_off_rate - base_attack) * 100.0
        benign_delta_pp = (benign_off_rate - base_benign) * 100.0
        attack_breach = (base_attack - attack_off_rate) > 0.01
        benign_breach = (benign_off_rate - base_benign) > 0.005
    else:
        attack_delta_pp = None
        benign_delta_pp = None
        attack_breach = False
        benign_breach = False

    if len(sev1) > 0 or attack_breach or benign_breach:
        status = "red"
    elif len(sev2) > 0 or len(alerts) > 0:
        status = "yellow"
    else:
        status = "green"

    critical_cases: List[Dict[str, Any]] = []
    for row in incidents:
        if not isinstance(row, Mapping):
            continue
        sev = str(row.get("severity", ""))
        if sev not in {"SEV-1", "SEV-2"}:
            continue
        critical_cases.append(
            {
                "incident_id": str(row.get("incident_id", "")),
                "severity": sev,
                "category": str(row.get("category", "")),
                "reason": str(row.get("reason", "")),
                "owner": str(row.get("owner", "")),
                "status": str(row.get("status", "open")),
            }
        )
    critical_cases = critical_cases[:10]

    top_fn = [x for x in _as_list(mismatch.get("fn_clusters")) if isinstance(x, Mapping)][:5]
    top_fp = [x for x in _as_list(mismatch.get("fp_clusters")) if isinstance(x, Mapping)][:5]
    top_families = [x for x in family_split if isinstance(x, Mapping)][:5]

    action_plan: List[Dict[str, Any]] = []
    if top_fn:
        first = top_fn[0]
        action_plan.append(
            {
                "priority": 1,
                "task": f"Reduce FN cluster {first.get('cluster_key', 'unknown')}",
                "owner": "Detection Engineering",
                "eta": "5 business days",
                "expected_impact": "Increase attack_off_rate",
            }
        )
    if top_fp:
        first = top_fp[0]
        action_plan.append(
            {
                "priority": 2,
                "task": f"Reduce FP cluster {first.get('cluster_key', 'unknown')}",
                "owner": "Security Operations",
                "eta": "5 business days",
                "expected_impact": "Decrease benign_off_rate",
            }
        )
    action_plan.append(
        {
            "priority": 3,
            "task": "Run blind-eval control pack and compare to previous week",
            "owner": "Security Operations",
            "eta": "3 business days",
            "expected_impact": "Regression detection before release",
        }
    )

    highlights = [
        f"Attack OFF rate: {attack_off_rate * 100.0:.2f}%",
        f"Benign OFF rate: {benign_off_rate * 100.0:.2f}%",
        f"Open incidents: {len(incidents)} (SEV-1={len(sev1)}, SEV-2={len(sev2)}, SEV-3={len(sev3)})",
    ]

    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "week": week_token,
        "status": status,
        "source_dashboard_feed": dashboard_feed_path,
        "source_incident_backlog": incident_backlog_path,
        "kpi": {
            "attack_volume": {
                "attack_sessions": int(volume.get("attack_sessions", 0) or 0),
                "attack_turn_events": int(volume.get("attack_turn_events", 0) or 0),
            },
            "attack_off_rate": float(attack_off_rate),
            "benign_off_rate": float(benign_off_rate),
            "family_split_top": top_families,
            "confusion": confusion,
        },
        "baseline_regression": {
            "baseline_status": baseline_status,
            "baseline_source": str(baseline.get("source", "")) if baseline_status == "provided" else "",
            "baseline_attack_off_rate": base_attack,
            "baseline_benign_off_rate": base_benign,
            "attack_off_rate_delta_pp": attack_delta_pp,
            "benign_off_rate_delta_pp": benign_delta_pp,
            "alert_attack_off_rate_breach": bool(attack_breach),
            "alert_benign_off_rate_breach": bool(benign_breach),
        },
        "incidents": {
            "open_total": int(len(incidents)),
            "sev_counts": {"SEV-1": len(sev1), "SEV-2": len(sev2), "SEV-3": len(sev3)},
            "critical_cases": critical_cases,
        },
        "top_mismatch_clusters": {
            "fn_clusters": top_fn,
            "fp_clusters": top_fp,
        },
        "alerts": alerts,
        "executive_summary": {
            "highlights": highlights,
            "top3": highlights[:3],
        },
        "next_week_plan": action_plan[:5],
    }


def render_weekly_markdown(summary: Mapping[str, Any]) -> str:
    kpi = summary.get("kpi", {}) if isinstance(summary.get("kpi"), Mapping) else {}
    baseline = (
        summary.get("baseline_regression", {})
        if isinstance(summary.get("baseline_regression"), Mapping)
        else {}
    )
    incidents = summary.get("incidents", {}) if isinstance(summary.get("incidents"), Mapping) else {}
    top = (
        summary.get("top_mismatch_clusters", {})
        if isinstance(summary.get("top_mismatch_clusters"), Mapping)
        else {}
    )

    lines: List[str] = []
    lines.append(f"# Weekly Security Report: {summary.get('week', '')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- status: {summary.get('status', 'green')}")
    for item in _as_list(summary.get("executive_summary", {}).get("highlights") if isinstance(summary.get("executive_summary"), Mapping) else []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## KPI Trends")
    attack_volume = kpi.get("attack_volume", {}) if isinstance(kpi.get("attack_volume"), Mapping) else {}
    lines.append(f"- attack_sessions: {int(attack_volume.get('attack_sessions', 0) or 0)}")
    lines.append(f"- attack_turn_events: {int(attack_volume.get('attack_turn_events', 0) or 0)}")
    lines.append(f"- attack_off_rate: {float(kpi.get('attack_off_rate', 0.0) or 0.0) * 100.0:.2f}%")
    lines.append(f"- benign_off_rate: {float(kpi.get('benign_off_rate', 0.0) or 0.0) * 100.0:.2f}%")
    lines.append("- top_families:")
    for row in _as_list(kpi.get("family_split_top")):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"  - {row.get('family', 'unknown')}: "
            f"{int(row.get('attack_sessions', 0) or 0)} "
            f"({float(row.get('share', 0.0) or 0.0) * 100.0:.2f}%)"
        )

    lines.append("")
    lines.append("## Baseline Regression")
    lines.append(f"- baseline_status: {baseline.get('baseline_status', 'missing')}")
    lines.append(f"- baseline_source: {baseline.get('baseline_source', '')}")
    lines.append(f"- attack_off_rate_delta_pp: {baseline.get('attack_off_rate_delta_pp', None)}")
    lines.append(f"- benign_off_rate_delta_pp: {baseline.get('benign_off_rate_delta_pp', None)}")
    lines.append(f"- attack_off_rate_breach: {baseline.get('alert_attack_off_rate_breach', False)}")
    lines.append(f"- benign_off_rate_breach: {baseline.get('alert_benign_off_rate_breach', False)}")

    lines.append("")
    lines.append("## Critical Cases")
    critical_cases = _as_list(incidents.get("critical_cases"))
    if not critical_cases:
        lines.append("- none")
    else:
        for row in critical_cases:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- {row.get('incident_id', '')} | {row.get('severity', '')} | "
                f"{row.get('category', '')} | {row.get('reason', '')}"
            )

    lines.append("")
    lines.append("## Top Mismatch Clusters")
    lines.append("- FN:")
    fn_clusters = _as_list(top.get("fn_clusters"))
    if not fn_clusters:
        lines.append("  - none")
    for row in fn_clusters:
        if not isinstance(row, Mapping):
            continue
        lines.append(f"  - {row.get('cluster_key', 'unknown')} (count={row.get('count', 0)})")
    lines.append("- FP:")
    fp_clusters = _as_list(top.get("fp_clusters"))
    if not fp_clusters:
        lines.append("  - none")
    for row in fp_clusters:
        if not isinstance(row, Mapping):
            continue
        lines.append(f"  - {row.get('cluster_key', 'unknown')} (count={row.get('count', 0)})")

    lines.append("")
    lines.append("## Next Week Plan")
    for row in _as_list(summary.get("next_week_plan")):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"- P{int(row.get('priority', 0) or 0)} | {row.get('task', '')} | "
            f"owner={row.get('owner', '')} | eta={row.get('eta', '')} | impact={row.get('expected_impact', '')}"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enterprise weekly security report from dashboard feed and incident backlog.")
    parser.add_argument("--dashboard-feed", required=True)
    parser.add_argument("--incident-backlog", required=True)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    week_token = _iso_week_token(now)

    out_md = _resolve_path(args.out_md)
    out_json = _resolve_path(args.out_json)
    if out_md is None:
        out_md = (ROOT / "enterprise" / "reports" / "weekly" / f"weekly_security_report_{week_token}.md").resolve()
    if out_json is None:
        out_json = (ROOT / "enterprise" / "reports" / "weekly" / f"weekly_security_report_{week_token}.json").resolve()

    feed_path = _resolve_path(args.dashboard_feed)
    backlog_path = _resolve_path(args.incident_backlog)
    if feed_path is None or not feed_path.exists():
        raise FileNotFoundError(f"dashboard feed not found: {feed_path}")
    if backlog_path is None or not backlog_path.exists():
        raise FileNotFoundError(f"incident backlog not found: {backlog_path}")

    baseline_path = _resolve_baseline_path(
        explicit_path=_resolve_path(args.baseline_json),
        weekly_dir=(ROOT / "enterprise" / "reports" / "weekly").resolve(),
        exclude_path=out_json,
    )
    baseline = _load_baseline_metrics(baseline_path)

    feed = _load_json(feed_path)
    backlog = _load_json(backlog_path)
    summary = build_weekly_summary(
        dashboard_feed=feed,
        incident_backlog=backlog,
        baseline=baseline,
        week_token=week_token,
        dashboard_feed_path=str(feed_path),
        incident_backlog_path=str(backlog_path),
    )
    markdown = render_weekly_markdown(summary)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_md": str(out_md),
                "out_json": str(out_json),
                "week": week_token,
                "baseline_status": summary.get("baseline_regression", {}).get("baseline_status", "missing"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
