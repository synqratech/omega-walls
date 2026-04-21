from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent

SEVERITY_ORDER = {"SEV-1": 0, "SEV-2": 1, "SEV-3": 2}

SEVERITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "SEV-1": {
        "owner": "Incident Commander",
        "sla": {
            "first_response": "15 min",
            "mitigation": "60 min",
            "rollback_decision": "120 min",
        },
        "escalation_path": "IC -> Head of Engineering -> Executive on-call",
    },
    "SEV-2": {
        "owner": "Detection Engineering",
        "sla": {
            "first_response": "60 min",
            "mitigation": "1 business day",
            "rollback_decision": "1 business day",
        },
        "escalation_path": "IC -> Detection Lead -> Product owner",
    },
    "SEV-3": {
        "owner": "Security Operations",
        "sla": {
            "first_response": "4 business hours",
            "mitigation": "3 business days",
            "rollback_decision": "optional",
        },
        "escalation_path": "Team owner + weekly review",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _next_incident_id(prefix: str, index: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{now}-{index:04d}"


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _cluster_severity(
    *,
    count: int,
    has_critical_breach: bool,
    sev2_cluster_count_threshold: int,
) -> str:
    if count >= sev2_cluster_count_threshold and not has_critical_breach:
        return "SEV-2"
    return "SEV-3"


def _build_alert_incident(
    *,
    alert: Mapping[str, Any],
    idx: int,
    high_volume: bool,
) -> Dict[str, Any]:
    severity = "SEV-1" if high_volume else "SEV-2"
    profile = SEVERITY_PROFILES[severity]
    metric = str(alert.get("metric", "unknown"))
    reason = str(alert.get("message", "metric breach"))
    evidence = [
        f"alert_id:{str(alert.get('id', 'unknown'))}",
        f"alert_type:{str(alert.get('type', 'unknown'))}",
        f"metric:{metric}",
    ]
    if "value" in alert:
        evidence.append(f"value:{alert.get('value')}")
    if "baseline" in alert:
        evidence.append(f"baseline:{alert.get('baseline')}")
    if "delta_pp" in alert:
        evidence.append(f"delta_pp:{alert.get('delta_pp')}")

    return {
        "incident_id": _next_incident_id("INC", idx),
        "severity": severity,
        "category": "metric_breach",
        "title": f"Metric breach: {metric}",
        "reason": reason,
        "owner": profile["owner"],
        "sla": profile["sla"],
        "escalation_path": profile["escalation_path"],
        "evidence_refs": evidence,
        "status": "open",
        "created_at_utc": _utc_now(),
    }


def _build_cluster_incident(
    *,
    cluster: Mapping[str, Any],
    kind: str,
    idx: int,
    has_critical_breach: bool,
    sev2_cluster_count_threshold: int,
) -> Dict[str, Any]:
    count = int(cluster.get("count", 0) or 0)
    severity = _cluster_severity(
        count=count,
        has_critical_breach=has_critical_breach,
        sev2_cluster_count_threshold=sev2_cluster_count_threshold,
    )
    profile = SEVERITY_PROFILES[severity]
    cluster_key = str(cluster.get("cluster_key", "unknown"))
    family = str(cluster.get("family", "unknown"))
    bucket = str(cluster.get("bucket", "unknown"))
    reason_sig = str(cluster.get("reason_signature", "unknown"))
    sample_sessions = _as_list(cluster.get("sample_session_ids"))
    sample_runs = _as_list(cluster.get("sample_run_ids"))

    evidence = [
        f"cluster_key:{cluster_key}",
        f"kind:{kind}",
        f"count:{count}",
        f"family:{family}",
        f"bucket:{bucket}",
        f"reason_signature:{reason_sig}",
    ]
    for sid in sample_sessions[:5]:
        evidence.append(f"session_id:{sid}")
    for rid in sample_runs[:5]:
        evidence.append(f"run_id:{rid}")

    return {
        "incident_id": _next_incident_id("INC", idx),
        "severity": severity,
        "category": f"{kind}_cluster",
        "title": f"{kind.upper()} cluster: {cluster_key}",
        "reason": f"{kind.upper()} cluster count={count} in window",
        "owner": profile["owner"],
        "sla": profile["sla"],
        "escalation_path": profile["escalation_path"],
        "evidence_refs": evidence,
        "status": "open",
        "created_at_utc": _utc_now(),
    }


def build_incident_backlog(
    *,
    dashboard_feed: Mapping[str, Any],
    dashboard_feed_path: str,
    high_volume_attack_sessions: int = 100,
    sev2_cluster_count_threshold: int = 8,
) -> Dict[str, Any]:
    kpi = dashboard_feed.get("kpi", {}) if isinstance(dashboard_feed.get("kpi"), Mapping) else {}
    volume = kpi.get("attack_volume", {}) if isinstance(kpi.get("attack_volume"), Mapping) else {}
    mismatch = (
        dashboard_feed.get("mismatch_clusters", {})
        if isinstance(dashboard_feed.get("mismatch_clusters"), Mapping)
        else {}
    )
    alerts = _as_list(dashboard_feed.get("alerts"))
    fn_clusters = _as_list(mismatch.get("fn_clusters"))
    fp_clusters = _as_list(mismatch.get("fp_clusters"))

    attack_sessions = int(volume.get("attack_sessions", 0) or 0)
    high_volume = attack_sessions >= int(high_volume_attack_sessions)
    critical_alerts = [
        a
        for a in alerts
        if isinstance(a, Mapping)
        and str(a.get("type", "")) in {"attack_off_rate_breach", "benign_off_rate_breach"}
    ]
    has_critical_breach = len(critical_alerts) > 0

    incidents: List[Dict[str, Any]] = []
    idx = 1

    for alert in critical_alerts:
        incidents.append(_build_alert_incident(alert=alert, idx=idx, high_volume=high_volume))
        idx += 1

    for cluster in fn_clusters:
        if not isinstance(cluster, Mapping):
            continue
        incidents.append(
            _build_cluster_incident(
                cluster=cluster,
                kind="fn",
                idx=idx,
                has_critical_breach=has_critical_breach,
                sev2_cluster_count_threshold=int(sev2_cluster_count_threshold),
            )
        )
        idx += 1

    for cluster in fp_clusters:
        if not isinstance(cluster, Mapping):
            continue
        incidents.append(
            _build_cluster_incident(
                cluster=cluster,
                kind="fp",
                idx=idx,
                has_critical_breach=has_critical_breach,
                sev2_cluster_count_threshold=int(sev2_cluster_count_threshold),
            )
        )
        idx += 1

    incidents.sort(
        key=lambda row: (
            SEVERITY_ORDER.get(str(row.get("severity", "SEV-3")), 99),
            -len(row.get("evidence_refs", [])),
            str(row.get("incident_id", "")),
        )
    )

    sev_counts = {"SEV-1": 0, "SEV-2": 0, "SEV-3": 0}
    for row in incidents:
        sev = str(row.get("severity", "SEV-3"))
        if sev in sev_counts:
            sev_counts[sev] += 1

    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "source_dashboard_feed": dashboard_feed_path,
        "summary": {
            "incidents_total": int(len(incidents)),
            "sev_counts": sev_counts,
            "high_volume_attack_sessions": bool(high_volume),
            "critical_breach_present": bool(has_critical_breach),
        },
        "incidents": incidents,
    }


def render_incident_markdown(backlog: Mapping[str, Any]) -> str:
    summary = backlog.get("summary", {}) if isinstance(backlog.get("summary"), Mapping) else {}
    incidents = _as_list(backlog.get("incidents"))

    lines: List[str] = []
    lines.append("# Incident Backlog v1")
    lines.append("")
    lines.append(f"- generated_at_utc: {backlog.get('generated_at_utc', '')}")
    lines.append(f"- source_dashboard_feed: {backlog.get('source_dashboard_feed', '')}")
    lines.append(f"- incidents_total: {int(summary.get('incidents_total', 0) or 0)}")
    sev_counts = summary.get("sev_counts", {}) if isinstance(summary.get("sev_counts"), Mapping) else {}
    lines.append(
        "- sev_counts: "
        f"SEV-1={int(sev_counts.get('SEV-1', 0) or 0)}, "
        f"SEV-2={int(sev_counts.get('SEV-2', 0) or 0)}, "
        f"SEV-3={int(sev_counts.get('SEV-3', 0) or 0)}"
    )
    lines.append("")
    lines.append("## Open Incidents")
    lines.append("")

    if not incidents:
        lines.append("- none")
        lines.append("")
        return "\n".join(lines)

    lines.append("| incident_id | severity | category | owner | SLA (first response) | reason |")
    lines.append("|---|---|---|---|---|---|")
    for row in incidents:
        sla = row.get("sla", {}) if isinstance(row.get("sla"), Mapping) else {}
        reason = str(row.get("reason", "")).replace("|", "/")
        lines.append(
            f"| {row.get('incident_id', '')} | {row.get('severity', '')} | "
            f"{row.get('category', '')} | {row.get('owner', '')} | "
            f"{sla.get('first_response', '')} | {reason} |"
        )

    lines.append("")
    lines.append("## Evidence Refs")
    lines.append("")
    for row in incidents:
        lines.append(f"### {row.get('incident_id', '')}")
        evidence = _as_list(row.get("evidence_refs"))
        if not evidence:
            lines.append("- (none)")
            continue
        for item in evidence:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def _resolve_path(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enterprise incident triage backlog from dashboard feed.")
    parser.add_argument("--dashboard-feed", required=True)
    parser.add_argument("--out-json", default="enterprise/incident/out/incident_backlog.json")
    parser.add_argument("--out-md", default="enterprise/incident/out/incident_backlog.md")
    parser.add_argument("--high-volume-attack-sessions", type=int, default=100)
    parser.add_argument("--sev2-cluster-count-threshold", type=int, default=8)
    args = parser.parse_args()

    feed_path = _resolve_path(args.dashboard_feed)
    if not feed_path.exists():
        raise FileNotFoundError(f"dashboard feed not found: {feed_path}")
    feed = _load_json(feed_path)

    backlog = build_incident_backlog(
        dashboard_feed=feed,
        dashboard_feed_path=str(feed_path),
        high_volume_attack_sessions=max(1, int(args.high_volume_attack_sessions)),
        sev2_cluster_count_threshold=max(1, int(args.sev2_cluster_count_threshold)),
    )
    markdown = render_incident_markdown(backlog)

    out_json = _resolve_path(args.out_json)
    out_md = _resolve_path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(backlog, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_json": str(out_json),
                "out_md": str(out_md),
                "incidents_total": int(backlog.get("summary", {}).get("incidents_total", 0)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

