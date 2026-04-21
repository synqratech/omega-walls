from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema  # type: ignore


RUN_TS_RE = re.compile(r"(?P<ts>\d{8}T\d{6}Z)")


@dataclass(frozen=True)
class RunRef:
    run_id: str
    run_ts: datetime
    report_json: Path
    rows_jsonl: Path
    sessions_total: int
    attack_sessions: int
    benign_sessions: int


@dataclass(frozen=True)
class SessionOutcome:
    run_id: str
    session_id: str
    family: str
    bucket: str
    label_session: str
    detected_off: bool
    first_off_turn: Optional[int]
    max_turn_p: float
    attack_turn_events: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _parse_run_ts(run_id: str, fallback_mtime: float) -> datetime:
    m = RUN_TS_RE.search(str(run_id))
    if m:
        try:
            return datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(float(fallback_mtime), tz=timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _extract_run_ref(report_json: Path, rows_jsonl: Path) -> RunRef:
    report = _load_json(report_json)
    summary_all = report.get("summary_all", {}) if isinstance(report.get("summary_all"), Mapping) else {}
    run_id = str(report.get("run_id", report_json.parent.name)).strip() or report_json.parent.name
    run_ts = _parse_run_ts(run_id=run_id, fallback_mtime=report_json.stat().st_mtime)
    return RunRef(
        run_id=run_id,
        run_ts=run_ts,
        report_json=report_json.resolve(),
        rows_jsonl=rows_jsonl.resolve(),
        sessions_total=int(summary_all.get("sessions_total", 0) or 0),
        attack_sessions=int(summary_all.get("attack_sessions", 0) or 0),
        benign_sessions=int(summary_all.get("benign_sessions", 0) or 0),
    )


def discover_stateful_runs(runs_root: Path) -> List[RunRef]:
    refs: List[RunRef] = []
    for report_json in runs_root.rglob("report.json"):
        rows_jsonl = report_json.with_name("rows.jsonl")
        if not rows_jsonl.exists():
            continue
        try:
            refs.append(_extract_run_ref(report_json=report_json, rows_jsonl=rows_jsonl))
        except Exception:
            continue
    refs.sort(key=lambda x: x.run_ts, reverse=True)
    return refs


def _reason_signature(label_session: str, detected_off: bool, first_off_turn: Optional[int], max_turn_p: float) -> str:
    if label_session == "attack":
        if detected_off:
            turn_text = str(first_off_turn) if first_off_turn is not None else "unknown"
            return f"detected:off_turn:{turn_text}"
        if max_turn_p <= 0.05:
            return "no_off:maxp<=0.05"
        if max_turn_p <= 0.20:
            return "no_off:maxp<=0.20"
        return "no_off:maxp>0.20"
    if detected_off:
        turn_text = str(first_off_turn) if first_off_turn is not None else "unknown"
        return f"false_off:turn:{turn_text}"
    return "correct_benign"


def load_session_outcomes(rows_jsonl: Path, run_id: str) -> List[SessionOutcome]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ln in rows_jsonl.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        sid = str(row.get("session_id", "")).strip()
        if not sid:
            continue
        grouped[sid].append(row)

    out: List[SessionOutcome] = []
    for sid, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        rows_sorted = sorted(rows, key=lambda r: int(r.get("turn_id", 0)))
        head = rows_sorted[0]
        family = str(head.get("family", "unknown")).strip() or "unknown"
        bucket = str(head.get("bucket", "core")).strip() or "core"
        label = str(head.get("label_session", "benign")).strip().lower()
        label_session = "attack" if label == "attack" else "benign"
        detected_off = any(bool(r.get("off", False)) for r in rows_sorted)
        first_off_turn = None
        for r in rows_sorted:
            if bool(r.get("off", False)):
                first_off_turn = int(r.get("turn_id", 0))
                break
        max_turn_p = 0.0
        attack_turn_events = 0
        for r in rows_sorted:
            try:
                max_turn_p = max(max_turn_p, float(r.get("max_p", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            if str(r.get("label_turn", "")).strip().lower() == "attack":
                attack_turn_events += 1
        out.append(
            SessionOutcome(
                run_id=run_id,
                session_id=sid,
                family=family,
                bucket=bucket,
                label_session=label_session,
                detected_off=bool(detected_off),
                first_off_turn=first_off_turn,
                max_turn_p=float(max_turn_p),
                attack_turn_events=int(attack_turn_events),
            )
        )
    return out


def _resolve_baseline_path(value: Optional[str]) -> Optional[Path]:
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.exists() else None
    weekly_dir = ROOT / "enterprise" / "reports" / "weekly"
    if not weekly_dir.exists():
        return None
    candidates = sorted(weekly_dir.glob("weekly_security_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_baseline_metrics(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"status": "missing"}
    payload = _load_json(path)
    # v1 weekly summary format (planned by this implementation).
    if isinstance(payload.get("kpi", {}), Mapping):
        kpi = payload.get("kpi", {})
        return {
            "status": "provided",
            "source": str(path.resolve()),
            "attack_off_rate": float(kpi.get("attack_off_rate", 0.0)),
            "benign_off_rate": float(kpi.get("benign_off_rate", 0.0)),
        }
    # Fallback: direct keys.
    if "attack_off_rate" in payload and "benign_off_rate" in payload:
        return {
            "status": "provided",
            "source": str(path.resolve()),
            "attack_off_rate": float(payload.get("attack_off_rate", 0.0)),
            "benign_off_rate": float(payload.get("benign_off_rate", 0.0)),
        }
    # Fallback for known eval reports.
    summary = payload.get("summary_all", {}) if isinstance(payload.get("summary_all"), Mapping) else {}
    if summary:
        return {
            "status": "provided",
            "source": str(path.resolve()),
            "attack_off_rate": float(summary.get("session_attack_off_rate", 0.0)),
            "benign_off_rate": float(summary.get("session_benign_off_rate", 0.0)),
        }
    return {"status": "missing"}


def _top_clusters(rows: Iterable[SessionOutcome], *, top_n: int = 10) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    sample_sessions: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    sample_runs: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for row in rows:
        reason = _reason_signature(
            label_session=row.label_session,
            detected_off=row.detected_off,
            first_off_turn=row.first_off_turn,
            max_turn_p=row.max_turn_p,
        )
        key = (row.family, row.bucket, reason)
        counts[key] += 1
        if len(sample_sessions[key]) < 5 and row.session_id not in sample_sessions[key]:
            sample_sessions[key].append(row.session_id)
        if len(sample_runs[key]) < 5 and row.run_id not in sample_runs[key]:
            sample_runs[key].append(row.run_id)
    out: List[Dict[str, Any]] = []
    for (family, bucket, reason), cnt in counts.most_common(top_n):
        cluster_key = f"{family}|{bucket}|{reason}"
        out.append(
            {
                "cluster_key": cluster_key,
                "family": family,
                "bucket": bucket,
                "reason_signature": reason,
                "count": int(cnt),
                "sample_session_ids": list(sample_sessions[(family, bucket, reason)]),
                "sample_run_ids": list(sample_runs[(family, bucket, reason)]),
            }
        )
    return out


def build_dashboard_feed(
    *,
    runs: List[RunRef],
    outcomes: List[SessionOutcome],
    window_days: int,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    attacks = [x for x in outcomes if x.label_session == "attack"]
    benign = [x for x in outcomes if x.label_session == "benign"]

    tp = sum(1 for x in attacks if x.detected_off)
    fn = sum(1 for x in attacks if not x.detected_off)
    fp = sum(1 for x in benign if x.detected_off)
    tn = sum(1 for x in benign if not x.detected_off)

    attack_off_rate = _safe_div(tp, tp + fn)
    benign_off_rate = _safe_div(fp, fp + tn)

    family_counter = Counter(x.family for x in attacks)
    attack_total = sum(family_counter.values())
    family_split = [
        {
            "family": fam,
            "attack_sessions": int(cnt),
            "share": _safe_div(cnt, attack_total),
        }
        for fam, cnt in family_counter.most_common()
    ]

    fn_rows = [x for x in attacks if not x.detected_off]
    fp_rows = [x for x in benign if x.detected_off]

    alerts: List[Dict[str, Any]] = []
    if baseline.get("status") == "provided":
        base_attack = float(baseline.get("attack_off_rate", 0.0))
        base_benign = float(baseline.get("benign_off_rate", 0.0))
        attack_delta_pp = (attack_off_rate - base_attack) * 100.0
        benign_delta_pp = (benign_off_rate - base_benign) * 100.0
        if (base_attack - attack_off_rate) > 0.01:
            alerts.append(
                {
                    "id": "alert_attack_off_rate_breach",
                    "severity": "high",
                    "type": "attack_off_rate_breach",
                    "metric": "attack_off_rate",
                    "message": "attack_off_rate is below baseline by more than 1.0 p.p.",
                    "value": float(attack_off_rate),
                    "baseline": float(base_attack),
                    "delta_pp": float(attack_delta_pp),
                    "threshold_pp": -1.0,
                }
            )
        if (benign_off_rate - base_benign) > 0.005:
            alerts.append(
                {
                    "id": "alert_benign_off_rate_breach",
                    "severity": "high",
                    "type": "benign_off_rate_breach",
                    "metric": "benign_off_rate",
                    "message": "benign_off_rate is above baseline by more than 0.5 p.p.",
                    "value": float(benign_off_rate),
                    "baseline": float(base_benign),
                    "delta_pp": float(benign_delta_pp),
                    "threshold_pp": 0.5,
                }
            )

    now = _utc_now()
    from_ts = now - timedelta(days=int(window_days))
    feed: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_iso(now),
        "window": {
            "days": int(window_days),
            "from_utc": _utc_iso(from_ts),
            "to_utc": _utc_iso(now),
            "runs_considered": int(len(runs)),
            "sessions_considered": int(len(outcomes)),
        },
        "source_runs": [
            {
                "run_id": x.run_id,
                "run_timestamp_utc": _utc_iso(x.run_ts),
                "report_json": str(x.report_json),
                "rows_jsonl": str(x.rows_jsonl),
                "sessions_total": int(x.sessions_total),
                "attack_sessions": int(x.attack_sessions),
                "benign_sessions": int(x.benign_sessions),
            }
            for x in runs
        ],
        "kpi": {
            "attack_volume": {
                "attack_sessions": int(len(attacks)),
                "attack_turn_events": int(sum(x.attack_turn_events for x in attacks)),
            },
            "family_split": family_split,
            "attack_off_rate": float(attack_off_rate),
            "benign_off_rate": float(benign_off_rate),
            "confusion": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "attack_sessions": int(len(attacks)),
                "benign_sessions": int(len(benign)),
            },
        },
        "mismatch_clusters": {
            "fn_clusters": _top_clusters(fn_rows, top_n=10),
            "fp_clusters": _top_clusters(fp_rows, top_n=10),
        },
        "alerts": alerts,
        "baseline": baseline,
    }
    return feed


def _render_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "<p>(none)</p>"
    parts = ["<table><thead><tr>"]
    for col in cols:
        parts.append(f"<th>{col}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                text = f"{val:.4f}"
            elif isinstance(val, list):
                text = ", ".join(str(x) for x in val)
            else:
                text = str(val)
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_dashboard_html(feed: Mapping[str, Any]) -> str:
    kpi = feed.get("kpi", {}) if isinstance(feed.get("kpi"), Mapping) else {}
    volume = kpi.get("attack_volume", {}) if isinstance(kpi.get("attack_volume"), Mapping) else {}
    confusion = kpi.get("confusion", {}) if isinstance(kpi.get("confusion"), Mapping) else {}
    mismatch = feed.get("mismatch_clusters", {}) if isinstance(feed.get("mismatch_clusters"), Mapping) else {}
    fn_clusters = mismatch.get("fn_clusters", []) if isinstance(mismatch.get("fn_clusters"), list) else []
    fp_clusters = mismatch.get("fp_clusters", []) if isinstance(mismatch.get("fp_clusters"), list) else []
    family_split = kpi.get("family_split", []) if isinstance(kpi.get("family_split"), list) else []
    alerts = feed.get("alerts", []) if isinstance(feed.get("alerts"), list) else []

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Enterprise Attack Dashboard v1</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    .kpi {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }}
    .card .v {{ font-size: 24px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 13px; }}
    h2 {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Enterprise Attack Dashboard v1</h1>
  <p>generated_at_utc: {feed.get("generated_at_utc","")}</p>
  <div class="kpi">
    <div class="card"><div>Attack sessions</div><div class="v">{int(volume.get("attack_sessions",0))}</div></div>
    <div class="card"><div>Attack off rate</div><div class="v">{float(kpi.get("attack_off_rate",0.0))*100.0:.2f}%</div></div>
    <div class="card"><div>Benign off rate</div><div class="v">{float(kpi.get("benign_off_rate",0.0))*100.0:.2f}%</div></div>
    <div class="card"><div>Critical alerts</div><div class="v">{sum(1 for a in alerts if str(a.get("severity",""))=="high")}</div></div>
  </div>

  <h2>Family Split</h2>
  {_render_table(family_split, ["family","attack_sessions","share"])}

  <h2>Confusion</h2>
  {_render_table([dict(confusion)], ["tp","fp","tn","fn","attack_sessions","benign_sessions"])}

  <h2>Top FN Clusters</h2>
  {_render_table(fn_clusters, ["cluster_key","count","sample_session_ids"])}

  <h2>Top FP Clusters</h2>
  {_render_table(fp_clusters, ["cluster_key","count","sample_session_ids"])}

  <h2>Alerts</h2>
  {_render_table(alerts, ["id","severity","type","metric","message","delta_pp"])}
</body>
</html>
"""


def validate_feed_schema(feed: Mapping[str, Any], schema_path: Path) -> None:
    schema = _load_json(schema_path)
    jsonschema.validate(instance=dict(feed), schema=schema)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enterprise attack dashboard feed + static HTML from stateful eval runs.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--out-json", default="enterprise/monitoring/out/dashboard_feed.json")
    parser.add_argument("--out-html", default="enterprise/monitoring/out/dashboard.html")
    parser.add_argument("--schema", default="schemas/enterprise_attack_dashboard_feed_v1.json")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (ROOT / runs_root).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs-root not found: {runs_root}")

    window_days = max(1, int(args.window_days))
    now = _utc_now()
    from_ts = now - timedelta(days=window_days)

    discovered = discover_stateful_runs(runs_root)
    runs = [x for x in discovered if x.run_ts >= from_ts]

    outcomes: List[SessionOutcome] = []
    for run in runs:
        outcomes.extend(load_session_outcomes(run.rows_jsonl, run.run_id))

    baseline_path = _resolve_baseline_path(args.baseline_json)
    baseline = load_baseline_metrics(baseline_path)

    feed = build_dashboard_feed(
        runs=runs,
        outcomes=outcomes,
        window_days=window_days,
        baseline=baseline,
    )

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = (ROOT / schema_path).resolve()
    validate_feed_schema(feed, schema_path)

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(feed, ensure_ascii=False, indent=2), encoding="utf-8")

    out_html = Path(args.out_html)
    if not out_html.is_absolute():
        out_html = (ROOT / out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_dashboard_html(feed), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_json": str(out_json),
                "out_html": str(out_html),
                "runs_considered": len(runs),
                "sessions_considered": len(outcomes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
