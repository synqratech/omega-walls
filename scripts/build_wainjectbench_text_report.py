from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _clean_series_name(name: str) -> str:
    base = name.strip()
    if base.endswith(".jsonl"):
        base = base[:-6]
    return base.replace("_", " ")


def _metric_label(name: str) -> str:
    mapping = {
        "attack_off_rate": "Attack off rate",
        "benign_off_rate": "Benign off rate",
        "precision": "Precision",
        "recall": "Recall",
    }
    return mapping.get(name, _clean_series_name(name))


def _nice_axis_max(value: float) -> float:
    if value <= 0.0:
        return 0.05
    if value <= 0.01:
        return 0.01
    if value <= 0.02:
        return 0.02
    if value <= 0.05:
        return 0.05
    if value <= 0.10:
        return 0.10
    if value <= 0.20:
        return 0.20
    return 1.0


def _bar_chart_svg(
    *,
    title: str,
    subtitle: str,
    rows: Iterable[Tuple[str, float]],
    out_path: Path,
    max_value: float = 1.0,
    width: int = 1120,
    left_margin: int = 290,
    right_margin: int = 56,
    top_margin: int = 96,
    row_height: int = 40,
    bar_height: int = 18,
    tick_count: int = 4,
    bar_fill: str = "#2563eb",
) -> None:
    items = list(rows)
    n = len(items)
    value_col_w = 94
    height = top_margin + 44 + max(1, n) * row_height
    chart_w = width - left_margin - right_margin - value_col_w

    def x_of(v: float) -> float:
        if max_value <= 0:
            return float(left_margin)
        p = min(max(float(v) / float(max_value), 0.0), 1.0)
        return float(left_margin) + p * float(chart_w)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>')
    lines.append(
        f'<text x="{left_margin}" y="40" font-size="30" font-family="Segoe UI, Arial, sans-serif" fill="#0f172a">{_svg_escape(title)}</text>'
    )
    lines.append(
        f'<text x="{left_margin}" y="64" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#475569">{_svg_escape(subtitle)}</text>'
    )

    tick_count = max(2, tick_count)
    for i in range(0, tick_count + 1):
        v = max_value * i / float(tick_count)
        x = x_of(v)
        color = "#cbd5e1" if i in (0, tick_count) else "#e2e8f0"
        lines.append(
            f'<line x1="{x:.2f}" y1="{top_margin-12}" x2="{x:.2f}" y2="{height-24}" stroke="{color}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{height-6}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="#64748b">{_pct(v)}</text>'
        )

    y = top_margin
    for name, value in items:
        x_end = x_of(value)
        y_mid = y + row_height / 2.0
        bar_y = y_mid - bar_height / 2.0
        lines.append(
            f'<text x="{left_margin-12}" y="{y_mid+5:.2f}" text-anchor="end" font-size="15" font-family="Segoe UI, Arial, sans-serif" fill="#0f172a">{_svg_escape(_clean_series_name(name))}</text>'
        )
        lines.append(
            f'<rect x="{left_margin}" y="{bar_y:.2f}" width="{chart_w:.2f}" height="{bar_height}" fill="#e2e8f0" rx="6" ry="6"/>'
        )
        lines.append(
            f'<rect x="{left_margin}" y="{bar_y:.2f}" width="{max(0.0, x_end-left_margin):.2f}" height="{bar_height}" fill="{bar_fill}" rx="6" ry="6"/>'
        )
        lines.append(
            f'<text x="{left_margin + chart_w + 10:.2f}" y="{y_mid+5:.2f}" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#0f172a">{_pct(value)}</text>'
        )
        y += row_height

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _extract_attack_file_rows(per_file: Dict[str, Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for name, payload in sorted(per_file.items()):
        if float(payload.get("attack_total", 0)) <= 0:
            continue
        rows.append((name, float(payload.get("attack_off_rate", 0.0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _extract_benign_file_rows(per_file: Dict[str, Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for name, payload in sorted(per_file.items()):
        if float(payload.get("benign_total", 0)) <= 0:
            continue
        rows.append((name, float(payload.get("benign_off_rate", 0.0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _build_md(
    *,
    report: Dict[str, Any],
    report_rel: str,
    rows_rel: str,
    summary_chart_rel: str,
    attack_chart_rel: str,
    benign_chart_rel: str,
) -> str:
    run_id = str(report.get("run_id", "unknown"))
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    per_file = report.get("per_file", {}) if isinstance(report.get("per_file"), dict) else {}
    baseline = report.get("baseline_compare", {}) if isinstance(report.get("baseline_compare"), dict) else {}
    delta = baseline.get("summary_delta", {}) if isinstance(baseline.get("summary_delta"), dict) else {}

    attack_rows = _extract_attack_file_rows(per_file)
    benign_rows = _extract_benign_file_rows(per_file)

    worst_attack = [r for r in sorted(attack_rows, key=lambda x: x[1])[:3]]
    top_attack = [r for r in sorted(attack_rows, key=lambda x: x[1], reverse=True)[:3]]
    worst_benign = [r for r in sorted(benign_rows, key=lambda x: x[1], reverse=True)[:3]]

    lines: List[str] = []
    lines.append("# WAInjectBench Text Eval Report (Run-Frozen)")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- date_utc: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}`")
    lines.append(f"- status: `{report.get('status')}`")
    lines.append(f"- comparability_status: `{report.get('comparability_status')}`")
    lines.append(f"- source_root: `{report.get('root')}`")
    lines.append(f"- samples_total: `{report.get('samples_total')}`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue")
    lines.append(".\\.venv\\Scripts\\python.exe scripts/eval_wainjectbench_text.py `")
    lines.append("  --profile dev `")
    lines.append("  --root data/WAInjectBench/text `")
    lines.append("  --seed 41 `")
    lines.append("  --weekly-regression")
    lines.append("```")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| attack_off_rate | `{summary.get('attack_off_rate', 0.0):.6f}` |")
    lines.append(f"| benign_off_rate | `{summary.get('benign_off_rate', 0.0):.6f}` |")
    lines.append(f"| precision | `{summary.get('precision', 0.0):.6f}` |")
    lines.append(f"| recall | `{summary.get('recall', 0.0):.6f}` |")
    lines.append(f"| tp / fp / tn / fn | `{summary.get('tp', 0)}/{summary.get('fp', 0)}/{summary.get('tn', 0)}/{summary.get('fn', 0)}` |")
    lines.append("")
    lines.append("![summary metrics](../assets/wainjectbench_text_eval/" + summary_chart_rel + ")")
    lines.append("")
    lines.append("## Family/File Breakdown")
    lines.append("")
    lines.append("### Attack Recall by Malicious File")
    lines.append("")
    lines.append("![attack by file](../assets/wainjectbench_text_eval/" + attack_chart_rel + ")")
    lines.append("")
    lines.append("### Benign Off Rate by Benign File")
    lines.append("")
    lines.append("![benign by file](../assets/wainjectbench_text_eval/" + benign_chart_rel + ")")
    lines.append("")
    lines.append("## Interpretation (Honest)")
    lines.append("")
    lines.append(f"- High precision profile: `precision={summary.get('precision', 0.0):.4f}` with low but non-zero benign offs (`{summary.get('benign_off_rate', 0.0):.4f}`).")
    lines.append(f"- Recall is moderate on this slice: `attack_off_rate={summary.get('attack_off_rate', 0.0):.4f}`.")
    if worst_attack:
        lines.append("- Main miss families (lowest attack_off_rate files):")
        for name, val in worst_attack:
            lines.append(f"  - `{name}`: `{val:.4f}`")
    if top_attack:
        lines.append("- Strong files (highest attack_off_rate):")
        for name, val in top_attack:
            lines.append(f"  - `{name}`: `{val:.4f}`")
    if worst_benign:
        lines.append("- Main benign FP contributors:")
        for name, val in worst_benign:
            lines.append(f"  - `{name}` benign_off_rate: `{val:.4f}`")
    lines.append("")
    lines.append("## Delta vs baseline_compare")
    lines.append("")
    if delta:
        lines.append("| metric | delta |")
        lines.append("|---|---:|")
        for key in ("attack_off_rate", "benign_off_rate", "precision", "recall"):
            if key in delta:
                lines.append(f"| {key} | `{float(delta[key]):+.6f}` |")
    else:
        lines.append("No baseline delta block in source report.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- report_json: `{report_rel}`")
    lines.append(f"- rows_jsonl: `{rows_rel}`")
    lines.append("")
    lines.append("## Comparability Note")
    lines.append("")
    lines.append("- This run remains `partial_comparison` per benchmark metadata.")
    lines.append("- No benchmark-maintainer detector leaderboard table is attached to WAInjectBench source card/readme.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WAInjectBench text eval markdown + SVG charts from report.json")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--out-doc", default="docs/implementation/33_wainjectbench_text_eval_2026-03-09.md")
    parser.add_argument("--assets-dir", default="docs/assets/wainjectbench_text_eval")
    args = parser.parse_args()

    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = (ROOT / report_path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    run_id = str(payload.get("run_id", "unknown"))
    safe_run = _safe_name(run_id)
    assets_base = Path(args.assets_dir)
    if not assets_base.is_absolute():
        assets_base = (ROOT / assets_base).resolve()
    out_assets = assets_base / safe_run
    out_assets.mkdir(parents=True, exist_ok=True)

    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    per_file = payload.get("per_file", {}) if isinstance(payload.get("per_file"), dict) else {}

    summary_rows = [
        (_metric_label("attack_off_rate"), float(summary.get("attack_off_rate", 0.0))),
        (_metric_label("benign_off_rate"), float(summary.get("benign_off_rate", 0.0))),
        (_metric_label("precision"), float(summary.get("precision", 0.0))),
        (_metric_label("recall"), float(summary.get("recall", 0.0))),
    ]
    attack_rows = _extract_attack_file_rows(per_file)
    benign_rows = _extract_benign_file_rows(per_file)
    benign_axis_max = _nice_axis_max(max((val for _, val in benign_rows), default=0.0))

    summary_svg = out_assets / "summary_metrics.svg"
    attack_svg = out_assets / "attack_off_by_file.svg"
    benign_svg = out_assets / "benign_off_by_file.svg"

    _bar_chart_svg(
        title="WAInjectBench Text - Summary Metrics",
        subtitle=f"run_id: {run_id}",
        rows=summary_rows,
        out_path=summary_svg,
        max_value=1.0,
        tick_count=4,
        bar_fill="#1d4ed8",
    )
    _bar_chart_svg(
        title="Attack Off Rate by Malicious File",
        subtitle=f"run_id: {run_id}",
        rows=attack_rows,
        out_path=attack_svg,
        max_value=1.0,
        tick_count=4,
        bar_fill="#2563eb",
    )
    _bar_chart_svg(
        title="Benign Off Rate by Benign File",
        subtitle=f"run_id: {run_id} | axis max: {_pct(benign_axis_max)}",
        rows=benign_rows,
        out_path=benign_svg,
        max_value=benign_axis_max,
        tick_count=4,
        bar_fill="#0ea5e9",
    )

    out_doc = Path(args.out_doc)
    if not out_doc.is_absolute():
        out_doc = (ROOT / out_doc).resolve()
    out_doc.parent.mkdir(parents=True, exist_ok=True)

    rows_jsonl = str(payload.get("artifacts", {}).get("rows_jsonl", ""))
    report_rel = report_path.relative_to(ROOT).as_posix() if report_path.is_relative_to(ROOT) else str(report_path)
    rows_rel = (
        Path(rows_jsonl).resolve().relative_to(ROOT).as_posix()
        if rows_jsonl and Path(rows_jsonl).exists() and Path(rows_jsonl).resolve().is_relative_to(ROOT)
        else rows_jsonl
    )

    summary_rel = f"{safe_run}/summary_metrics.svg"
    attack_rel = f"{safe_run}/attack_off_by_file.svg"
    benign_rel = f"{safe_run}/benign_off_by_file.svg"
    md = _build_md(
        report=payload,
        report_rel=report_rel,
        rows_rel=rows_rel,
        summary_chart_rel=summary_rel,
        attack_chart_rel=attack_rel,
        benign_chart_rel=benign_rel,
    )
    out_doc.write_text(md, encoding="utf-8")

    result = {
        "status": "ok",
        "run_id": run_id,
        "report_json": str(report_path),
        "out_doc": str(out_doc),
        "assets_dir": str(out_assets),
        "assets": {
            "summary_metrics_svg": str(summary_svg),
            "attack_off_by_file_svg": str(attack_svg),
            "benign_off_by_file_svg": str(benign_svg),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
