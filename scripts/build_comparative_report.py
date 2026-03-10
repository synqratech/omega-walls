from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _latest_json(root: Path, filename: str = "report.json") -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted([p / filename for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _latest_contour_manifest(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted([p / "manifest.json" for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _extract_internal_summary(contour: Mapping[str, Any]) -> Dict[str, Any]:
    reports = contour.get("reports", {}) if isinstance(contour.get("reports"), Mapping) else {}
    out: Dict[str, Any] = {}
    for key in ("rule_cycle", "strict_pi", "attachment", "bipia"):
        rpt = reports.get(key)
        if isinstance(rpt, Mapping):
            out[key] = {
                "status": rpt.get("status"),
                "summary": rpt.get("summary", {}),
            }
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Comparative Report")
    lines.append("")
    lines.append(f"- run_id: `{payload.get('run_id')}`")
    lines.append(f"- generated_utc: `{payload.get('generated_utc')}`")
    lines.append("")

    lines.append("## Internal contour")
    contour = payload.get("internal_contour", {})
    if isinstance(contour, Mapping):
        for name, row in sorted((contour.get("summary", {}) or {}).items()):
            if not isinstance(row, Mapping):
                continue
            lines.append(f"- {name}: status=`{row.get('status')}`")
    lines.append("")

    for section_name in ("direct_comparison", "partial_comparison", "non_comparable"):
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        section = payload.get(section_name, [])
        if isinstance(section, list) and section:
            for row in section:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    f"- {row.get('benchmark')}: comparability=`{row.get('comparability_status')}`, source=`{row.get('source_url')}`"
                )
                omega = row.get("omega_metrics")
                if isinstance(omega, Mapping):
                    lines.append(
                        f"  - omega attack_off_rate=`{omega.get('attack_off_rate')}`, benign_off_rate=`{omega.get('benign_off_rate')}`"
                    )
                if row.get("benchmark") == "PINT" and isinstance(omega, Mapping):
                    lines.append(f"  - omega pint_score_pct=`{omega.get('pint_score_pct')}`")
        else:
            lines.append("- (none)")
        lines.append("")

    lines.append("## Methodology limits")
    limits = payload.get("methodology_limits", [])
    if isinstance(limits, list) and limits:
        for x in limits:
            lines.append(f"- {x}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_comparative_payload(
    *,
    run_id: str,
    contour_manifest: Optional[Mapping[str, Any]],
    pint_report: Optional[Mapping[str, Any]],
    wainject_report: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    direct: List[Dict[str, Any]] = []
    partial: List[Dict[str, Any]] = []
    non_comp: List[Dict[str, Any]] = []
    limits: List[str] = []

    if isinstance(pint_report, Mapping):
        pint_external = pint_report.get("external_benchmark", {}) if isinstance(pint_report.get("external_benchmark"), Mapping) else {}
        pint_row = {
            "benchmark": "PINT",
            "source_url": pint_external.get("source_url"),
            "source_date": pint_external.get("source_date"),
            "metric_mapping": pint_external.get("metric_mapping"),
            "comparability_status": (
                "direct_comparison"
                if bool(pint_report.get("dataset_ready", False)) and bool((pint_external.get("public_scoreboard", []) or []))
                else "non_comparable"
            ),
            "omega_metrics": pint_report.get("summary", {}),
            "public_baselines": pint_external.get("public_scoreboard", []),
        }
        if pint_row["comparability_status"] == "direct_comparison":
            direct.append(pint_row)
        else:
            non_comp.append(pint_row)
            limits.append("PINT local dataset was not available; direct head-to-head disabled.")
    else:
        limits.append("PINT report missing.")

    if isinstance(wainject_report, Mapping):
        wa_external = wainject_report.get("external_benchmark", {}) if isinstance(wainject_report.get("external_benchmark"), Mapping) else {}
        wa_row = {
            "benchmark": "WAInjectBench",
            "source_url": wa_external.get("source_url"),
            "source_date": wa_external.get("source_date"),
            "metric_mapping": wa_external.get("metric_mapping"),
            "comparability_status": "partial_comparison" if bool(wainject_report.get("dataset_ready", False)) else "non_comparable",
            "omega_metrics": wainject_report.get("summary", {}),
            "public_baselines": [],
        }
        if wa_row["comparability_status"] == "partial_comparison":
            partial.append(wa_row)
            limits.append("WAInjectBench has no benchmark-maintainer detector leaderboard table in official source card/readme.")
        else:
            non_comp.append(wa_row)
            limits.append("WAInjectBench local dataset was not available.")
    else:
        limits.append("WAInjectBench report missing.")

    if len(direct) == 0:
        limits.append("No direct external benchmark comparison available.")

    contour_ref = None
    contour_summary = {}
    if isinstance(contour_manifest, Mapping):
        contour_ref = contour_manifest.get("artifacts", {}).get("manifest_json")
        contour_summary = _extract_internal_summary(contour_manifest)

    return {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_bar": "benchmark-maintainer only",
        "internal_contour": {
            "manifest_ref": contour_ref,
            "summary": contour_summary,
        },
        "direct_comparison": direct,
        "partial_comparison": partial,
        "non_comparable": non_comp,
        "methodology_limits": sorted(set(str(x) for x in limits if str(x).strip())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified comparative report from internal contour and external benchmark anchors.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--contour-manifest", default=None)
    parser.add_argument("--pint-report", default=None)
    parser.add_argument("--wainject-report", default=None)
    parser.add_argument("--artifacts-root", default="artifacts/comparative_report")
    args = parser.parse_args()

    contour_path = (ROOT / str(args.contour_manifest)).resolve() if args.contour_manifest else _latest_contour_manifest((ROOT / "artifacts" / "post_patch_contour").resolve())
    pint_path = (ROOT / str(args.pint_report)).resolve() if args.pint_report else _latest_json((ROOT / "artifacts" / "pint_eval").resolve())
    wa_path = (ROOT / str(args.wainject_report)).resolve() if args.wainject_report else _latest_json((ROOT / "artifacts" / "wainject_eval").resolve())

    contour_manifest = _load_json(contour_path) if contour_path and contour_path.exists() else None
    pint_report = _load_json(pint_path) if pint_path and pint_path.exists() else None
    wainject_report = _load_json(wa_path) if wa_path and wa_path.exists() else None

    run_id = str(args.run_id or f"comparative_{_utc_compact_now()}")
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"

    payload = build_comparative_payload(
        run_id=run_id,
        contour_manifest=contour_manifest,
        pint_report=pint_report,
        wainject_report=wainject_report,
    )
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_render_markdown(payload), encoding="utf-8")

    out_payload = {
        "run_id": run_id,
        "status": "ok",
        "artifacts": {
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
        "inputs": {
            "contour_manifest": str(contour_path) if contour_path else None,
            "pint_report": str(pint_path) if pint_path else None,
            "wainject_report": str(wa_path) if wa_path else None,
        },
    }
    print(json.dumps(out_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
