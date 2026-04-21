from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_support_stateful_vs_stateless import run_eval  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    overall = report.get("metrics", {}).get("overall", {})
    stateful = overall.get("stateful_target", {})
    baseline_d = overall.get("baseline_d_bare_llm_detector", None)
    baseline_d_row = (
        {
            "session_attack_off_rate": float(baseline_d.get("session_attack_off_rate", 0.0) or 0.0),
            "session_benign_off_rate": float(baseline_d.get("session_benign_off_rate", 0.0) or 0.0),
        }
        if isinstance(baseline_d, dict)
        else None
    )
    return {
        "generated_at_utc": _utc_now_iso(),
        "run_id": report.get("run_id"),
        "stateful_target": {
            "session_attack_off_rate": float(stateful.get("session_attack_off_rate", 0.0) or 0.0),
            "session_benign_off_rate": float(stateful.get("session_benign_off_rate", 0.0) or 0.0),
        },
        "baseline_d_bare_llm_detector": baseline_d_row,
        "comparisons_matched_benign_rate": report.get("comparisons_matched_benign_rate", {}),
        "artifacts": report.get("artifacts", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run blind red-team holdout eval (stateful vs stateless + optional bare LLM D).")
    parser.add_argument("--packs-root", required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--stateful-mode", choices=["pi0", "hybrid", "hybrid_api"], default="hybrid_api")
    parser.add_argument("--strict-projector", action="store_true")
    parser.add_argument("--allow-api-fallback", action="store_true")
    parser.add_argument("--require-semantic-active", action="store_true")
    parser.add_argument("--enable-stateful-support-tuning", action="store_true")
    parser.add_argument("--api-model", default="gpt-5.4-mini")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-timeout-sec", type=float, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--api-cache-path", default=None)
    parser.add_argument("--api-error-log-path", default=None)
    parser.add_argument("--baseline-b-calibration", choices=["benign_q95"], default="benign_q95")
    parser.add_argument("--baseline-c-mode", choices=["prefix_scan"], default="prefix_scan")
    parser.add_argument("--baseline-d-enable", action="store_true")
    parser.add_argument("--baseline-d-model", default="gpt-5.4-mini")
    parser.add_argument("--baseline-d-base-url", default=None)
    parser.add_argument("--baseline-d-timeout-sec", type=float, default=None)
    parser.add_argument("--baseline-d-retries", type=int, default=None)
    parser.add_argument("--baseline-d-calibration", choices=["benign_q95"], default="benign_q95")
    parser.add_argument("--baseline-d-mode", choices=["per_turn_only"], default="per_turn_only")
    parser.add_argument("--artifacts-root", default="artifacts/support_family_eval_compare_blind")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    report = run_eval(
        packs_root=(ROOT / str(args.packs_root)).resolve(),
        profile=str(args.profile),
        stateful_mode=str(args.stateful_mode),
        strict_projector=bool(args.strict_projector),
        allow_api_fallback=bool(args.allow_api_fallback),
        require_semantic_active=bool(args.require_semantic_active),
        enable_stateful_support_tuning=bool(args.enable_stateful_support_tuning),
        api_model=(str(args.api_model) if args.api_model else None),
        api_base_url=(str(args.api_base_url) if args.api_base_url else None),
        api_timeout_sec=(float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
        api_retries=(int(args.api_retries) if args.api_retries is not None else None),
        api_cache_path=(str(args.api_cache_path) if args.api_cache_path else None),
        api_error_log_path=(str(args.api_error_log_path) if args.api_error_log_path else None),
        baseline_b_calibration=str(args.baseline_b_calibration),
        baseline_c_mode=str(args.baseline_c_mode),
        artifacts_root=(ROOT / str(args.artifacts_root)).resolve(),
        seed=int(args.seed),
        baseline_d_enable=bool(args.baseline_d_enable),
        baseline_d_model=(str(args.baseline_d_model) if args.baseline_d_model else None),
        baseline_d_base_url=(str(args.baseline_d_base_url) if args.baseline_d_base_url else None),
        baseline_d_timeout_sec=(float(args.baseline_d_timeout_sec) if args.baseline_d_timeout_sec is not None else None),
        baseline_d_retries=(int(args.baseline_d_retries) if args.baseline_d_retries is not None else None),
        baseline_d_calibration=str(args.baseline_d_calibration),
        baseline_d_mode=str(args.baseline_d_mode),
    )
    summary = _summary_from_report(report)
    summary_path = Path(str(report.get("artifacts", {}).get("report_json", ""))).resolve().parent / "blind_holdout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": report, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
