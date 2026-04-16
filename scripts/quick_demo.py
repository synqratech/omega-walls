from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PACK = "tests/data/session_benchmark/agentdojo_cocktail_mini_smoke_v1.jsonl"
DEFAULT_ARTIFACTS_ROOT = "artifacts/quick_demo"
DEFAULT_API_MODEL = "gpt-5.4-mini"
DEFAULT_PROFILE = "dev"
DEFAULT_AGENTDOJO_RUNS_ROOT = "data/AgentDojo/runs"

EXIT_OK = 0
EXIT_MISSING_KEY = 2
EXIT_RUNTIME_ERROR = 3


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "\"":
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:  # noqa: BLE001
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
    return None


def _run_cmd(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_report_path(payload: Mapping[str, Any], artifacts_root: Path) -> Optional[Path]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        report_json = artifacts.get("report_json")
        if report_json:
            path = Path(str(report_json)).resolve()
            if path.exists():
                return path

    if artifacts_root.exists():
        candidates = sorted(
            (p for p in artifacts_root.glob("**/report.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
    return None


def _print_runtime_failure(*, title: str, proc: subprocess.CompletedProcess[str]) -> None:
    print(f"[quick-demo] {title}", file=sys.stderr)
    if proc.stdout.strip():
        print("[quick-demo] stdout:", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
    if proc.stderr.strip():
        print("[quick-demo] stderr:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)


def _build_pack_if_needed(args: argparse.Namespace, artifacts_root: Path) -> Optional[Path]:
    if str(args.dataset_source) == "built_in":
        pack_path = (ROOT / str(args.pack)).resolve()
        if not pack_path.exists():
            print(f"[quick-demo] Pack not found: {pack_path}", file=sys.stderr)
            return None
        return pack_path

    pack_dir = (artifacts_root / "generated_pack").resolve()
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_path = (pack_dir / "agentdojo_cocktail_mini_generated.jsonl").resolve()
    meta_path = pack_path.with_suffix(".meta.json")

    argv = [
        sys.executable,
        str((ROOT / "scripts" / "build_agentdojo_cocktail_mini_pack.py").resolve()),
        "--runs-root",
        str((ROOT / str(args.agentdojo_runs_root)).resolve()),
        "--out",
        str(pack_path),
        "--meta-out",
        str(meta_path),
    ]
    proc = _run_cmd(argv)
    if proc.returncode != 0:
        _print_runtime_failure(title="AgentDojo mini pack build failed.", proc=proc)
        return None
    payload = _extract_first_json_object(proc.stdout)
    if not isinstance(payload, dict):
        print("[quick-demo] AgentDojo mini pack build produced non-JSON output.", file=sys.stderr)
        return None
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        print(f"[quick-demo] AgentDojo mini pack build status is not ok: {status}", file=sys.stderr)
        return None
    if not pack_path.exists():
        print(f"[quick-demo] Generated pack path missing: {pack_path}", file=sys.stderr)
        return None
    return pack_path


def _semantic_fallback_active(report: Mapping[str, Any]) -> bool:
    projector = report.get("projector", {})
    if not isinstance(projector, Mapping):
        return False
    core_runtime = projector.get("core_runtime", {})
    cross_runtime = projector.get("cross_runtime", {})
    core_active = (
        core_runtime.get("semantic_active")
        if isinstance(core_runtime, Mapping)
        else None
    )
    cross_active = (
        cross_runtime.get("semantic_active")
        if isinstance(cross_runtime, Mapping)
        else None
    )
    return (core_active is False) or (cross_active is False)


def run_quick_demo(args: argparse.Namespace) -> int:
    if str(args.mode) == "hybrid_api":
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            print(
                "[quick-demo] OPENAI_API_KEY is required for mode=hybrid_api.\n"
                "Set it before running:\n"
                "  PowerShell: $env:OPENAI_API_KEY='sk-...'\n"
                "  Bash: export OPENAI_API_KEY='sk-...'",
                file=sys.stderr,
            )
            return EXIT_MISSING_KEY

    artifacts_root = (ROOT / str(args.artifacts_root)).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    pack_path = _build_pack_if_needed(args, artifacts_root)
    if pack_path is None:
        return EXIT_RUNTIME_ERROR

    eval_argv = [
        sys.executable,
        str((ROOT / "scripts" / "eval_agentdojo_stateful_mini.py").resolve()),
        "--profile",
        str(args.profile),
        "--mode",
        str(args.mode),
        "--pack",
        str(pack_path),
        "--artifacts-root",
        str(artifacts_root),
    ]
    if bool(args.strict_projector):
        eval_argv.append("--strict-projector")
    if args.api_model:
        eval_argv.extend(["--api-model", str(args.api_model)])

    proc = _run_cmd(eval_argv)
    if proc.returncode != 0:
        _print_runtime_failure(title="Demo eval run failed.", proc=proc)
        return EXIT_RUNTIME_ERROR

    payload = _extract_first_json_object(proc.stdout)
    if not isinstance(payload, dict):
        print("[quick-demo] Eval output is not parseable JSON.", file=sys.stderr)
        if proc.stdout.strip():
            print(proc.stdout, file=sys.stderr)
        if proc.stderr.strip():
            print(proc.stderr, file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    report_path = _resolve_report_path(payload, artifacts_root)
    if report_path is None or not report_path.exists():
        print("[quick-demo] report.json not found after eval run.", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[quick-demo] report.json parse failed: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    if not isinstance(report, dict):
        print("[quick-demo] report.json must contain a JSON object.", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    summary_all = report.get("summary_all", {})
    if not isinstance(summary_all, Mapping):
        print("[quick-demo] report.json missing summary_all object.", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    cocktail = report.get("cocktail_metrics", {})
    if not isinstance(cocktail, Mapping):
        cocktail = {}

    attack_off = _safe_float(summary_all.get("session_attack_off_rate"))
    benign_off = _safe_float(summary_all.get("session_benign_off_rate"))
    mssr_core = _safe_float(cocktail.get("mssr_core"))
    mssr_cross_primary = _safe_float(cocktail.get("mssr_cross_primary"))
    tp = _safe_int(summary_all.get("tp"))
    blocked_observed = tp > 0

    print("Quick Demo Summary")
    print(f"- mode: {args.mode}")
    print(f"- pack: {pack_path}")
    print(f"- session_attack_off_rate: {attack_off:.4f}")
    print(f"- session_benign_off_rate: {benign_off:.4f}")
    print(f"- mssr_core: {mssr_core:.4f}")
    print(f"- mssr_cross_primary: {mssr_cross_primary:.4f}")
    if blocked_observed:
        print("- blocked behavior observed: yes")
    else:
        print("- blocked behavior observed: no")
    print(f"- artifacts: {report_path.parent}")

    if _semantic_fallback_active(report):
        print("WARNING: semantic fallback active (semantic_active=false).")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="5-minute OSS quick demo: run AgentDojo mini pack and print compact block summary."
    )
    parser.add_argument(
        "--mode",
        default="hybrid_api",
        choices=["hybrid_api", "pi0", "hybrid"],
        help="Runtime mode for quick demo (default: hybrid_api).",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument(
        "--strict-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict projector mode (default: true).",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["built_in", "agentdojo_runs"],
        default="built_in",
        help="Dataset source for demo pack (default: built_in).",
    )
    parser.add_argument("--pack", default=DEFAULT_PACK, help="Path to prebuilt session pack for built_in mode.")
    parser.add_argument(
        "--agentdojo-runs-root",
        default=DEFAULT_AGENTDOJO_RUNS_ROOT,
        help="Path to AgentDojo runs root used only when --dataset-source=agentdojo_runs.",
    )
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_quick_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
