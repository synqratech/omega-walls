from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = ROOT / "plugins" / "openclaw-omega-guard"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_compact() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _extract_json_blob(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("stdout does not contain JSON payload")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _run_command(
    *,
    name: str,
    argv: Sequence[str],
    cwd: Path,
    out_dir: Path,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
    )
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    parsed: Dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = _extract_json_blob(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
    return {
        "name": name,
        "argv": list(argv),
        "exit_code": int(proc.returncode),
        "cwd": str(cwd),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "parse_error": parse_error,
        "parsed_json": parsed,
    }


def _wait_for_healthz(*, base_url: str, timeout_sec: int) -> bool:
    deadline = time.time() + float(timeout_sec)
    url = f"{base_url.rstrip('/')}/healthz"
    while time.time() < deadline:
        try:
            req = urllib_request.Request(url=url, method="GET")
            with urllib_request.urlopen(req, timeout=1.5) as resp:  # noqa: S310
                if int(getattr(resp, "status", 0)) == 200:
                    return True
        except (urllib_error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _phase1(
    *,
    python_exec: str,
    profile: str,
    strict: bool,
    run_dir: Path,
) -> Dict[str, Any]:
    output_path = run_dir / "phase1_langchain.json"
    command = [
        python_exec,
        str((ROOT / "scripts" / "smoke_langchain_real_workflow.py").resolve()),
        "--profile",
        profile,
        "--projector-mode",
        "pi0",
        "--output",
        str(output_path),
    ]
    if strict:
        command.append("--strict")
    result = _run_command(name="phase1_langchain", argv=command, cwd=ROOT, out_dir=run_dir)
    payload: Dict[str, Any] = {}
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    elif isinstance(result.get("parsed_json"), dict):
        payload = dict(result["parsed_json"])
    return {
        "status": "ok" if int(result["exit_code"]) == 0 and payload.get("status") == "ok" else "fail",
        "command": result,
        "report_path": str(output_path),
        "report": payload,
    }


def _phase2(
    *,
    python_exec: str,
    profile: str,
    strict: bool,
    run_dir: Path,
    api_host: str,
    api_port: int,
) -> Dict[str, Any]:
    api_log_path = run_dir / "api.log"
    api_env = os.environ.copy()
    hmac_secret = secrets.token_hex(24)
    api_env["OMEGA_API_HMAC_SECRET"] = hmac_secret
    # Keep strict auth (HMAC+API key), but allow local non-TLS loopback for stand validation.
    api_env["OMEGA__API__SECURITY__REQUIRE_HTTPS"] = "false"
    api_env["OMEGA__API__SECURITY__TRANSPORT_MODE"] = "disabled"
    api_cmd = [
        python_exec,
        "-m",
        "omega.api.cli",
        "--profile",
        profile,
        "--host",
        api_host,
        "--port",
        str(int(api_port)),
    ]
    if strict:
        api_cmd.extend(["--no-proxy-headers"])

    api_proc: subprocess.Popen[str] | None = None
    hard_errors: list[str] = []
    command_runs: list[Dict[str, Any]] = []
    smoke_payload: Dict[str, Any] = {}
    local_api_payload: Dict[str, Any] = {}

    try:
        with api_log_path.open("w", encoding="utf-8") as api_log:
            api_proc = subprocess.Popen(  # noqa: S603
                api_cmd,
                cwd=str(ROOT),
                stdout=api_log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=api_env,
            )
            base_url = f"http://{api_host}:{int(api_port)}"
            if not _wait_for_healthz(base_url=base_url, timeout_sec=30):
                hard_errors.append("api_boot_failure")
                return {
                    "status": "fail",
                    "hard_errors": hard_errors,
                    "api_command": api_cmd,
                    "api_log": str(api_log_path),
                    "commands": [],
                    "smoke_payload": {},
                    "local_api_payload": {},
                }

            plugin_env = os.environ.copy()
            plugin_env["OMEGA_OPENCLAW_API_BASE_URL"] = base_url
            plugin_env["OMEGA_OPENCLAW_API_KEY"] = "dev-api-key"
            plugin_env["OMEGA_OPENCLAW_HMAC_SECRET"] = hmac_secret
            plugin_env["OMEGA_OPENCLAW_TENANT_ID"] = f"tenant-{_utc_compact()}"
            npm_bin = shutil.which("npm") or shutil.which("npm.cmd") or "npm"

            if not (PLUGIN_ROOT / "node_modules").exists():
                setup_result = _run_command(
                    name="phase2_npm_install",
                    argv=[npm_bin, "install", "--no-audit", "--no-fund"],
                    cwd=PLUGIN_ROOT,
                    out_dir=run_dir,
                    env=plugin_env,
                )
                command_runs.append(setup_result)
                if int(setup_result["exit_code"]) != 0:
                    return {
                        "status": "fail",
                        "hard_errors": hard_errors,
                        "api_command": api_cmd,
                        "api_log": str(api_log_path),
                        "commands": command_runs,
                        "smoke_payload": {},
                        "local_api_payload": {},
                    }

            commands = [
                ("phase2_typecheck", [npm_bin, "run", "typecheck"]),
                ("phase2_test", [npm_bin, "run", "test"]),
                ("phase2_build", [npm_bin, "run", "build"]),
                ("phase2_smoke", [npm_bin, "run", "smoke"]),
                ("phase2_smoke_local_api", [npm_bin, "run", "smoke:local-api"]),
            ]
            for name, argv in commands:
                result = _run_command(name=name, argv=argv, cwd=PLUGIN_ROOT, out_dir=run_dir, env=plugin_env)
                command_runs.append(result)
                if int(result["exit_code"]) != 0:
                    break
                if name == "phase2_smoke" and isinstance(result.get("parsed_json"), dict):
                    smoke_payload = dict(result["parsed_json"])
                if name == "phase2_smoke_local_api" and isinstance(result.get("parsed_json"), dict):
                    local_api_payload = dict(result["parsed_json"])
    finally:
        if api_proc is not None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()
                api_proc.wait(timeout=5)

    failed_commands = [row for row in command_runs if int(row.get("exit_code", 1)) != 0]
    status = "ok" if not failed_commands and not hard_errors else "fail"
    return {
        "status": status,
        "hard_errors": hard_errors,
        "api_command": api_cmd,
        "api_log": str(api_log_path),
        "commands": command_runs,
        "smoke_payload": smoke_payload,
        "local_api_payload": local_api_payload,
    }


def _gates(phase1: Mapping[str, Any], phase2: Mapping[str, Any]) -> Dict[str, bool]:
    phase1_summary = dict((phase1.get("report") or {}).get("summary", {}) or {})
    smoke_payload = dict(phase2.get("smoke_payload", {}) or {})
    local_api_payload = dict(phase2.get("local_api_payload", {}) or {})
    sample_block = dict(smoke_payload.get("sample_block_decision", {}) or {})
    sample_approval = dict(smoke_payload.get("sample_require_approval_decision", {}) or {})

    blocked_input_seen = bool(phase1_summary.get("blocked_input_seen"))
    blocked_tool_seen = bool(phase1_summary.get("blocked_tool_seen")) or bool(sample_block.get("block"))
    require_approval_seen = bool(phase1_summary.get("require_approval_seen")) or bool(sample_approval.get("requireApproval"))
    webfetch_guard_seen = bool(smoke_payload.get("webfetch_guard_seen"))
    outage_fail_closed_seen = bool(phase1_summary.get("outage_fail_closed_seen"))
    orphan_executions_zero = bool(phase1_summary.get("orphan_executions_zero"))
    gateway_coverage_ok = bool(phase1_summary.get("gateway_coverage_ok"))
    session_reset_seen = bool(local_api_payload.get("session_reset_seen"))

    return {
        "blocked_input_seen": blocked_input_seen,
        "blocked_tool_seen": blocked_tool_seen,
        "require_approval_seen": require_approval_seen,
        "webfetch_guard_seen": webfetch_guard_seen,
        "outage_fail_closed_seen": outage_fail_closed_seen,
        "orphan_executions_zero": orphan_executions_zero,
        "gateway_coverage_ok": gateway_coverage_ok,
        "session_reset_seen": session_reset_seen,
    }


def aggregate_status(
    *,
    strict: bool,
    phase1_status: str,
    phase2_status: str,
    gates: Mapping[str, bool],
    hard_errors: Sequence[str],
) -> str:
    if hard_errors:
        return "fail"
    all_gates = all(bool(v) for v in gates.values())
    if phase1_status == "ok" and phase2_status == "ok" and all_gates:
        return "ok"
    if strict:
        return "fail"
    return "partial"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real-workflow validation stand: LangChain mini-agent + OpenClaw plugin local e2e")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--artifacts-root", default="artifacts/real_agent_stand")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8080)
    args = parser.parse_args(argv)

    started = _utc_now()
    run_id = f"real_agent_stand_{_utc_compact()}"
    run_dir = Path(args.artifacts_root)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    phase1 = _phase1(python_exec=sys.executable, profile=str(args.profile), strict=bool(args.strict), run_dir=run_dir)
    phase2 = _phase2(
        python_exec=sys.executable,
        profile=str(args.profile),
        strict=bool(args.strict),
        run_dir=run_dir,
        api_host=str(args.api_host),
        api_port=int(args.api_port),
    )
    phase2_path = run_dir / "phase2_openclaw.json"
    phase2_path.write_text(json.dumps(phase2, ensure_ascii=True, indent=2), encoding="utf-8")
    gates = _gates(phase1, phase2)
    hard_errors = list(phase2.get("hard_errors", []) or [])
    overall_status = aggregate_status(
        strict=bool(args.strict),
        phase1_status=str(phase1.get("status", "fail")),
        phase2_status=str(phase2.get("status", "fail")),
        gates=gates,
        hard_errors=hard_errors,
    )

    report = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": _utc_now().isoformat(),
        "profile": str(args.profile),
        "strict": bool(args.strict),
        "artifacts_dir": str(run_dir),
        "phases": {
            "phase1_langchain": phase1,
            "phase2_openclaw": phase2,
        },
        "artifacts": {
            "phase1_langchain_json": str(run_dir / "phase1_langchain.json"),
            "phase2_openclaw_json": str(phase2_path),
            "api_log": str(run_dir / "api.log"),
        },
        "gates": gates,
        "overall_status": overall_status,
        "hard_errors": hard_errors,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if overall_status == "fail":
        return 1
    if bool(args.strict) and overall_status != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
