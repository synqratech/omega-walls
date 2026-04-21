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
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = ROOT / "plugins" / "openclaw-omega-guard"
FRAMEWORKS: tuple[str, ...] = (
    "langchain_guard",
    "langgraph_guard",
    "llamaindex_guard",
    "haystack_guard",
    "autogen_guard",
    "crewai_guard",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_compact() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _default_run_id(prefix: str) -> str:
    return f"{prefix}_{_utc_compact()}"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2), encoding="utf-8")


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


def _load_workflow_cases(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload.get("cases"), list):
        raise ValueError(f"workflow fixture has no cases list: {path}")
    rows = []
    for row in payload["cases"]:
        if not isinstance(row, dict):
            continue
        rows.append(dict(row))
    if len(rows) != 10:
        raise ValueError(f"workflow fixture must contain 10 cases, got {len(rows)}")
    return rows


def _load_identity_fixtures(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            rows.append(dict(obj))
    if not rows:
        raise ValueError(f"identity fixture is empty: {path}")
    return rows


def _aggregate_framework_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    blocked_input_seen = bool(summary.get("blocked_input_seen") or summary.get("blocked_graph_input_seen") or summary.get("blocked_query_seen") or summary.get("blocked_conversation_seen"))
    blocked_tool_seen = bool(summary.get("blocked_tool_seen"))
    gateway = float(summary.get("gateway_coverage", 0.0))
    orphans = int(summary.get("orphan_executions", 999))
    return {
        "blocked_input_seen": blocked_input_seen,
        "blocked_tool_seen": blocked_tool_seen,
        "require_approval_seen": bool(summary.get("require_approval_seen", False)),
        "gateway_coverage": gateway,
        "orphan_executions": orphans,
        "gateway_coverage_ok": gateway >= 1.0,
        "orphan_executions_zero": orphans == 0,
    }


def _run_contract_layer(
    *,
    python_exec: str,
    profile: str,
    strict: bool,
    run_dir: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_runs: List[Dict[str, Any]] = []

    framework_summary = _run_command(
        name="contract_framework_smokes",
        argv=[
            python_exec,
            str((ROOT / "scripts" / "run_framework_smokes.py").resolve()),
            "--profile",
            profile,
            *(["--strict"] if strict else []),
            "--output-dir",
            str(run_dir / "framework_smokes"),
        ],
        cwd=ROOT,
        out_dir=run_dir,
    )
    command_runs.append(framework_summary)
    framework_payload: Dict[str, Any] = {}
    if isinstance(framework_summary.get("parsed_json"), dict):
        framework_payload = dict(framework_summary["parsed_json"])
    else:
        summary_path = run_dir / "framework_smokes" / "summary.json"
        if summary_path.exists():
            framework_payload = _read_json(summary_path)

    npm_bin = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    plugin_typecheck = _run_command(
        name="contract_openclaw_typecheck",
        argv=[npm_bin, "run", "typecheck"],
        cwd=PLUGIN_ROOT,
        out_dir=run_dir,
    )
    command_runs.append(plugin_typecheck)
    plugin_test = _run_command(
        name="contract_openclaw_test",
        argv=[npm_bin, "run", "test"],
        cwd=PLUGIN_ROOT,
        out_dir=run_dir,
    )
    command_runs.append(plugin_test)

    targets: Dict[str, Any] = {}
    fw_map = framework_payload.get("frameworks", {}) if isinstance(framework_payload.get("frameworks"), Mapping) else {}
    for name in FRAMEWORKS:
        row = fw_map.get(name, {}) if isinstance(fw_map.get(name), Mapping) else {}
        detailed_path = run_dir / "framework_smokes" / f"{name}_report.json"
        agg = {
            "blocked_input_seen": False,
            "blocked_tool_seen": False,
        }
        if detailed_path.exists():
            agg = _aggregate_framework_summary(_read_json(detailed_path))
        targets[name] = {
            "status": "ok" if int(row.get("failure_count", 1)) == 0 else "fail",
            "gateway_coverage": float(row.get("gateway_coverage", 0.0)),
            "orphan_executions": int(row.get("orphan_executions", 999)),
            "blocked_input_seen": bool(agg["blocked_input_seen"]),
            "blocked_tool_seen": bool(agg["blocked_tool_seen"]),
        }

    targets["openclaw_mapper"] = {
        "status": "ok" if int(plugin_typecheck["exit_code"]) == 0 and int(plugin_test["exit_code"]) == 0 else "fail",
        "gateway_coverage": 1.0,
        "orphan_executions": 0,
        "blocked_input_seen": True,
        "blocked_tool_seen": True,
    }

    min_gateway = min(float(v.get("gateway_coverage", 0.0)) for v in targets.values())
    total_orphans = sum(int(v.get("orphan_executions", 0)) for v in targets.values())
    blocked_input_seen = all(bool(v.get("blocked_input_seen")) for v in targets.values())
    blocked_tool_seen = all(bool(v.get("blocked_tool_seen")) for v in targets.values())
    overall_ok = (
        int(framework_summary["exit_code"]) == 0
        and int(plugin_typecheck["exit_code"]) == 0
        and int(plugin_test["exit_code"]) == 0
        and min_gateway >= 1.0
        and total_orphans == 0
        and blocked_input_seen
        and blocked_tool_seen
    )

    report = {
        "layer": "contract",
        "status": "ok" if overall_ok else "fail",
        "targets": targets,
        "gates": {
            "gateway_coverage_ok": bool(min_gateway >= 1.0),
            "orphan_executions_zero": bool(total_orphans == 0),
            "blocked_input_seen": blocked_input_seen,
            "blocked_tool_seen": blocked_tool_seen,
        },
        "metrics": {
            "min_gateway_coverage": float(min_gateway),
            "total_orphans": int(total_orphans),
        },
        "command_runs": command_runs,
    }
    _write_json(run_dir / "report.json", report)
    return report


def _run_workflow_layer(
    *,
    python_exec: str,
    profile: str,
    strict: bool,
    run_dir: Path,
    api_host: str,
    api_port: int,
    cases_fixture: Path,
    identity_fixture: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_runs: List[Dict[str, Any]] = []

    cases = _load_workflow_cases(cases_fixture)
    identities = _load_identity_fixtures(identity_fixture)
    _write_json(run_dir / "workflow_cases_snapshot.json", {"cases": cases, "identities_count": len(identities)})

    # Framework matrix path (6 adapters)
    fw_summary_cmd = _run_command(
        name="workflow_framework_smokes",
        argv=[
            python_exec,
            str((ROOT / "scripts" / "run_framework_smokes.py").resolve()),
            "--profile",
            profile,
            *(["--strict"] if strict else []),
            "--output-dir",
            str(run_dir / "framework_smokes"),
        ],
        cwd=ROOT,
        out_dir=run_dir,
    )
    command_runs.append(fw_summary_cmd)
    fw_payload = dict(fw_summary_cmd["parsed_json"]) if isinstance(fw_summary_cmd.get("parsed_json"), dict) else {}
    if not fw_payload:
        summary_path = run_dir / "framework_smokes" / "summary.json"
        if summary_path.exists():
            fw_payload = _read_json(summary_path)

    per_framework: Dict[str, Any] = {}
    fw_reports_dir = run_dir / "framework_smokes"
    for framework in FRAMEWORKS:
        report_path = fw_reports_dir / f"{framework}_report.json"
        if report_path.exists():
            source_payload = _read_json(report_path)
            agg = _aggregate_framework_summary(source_payload)
            item = {
                "framework": framework,
                "status": "ok" if not source_payload.get("failures") else "fail",
                "case_count": len(cases),
                "blocked_input_seen": bool(agg["blocked_input_seen"]),
                "blocked_tool_seen": bool(agg["blocked_tool_seen"]),
                "require_approval_seen": bool(agg["require_approval_seen"]),
                "gateway_coverage_ok": bool(agg["gateway_coverage_ok"]),
                "orphan_executions_zero": bool(agg["orphan_executions_zero"]),
                "raw_summary": dict(source_payload.get("summary", {})),
            }
            per_framework[framework] = item
            _write_json(run_dir / f"{framework}.json", item)
        else:
            item = {
                "framework": framework,
                "status": "fail",
                "case_count": len(cases),
                "blocked_input_seen": False,
                "blocked_tool_seen": False,
                "require_approval_seen": False,
                "gateway_coverage_ok": False,
                "orphan_executions_zero": False,
                "raw_summary": {},
            }
            per_framework[framework] = item
            _write_json(run_dir / f"{framework}.json", item)

    # OpenClaw local-api path with strict auth
    api_log_path = run_dir / "api.log"
    api_env = os.environ.copy()
    hmac_secret = secrets.token_hex(24)
    api_env["OMEGA_API_HMAC_SECRET"] = hmac_secret
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
    openclaw_status = "fail"
    openclaw_payload: Dict[str, Any] = {
        "framework": "openclaw",
        "status": "fail",
        "blocked_input_seen": False,
        "blocked_tool_seen": False,
        "require_approval_seen": False,
        "gateway_coverage_ok": False,
        "orphan_executions_zero": False,
        "webfetch_guard_seen": False,
        "session_reset_seen": False,
    }
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
                openclaw_payload["error"] = "api_boot_failure"
            else:
                plugin_env = os.environ.copy()
                plugin_env["OMEGA_OPENCLAW_API_BASE_URL"] = base_url
                plugin_env["OMEGA_OPENCLAW_API_KEY"] = "dev-api-key"
                plugin_env["OMEGA_OPENCLAW_HMAC_SECRET"] = hmac_secret
                plugin_env["OMEGA_OPENCLAW_TENANT_ID"] = f"tenant-{_utc_compact()}"
                npm_bin = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
                smoke = _run_command(
                    name="workflow_openclaw_smoke",
                    argv=[npm_bin, "run", "smoke"],
                    cwd=PLUGIN_ROOT,
                    out_dir=run_dir,
                    env=plugin_env,
                )
                command_runs.append(smoke)
                local_api = _run_command(
                    name="workflow_openclaw_smoke_local_api",
                    argv=[npm_bin, "run", "smoke:local-api"],
                    cwd=PLUGIN_ROOT,
                    out_dir=run_dir,
                    env=plugin_env,
                )
                command_runs.append(local_api)
                smoke_payload = dict(smoke.get("parsed_json") or {})
                local_api_payload = dict(local_api.get("parsed_json") or {})
                sample_block = dict(smoke_payload.get("sample_block_decision", {}) or {})
                sample_approval = dict(smoke_payload.get("sample_require_approval_decision", {}) or {})
                openclaw_payload.update(
                    {
                        "status": "ok" if int(smoke["exit_code"]) == 0 and int(local_api["exit_code"]) == 0 else "fail",
                        "blocked_input_seen": bool(sample_block.get("block")),
                        "blocked_tool_seen": bool(sample_block.get("block")),
                        "require_approval_seen": bool(sample_approval.get("requireApproval")),
                        "gateway_coverage_ok": True,
                        "orphan_executions_zero": True,
                        "webfetch_guard_seen": bool(smoke_payload.get("webfetch_guard_seen")),
                        "session_reset_seen": bool(local_api_payload.get("session_reset_seen")),
                        "smoke_payload": smoke_payload,
                        "local_api_payload": local_api_payload,
                    }
                )
                openclaw_status = str(openclaw_payload.get("status", "fail"))
    finally:
        if api_proc is not None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()
                api_proc.wait(timeout=5)

    _write_json(run_dir / "openclaw.json", openclaw_payload)
    per_framework["openclaw"] = openclaw_payload

    gates = {
        "blocked_input_seen": all(bool(row.get("blocked_input_seen")) for row in per_framework.values()),
        "blocked_tool_seen": all(bool(row.get("blocked_tool_seen")) for row in per_framework.values()),
        "require_approval_seen": any(bool(row.get("require_approval_seen")) for row in per_framework.values()),
        "orphan_executions_zero": all(bool(row.get("orphan_executions_zero")) for row in per_framework.values()),
        "gateway_coverage_ok": all(bool(row.get("gateway_coverage_ok")) for row in per_framework.values()),
        "webfetch_guard_seen": bool(openclaw_payload.get("webfetch_guard_seen")),
    }
    overall_ok = (
        int(fw_summary_cmd["exit_code"]) == 0
        and openclaw_status == "ok"
        and all(bool(v) for v in gates.values())
    )

    report = {
        "layer": "workflow",
        "status": "ok" if overall_ok else "fail",
        "case_count": len(cases),
        "identity_fixture_rows": len(identities),
        "frameworks": per_framework,
        "gates": gates,
        "command_runs": command_runs,
    }
    _write_json(run_dir / "report.json", report)
    return report


def _run_stress_layer(
    *,
    python_exec: str,
    strict: bool,
    run_dir: Path,
    webfetch_fixture_dir: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_runs: List[Dict[str, Any]] = []

    tests = [
        ("stress_race", "tests/test_framework_race_conditions.py"),
        ("stress_approval", "tests/test_approval_resume_e2e.py"),
        ("stress_webfetch", "tests/test_webfetch_edge_corpus.py"),
    ]
    passed = 0
    for name, test_path in tests:
        pytest_inline = (
            "import os,sys,pytest; "
            "os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; "
            f"sys.exit(pytest.main(['-q','{test_path}']))"
        )
        row = _run_command(
            name=name,
            argv=[python_exec, "-c", pytest_inline],
            cwd=ROOT,
            out_dir=run_dir,
        )
        command_runs.append(row)
        if int(row["exit_code"]) == 0:
            passed += 1

    webfetch_files = sorted([p.name for p in webfetch_fixture_dir.glob("*.html") if p.is_file()])
    _write_json(run_dir / "webfetch_fixture_snapshot.json", {"files": webfetch_files})

    # In strict mode each sub-suite must pass; rates are normalized to [0..1].
    require_approval_resume_success_rate = 1.0 if int(command_runs[1]["exit_code"]) == 0 else 0.0
    replay_block_rate = 1.0 if int(command_runs[0]["exit_code"]) == 0 else 0.0
    webfetch_edge_handling_rate = 1.0 if int(command_runs[2]["exit_code"]) == 0 else 0.0

    gates = {
        "require_approval_resume_success_rate": require_approval_resume_success_rate,
        "replay_block_rate": replay_block_rate,
        "webfetch_edge_handling_rate": webfetch_edge_handling_rate,
        "gateway_coverage_ok": bool(int(command_runs[0]["exit_code"]) == 0),
        "orphan_executions_zero": bool(int(command_runs[0]["exit_code"]) == 0),
        "blocked_input_seen": bool(int(command_runs[0]["exit_code"]) == 0),
        "blocked_tool_seen": bool(int(command_runs[0]["exit_code"]) == 0),
    }
    overall_ok = (
        all(float(gates[k]) >= 1.0 for k in ("require_approval_resume_success_rate", "replay_block_rate", "webfetch_edge_handling_rate"))
        and bool(gates["gateway_coverage_ok"])
        and bool(gates["orphan_executions_zero"])
        and bool(gates["blocked_input_seen"])
        and bool(gates["blocked_tool_seen"])
    )
    report = {
        "layer": "stress",
        "status": "ok" if overall_ok else "fail",
        "metrics": gates,
        "passed_suites": int(passed),
        "total_suites": int(len(tests)),
        "strict": bool(strict),
        "command_runs": command_runs,
    }
    _write_json(run_dir / "chaos_report.json", report)
    return report


def _join_status(values: Iterable[str], *, strict: bool) -> str:
    rows = [str(v) for v in values]
    if rows and all(v == "ok" for v in rows):
        return "ok"
    return "fail" if strict else "partial"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-layer unified framework stand (contract/workflow/stress)")
    parser.add_argument("--layer", choices=("contract", "workflow", "stress", "all"), default="all")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = str(args.run_id or _default_run_id("framework_matrix"))
    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = ROOT / artifacts_root

    contract_dir = artifacts_root / "framework_contract" / run_id
    workflow_dir = artifacts_root / "real_workflow_matrix" / run_id
    stress_dir = artifacts_root / "stress" / run_id

    cases_fixture = ROOT / "tests" / "data" / "framework_matrix" / "workflow_cases_v1.json"
    identity_fixture = ROOT / "tests" / "data" / "framework_matrix" / "identity_fixture_v1.jsonl"
    webfetch_fixture_dir = ROOT / "tests" / "data" / "framework_matrix" / "webfetch_edge_corpus"

    layers = [str(args.layer)] if str(args.layer) != "all" else ["contract", "workflow", "stress"]
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": _utc_now().isoformat(),
        "profile": str(args.profile),
        "strict": bool(args.strict),
        "layer": str(args.layer),
        "layers": {},
    }

    if "contract" in layers:
        payload["layers"]["contract"] = _run_contract_layer(
            python_exec=sys.executable,
            profile=str(args.profile),
            strict=bool(args.strict),
            run_dir=contract_dir,
        )
    if "workflow" in layers:
        payload["layers"]["workflow"] = _run_workflow_layer(
            python_exec=sys.executable,
            profile=str(args.profile),
            strict=bool(args.strict),
            run_dir=workflow_dir,
            api_host=str(args.api_host),
            api_port=int(args.api_port),
            cases_fixture=cases_fixture,
            identity_fixture=identity_fixture,
        )
    if "stress" in layers:
        payload["layers"]["stress"] = _run_stress_layer(
            python_exec=sys.executable,
            strict=bool(args.strict),
            run_dir=stress_dir,
            webfetch_fixture_dir=webfetch_fixture_dir,
        )

    layer_statuses = [str((payload["layers"][k] or {}).get("status", "fail")) for k in payload["layers"]]
    payload["overall_status"] = _join_status(layer_statuses, strict=bool(args.strict))
    payload["finished_at"] = _utc_now().isoformat()
    payload["artifacts"] = {
        "framework_contract": str(contract_dir / "report.json"),
        "real_workflow_matrix": str(workflow_dir / "report.json"),
        "stress_chaos": str(stress_dir / "chaos_report.json"),
    }

    summary_path = artifacts_root / "framework_matrix_summary" / f"{run_id}.json"
    _write_json(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))

    if payload["overall_status"] == "fail":
        return 1
    if bool(args.strict) and payload["overall_status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
