#!/usr/bin/env python3
"""FW-008 gate: validate runtime docker image metadata and hygiene."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _load_inspect(image_ref: str) -> Dict[str, Any]:
    proc = _run(["docker", "image", "inspect", image_ref])
    if proc.returncode != 0:
        raise RuntimeError(f"docker image inspect failed: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(proc.stdout)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("docker image inspect returned empty payload")
    obj = payload[0]
    if not isinstance(obj, dict):
        raise RuntimeError("docker image inspect payload shape invalid")
    return obj


def _load_history(image_ref: str) -> List[Dict[str, str]]:
    proc = _run(["docker", "history", "--no-trunc", "--format", "{{json .}}", image_ref])
    if proc.returncode != 0:
        return []
    rows: List[Dict[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append({str(k): str(v) for k, v in item.items()})
    return rows


def _check_banned_paths(image_ref: str, banned: List[str]) -> Dict[str, Any]:
    script = (
        "import pathlib,json; "
        "banned=" + repr(list(banned)) + "; "
        "base=pathlib.Path('/app'); "
        "present=[x for x in banned if (base / x).exists()]; "
        "print(json.dumps({'present':present}))"
    )
    proc = _run(["docker", "run", "--rm", "--entrypoint", "python", image_ref, "-c", script])
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "present": []}
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except Exception:
        return {"ok": False, "error": "failed to parse banned path check output", "present": []}
    present = payload.get("present", [])
    if not isinstance(present, list):
        present = []
    return {"ok": len(present) == 0, "present": [str(x) for x in present], "error": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check docker image metadata + runtime hygiene for FW-008")
    parser.add_argument("--image", required=True, help="Docker image reference to inspect")
    parser.add_argument(
        "--banned-path",
        action="append",
        default=[],
        help="Path under /app that must not exist in runtime image (repeatable)",
    )
    args = parser.parse_args()

    image_ref = str(args.image).strip()
    if not image_ref:
        raise SystemExit("--image is required")

    inspect_obj = _load_inspect(image_ref)
    cfg = inspect_obj.get("Config", {}) if isinstance(inspect_obj.get("Config", {}), dict) else {}
    entrypoint = list(cfg.get("Entrypoint", []) or [])
    cmd = list(cfg.get("Cmd", []) or [])
    exposed = cfg.get("ExposedPorts", {}) if isinstance(cfg.get("ExposedPorts", {}), dict) else {}
    healthcheck = cfg.get("Healthcheck", {}) if isinstance(cfg.get("Healthcheck", {}), dict) else {}
    healthcheck_test = list(healthcheck.get("Test", []) or [])
    size_bytes = int(inspect_obj.get("Size", 0) or 0)
    history = _load_history(image_ref)

    checks: Dict[str, bool] = {}
    checks["entrypoint_contains_omega_walls_api"] = any("omega-walls-api" in str(x) for x in entrypoint)
    checks["exposes_8080_tcp"] = "8080/tcp" in exposed
    checks["has_healthcheck"] = len(healthcheck_test) > 0
    checks["healthcheck_targets_healthz"] = any("/healthz" in str(x) for x in healthcheck_test)
    checks["cmd_contains_quickstart"] = "--profile" in cmd and "quickstart" in cmd
    checks["cmd_binds_0_0_0_0_8080"] = ("--host" in cmd and "0.0.0.0" in cmd and "--port" in cmd and "8080" in cmd)

    banned_paths = [str(x).strip() for x in list(args.banned_path or []) if str(x).strip()]
    if not banned_paths:
        banned_paths = [
            "tests",
            "redteam",
            "data",
            "notebooks",
            "artifacts",
            "docs",
            "cloud",
        ]
    banned_check = _check_banned_paths(image_ref=image_ref, banned=banned_paths)
    checks["banned_paths_absent"] = bool(banned_check.get("ok", False))

    ok = all(bool(v) for v in checks.values())
    payload = {
        "status": "ok" if ok else "fail",
        "image": image_ref,
        "checks": checks,
        "inspect_summary": {
            "entrypoint": entrypoint,
            "cmd": cmd,
            "exposed_ports": sorted(list(exposed.keys())),
            "healthcheck_test": healthcheck_test,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        },
        "banned_paths": {
            "configured": banned_paths,
            "present": list(banned_check.get("present", [])),
            "error": str(banned_check.get("error", "")),
        },
        "history_layers": history,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
