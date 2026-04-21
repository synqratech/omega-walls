#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates" / "secure_agent_template"
FRAMEWORKS = ("none", "langchain", "langgraph", "llamaindex", "haystack", "autogen", "crewai")
PROJECTOR_MODES = ("pi0", "hybrid", "hybrid_api", "pitheta")
GUARD_MODES = ("monitor", "enforce")


def _sanitize_package_name(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    value = value.lower()
    if not value:
        value = "omega_secure_agent"
    if value[0].isdigit():
        value = f"pkg_{value}"
    return value


def _resolve_run_copy():
    try:
        from copier import run_copy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Copier is required. Install it with: python -m pip install copier"
        ) from exc
    return run_copy


def _run_copier(*, out_dir: Path, data: Dict[str, Any]) -> None:
    run_copy = _resolve_run_copy()
    sig = inspect.signature(run_copy)
    kwargs: Dict[str, Any] = {}

    if "src_path" in sig.parameters:
        kwargs["src_path"] = str(TEMPLATE_DIR)
    if "dst_path" in sig.parameters:
        kwargs["dst_path"] = str(out_dir)
    if "data" in sig.parameters:
        kwargs["data"] = data
    if "defaults" in sig.parameters:
        kwargs["defaults"] = True
    if "unsafe" in sig.parameters:
        kwargs["unsafe"] = True
    if "quiet" in sig.parameters:
        kwargs["quiet"] = True
    if "overwrite" in sig.parameters:
        kwargs["overwrite"] = True

    if "src_path" not in kwargs or "dst_path" not in kwargs:
        run_copy(str(TEMPLATE_DIR), str(out_dir), data=data)
        return

    try:
        run_copy(**kwargs)
    except TypeError:
        run_copy(str(TEMPLATE_DIR), str(out_dir), data=data)


def _ensure_out_dir_ready(out_dir: Path) -> None:
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries:
            raise RuntimeError(f"Output directory is not empty: {out_dir}")
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FW-007 secure-agent-template scaffold")
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    parser.add_argument("--out", required=True, help="Output directory for generated project")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--package-name", default=None)
    parser.add_argument("--python-version", default="3.13")
    parser.add_argument("--projector-mode", choices=PROJECTOR_MODES, default="pi0")
    parser.add_argument("--guard-mode", choices=GUARD_MODES, default="monitor")
    parser.add_argument("--include-ci", dest="include_ci", action="store_true", default=True)
    parser.add_argument("--no-ci", dest="include_ci", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not TEMPLATE_DIR.exists():
        print(json.dumps({"status": "error", "error": f"template missing: {TEMPLATE_DIR}"}))
        return 3

    out_dir = Path(args.out).resolve()
    project_name = str(args.project_name or out_dir.name or "omega-secure-agent").strip() or "omega-secure-agent"
    package_name = _sanitize_package_name(str(args.package_name or project_name))

    try:
        _ensure_out_dir_ready(out_dir)
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 4

    data = {
        "project_name": project_name,
        "package_name": package_name,
        "python_version": str(args.python_version),
        "framework": str(args.framework),
        "projector_mode": str(args.projector_mode),
        "guard_mode": str(args.guard_mode),
        "include_ci": bool(args.include_ci),
    }

    try:
        _run_copier(out_dir=out_dir, data=data)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 5

    if not bool(args.include_ci):
        ci_dir = out_dir / ".github"
        if ci_dir.exists():
            shutil.rmtree(ci_dir)

    payload = {
        "status": "ok",
        "template": str(TEMPLATE_DIR),
        "output": str(out_dir),
        "project_name": project_name,
        "package_name": package_name,
        "framework": str(args.framework),
        "projector_mode": str(args.projector_mode),
        "guard_mode": str(args.guard_mode),
        "include_ci": bool(args.include_ci),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
