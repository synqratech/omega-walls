from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def _project_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(((payload.get("project") or {}).get("version")) or "").strip()


def _pick_artifacts(*, dist_dir: Path, version: str) -> tuple[Path, Path]:
    wheel = dist_dir / f"omega_walls-{version}-py3-none-any.whl"
    sdist = dist_dir / f"omega_walls-{version}.tar.gz"
    if not wheel.exists():
        raise RuntimeError(f"wheel not found: {wheel}")
    if not sdist.exists():
        raise RuntimeError(f"sdist not found: {sdist}")
    return wheel, sdist


def _read_wheel_member(wheel: Path, member: str) -> str:
    with zipfile.ZipFile(wheel) as zf:
        return zf.read(member).decode("utf-8")


def _read_sdist_member(sdist: Path, suffix_member: str) -> str:
    with tarfile.open(sdist, "r:gz") as tf:
        names = tf.getnames()
        matches = [n for n in names if n.endswith(suffix_member)]
        if not matches:
            raise RuntimeError(f"sdist member not found: {suffix_member}")
        member = matches[0]
        handle = tf.extractfile(member)
        if handle is None:
            raise RuntimeError(f"cannot extract sdist member: {member}")
        return handle.read().decode("utf-8")


def _evaluate_text_contracts(*, quickstart: str, sensitive_rules: str, cli_text: str, adapter_core: str) -> dict[str, bool]:
    return {
        "quickstart_monitor_default": "guard_mode: monitor" in quickstart,
        "sensitive_rules_only_semantic_mode": "semantic_mode: rules_only" in sensitive_rules,
        "cli_no_args_safe_path": 'if not argv or argv[0] in {"-h", "--help", "help"}:' in cli_text,
        "tool_block_contract_freeze_action": (
            'payload["action"] = cls.resolve_tool_block_action(gate_decision)' in adapter_core
            and 'return "TOOL_FREEZE"' in adapter_core
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release wheel/sdist contain required launch blocker fixes.")
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    version = str(args.version or _project_version()).strip()
    if not version:
        raise RuntimeError("cannot resolve project version")
    dist_dir = (ROOT / str(args.dist_dir)).resolve()
    wheel, sdist = _pick_artifacts(dist_dir=dist_dir, version=version)

    checks = _evaluate_text_contracts(
        quickstart=_read_wheel_member(wheel, "omega/config/resources/profiles/quickstart.yml"),
        sensitive_rules=_read_wheel_member(wheel, "omega/config/resources/profiles/sensitive_rules_only.yml"),
        cli_text=_read_wheel_member(wheel, "omega/cli.py"),
        adapter_core=_read_wheel_member(wheel, "omega/adapters/core.py"),
    )
    sdist_checks = _evaluate_text_contracts(
        quickstart=_read_sdist_member(sdist, "omega/config/resources/profiles/quickstart.yml"),
        sensitive_rules=_read_sdist_member(sdist, "omega/config/resources/profiles/sensitive_rules_only.yml"),
        cli_text=_read_sdist_member(sdist, "omega/cli.py"),
        adapter_core=_read_sdist_member(sdist, "omega/adapters/core.py"),
    )

    report = {
        "event": "prepublish_artifact_check_v1",
        "status": "ok",
        "version": version,
        "wheel": str(wheel),
        "sdist": str(sdist),
        "wheel_checks": checks,
        "sdist_checks": sdist_checks,
    }
    failed = [f"wheel:{k}" for k, ok in checks.items() if not bool(ok)]
    failed.extend([f"sdist:{k}" for k, ok in sdist_checks.items() if not bool(ok)])
    if failed:
        report["status"] = "fail"
        report["failed"] = failed
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 1

    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

