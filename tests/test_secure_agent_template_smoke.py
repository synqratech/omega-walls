from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end >= start, f"stdout does not contain JSON object:\n{text}"
    payload = json.loads(text[start : end + 1])
    assert isinstance(payload, dict)
    return payload


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_secure_agent_template_render_and_smoke(tmp_path: Path) -> None:
    pytest.importorskip("copier")

    out_dir = tmp_path / "secure-agent-none"
    init_proc = _run(
        [
            sys.executable,
            "scripts/init_secure_agent_template.py",
            "--framework",
            "none",
            "--out",
            str(out_dir),
            "--project-name",
            "FW007 None Project",
        ],
        cwd=ROOT,
    )
    assert init_proc.returncode == 0, init_proc.stdout + init_proc.stderr
    init_payload = _extract_json(init_proc.stdout)
    assert init_payload.get("status") == "ok"
    assert out_dir.exists()
    for rel in ("README.md", "app.py", "scripts/smoke.py", "config/local_dev.yml", "requirements.txt"):
        content = (out_dir / rel).read_text(encoding="utf-8")
        assert "{{" not in content and "}}" not in content, f"unresolved template token in {rel}"

    smoke_proc = _run([sys.executable, str(out_dir / "scripts" / "smoke.py")], cwd=ROOT)
    assert smoke_proc.returncode == 0, smoke_proc.stdout + smoke_proc.stderr
    smoke_payload = _extract_json(smoke_proc.stdout)
    assert smoke_payload.get("status") == "ok"
    samples = smoke_payload.get("samples", {})
    assert isinstance(samples, dict)
    attack = samples.get("attack", {})
    assert isinstance(attack, dict)
    assert str(attack.get("actual_action", "")).upper() == "ALLOW"
    assert str(attack.get("intended_action", "")).upper() not in {"", "ALLOW"}


def test_secure_agent_template_framework_wiring_render(tmp_path: Path) -> None:
    pytest.importorskip("copier")

    out_dir = tmp_path / "secure-agent-langchain"
    init_proc = _run(
        [
            sys.executable,
            "scripts/init_secure_agent_template.py",
            "--framework",
            "langchain",
            "--out",
            str(out_dir),
            "--project-name",
            "FW007 LangChain Project",
            "--no-ci",
        ],
        cwd=ROOT,
    )
    assert init_proc.returncode == 0, init_proc.stdout + init_proc.stderr
    assert not (out_dir / ".github").exists()

    app_proc = _run([sys.executable, str(out_dir / "app.py"), "--framework-check"], cwd=ROOT)
    assert app_proc.returncode == 0, app_proc.stdout + app_proc.stderr
    app_payload = _extract_json(app_proc.stdout)
    assert app_payload.get("status") == "ok"
    integration = app_payload.get("integration", {})
    assert isinstance(integration, dict)
    assert integration.get("framework") == "langchain"
    assert integration.get("enabled") is True
    assert str(integration.get("guard_class", "")).strip() == "OmegaLangChainGuard"
