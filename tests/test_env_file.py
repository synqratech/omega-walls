from __future__ import annotations

import os
from pathlib import Path

from omega.env_file import load_repo_env_file


def test_load_repo_env_file_accepts_utf8_bom(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("\ufeffOPENAI_API_KEY=sk-test-bom\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_repo_env_file(env_path=env_path, override=True)

    assert loaded == env_path.resolve()
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-bom"
