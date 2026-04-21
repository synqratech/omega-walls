from __future__ import annotations

from pathlib import Path


def test_fw001_core_suite_file_is_non_empty_and_paths_exist():
    root = Path(__file__).resolve().parents[1]
    suite_file = root / "tests" / "fw001_core_suite.txt"
    lines = [line.strip() for line in suite_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert lines, "fw001 core suite cannot be empty"
    for rel in lines:
        assert rel.startswith("tests/")
        assert rel.endswith(".py")
        path = root / rel
        assert path.exists(), f"missing test path in fw001 core suite: {rel}"
        assert path.is_file()
