from __future__ import annotations

from pathlib import Path


def test_refactor_guard_suite_is_valid_and_core_coverage_present() -> None:
    root = Path(__file__).resolve().parents[1]
    suite_file = root / "tests" / "refactor_guard_suite.txt"
    lines = [line.strip() for line in suite_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert lines, "refactor guard suite cannot be empty"
    assert len(lines) == len(set(lines)), "refactor guard suite contains duplicate test paths"

    for rel in lines:
        assert rel.startswith("tests/")
        assert rel.endswith(".py")
        path = root / rel
        assert path.exists(), f"missing test path in refactor guard suite: {rel}"
        assert path.is_file()

    must_have = {
        "tests/test_refactor_behavior_lock.py",
        "tests/test_refactor_characterization_snapshots.py",
        "tests/test_api_session_runtime.py",
        "tests/test_harness_monitor_mode.py",
        "tests/test_pi0.py",
        "tests/test_api_hybrid_projector.py",
        "tests/test_config_loader_package_resources.py",
    }
    missing = sorted(must_have - set(lines))
    assert not missing, f"refactor guard suite missing required coverage files: {missing}"
