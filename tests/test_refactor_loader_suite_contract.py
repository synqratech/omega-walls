from __future__ import annotations

from pathlib import Path


def test_refactor_loader_suite_is_valid_and_has_required_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    suite_file = root / "tests" / "refactor_loader_suite.txt"
    lines = [line.strip() for line in suite_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert lines, "refactor loader suite cannot be empty"
    assert len(lines) == len(set(lines)), "refactor loader suite contains duplicate test paths"

    for rel in lines:
        assert rel.startswith("tests/")
        assert rel.endswith(".py")
        path = root / rel
        assert path.exists(), f"missing test path in refactor loader suite: {rel}"
        assert path.is_file()

    must_have = {
        "tests/test_refactor_behavior_lock.py",
        "tests/test_refactor_characterization_snapshots.py",
        "tests/test_config_loader_package_resources.py",
        "tests/test_quickstart_profile.py",
        "tests/test_profile_surface_parity.py",
        "tests/test_config_loader_negative_cases.py",
        "tests/test_integration_harness.py",
        "tests/test_api_attachment_security.py",
        "tests/test_api_hybrid_projector.py",
    }
    missing = sorted(must_have - set(lines))
    assert not missing, f"refactor loader suite missing required coverage files: {missing}"

