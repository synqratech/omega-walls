from __future__ import annotations

import scripts.run_framework_matrix_stand as stand


def test_join_status_strict_and_non_strict() -> None:
    assert stand._join_status(["ok", "ok"], strict=True) == "ok"  # noqa: SLF001
    assert stand._join_status(["ok", "fail"], strict=True) == "fail"  # noqa: SLF001
    assert stand._join_status(["ok", "fail"], strict=False) == "partial"  # noqa: SLF001


def test_default_run_id_is_prefixed() -> None:
    run_id = stand._default_run_id("framework_matrix")  # noqa: SLF001
    assert run_id.startswith("framework_matrix_")
    assert len(run_id) > len("framework_matrix_")

