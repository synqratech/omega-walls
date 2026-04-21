from __future__ import annotations

from pathlib import Path
import time

from omega.adapters.core import AdapterSessionContext, OmegaAdapterRuntime


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "data" / "framework_matrix" / "webfetch_edge_corpus"


def test_webfetch_edge_corpus_files_exist() -> None:
    root = _fixture_dir()
    names = sorted(p.name for p in root.glob("*.html"))
    assert names == ["bad_charset.html", "broken.html", "empty.html", "huge_page.html", "timeout.html"]


def test_webfetch_edge_corpus_processed_without_runtime_crash() -> None:
    runtime = OmegaAdapterRuntime(profile="dev", projector_mode="pi0")
    root = _fixture_dir()
    for idx, path in enumerate(sorted(root.glob("*.html")), start=1):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name == "timeout.html":
            # Simulate delayed upstream content that still must be processed safely.
            text = (text + "\n") * 200
        started = time.perf_counter()
        decision = runtime.check_model_input(
            messages_text=text,
            ctx=AdapterSessionContext(session_id=f"wf-edge-{idx}", actor_id=f"actor-{idx}", metadata={"source": path.name}),
        )
        elapsed = time.perf_counter() - started
        assert decision.session_id == f"wf-edge-{idx}"
        assert isinstance(decision.control_outcome, str)
        assert isinstance(bool(decision.off), bool)
        # Guardrail: edge fixtures should be handled quickly in local runtime.
        assert elapsed < 5.0
