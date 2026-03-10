from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omega.eval.bipia_manifest import BIPIAManifestInput, build_bipia_manifest, verify_qa_abstract_md5


FIXTURE_ROOT = "tests/data/bipia_fixture/benchmark"


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_verify_qa_abstract_md5_fixture_ok():
    ok, details = verify_qa_abstract_md5(FIXTURE_ROOT)
    assert ok is True
    assert details["qa"]["train.jsonl"]["ok"] is True
    assert details["qa"]["test.jsonl"]["ok"] is True
    assert details["abstract"]["train.jsonl"]["ok"] is True
    assert details["abstract"]["test.jsonl"]["ok"] is True


def test_build_manifest_requires_commit_in_strict(monkeypatch):
    monkeypatch.delenv("BIPIA_COMMIT", raising=False)
    with pytest.raises(ValueError):
        build_bipia_manifest(
            BIPIAManifestInput(
                benchmark_root=FIXTURE_ROOT,
                split="test",
                mode="sampled",
                seed_pack=[41],
                config_refs={"resolved_config_sha256": "x"},
                strict=True,
            )
        )


def test_build_manifest_ok_with_commit(monkeypatch):
    monkeypatch.setenv("BIPIA_COMMIT", "deadbeef")
    manifest = build_bipia_manifest(
        BIPIAManifestInput(
            benchmark_root=FIXTURE_ROOT,
            split="test",
            mode="sampled",
            seed_pack=[41, 42],
            config_refs={"resolved_config_sha256": "x"},
            strict=True,
        )
    )
    assert manifest["bipia_commit"] == "deadbeef"
    assert manifest["data_readiness"]["qa_abstract_md5_ok"] is True
    assert manifest["missing_files"] == []


def test_build_manifest_fails_on_md5_mismatch(monkeypatch):
    monkeypatch.setenv("BIPIA_COMMIT", "deadbeef")
    dst = _mk_local_tmp("bipia-manifest") / "benchmark"
    src = Path(FIXTURE_ROOT)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(path.read_bytes())
    (dst / "qa" / "test.jsonl").write_text('{"context":"tampered","question":"q","ideal":["x"]}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        build_bipia_manifest(
            BIPIAManifestInput(
                benchmark_root=str(dst),
                split="test",
                mode="sampled",
                seed_pack=[41],
                config_refs={"resolved_config_sha256": "x"},
                strict=True,
            )
        )
