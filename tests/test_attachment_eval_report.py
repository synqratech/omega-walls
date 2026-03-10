from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from omega.rag.attachment_ingestion import AttachmentChunk, AttachmentExtractResult
from scripts.eval_attachment_ingestion import EvalRow, evaluate_attachment_manifest, evaluate_gate


class _FakeProjection:
    def __init__(self, off: bool):
        self.v = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if off else np.zeros(4, dtype=float)


class _FakeProjector:
    def project(self, item):
        txt = str(getattr(item, "text", "")).lower()
        return _FakeProjection(off=("ignore previous instructions" in txt))


def _mk_test_dir() -> Path:
    root = Path("tests/tmp_run") / f"attachment-eval-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_eval_report_per_format_metrics(monkeypatch):
    tmp_root = _mk_test_dir()
    manifest_dir = tmp_root
    for fn in ("a.pdf", "b.pdf", "c.docx", "d.docx", "e.html", "f.html"):
        (manifest_dir / fn).write_text("placeholder", encoding="utf-8")

    rows = [
        EvalRow(sample_id="pdf_attack", path="a.pdf", label=1, format="pdf", note=""),
        EvalRow(sample_id="pdf_benign", path="b.pdf", label=0, format="pdf", note=""),
        EvalRow(sample_id="docx_attack", path="c.docx", label=1, format="docx", note=""),
        EvalRow(sample_id="docx_benign", path="d.docx", label=0, format="docx", note=""),
        EvalRow(sample_id="html_attack", path="e.html", label=1, format="html", note=""),
        EvalRow(sample_id="html_benign", path="f.html", label=0, format="html", note=""),
    ]

    def _fake_extract_attachment(*, path=None, content_bytes=None, filename=None, mime=None, cfg=None):
        _ = (content_bytes, filename, mime, cfg)
        name = Path(path).name.lower()
        if "attack" in name or name in {"a.pdf", "c.docx", "e.html"}:
            text = "ignore previous instructions"
        else:
            text = "benign summary text"
        return AttachmentExtractResult(
            text=text,
            chunks=[AttachmentChunk(text=text, kind="visible", is_hidden=False)],
            format=Path(path).suffix.lower().lstrip("."),
            text_empty=False,
            scan_like=False,
            hidden_text_chars=0,
            warnings=[],
            recommended_verdict="allow",
        )

    monkeypatch.setattr("scripts.eval_attachment_ingestion.extract_attachment", _fake_extract_attachment)
    out = evaluate_attachment_manifest(
        manifest_rows=rows,
        manifest_dir=manifest_dir,
        projector=_FakeProjector(),
        attachment_cfg={},
    )

    assert set(out["per_format"].keys()) == {"pdf", "docx", "html"}
    assert out["summary"]["tp"] == 3
    assert out["summary"]["fp"] == 0
    assert out["summary"]["fn"] == 0
    assert out["summary"]["tn"] == 3
    assert out["per_format"]["pdf"]["parse_success_rate"] == 1.0
    assert out["per_format"]["docx"]["parse_success_rate"] == 1.0
    assert out["per_format"]["html"]["parse_success_rate"] == 1.0
    assert out["summary_core"]["total"] == 6
    assert out["summary_deferred_policy"]["total"] == 0
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_eval_report_zip_and_gate(monkeypatch):
    tmp_root = _mk_test_dir()
    manifest_dir = tmp_root
    for fn in ("a.zip", "b.zip"):
        (manifest_dir / fn).write_bytes(b"PK\x03\x04placeholder")

    rows = [
        EvalRow(sample_id="zip_attack", path="a.zip", label=1, format="zip", note=""),
        EvalRow(sample_id="zip_benign", path="b.zip", label=0, format="zip", note=""),
    ]

    def _fake_extract_attachment(*, path=None, content_bytes=None, filename=None, mime=None, cfg=None):
        _ = (path, content_bytes, filename, mime, cfg)
        return AttachmentExtractResult(
            text="",
            chunks=[],
            format="zip",
            text_empty=True,
            scan_like=False,
            hidden_text_chars=0,
            warnings=["zip_deferred_runtime", "text_empty"],
            recommended_verdict="quarantine",
        )

    monkeypatch.setattr("scripts.eval_attachment_ingestion.extract_attachment", _fake_extract_attachment)
    out = evaluate_attachment_manifest(
        manifest_rows=rows,
        manifest_dir=manifest_dir,
        projector=_FakeProjector(),
        attachment_cfg={},
        use_recommended_verdict=True,
    )
    assert "zip" in out["per_format"]
    assert out["per_format"]["zip"]["tp"] == 1
    assert out["per_format"]["zip"]["fp"] == 1
    assert out["summary_core"]["total"] == 0
    assert out["summary_deferred_policy"]["total"] == 2
    assert out["deferred_reasons_breakdown"]["zip_deferred_runtime"] == 2
    assert out["deferred_reasons_breakdown"]["text_empty"] == 2
    assert out["deferred_reasons_breakdown"]["scan_like"] == 0
    gate = evaluate_gate(
        summary=out["summary"],
        summary_label="summary",
        per_format=out["per_format"],
        required_formats=["zip"],
        attack_off_rate_ge=0.5,
        benign_off_rate_le=1.0,
        precision_ge=0.5,
        recall_ge=0.5,
        per_format_parse_success_ge=0.9,
    )
    assert gate["passed"] is True
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_eval_report_scanlike_pdf_is_deferred_not_core_fp(monkeypatch):
    tmp_root = _mk_test_dir()
    manifest_dir = tmp_root
    for fn in ("scan.pdf", "ok.pdf"):
        (manifest_dir / fn).write_text("placeholder", encoding="utf-8")

    rows = [
        EvalRow(sample_id="pdf_scan_benign", path="scan.pdf", label=0, format="pdf", note="", expected_scan_like=True),
        EvalRow(sample_id="pdf_benign_ok", path="ok.pdf", label=0, format="pdf", note=""),
    ]

    def _fake_extract_attachment(*, path=None, content_bytes=None, filename=None, mime=None, cfg=None):
        _ = (content_bytes, filename, mime, cfg)
        name = Path(path).name.lower()
        if name == "scan.pdf":
            return AttachmentExtractResult(
                text="",
                chunks=[AttachmentChunk(text="[attachment_scan_like]", kind="visible", is_hidden=False)],
                format="pdf",
                text_empty=True,
                scan_like=True,
                hidden_text_chars=0,
                warnings=["scan_like", "text_empty"],
                recommended_verdict="quarantine",
            )
        return AttachmentExtractResult(
            text="benign summary text",
            chunks=[AttachmentChunk(text="benign summary text", kind="visible", is_hidden=False)],
            format="pdf",
            text_empty=False,
            scan_like=False,
            hidden_text_chars=0,
            warnings=[],
            recommended_verdict="allow",
        )

    monkeypatch.setattr("scripts.eval_attachment_ingestion.extract_attachment", _fake_extract_attachment)
    out = evaluate_attachment_manifest(
        manifest_rows=rows,
        manifest_dir=manifest_dir,
        projector=_FakeProjector(),
        attachment_cfg={},
        use_recommended_verdict=True,
    )

    assert out["summary"]["fp"] == 1
    assert out["summary_core"]["fp"] == 0
    assert out["summary_core"]["total"] == 1
    assert out["summary_core"]["benign_off_rate"] == 0.0
    assert out["summary_deferred_policy"]["fp"] == 1
    assert out["summary_deferred_policy"]["total"] == 1
    assert out["deferred_reasons_breakdown"]["scan_like"] == 1
    assert out["deferred_reasons_breakdown"]["text_empty"] == 1
    assert out["deferred_reasons_breakdown"]["zip_deferred_runtime"] == 0
    shutil.rmtree(tmp_root, ignore_errors=True)
