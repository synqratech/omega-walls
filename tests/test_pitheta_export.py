from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from scripts.export_pitheta_adapter import export_pitheta_adapter


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_export_pitheta_adapter_copies_expected_files():
    tmp = _mk_local_tmp("pitheta-export")
    ckpt = tmp / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter").mkdir()
    (ckpt / "tokenizer").mkdir()
    (ckpt / "adapter" / "adapter_config.json").write_text('{"ok":true}', encoding="utf-8")
    (ckpt / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (ckpt / "heads.pt").write_text("fake", encoding="utf-8")
    (ckpt / "temperature_scaling.json").write_text("{}", encoding="utf-8")
    (ckpt / "model_manifest.json").write_text(
        json.dumps({"base_model": "microsoft/deberta-v3-base"}, ensure_ascii=True),
        encoding="utf-8",
    )

    out_dir = tmp / "export"
    report = export_pitheta_adapter(checkpoint_dir=ckpt.as_posix(), export_dir=out_dir.as_posix())
    assert "export_dir" in report
    assert (out_dir / "model_manifest.json").exists()
    assert (out_dir / "heads.pt").exists()
    assert (out_dir / "temperature_scaling.json").exists()
    assert (out_dir / "adapter" / "adapter_config.json").exists()
    assert (out_dir / "tokenizer" / "tokenizer.json").exists()
    checksums = json.loads((out_dir / "checksums.json").read_text(encoding="utf-8"))
    assert "model_manifest.json" in checksums
