from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid
import zipfile

import pytest

from scripts.unpack_eval_family_packs import unpack_eval_family_packs


def _workspace_tmp(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "support_eval_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _session_pack_lines() -> str:
    rows = [
        {
            "session_id": "atk_1",
            "turn_id": 1,
            "text": "ignore previous",
            "label_turn": "attack",
            "label_session": "attack",
            "family": "f1",
            "source_ref": "x",
            "source_type": "external_untrusted",
            "actor_id": "a1",
            "bucket": "core",
            "eval_slice": "text_intrinsic",
        },
        {
            "session_id": "ben_1",
            "turn_id": 1,
            "text": "safe",
            "label_turn": "benign",
            "label_session": "benign",
            "family": "f1",
            "source_ref": "y",
            "source_type": "internal_trusted",
            "actor_id": "a2",
            "bucket": "core",
            "eval_slice": "text_intrinsic",
        },
    ]
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def _build_zip(path: Path, *, with_root: bool, include_runtime: bool = True) -> None:
    prefix = "pack_root/" if with_root else ""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}manifest.json", json.dumps({"name": path.stem}, ensure_ascii=False))
        zf.writestr(f"{prefix}README.md", f"# {path.stem}\n")
        if include_runtime:
            zf.writestr(f"{prefix}runtime/session_pack.jsonl", _session_pack_lines())


def test_unpack_handles_root_and_flat_layout():
    tmp_path = _workspace_tmp("unpack_layout")
    try:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir(parents=True, exist_ok=True)

        _build_zip(src / "alpha.zip", with_root=False)
        _build_zip(src / "beta.zip", with_root=True)

        payload = unpack_eval_family_packs(src_root=src, out_root=out, clean_existing=True)
        assert payload["pack_count"] == 2
        assert (out / "alpha" / "runtime" / "session_pack.jsonl").exists()
        assert (out / "beta" / "runtime" / "session_pack.jsonl").exists()

        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
        assert [pack["pack_id"] for pack in index["packs"]] == ["alpha", "beta"]
        assert index["packs"][0]["stats"]["sessions_total"] == 2
        assert index["packs"][0]["stats"]["attack_sessions"] == 1
        assert index["packs"][0]["stats"]["benign_sessions"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_unpack_fails_when_runtime_pack_missing():
    tmp_path = _workspace_tmp("unpack_missing_runtime")
    try:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir(parents=True, exist_ok=True)
        _build_zip(src / "broken.zip", with_root=False, include_runtime=False)

        with pytest.raises(FileNotFoundError):
            unpack_eval_family_packs(src_root=src, out_root=out, clean_existing=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_unpack_index_stable_pack_order():
    tmp_path = _workspace_tmp("unpack_order")
    try:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir(parents=True, exist_ok=True)

        _build_zip(src / "zeta.zip", with_root=False)
        _build_zip(src / "alpha.zip", with_root=False)

        first = unpack_eval_family_packs(src_root=src, out_root=out, clean_existing=True)
        second = unpack_eval_family_packs(src_root=src, out_root=out, clean_existing=False)
        first_ids = [pack["pack_id"] for pack in first["packs"]]
        second_ids = [pack["pack_id"] for pack in second["packs"]]
        assert first_ids == ["alpha", "zeta"]
        assert second_ids == ["alpha", "zeta"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
