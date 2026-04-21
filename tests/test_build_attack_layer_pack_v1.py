from __future__ import annotations

import json
from pathlib import Path

from scripts.build_attack_layer_pack_v1 import build_attack_layer_pack_v1
from scripts.run_attack_layer_cycle import _load_manifest_rows, validate_manifest_rows


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        rows.append(obj)
    return rows


def test_build_attack_layer_pack_v1_layout_and_contract(tmp_path: Path):
    out_root = tmp_path / "attack_layers" / "v1"
    meta = build_attack_layer_pack_v1(out_root=out_root, seed=41)
    assert meta["status"] == "ok"

    for layer in (
        "fragmentation",
        "context_accumulation",
        "tool_chain",
        "role_persona",
        "obfuscation",
        "refusal_erosion",
        "benign_stability",
        "cross_session",
        "temporal_deferred",
    ):
        assert (out_root / layer / "manifest.jsonl").exists()

    manifest_all = out_root / "manifest_all.jsonl"
    assert manifest_all.exists()
    rows = _read_jsonl(manifest_all)
    assert len(rows) > 0
    assert not any(str(r.get("layer")) == "temporal_deferred" for r in rows)

    parsed = _load_manifest_rows(manifest_all)
    validate_manifest_rows(parsed)
    assert len(parsed) == len(rows)


def test_build_attack_layer_pack_v1_deterministic(tmp_path: Path):
    out_root = tmp_path / "attack_layers" / "v1"
    build_attack_layer_pack_v1(out_root=out_root, seed=41)
    first_all = (out_root / "manifest_all.jsonl").read_text(encoding="utf-8")
    first_meta = (out_root / "manifest.meta.json").read_text(encoding="utf-8")

    build_attack_layer_pack_v1(out_root=out_root, seed=41)
    second_all = (out_root / "manifest_all.jsonl").read_text(encoding="utf-8")
    second_meta = (out_root / "manifest.meta.json").read_text(encoding="utf-8")

    assert first_all == second_all
    assert first_meta == second_meta
