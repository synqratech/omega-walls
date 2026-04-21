from __future__ import annotations

import json
from pathlib import Path

from scripts.build_partner_policy_pack import build_partner_policy_pack


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        out.append(obj)
    return out


def test_build_partner_policy_pack_layout_and_counts(tmp_path: Path):
    out_root = tmp_path / "partner_policy_pack" / "v1"
    result = build_partner_policy_pack(out_root=out_root, seed=41)

    assert result["status"] == "ok"
    for folder in (
        "attack_explicit",
        "attack_cocktail",
        "attack_distributed",
        "attack_session_timed",
        "benign_pass",
        "benign_warn",
    ):
        assert (out_root / folder).is_dir()

    stateless_rows = _read_jsonl(out_root / "manifest_stateless.jsonl")
    session_rows = _read_jsonl(out_root / "manifest_session.jsonl")
    meta = json.loads((out_root / "manifest.meta.json").read_text(encoding="utf-8"))

    assert len(stateless_rows) == 24
    assert len({str(r["case_id"]) for r in stateless_rows}) == 24
    assert len({str(r["case_id"]) for r in session_rows}) == 24
    assert meta["counts"]["cases_total"] == 48
    assert meta["counts"]["stateless_cases"] == 24
    assert meta["counts"]["session_cases"] == 24


def test_build_partner_policy_pack_is_deterministic(tmp_path: Path):
    out_root = tmp_path / "partner_policy_pack" / "v1"
    build_partner_policy_pack(out_root=out_root, seed=41)
    first_stateless = (out_root / "manifest_stateless.jsonl").read_text(encoding="utf-8")
    first_session = (out_root / "manifest_session.jsonl").read_text(encoding="utf-8")
    first_meta = (out_root / "manifest.meta.json").read_text(encoding="utf-8")

    build_partner_policy_pack(out_root=out_root, seed=41)
    second_stateless = (out_root / "manifest_stateless.jsonl").read_text(encoding="utf-8")
    second_session = (out_root / "manifest_session.jsonl").read_text(encoding="utf-8")
    second_meta = (out_root / "manifest.meta.json").read_text(encoding="utf-8")

    assert first_stateless == second_stateless
    assert first_session == second_session
    assert first_meta == second_meta
