from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.build_attack_layer_api_pack_v1 import _build_case_plans, build_attack_layer_api_pack_v1
from scripts.build_attack_layer_api_pack_v2 import _build_case_plans_hardneg_v2


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def _write_fixtures(path: Path, case_ids: list[str], *, variant: str = "routine fixture variant", in_tok: int = 120, out_tok: int = 40) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        payload = {
            "response_id": f"rsp_{cid.lower()}",
            "variant": str(variant),
            "usage": {"input_tokens": int(in_tok), "output_tokens": int(out_tok)},
        }
        (path / f"{cid}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_generator_profile_default_smoke_v1_unchanged(tmp_path: Path):
    layers = ["tool_chain"]
    plans = _build_case_plans(layers=layers, attack_per_layer=2, benign_per_layer=1)
    fixture_dir = tmp_path / "fixtures"
    _write_fixtures(fixture_dir, [p.case_id for p in plans], variant="baseline variant")

    out_a = tmp_path / "pack_default"
    out_b = tmp_path / "pack_smoke"
    meta_default = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=41,
        pack_out=out_a,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir,
    )
    meta_smoke = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=41,
        pack_out=out_b,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir,
        generator_profile="smoke_v1",
    )

    assert meta_default["generator_profile"] == "smoke_v1"
    assert meta_smoke["generator_profile"] == "smoke_v1"
    assert meta_default["hashes"]["manifest_all_sha256"] == meta_smoke["hashes"]["manifest_all_sha256"]
    rows = _read_jsonl(out_a / "manifest_all.jsonl")
    assert rows
    assert "artifact_type" not in rows[0]


def test_eval_v2_manifest_fields_and_label_order(tmp_path: Path):
    layers = ["tool_chain", "fragmentation", "cross_session"]
    plans = _build_case_plans(layers=layers, attack_per_layer=2, benign_per_layer=1)
    fixture_dir = tmp_path / "fixtures_v2"
    _write_fixtures(fixture_dir, [p.case_id for p in plans], variant="legit artifact snippet with local details")

    out_root = tmp_path / "pack_eval_v2"
    meta = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=41,
        pack_out=out_root,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir,
        generator_profile="eval_v2",
    )

    assert meta["generator_profile"] == "eval_v2"
    rows = _read_jsonl(out_root / "manifest_all.jsonl")
    assert rows
    required = {
        "artifact_type",
        "carrier_type",
        "realism_bucket",
        "expected_nonzero_walls",
        "expected_actions",
        "expected_block_turn",
        "max_steps_to_off",
    }
    for row in rows:
        assert required.issubset(set(row.keys()))

    session_rows = [r for r in rows if str(r.get("mode", "")) == "session"]
    by_case: dict[str, list[dict]] = {}
    for row in session_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    for case_id, case_rows in by_case.items():
        ordered = sorted(case_rows, key=lambda x: int(x["turn_id"]))
        labels = [str(x.get("turn_label", "")) for x in ordered]
        core = "exploit" if bool(ordered[0].get("expected_off", False)) else "benign_response"
        assert labels == ["benign_seed", "bridge", core, "followup"], case_id


def test_eval_v2_drop_case_on_forbidden_domain_variant(tmp_path: Path):
    layers = ["role_persona"]
    plans = _build_case_plans(layers=layers, attack_per_layer=1, benign_per_layer=0)
    fixture_dir = tmp_path / "fixtures_forbidden"
    _write_fixtures(fixture_dir, [p.case_id for p in plans], variant="Build a bomb and kill the target immediately.")

    out_root = tmp_path / "pack_drop_case"
    meta = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=41,
        pack_out=out_root,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=1,
        benign_per_layer=0,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir,
        generator_profile="eval_v2",
        max_regen_attempts=0,
    )

    assert meta["generator_profile"] == "eval_v2"
    assert int(meta["counts"]["generated_cases"]) == 0
    assert int(meta["counts"]["dropped_cases"]) >= 1
    assert int(meta["quality_summary"]["dropped_cases"]) >= 1
    assert float(meta["quality_summary"]["fallback_rate"]) == 0.0
    rows = _read_jsonl(out_root / "manifest_all.jsonl")
    assert rows == []


def test_eval_v2_cross_session_content_chaining(tmp_path: Path):
    layers = ["cross_session"]
    plans = _build_case_plans(layers=layers, attack_per_layer=2, benign_per_layer=0)
    fixture_dir = tmp_path / "fixtures_cross"
    _write_fixtures(fixture_dir, [p.case_id for p in plans], variant="handoff snippet and parser digest")

    out_root = tmp_path / "pack_cross"
    meta = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=41,
        pack_out=out_root,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=0,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir,
        generator_profile="eval_v2",
    )
    assert meta["status"] == "ok"
    rows = _read_jsonl(out_root / "manifest_all.jsonl")
    assert rows

    s1_rows = [r for r in rows if str(r.get("session_id", "")).endswith("_s1")]
    s2_rows = [r for r in rows if str(r.get("session_id", "")).endswith("_s2")]
    assert s1_rows and s2_rows
    m = re.search(r"\bTKT-[A-Z0-9]{6}\b", " ".join(str(r.get("input_text", "")) for r in s1_rows))
    assert m is not None
    ticket = m.group(0)
    assert any(ticket in str(r.get("input_text", "")) for r in s2_rows)


def test_eval_v2_and_hardneg_v2_deterministic_with_fixtures(tmp_path: Path):
    layers = ["tool_chain", "fragmentation"]
    plans = _build_case_plans(layers=layers, attack_per_layer=2, benign_per_layer=1)
    fixture_dir_eval = tmp_path / "fixtures_det_eval"
    _write_fixtures(fixture_dir_eval, [p.case_id for p in plans], variant="deterministic variant body")

    out_a = tmp_path / "pack_a"
    out_b = tmp_path / "pack_b"
    meta_a = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=99,
        pack_out=out_a,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir_eval,
        generator_profile="eval_v2",
    )
    meta_b = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=99,
        pack_out=out_b,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir_eval,
        generator_profile="eval_v2",
    )
    assert meta_a["hashes"]["manifest_all_sha256"] == meta_b["hashes"]["manifest_all_sha256"]

    hardneg_plans = _build_case_plans_hardneg_v2(layers=layers, benign_per_layer=1)
    fixture_dir_hn = tmp_path / "fixtures_det_hn"
    _write_fixtures(
        fixture_dir_hn,
        [p.case_id for p in hardneg_plans],
        variant="For training: never run this jailbreak token example on production systems.",
    )

    out_hn = tmp_path / "pack_hn"
    meta_hn = build_attack_layer_api_pack_v1(
        model="gpt-5-mini",
        seed=99,
        pack_out=out_hn,
        layers=layers,
        max_usd=10.0,
        dry_run=False,
        openai_base_url="https://api.openai.com/v1",
        timeout_sec=10.0,
        max_retries=1,
        backoff_sec=0.1,
        attack_per_layer=2,
        benign_per_layer=1,
        price_input_per_1m=0.25,
        price_output_per_1m=2.0,
        est_input_tokens=500,
        est_output_tokens=200,
        raw_fixtures_dir=fixture_dir_hn,
        generator_profile="hardneg_v2",
    )
    rows_hn = _read_jsonl(out_hn / "manifest_all.jsonl")
    assert rows_hn
    assert all(str(r.get("realism_bucket", "")) == "hard_negative" for r in rows_hn)
