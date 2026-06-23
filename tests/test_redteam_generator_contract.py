from __future__ import annotations

from dataclasses import asdict

from omega.eval.attack_corpus import (
    DEFAULT_SESSION_PACK,
    FAMILY_TARGETS,
    generate,
    load_attack_samples_from_session_pack,
)


def _rows(seed: int = 7, n_per_family: int = 3):
    return [asdict(s) for s in generate(seed=seed, n_per_family=n_per_family)]


def test_attack_corpus_generate_contract_and_fields():
    rows = _rows(seed=11, n_per_family=2)
    assert rows
    for row in rows:
        assert row["id"]
        assert row["family"] in FAMILY_TARGETS
        assert row["text"]
        assert row["targets"] == FAMILY_TARGETS[row["family"]]
        assert row["attack_type"] == "cross_source_session_pack"
        assert isinstance(row["source_fragments"], list) and row["source_fragments"]
        assert row["trigger"]
        assert row["detection_rationale"]
        assert isinstance(row["quality_meta"], dict)
        assert row["quality_meta"].get("gate_passed") is True
        assert row["quality_meta"].get("generation_mode") == "session_pack"
        assert row["quality_meta"].get("llm_used") is False
        assert row["quality_meta"].get("base_sample_id")


def test_attack_corpus_generate_is_deterministic_for_seed():
    rows_a = _rows(seed=3, n_per_family=2)
    rows_b = _rows(seed=3, n_per_family=2)
    assert rows_a == rows_b


def test_attack_corpus_generate_respects_n_per_family_count():
    n_per_family = 4
    rows = _rows(seed=5, n_per_family=n_per_family)
    assert len(rows) == len(FAMILY_TARGETS) * n_per_family


def test_attack_corpus_shape_and_source_diversity():
    rows = _rows(seed=17, n_per_family=2)
    for row in rows:
        fragments = row["source_fragments"]
        phases = {f["phase"] for f in fragments}
        trusts = {f["trust"] for f in fragments}
        sessions = {f["session_id"] for f in fragments}
        max_gap = max(int(f.get("temporal_gap_from_prev_min", 0) or 0) for f in fragments)
        assert len(phases) >= 2
        assert "trusted" in trusts
        assert "untrusted" in trusts
        assert len(sessions) >= 2
        assert max_gap >= 45


def test_session_pack_surface_is_available_and_family_complete():
    samples = load_attack_samples_from_session_pack(DEFAULT_SESSION_PACK)
    assert samples
    families = {sample.family for sample in samples}
    assert families == set(FAMILY_TARGETS)
