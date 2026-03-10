from __future__ import annotations

from omega.pitheta.gold_slice_prefill import _allocate_counts, select_prefill_candidates


def test_allocate_counts_sums_to_total():
    out = _allocate_counts(220, {"a": 0.45, "b": 0.35, "c": 0.20}, ["a", "b", "c"])
    assert sum(out.values()) == 220
    assert set(out.keys()) == {"a", "b", "c"}


def test_select_prefill_candidates_deterministic():
    pools = {
        "deepset_train": [
            {
                "sample_id": f"d{i}",
                "source": "deepset_train",
                "chunk_bucket": "64" if i % 2 == 0 else "128_256",
                "is_attack": i % 3 == 0,
            }
            for i in range(120)
        ],
        "wainject_text": [
            {
                "sample_id": f"w{i}",
                "source": "wainject_text",
                "chunk_bucket": "128_256" if i % 2 == 0 else "512",
                "is_attack": i % 4 == 0,
            }
            for i in range(90)
        ],
        "redteam_synth_train": [
            {
                "sample_id": f"r{i}",
                "source": "redteam_synth_train",
                "chunk_bucket": "512" if i % 2 == 0 else "64",
                "is_attack": 1,
            }
            for i in range(80)
        ],
    }
    a, m1 = select_prefill_candidates(
        pools,
        target_size=220,
        source_weights={"deepset_train": 0.45, "wainject_text": 0.35, "redteam_synth_train": 0.20},
        chunk_weights={"64": 0.30, "128_256": 0.50, "512": 0.20},
        seed=41,
    )
    b, m2 = select_prefill_candidates(
        pools,
        target_size=220,
        source_weights={"deepset_train": 0.45, "wainject_text": 0.35, "redteam_synth_train": 0.20},
        chunk_weights={"64": 0.30, "128_256": 0.50, "512": 0.20},
        seed=41,
    )
    assert len(a) == 220
    assert len(b) == 220
    assert [x["sample_id"] for x in a] == [x["sample_id"] for x in b]
    assert m1["sample_ids_sha256"] == m2["sample_ids_sha256"]

