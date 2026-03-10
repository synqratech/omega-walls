from __future__ import annotations

from typing import Any, Dict

import numpy as np

from omega.api.chunk_pipeline import score_chunks
from omega.interfaces.contracts_v1 import ContentItem, ProjectionEvidence, ProjectionResult


class _ProjectorStub:
    def __init__(self, rows: Dict[str, Dict[str, Any]]):
        self.rows = rows

    def project(self, item: ContentItem) -> ProjectionResult:
        row = self.rows.get(item.doc_id, {})
        v = np.array(row.get("v", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        matches = dict(row.get("matches", {}))
        return ProjectionResult(
            doc_id=item.doc_id,
            v=v,
            evidence=ProjectionEvidence(
                polarity=[1 if float(x) > 0 else 0 for x in v.tolist()],
                debug_scores_raw=[float(x) for x in v.tolist()],
                matches=matches,
            ),
        )


def _items(n: int) -> list[ContentItem]:
    out = []
    for i in range(n):
        out.append(
            ContentItem(
                doc_id=f"d{i}",
                source_id="src",
                source_type="other",
                trust="untrusted",
                text=f"chunk-{i}",
            )
        )
    return out


def test_chunk_pipeline_scores_worst_synergy_confidence():
    rows = {
        "d0": {
            "v": [0.82, 0.00, 0.76, 0.00],
            "matches": {"gapped_rule_ids": ["ovr_01"], "sql_db_pair_intent": True},
        },
        "d1": {
            "v": [0.12, 0.66, 0.00, 0.00],
            "matches": {"secret_leak_markers": ["return embeddings"]},
        },
    }
    projector = _ProjectorStub(rows)
    result = score_chunks(
        projector=projector,
        items=_items(2),
        walls=["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        cfg={
            "wall_trigger_threshold": 0.12,
            "synergy_pairs": [
                ["override_instructions", "tool_or_action_abuse"],
                ["override_instructions", "secret_exfiltration"],
            ],
        },
    )
    assert result.worst_chunk_score >= 0.80
    assert result.pattern_synergy > 0.0
    assert result.confidence > 0.0
    assert result.doc_score >= 0.55
    assert "override_instructions+tool_or_action_abuse" in result.pair_hits
    assert "gapped:ovr_01" in result.rule_ids
    assert "tool:sql_db_pair_intent" in result.rule_ids
    assert "d0" in result.triggered_chunk_ids
    assert any(r.startswith("chunk_worst") for r in result.reasons)


def test_chunk_pipeline_is_deterministic():
    rows = {
        "d0": {"v": [0.40, 0.00, 0.00, 0.00], "matches": {}},
        "d1": {"v": [0.38, 0.00, 0.00, 0.00], "matches": {}},
        "d2": {"v": [0.02, 0.00, 0.00, 0.00], "matches": {}},
    }
    projector = _ProjectorStub(rows)
    kwargs = {
        "projector": projector,
        "items": _items(3),
        "walls": ["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        "cfg": {"wall_trigger_threshold": 0.12},
    }
    a = score_chunks(**kwargs)
    b = score_chunks(**kwargs)
    assert a.doc_score == b.doc_score
    assert a.worst_chunk_score == b.worst_chunk_score
    assert a.pattern_synergy == b.pattern_synergy
    assert a.confidence == b.confidence
    assert a.top_chunks == b.top_chunks
