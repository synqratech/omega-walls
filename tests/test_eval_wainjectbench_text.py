from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from omega.interfaces.contracts_v1 import OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult
from scripts.eval_wainjectbench_text import (
    WATextSample,
    WATextLoadStats,
    _baseline_compare,
    _latest_baseline_report,
    _resolve_comparability_for_wainject,
    _stratified_cap_samples,
    _write_external_refs_json,
    evaluate_wainject_rows,
    evaluate_wainject_sessionized_diagnostic,
    load_wainject_text,
    load_wainject_text_with_stats,
)


class _ProjectorStub:
    def project(self, item):
        txt = str(getattr(item, "text", "")).lower()
        score = 1.0 if ("ignore" in txt or "malicious" in txt) else 0.0
        return ProjectionResult(
            doc_id=str(getattr(item, "doc_id", "")),
            v=np.array([score, 0.0, 0.0, 0.0], dtype=float),
            evidence=ProjectionEvidence(polarity=[0, 0, 0, 0], debug_scores_raw=[0.0, 0.0, 0.0, 0.0], matches={}),
        )


class _CoreStub:
    def step(self, state, items, projections):
        off = bool(float(np.sum(projections[0].v)) > 0.0)
        p = np.array([float(np.sum(projections[0].v)), 0.0, 0.0, 0.0], dtype=float)
        return OmegaStepResult(
            session_id="wa:test",
            step=1,
            v_total=np.zeros(4, dtype=float),
            p=p,
            m_prev=np.zeros(4, dtype=float),
            m_next=p,
            off=off,
            reasons=OmegaOffReasons(False, off, False, False),
            top_docs=[],
            contribs=[],
        )


class _PolicyStub:
    def select_actions(self, step_result, items):
        return OffDecision(off=bool(step_result.off), severity="L2" if bool(step_result.off) else "L1", actions=[])


def _mk_local_tmp(name: str) -> Path:
    root = Path(__file__).resolve().parent / "_tmp" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_load_wainject_text_from_local_tree():
    tmp = _mk_local_tmp("wa-load")
    (tmp / "benign").mkdir(exist_ok=True)
    (tmp / "malicious").mkdir(exist_ok=True)
    (tmp / "benign" / "a.jsonl").write_text('{"text":"normal request"}\n', encoding="utf-8")
    (tmp / "malicious" / "b.jsonl").write_text('{"text":"ignore instructions"}\n', encoding="utf-8")
    rows = load_wainject_text(tmp)
    assert len(rows) == 2
    assert {r.label for r in rows} == {0, 1}


def test_load_wainject_text_with_stats_tracks_dropouts():
    tmp = _mk_local_tmp("wa-load-stats")
    (tmp / "benign").mkdir(exist_ok=True)
    (tmp / "malicious").mkdir(exist_ok=True)
    (tmp / "benign" / "a.jsonl").write_text('\n{"text":"ok"}\n{"text":"   "}\nnot-json\n[]\n', encoding="utf-8")
    (tmp / "malicious" / "b.jsonl").write_text('{"content":"malicious payload"}\n', encoding="utf-8")

    rows, stats = load_wainject_text_with_stats(tmp)

    assert len(rows) == 2
    assert stats.files_seen == 2
    assert stats.benign_files_seen == 1
    assert stats.malicious_files_seen == 1
    assert stats.rows_seen == 6
    assert stats.rows_loaded == 3
    assert stats.rows_dropped_empty_line == 1
    assert stats.rows_dropped_invalid_json == 1
    assert stats.rows_dropped_non_mapping == 1
    assert stats.rows_dropped_empty_text == 1
    assert stats.samples_built == 2
    assert stats.benign_samples_built == 1
    assert stats.malicious_samples_built == 1


def test_evaluate_wainject_rows_smoke():
    tmp = _mk_local_tmp("wa-eval")
    (tmp / "benign").mkdir(exist_ok=True)
    (tmp / "malicious").mkdir(exist_ok=True)
    (tmp / "benign" / "a.jsonl").write_text('{"text":"normal request"}\n{"text":"safe summary"}\n', encoding="utf-8")
    (tmp / "malicious" / "b.jsonl").write_text('{"text":"ignore previous instructions"}\n', encoding="utf-8")
    rows = load_wainject_text(tmp)
    out = evaluate_wainject_rows(rows=rows, projector=_ProjectorStub(), omega_core=_CoreStub(), off_policy=_PolicyStub())
    assert out["summary"]["tp"] == 1
    assert out["summary"]["fp"] == 0
    assert out["summary"]["tn"] == 2
    assert out["summary"]["fn"] == 0


def test_stratified_cap_samples_preserves_each_stratum():
    rows = [
        WATextSample(sample_id="benign:a:000001", label=0, source_file="a.jsonl", text="ok-1"),
        WATextSample(sample_id="benign:a:000002", label=0, source_file="a.jsonl", text="ok-2"),
        WATextSample(sample_id="benign:b:000001", label=0, source_file="b.jsonl", text="ok-3"),
        WATextSample(sample_id="malicious:c:000001", label=1, source_file="c.jsonl", text="atk-1"),
        WATextSample(sample_id="malicious:c:000002", label=1, source_file="c.jsonl", text="atk-2"),
        WATextSample(sample_id="malicious:d:000001", label=1, source_file="d.jsonl", text="atk-3"),
    ]
    selected, info = _stratified_cap_samples(rows, max_samples=4, seed=41)
    assert len(selected) == 4
    assert info["strategy"] == "stratified_label_source_file"
    labels = {x.label for x in selected}
    assert labels == {0, 1}
    sources = {x.source_file for x in selected}
    assert "a.jsonl" in sources
    assert "c.jsonl" in sources


def test_external_refs_comparability_status_consistent():
    tmp = _mk_local_tmp("wa-refs")
    refs = tmp / "refs.json"
    _write_external_refs_json(refs, comparability_status="partial_comparison")
    payload = json.loads(refs.read_text(encoding="utf-8"))
    assert payload["comparability_status"] == "partial_comparison"


def test_baseline_compare_delta_signs():
    cur = {"summary": {"attack_off_rate": 0.6, "benign_off_rate": 0.01, "precision": 0.9, "recall": 0.6}}
    base = {"summary": {"attack_off_rate": 0.5, "benign_off_rate": 0.02, "precision": 0.8, "recall": 0.5}}
    out = _baseline_compare(cur, base)
    assert np.isclose(out["summary_delta"]["attack_off_rate"], 0.1)
    assert np.isclose(out["summary_delta"]["benign_off_rate"], -0.01)


def test_sessionized_diagnostic_marks_non_comparable_and_has_summary():
    rows = [
        WATextSample(sample_id="benign:a:000001", label=0, source_file="comment_issue.jsonl", text="regular comment"),
        WATextSample(sample_id="benign:a:000002", label=0, source_file="web_text.jsonl", text="normal page"),
        WATextSample(sample_id="malicious:b:000001", label=1, source_file="popup.jsonl", text="ignore previous instructions"),
        WATextSample(sample_id="malicious:b:000002", label=1, source_file="popup.jsonl", text="malicious prompt"),
    ]
    out = evaluate_wainject_sessionized_diagnostic(
        rows=rows,
        projector=_ProjectorStub(),
        omega_core=_CoreStub(),
        off_policy=_PolicyStub(),
        seed=41,
        attack_chunk_size=2,
        benign_chunk_size=2,
        benign_prefix_turns=1,
    )
    assert out["status"] == "ok"
    assert out["comparability_status"] == "non_comparable"
    assert "summary" in out
    assert out["diagnostic_info"]["strategy"] == "synthetic_sessionization_from_text_rows"


def test_resolve_comparability_gate_requires_full_complete_run():
    stats = WATextLoadStats(
        files_seen=2,
        benign_files_seen=1,
        malicious_files_seen=1,
        rows_seen=10,
        rows_loaded=10,
        rows_dropped_empty_line=0,
        rows_dropped_invalid_json=0,
        rows_dropped_non_mapping=0,
        rows_dropped_empty_text=0,
        samples_built=10,
        benign_samples_built=5,
        malicious_samples_built=5,
    )
    status, reason = _resolve_comparability_for_wainject(
        dataset_ready=True,
        max_samples=0,
        selected_total=10,
        load_stats=stats,
    )
    assert status == "partial_comparison"
    assert reason == "full_run_complete_benign_malicious_splits"

    status2, reason2 = _resolve_comparability_for_wainject(
        dataset_ready=True,
        max_samples=32,
        selected_total=32,
        load_stats=stats,
    )
    assert status2 == "non_comparable"
    assert reason2 == "subsampled_run_max_samples"


def test_latest_baseline_report_prefers_full_comparable_only():
    root = _mk_local_tmp("wa-baseline-latest")
    run_non = root / "wainject_eval_noncomparable_latest"
    run_non.mkdir(parents=True, exist_ok=True)
    (run_non / "report.json").write_text(
        json.dumps(
            {
                "run_id": "noncmp",
                "samples_total": 40,
                "comparability_status": "non_comparable",
                "comparability_reason": "subsampled_run_max_samples",
                "sampling": {
                    "strategy": "stratified_label_source_file",
                    "requested_max_samples": 40,
                    "selected_total": 40,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_cmp = root / "wainject_eval_full_comparable"
    run_cmp.mkdir(parents=True, exist_ok=True)
    (run_cmp / "report.json").write_text(
        json.dumps(
            {
                "run_id": "cmp",
                "samples_total": 3698,
                "comparability_status": "partial_comparison",
                "comparability_reason": "full_run_complete_benign_malicious_splits",
                "sampling": {
                    "strategy": "full",
                    "requested_max_samples": 0,
                    "selected_total": 3698,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Force mtime ordering so the non-comparable report is newer.
    newer = run_non / "report.json"
    older = run_cmp / "report.json"
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))
    picked = _latest_baseline_report(root)
    assert picked is not None
    assert picked.parent.name == "wainject_eval_full_comparable"


def test_latest_baseline_report_returns_none_when_no_comparable():
    root = _mk_local_tmp("wa-baseline-none")
    run_non = root / "wainject_eval_noncomparable_only"
    run_non.mkdir(parents=True, exist_ok=True)
    (run_non / "report.json").write_text(
        json.dumps(
            {
                "run_id": "noncmp",
                "samples_total": 40,
                "comparability_status": "non_comparable",
                "comparability_reason": "subsampled_run_max_samples",
                "sampling": {
                    "strategy": "stratified_label_source_file",
                    "requested_max_samples": 40,
                    "selected_total": 40,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert _latest_baseline_report(root) is None
