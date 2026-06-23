from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import uuid

from scripts.build_session_benchmark_pack import build_session_pack, write_session_pack


def _group(rows):
    out = defaultdict(list)
    for row in rows:
        out[str(row["session_id"])].append(row)
    return out


def test_session_pack_builder_is_deterministic():
    root = Path(__file__).resolve().parent.parent
    a = build_session_pack(repo_root=root, seed=41)
    b = build_session_pack(repo_root=root, seed=41)
    assert [
        (
            r["session_id"],
            r["turn_id"],
            r["family"],
            r["label_turn"],
            r["label_session"],
            r["bucket"],
            r["eval_slice"],
            r["actor_id"],
            r["text"],
        )
        for r in a
    ] == [
        (
            r["session_id"],
            r["turn_id"],
            r["family"],
            r["label_turn"],
            r["label_session"],
            r["bucket"],
            r["eval_slice"],
            r["actor_id"],
            r["text"],
        )
        for r in b
    ]


def test_session_pack_builder_schema_and_balance():
    root = Path(__file__).resolve().parent.parent
    rows = build_session_pack(repo_root=root, seed=41)
    sessions = _group(rows)

    assert len(sessions) == 300
    attack_sessions = [sid for sid, turns in sessions.items() if str(turns[0]["label_session"]) == "attack"]
    benign_sessions = [sid for sid, turns in sessions.items() if str(turns[0]["label_session"]) == "benign"]
    assert len(attack_sessions) == 210
    assert len(benign_sessions) == 90

    family_set = {str(turns[0]["family"]) for turns in sessions.values()}
    assert family_set == {
        "cocktail",
        "distributed_wo_explicit",
        "roleplay_escalation",
        "tool_exfil_chain",
        "benign_long_context",
    }

    mixed_attack = 0
    cross_sessions = 0
    seen_context_required = 0
    for sid, turns in sessions.items():
        turn_ids = sorted(int(t["turn_id"]) for t in turns)
        assert turn_ids == list(range(1, len(turns) + 1))
        assert 3 <= len(turns) <= 8
        if str(turns[0]["bucket"]) == "cross_session":
            cross_sessions += 1
        if any(str(t.get("eval_slice", "")) == "context_required" for t in turns):
            seen_context_required += 1
        if str(turns[0]["label_session"]) == "attack":
            labels = {str(t["label_turn"]) for t in turns}
            if {"attack", "benign"}.issubset(labels):
                mixed_attack += 1

    assert cross_sessions == 30
    assert mixed_attack == 42
    assert seen_context_required > 0


def test_write_session_pack_dual_runtime_contract():
    root = Path(__file__).resolve().parent.parent
    rows = build_session_pack(repo_root=root, seed=41)
    out_root = Path("tmp_codex_pytest") / "session_pack_builder" / f"dual_{uuid.uuid4().hex}"
    try:
        result = write_session_pack(rows, out_root=out_root, seed=41)
        runtime_path = Path(result["runtime_jsonl"])
        labels_path = Path(result["labels_jsonl"])
        assert runtime_path.exists()
        assert labels_path.exists()

        runtime_rows = [json.loads(x) for x in runtime_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert runtime_rows
        for row in runtime_rows[:20]:
            assert set(row.keys()) == {"session_id", "turn_id", "text", "source_type", "source_id"}
            assert str(row["session_id"]).startswith("s_")
            assert str(row["source_id"]).startswith("src_")
            assert "label" not in "".join(row.keys())
            assert "family" not in "".join(row.keys())
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
