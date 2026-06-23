from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from scripts.build_stateful_only_support_pack import _write_pack


def test_write_pack_dual_runtime_contract():
    out_root = Path("tmp_codex_pytest") / "support_pack_builder" / f"dual_{uuid.uuid4().hex}"
    try:
        rows = [
            {
                "session_id": "legacy_attack_session",
                "turn_id": 1,
                "text": "hello",
                "label_turn": "benign",
                "label_session": "attack",
                "family": "xsrc_override_secret",
                "source_ref": "legacy:src:1",
                "source_type": "external_untrusted",
                "actor_id": "legacy_actor_attack",
                "bucket": "cross_session",
                "eval_slice": "context_required",
                "meta_phase": "delivery",
                "meta_rel_time_min": 7,
            }
        ]
        scenes = [
            {
                "scene_id": "scene_001",
                "session_id": "legacy_attack_session",
                "label_session": "attack",
                "pair_id": "pair_1",
                "family": "xsrc_override_secret",
                "source_session": "session_a",
            }
        ]
        built = _write_pack(
            out_root=out_root,
            pack_id="pack_test",
            rows=rows,
            scenes=scenes,
            stats={"sessions_total": 1, "attack_sessions": 1, "benign_sessions": 0, "turns_total": 1, "attack_turns": 0, "benign_turns": 1},
            source_inputs=[],
        )
        runtime_path = built.runtime_pack_path
        labels_path = built.pack_root / "labels" / "session_pack_labels.jsonl"
        assert runtime_path.exists()
        assert labels_path.exists()

        runtime_rows = [json.loads(x) for x in runtime_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert runtime_rows
        row = runtime_rows[0]
        assert set(row.keys()) == {"session_id", "turn_id", "text", "source_type", "source_id"}
        assert str(row["session_id"]).startswith("s_")
        assert str(row["source_id"]).startswith("src_")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
