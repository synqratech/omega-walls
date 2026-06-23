from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from scripts.build_agent3sigma_stage_pack import main as build_agent3sigma_stage_pack_main
from scripts.session_pack_leakage import load_dual_session_pack


def test_agent3sigma_builder_emits_opaque_runtime_source_ids() -> None:
    out_root = Path("tmp_codex_pytest") / "agent3sigma_pack_builder" / f"dual_{uuid.uuid4().hex}"
    try:
        rc = build_agent3sigma_stage_pack_main(["--out-root", str(out_root)])
        assert rc == 0

        runtime_path = out_root / "runtime" / "session_pack.jsonl"
        labels_path = out_root / "labels" / "session_pack_labels.jsonl"
        assert runtime_path.exists()
        assert labels_path.exists()

        runtime_rows = [json.loads(x) for x in runtime_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert runtime_rows
        for row in runtime_rows[:20]:
            assert set(row.keys()) == {"session_id", "turn_id", "text", "source_type", "source_id"}
            assert str(row["session_id"]).startswith("s_")
            assert str(row["source_id"]).startswith("src_")
            assert str(row["source_id"]).count("_") == 1

        merged_rows = load_dual_session_pack(runtime_pack_path=runtime_path, labels_pack_path=labels_path)
        assert len(merged_rows) == len(runtime_rows)
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
