from __future__ import annotations

import json
from pathlib import Path

from omega.interfaces.contracts_v1 import ContentItem


def load_jsonl(path: str):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def mk_item(doc_id: str, text: str, source_id: str = "synthetic:test") -> ContentItem:
    return ContentItem(
        doc_id=doc_id,
        source_id=source_id,
        source_type="other",
        trust="untrusted",
        text=text,
    )
