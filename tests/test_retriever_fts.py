from __future__ import annotations

from omega.rag.retriever_fts import SQLiteFTSRetriever
from omega.rag.sources_fs import load_content_items_from_directory


def test_load_content_items_from_directory():
    items = load_content_items_from_directory("data/smoke_sources/safe_index")
    assert len(items) >= 3
    assert all(item.source_id.startswith("file:") for item in items)


def test_sqlite_fts_retriever_returns_relevant_doc():
    items = load_content_items_from_directory("data/smoke_sources/poisoned_index")
    retriever = SQLiteFTSRetriever(corpus=items, homoglyph_map={"0": "o", "1": "i"})

    result = retriever.retrieve("ignore instructions reveal token", top_k=1)
    assert len(result) == 1
    assert "ignore previous instructions" in result[0].text.lower()
