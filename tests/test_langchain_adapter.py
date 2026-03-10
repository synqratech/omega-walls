from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest

from omega.config.loader import load_resolved_config
from omega.integrations.langchain_adapter import (
    LangChainRetrieverAdapter,
    build_langchain_bm25_adapter_from_directory,
)


@dataclass
class _Doc:
    page_content: str
    metadata: dict


class _DummyRetriever:
    def invoke(self, query: str, config=None):  # noqa: ANN001
        del query
        del config
        return [
            _Doc(page_content="wiki guidance", metadata={"doc_id": "d1", "source_id": "wiki:1", "source_type": "wiki"}),
            _Doc(page_content="   ", metadata={"doc_id": "drop", "source_id": "web:drop", "source_type": "web"}),
            _Doc(page_content="fallback type", metadata={"id": "d3", "source": "ticket:42"}),
        ]


def test_langchain_adapter_contract_and_empty_text_filtering():
    cfg = load_resolved_config(profile="dev").resolved
    adapter = LangChainRetrieverAdapter.from_config(retriever=_DummyRetriever(), config=cfg)
    items = adapter.search("q", 5)

    assert len(items) == 2
    assert items[0].doc_id == "d1"
    assert items[0].source_type == "wiki"
    assert items[0].trust == "trusted"
    assert items[1].doc_id == "d3"
    assert items[1].source_type == "other"
    assert items[1].trust == "untrusted"


def test_langchain_adapter_uses_source_policy_from_config():
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    cfg["source_policy"]["source_type_to_trust"]["wiki"] = "untrusted"
    adapter = LangChainRetrieverAdapter.from_config(retriever=_DummyRetriever(), config=cfg)
    items = adapter.search("q", 1)
    assert len(items) == 1
    assert items[0].source_type == "wiki"
    assert items[0].trust == "untrusted"


def test_langchain_bm25_builder_from_directory():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_community")

    cfg = load_resolved_config(profile="dev").resolved
    adapter = build_langchain_bm25_adapter_from_directory(
        root_dir="data/smoke_sources/safe_index",
        config=cfg,
        k=2,
    )
    items = adapter.search("rotate api keys", 2)
    assert len(items) >= 1
    assert all(item.text for item in items)
