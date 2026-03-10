from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest

from omega.config.loader import load_resolved_config
from omega.integrations.llamaindex_adapter import (
    LlamaIndexRetrieverAdapter,
    build_llamaindex_vector_adapter_from_directory,
)


@dataclass
class _Node:
    text: str
    metadata: dict
    id_: str | None = None


@dataclass
class _NodeWithScore:
    node: _Node
    score: float


class _DummyRetriever:
    def retrieve(self, query: str):  # noqa: ANN001
        del query
        return [
            _NodeWithScore(
                node=_Node(text="ticket evidence", metadata={"source_id": "ticket:1", "source_type": "ticket"}, id_="n1"),
                score=0.9,
            ),
            _NodeWithScore(
                node=_Node(text="   ", metadata={"source_id": "web:drop", "source_type": "web"}, id_="drop"),
                score=0.5,
            ),
            _NodeWithScore(node=_Node(text="fallback node", metadata={"source": "x:1"}), score=0.2),
        ]


def test_llamaindex_adapter_contract_and_empty_text_filtering():
    cfg = load_resolved_config(profile="dev").resolved
    adapter = LlamaIndexRetrieverAdapter.from_config(retriever=_DummyRetriever(), config=cfg)
    items = adapter.search("q", 5)

    assert len(items) == 2
    assert items[0].doc_id == "n1"
    assert items[0].source_type == "ticket"
    assert items[0].trust == "semi"
    assert items[1].source_type == "other"
    assert items[1].trust == "untrusted"


def test_llamaindex_adapter_uses_source_policy_from_config():
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    cfg["source_policy"]["source_type_to_trust"]["ticket"] = "trusted"
    adapter = LlamaIndexRetrieverAdapter.from_config(retriever=_DummyRetriever(), config=cfg)
    items = adapter.search("q", 1)
    assert len(items) == 1
    assert items[0].source_type == "ticket"
    assert items[0].trust == "trusted"


def test_llamaindex_vector_builder_from_directory():
    pytest.importorskip("llama_index.core")

    cfg = load_resolved_config(profile="dev").resolved
    adapter = build_llamaindex_vector_adapter_from_directory(
        root_dir="data/smoke_sources/safe_index",
        config=cfg,
        similarity_top_k=2,
    )
    items = adapter.search("rotate api keys", 2)
    assert len(items) >= 1
    assert all(item.text for item in items)
