from __future__ import annotations

from dataclasses import dataclass

from omega.config.loader import load_resolved_config
from omega.rag.retriever_adapters import ExternalRetrieverAdapter, SQLiteFTSRetrieverAdapter
from omega.rag.retriever_prod_adapter import RetrieverProdAdapter, build_retriever_prod_adapter
from omega.rag.retriever_provider import ExternalRetrieverProvider


@dataclass
class FakeRetrieverClient:
    def search(self, query: str, k: int):
        return [
            {
                "id": "hit-1",
                "text": f"{query} from web",
                "source_id": "web:https://example.com",
                "source_type": "web",
                "metadata": {"rank": 1},
            },
            {
                "id": "hit-2",
                "text": f"{query} from ticket",
                "source_id": "ticket:42",
                "source_type": "ticket",
                "metadata": {"rank": 2},
            },
        ][:k]


def test_external_retriever_adapter_maps_content_items():
    cfg = load_resolved_config(profile="dev").resolved
    adapter = ExternalRetrieverAdapter.from_config(client=FakeRetrieverClient(), config=cfg)
    items = adapter.search("rotation", 2)

    assert len(items) == 2
    assert items[0].doc_id == "hit-1"
    assert items[0].trust == "untrusted"
    assert items[1].trust == "semi"


def test_prod_adapter_fallbacks_and_empty_text_drop():
    cfg = load_resolved_config(profile="dev").resolved
    provider = ExternalRetrieverProvider(
        client=FakeRetrieverClientWithMissingFields()
    )
    adapter = RetrieverProdAdapter.from_config(provider=provider, config=cfg)
    items = adapter.search("rotation", 10)

    # One record with empty text must be dropped.
    assert len(items) == 2
    assert items[0].source_type == "other"
    assert items[0].trust == "untrusted"
    assert items[1].source_type == "ticket"
    assert items[1].trust == "semi"


def test_sqlite_fts_retriever_adapter_contract():
    cfg = load_resolved_config(profile="dev").resolved
    adapter = SQLiteFTSRetrieverAdapter.from_directory("data/smoke_sources/safe_index", config=cfg)
    items = adapter.search("rotate api keys", 2)

    assert len(items) >= 1
    assert all(hasattr(item, "doc_id") for item in items)
    assert all(item.trust in {"trusted", "semi", "untrusted"} for item in items)


def test_build_retriever_prod_adapter_sqlite_backend():
    cfg = load_resolved_config(profile="dev").resolved
    adapter = build_retriever_prod_adapter(config=cfg, source_root="data/smoke_sources/safe_index")
    items = adapter.search("credentials", 2)
    assert len(items) >= 1


@dataclass
class FakeRetrieverClientWithMissingFields:
    def search(self, query: str, k: int):
        del query
        del k
        return [
            {
                "id": "hit-missing",
                "text": "legit content",
                "source_id": "x:1",
                "metadata": {"rank": 1},
            },
            {
                "id": "hit-ticket",
                "text": "support details",
                "source_id": "ticket:55",
                "source_type": "ticket",
            },
            {
                "id": "hit-empty",
                "text": "   ",
                "source_id": "web:bad",
                "source_type": "web",
            },
        ]
