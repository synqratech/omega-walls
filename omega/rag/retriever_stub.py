"""Deterministic retriever stub for tests and local harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from omega.interfaces.contracts_v1 import ContentItem


@dataclass
class RetrieverStub:
    packets_by_query: Dict[str, List[ContentItem]] = field(default_factory=dict)

    def search(self, query: str, k: int) -> List[ContentItem]:
        return list(self.packets_by_query.get(query, []))[:k]

    def retrieve(self, query: str) -> List[ContentItem]:
        return list(self.packets_by_query.get(query, []))
