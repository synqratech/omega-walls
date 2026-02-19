"""Retriever protocol for production-oriented adapters."""

from __future__ import annotations

from typing import List, Protocol

from omega.interfaces.contracts_v1 import ContentItem


class Retriever(Protocol):
    def search(self, query: str, k: int) -> List[ContentItem]:
        ...
