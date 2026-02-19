"""Simple keyword overlap retriever for local smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from omega.interfaces.contracts_v1 import ContentItem
from omega.projector.normalize import normalize_text, tokenize


@dataclass
class KeywordRetriever:
    corpus: List[ContentItem]
    homoglyph_map: dict[str, str]

    def search(self, query: str, k: int) -> List[ContentItem]:
        return self.retrieve(query=query, top_k=k)

    def retrieve(self, query: str, top_k: int = 4) -> List[ContentItem]:
        q_tokens = set(tokenize(normalize_text(query, self.homoglyph_map)))
        scored = []
        for item in self.corpus:
            t_tokens = set(tokenize(normalize_text(item.text, self.homoglyph_map)))
            overlap = len(q_tokens & t_tokens)
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
