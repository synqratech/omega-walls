"""SQLite FTS-based retriever with BM25 fallback."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.projector.normalize import normalize_text, tokenize


@dataclass
class SQLiteFTSRetriever:
    corpus: List[ContentItem]
    homoglyph_map: Dict[str, str]

    def __post_init__(self) -> None:
        self._by_id = {item.doc_id: item for item in self.corpus}
        self._conn = sqlite3.connect(":memory:")
        self._use_fts = True

        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id UNINDEXED, source_id UNINDEXED, source_type UNINDEXED, trust UNINDEXED, text)"
            )
            self._conn.executemany(
                "INSERT INTO docs_fts(doc_id, source_id, source_type, trust, text) VALUES (?,?,?,?,?)",
                [(d.doc_id, d.source_id, d.source_type, d.trust, d.text) for d in self.corpus],
            )
        except sqlite3.OperationalError:
            self._use_fts = False

        self._fallback_docs: Dict[str, Counter[str]] = {}
        self._doc_len: Dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0
        if not self._use_fts:
            total_len = 0
            for item in self.corpus:
                tokens = tokenize(normalize_text(item.text, self.homoglyph_map))
                cnt = Counter(tokens)
                self._fallback_docs[item.doc_id] = cnt
                self._doc_len[item.doc_id] = len(tokens)
                total_len += len(tokens)
                for tok in cnt:
                    self._df[tok] += 1
            self._avg_len = total_len / max(1, len(self.corpus))

    def _sanitize_query_tokens(self, query: str) -> List[str]:
        tokens = tokenize(normalize_text(query, self.homoglyph_map))
        valid = re.compile(r"^[a-z0-9_]+$")
        clean = [tok for tok in tokens if valid.match(tok)]
        return clean

    def _retrieve_fts(self, query: str, top_k: int) -> List[ContentItem]:
        tokens = self._sanitize_query_tokens(query)
        if not tokens:
            return self.corpus[:top_k]

        terms = []
        for tok in tokens:
            if len(tok) >= 4:
                terms.append(f"{tok}*")
            else:
                terms.append(tok)
        match_expr = " OR ".join(terms)

        rows = self._conn.execute(
            "SELECT doc_id, bm25(docs_fts) AS score FROM docs_fts WHERE docs_fts MATCH ? ORDER BY score ASC LIMIT ?",
            (match_expr, int(top_k)),
        ).fetchall()

        if not rows:
            return self.corpus[:top_k]
        return [self._by_id[row[0]] for row in rows if row[0] in self._by_id]

    def _retrieve_fallback(self, query: str, top_k: int) -> List[ContentItem]:
        tokens = self._sanitize_query_tokens(query)
        if not tokens:
            return self.corpus[:top_k]

        N = max(1, len(self.corpus))
        k1 = 1.5
        b = 0.75

        scored = []
        for doc_id, tf in self._fallback_docs.items():
            score = 0.0
            dl = self._doc_len[doc_id]
            for tok in tokens:
                f = tf.get(tok, 0)
                if f == 0:
                    continue
                df = self._df.get(tok, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = f + k1 * (1 - b + b * dl / max(1.0, self._avg_len))
                score += idf * (f * (k1 + 1) / denom)
            scored.append((score, doc_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._by_id[doc_id] for score, doc_id in scored[:top_k] if score > 0.0]

    def retrieve(self, query: str, top_k: int = 4) -> List[ContentItem]:
        if self._use_fts:
            return self._retrieve_fts(query, top_k)
        return self._retrieve_fallback(query, top_k)

    def search(self, query: str, k: int) -> List[ContentItem]:
        return self.retrieve(query=query, top_k=k)
