"""Hybrid retrieval: BM25 + dense vectors (ChromaDB), fused with reciprocal rank fusion."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .lexical import BM25Index, tokenize
from .vectorstore import VectorStore

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    key: tuple[str, str]          # (act, section)
    chunk: Chunk                  # best-ranked chunk of that section
    score: float
    rank: int


@dataclass
class Retrieval:
    query: str
    hits: list[Hit]
    signals: dict = field(default_factory=dict)   # dense_top, coverage: used for abstention


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over ranked lists of ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, store: VectorStore | None = None, pool: int = 50):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder.fit([c.index_text for c in chunks])
        self.bm25 = BM25Index([c.index_text for c in chunks])
        self.store = store or VectorStore()
        if self.store.count() == 0:
            self.store.add(chunks, self.embedder.embed([c.index_text for c in chunks]))
        self.pool = pool

    def _bm25_ranking(self, query: str) -> list[tuple[str, float]]:
        s = self.bm25.scores(query)
        order = np.argsort(-s)[: self.pool]
        return [(self.chunks[i].chunk_id, float(s[i])) for i in order if s[i] > 0]

    def _dense_ranking(self, query: str) -> list[tuple[str, float]]:
        return self.store.query(self.embedder.embed([query])[0], k=self.pool)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        bm = self._bm25_ranking(query)
        de = self._dense_ranking(query)
        if mode == "bm25":
            ranked = bm
        elif mode == "dense":
            ranked = de
        else:
            ranked = rrf([[c for c, _ in bm], [c for c, _ in de]])
        hits: list[Hit] = []
        seen: set[tuple[str, str]] = set()
        for cid, score in ranked:
            ch = self.by_id[cid]
            if ch.key in seen:
                continue
            seen.add(ch.key)
            hits.append(Hit(ch.key, ch, float(score), len(hits) + 1))
            if len(hits) >= k:
                break
        return Retrieval(query, hits, self._signals(query, hits, de))

    def _signals(self, query: str, hits: list[Hit], dense: list[tuple[str, float]]) -> dict:
        qt = set(tokenize(query))
        if not hits or not qt:
            return {"dense_top": 0.0, "coverage": 0.0}
        top_tokens = set(tokenize(hits[0].chunk.index_text))
        coverage = len(qt & top_tokens) / len(qt)
        dense_top = dense[0][1] if dense else 0.0
        return {"dense_top": float(dense_top), "coverage": float(coverage)}
