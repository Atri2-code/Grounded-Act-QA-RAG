"""End-to-end pipeline: load, chunk, embed, index, retrieve, answer."""
from __future__ import annotations

from pathlib import Path

from .answer import Answer, ExtractiveAnswerer
from .chunking import chunk_corpus
from .data import Section, load_corpus
from .embeddings import make_embedder
from .retrieval import Retrieval, Retriever
from .vectorstore import VectorStore


class Pipeline:
    def __init__(self, sections: list[Section], embedder: str = "lsa", max_chars: int = 700,
                 persist_dir: str | None = None, seed: int = 42):
        self.sections = {s.key: s for s in sections}
        self.chunks = chunk_corpus(sections, max_chars=max_chars)
        emb = make_embedder(embedder, seed=seed) if embedder == "lsa" else make_embedder(embedder)
        self.retriever = Retriever(self.chunks, emb, VectorStore(persist_dir))
        self.answerer = ExtractiveAnswerer(self.retriever, self.sections)

    @classmethod
    def from_dir(cls, data_dir: str | Path, **kw) -> "Pipeline":
        return cls(load_corpus(data_dir), **kw)

    def retrieve(self, question: str, k: int = 5, mode: str = "hybrid") -> Retrieval:
        return self.retriever.search(question, k=k, mode=mode)

    def ask(self, question: str, k: int = 5, mode: str = "hybrid", answerer=None) -> Answer:
        return (answerer or self.answerer).answer(self.retrieve(question, k=k, mode=mode))
