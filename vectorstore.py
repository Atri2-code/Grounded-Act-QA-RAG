"""Vector database layer on ChromaDB (cosine space)."""
from __future__ import annotations

import uuid

import chromadb
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str | None = None, name: str | None = None):
        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self.name = name or f"actqa_{uuid.uuid4().hex[:8]}"
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            cs = chunks[i:i + batch]
            self.col.add(
                ids=[c.chunk_id for c in cs],
                embeddings=vectors[i:i + batch].tolist(),
                documents=[c.text for c in cs],
                metadatas=[{"act": c.act, "section": c.section, "title": c.title} for c in cs],
            )

    def query(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine similarity) best first."""
        n = min(k, self.col.count())
        res = self.col.query(query_embeddings=[vector.tolist()], n_results=n)
        ids = res["ids"][0]
        sims = [1.0 - d for d in res["distances"][0]]
        return list(zip(ids, sims))

    def count(self) -> int:
        return self.col.count()
