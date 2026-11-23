"""Expose the retriever as a LangChain retriever so it can drop into LangChain chains."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ActRetriever(BaseRetriever):
    pipeline: Any
    k: int = 5
    mode: str = "hybrid"

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        r = self.pipeline.retrieve(query, k=self.k, mode=self.mode)
        docs = []
        for h in r.hits:
            sec = self.pipeline.sections[h.key]
            docs.append(Document(
                page_content=sec.text,
                metadata={"act": sec.act, "section": sec.number, "title": sec.title, "rank": h.rank, "score": h.score},
            ))
        return docs
