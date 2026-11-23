"""Embedders. The default is offline (TF-IDF + truncated SVD, i.e. LSA); a neural one is optional."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexical import analyze


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class LSAEmbedder:
    """Dense vectors from TF-IDF followed by truncated SVD. Fit on the corpus only, never on questions."""

    name = "lsa"

    def __init__(self, dims: int = 256, seed: int = 42):
        self.dims = dims
        self.seed = seed
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True, min_df=1)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.svd.fit(X)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        assert self.svd is not None, "call fit() first"
        return _l2(self.svd.transform(self.vec.transform(texts)))


class SentenceTransformerEmbedder:
    """Optional neural embedder (pip install sentence-transformers). Not used for the shipped results."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"st:{model_name}"
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(self.model.encode(texts, show_progress_bar=False)))


def make_embedder(kind: str = "lsa", **kw) -> Embedder:
    if kind == "lsa":
        return LSAEmbedder(**kw)
    if kind.startswith("st:"):
        return SentenceTransformerEmbedder(kind[3:])
    raise ValueError(f"unknown embedder: {kind}")
