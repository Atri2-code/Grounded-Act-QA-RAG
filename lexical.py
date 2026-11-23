"""BM25 keyword retrieval."""
from __future__ import annotations

import re

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its of on or "
    "that the their there this to under was what when where which who whom why will with would you "
    "your my me they them any".split()
)


_STEM = PorterStemmer()
_CACHE: dict[str, str] = {}


def stem(tok: str) -> str:
    s = _CACHE.get(tok)
    if s is None:
        s = _CACHE[tok] = _STEM.stem(tok)
    return s


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stop words, Porter-stem (so 'magistrates' matches 'magistrate')."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def analyze(text: str) -> list[str]:
    """Analyzer for the TF-IDF vectorizer: stemmed unigrams plus bigrams."""
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class BM25Index:
    def __init__(self, docs: list[str]):
        self.tokens = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def idf(self, term: str) -> float:
        return float(self.bm25.idf.get(term, 0.0))
