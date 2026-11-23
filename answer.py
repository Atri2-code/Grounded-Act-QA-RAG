"""Grounded answering. The default answerer is extractive: it can only return sentences from the Act."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import split_sentences
from .data import Section
from .lexical import tokenize
from .retrieval import Retrieval, Retriever

NOT_FOUND = "I could not find this in the indexed Acts."


@dataclass
class Answer:
    text: str | None
    citations: list[tuple[str, str]]
    abstained: bool
    signals: dict = field(default_factory=dict)
    related: list[tuple[str, str]] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "answer": self.text if not self.abstained else NOT_FOUND,
            "abstained": self.abstained,
            "citations": [f"{a} s.{s}" for a, s in self.citations],
            "related": [f"{a} s.{s}" for a, s in self.related],
            "signals": self.signals,
            "reason": self.reason,
        }


def content_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def grounded_fraction(answer: str, evidence: list[str], min_recall: float = 0.8) -> float:
    """Fraction of answer sentences whose content words are (almost) all present in the evidence."""
    ev = set()
    for e in evidence:
        ev |= content_tokens(e)
    sents = [s for s in split_sentences(re.sub(r"\[[A-Z]+ §\w+\]", "", answer)) if content_tokens(s)]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        toks = content_tokens(s)
        if len(toks & ev) / len(toks) >= min_recall:
            ok += 1
    return ok / len(sents)


class ExtractiveAnswerer:
    """Pick the best sentences of the top section. Abstain when retrieval evidence is weak.

    Defaults are the values tuned on the dev split (see results/report.md): at least 70% of the
    question's content words must appear in the top chunk, and three sentences are returned.
    """

    def __init__(self, retriever: Retriever, sections: dict[tuple[str, str], Section],
                 tau_dense: float = 0.0, tau_coverage: float = 0.7, n_sentences: int = 3):
        self.retriever = retriever
        self.sections = sections
        self.tau_dense = tau_dense
        self.tau_coverage = tau_coverage
        self.n_sentences = n_sentences

    def should_abstain(self, signals: dict) -> bool:
        return signals["dense_top"] < self.tau_dense or signals["coverage"] < self.tau_coverage

    def answer(self, retrieval: Retrieval) -> Answer:
        if not retrieval.hits:
            return Answer(None, [], True, retrieval.signals, reason="no results")
        if self.should_abstain(retrieval.signals):
            return Answer(None, [], True, retrieval.signals, reason="weak evidence")
        top = retrieval.hits[0]
        sec = self.sections[top.key]
        q = content_tokens(retrieval.query)
        sents = split_sentences(sec.text)
        scored = []
        for i, s in enumerate(sents):
            toks = content_tokens(s)
            score = sum(self.retriever.bm25.idf(t) for t in toks & q)
            scored.append((score, i, s))
        best = sorted(scored, key=lambda x: (-x[0], x[1]))[: self.n_sentences]
        text = " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))
        related = [h.key for h in retrieval.hits[1:3]]
        return Answer(text, [top.key], False, retrieval.signals, related)
