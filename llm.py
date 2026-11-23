"""Optional generative answerer on the Anthropic API.

The model may only use the retrieved sections, must cite them, and must say NOT_FOUND otherwise.
Every answer is then checked: citations must be among the retrieved sections and each sentence must
be grounded in the cited text. Anything that fails the check is turned into an abstention.
The client is injected so the logic is testable offline; this module is not run in the shipped results.
"""
from __future__ import annotations

import os
import re

from .answer import Answer, grounded_fraction
from .data import Section
from .retrieval import Retrieval

SYSTEM = (
    "You answer questions about Indian Acts using ONLY the sections provided. "
    "Cite every claim like [NIA §138]. If the sections do not contain the answer, reply exactly NOT_FOUND. "
    "A fluent wrong answer is worse than no answer."
)
_CITE = re.compile(r"\[([A-Z]+) §(\w+)\]")


def build_prompt(question: str, sections: list[Section]) -> str:
    ctx = "\n\n".join(f"[{s.act} §{s.number}] {s.title}\n{s.text}" for s in sections)
    return f"Sections:\n{ctx}\n\nQuestion: {question}"


class LLMAnswerer:
    def __init__(self, sections: dict[tuple[str, str], Section], client=None, model: str | None = None,
                 top_sections: int = 3, min_grounded: float = 1.0):
        if client is None:
            import anthropic  # lazy import; needs ANTHROPIC_API_KEY
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("ACTQA_MODEL", "claude-sonnet-5-5")
        self.sections = sections
        self.top_sections = top_sections
        self.min_grounded = min_grounded

    def answer(self, retrieval: Retrieval) -> Answer:
        picked = [self.sections[h.key] for h in retrieval.hits[: self.top_sections]]
        if not picked:
            return Answer(None, [], True, retrieval.signals, reason="no results")
        msg = self.client.messages.create(
            model=self.model, max_tokens=500, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(retrieval.query, picked)}],
        )
        text = msg.content[0].text.strip()
        if text.upper().startswith("NOT_FOUND"):
            return Answer(None, [], True, retrieval.signals, reason="model said not found")
        allowed = {s.key for s in picked}
        cites = [(a, n) for a, n in _CITE.findall(text)]
        if not cites or any(c not in allowed for c in cites):
            return Answer(None, [], True, retrieval.signals, reason="missing or invalid citation")
        evidence = [self.sections[c].text for c in set(cites)]
        if grounded_fraction(text, evidence) < self.min_grounded:
            return Answer(None, [], True, retrieval.signals, reason="ungrounded sentence")
        return Answer(text, list(dict.fromkeys(cites)), False, retrieval.signals)
