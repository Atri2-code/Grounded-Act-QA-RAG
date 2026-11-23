"""Section-aware chunking: a chunk never crosses a section boundary."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Section

_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(“\"\d\[])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    act: str
    section: str
    title: str
    header: str
    text: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.act, self.section)

    @property
    def index_text(self) -> str:
        """Text that is embedded and indexed: the header gives each chunk its context."""
        return f"{self.header}. {self.text}"


def chunk_section(sec: Section, max_chars: int = 700, overlap_sentences: int = 1) -> list[Chunk]:
    sents = split_sentences(sec.text)
    windows: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if cur and size + len(s) > max_chars:
            windows.append(cur)
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(x) for x in cur)
        cur.append(s)
        size += len(s)
    if cur:
        windows.append(cur)
    # drop a trailing window that only repeats the overlap sentences
    if len(windows) > 1 and len(windows[-1]) <= overlap_sentences:
        windows.pop()
    return [
        Chunk(f"{sec.act}-{sec.number}-{i}", sec.act, sec.number, sec.title, sec.header, " ".join(w))
        for i, w in enumerate(windows)
    ]


def chunk_corpus(sections: list[Section], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for sec in sections:
        out.extend(chunk_section(sec, **kw))
    return out
