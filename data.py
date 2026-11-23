"""Load Act sections from JSON files into a common Section record."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Short id -> (file name, display name, field names for the section number, title, text)
ACTS = {
    "NIA": ("nia.json", "Negotiable Instruments Act, 1881", ("section", "section_title", "section_desc")),
    "MVA": ("MVA.json", "Motor Vehicles Act, 1988", ("section", "title", "description")),
}


@dataclass(frozen=True)
class Section:
    act: str
    number: str
    title: str
    text: str
    act_name: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.act, self.number)

    @property
    def header(self) -> str:
        return f"{self.act_name or self.act}, section {self.number}: {self.title}"


_FOOTNOTE_OPEN = re.compile(r"(?<![\w])\d{1,2}\[")      # "1[(1) ..." amendment markers
_FOOTNOTE_STARS = re.compile(r"\d{1,2}\*{2,}")          # "2***" omitted-text markers
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Remove amendment footnote markers and collapse whitespace."""
    text = _FOOTNOTE_OPEN.sub("", text)
    text = _FOOTNOTE_STARS.sub("", text)
    text = text.replace("]", "")                      # closing brackets left behind by amendment markers
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lowercase, straighten quotes and collapse whitespace, for phrase matching."""
    text = clean(text).lower()
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")):
        text = text.replace(a, b)
    return text


def load_act(path: Path, act: str, act_name: str, fields: tuple[str, str, str]) -> list[Section]:
    num_f, title_f, text_f = fields
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Section] = []
    for row in rows:
        text = clean(str(row.get(text_f, "") or ""))
        title = clean(str(row.get(title_f, "") or ""))
        if len(text) < 25 and "omitted" in text.lower():
            continue
        if not text:
            continue
        out.append(Section(act, str(row[num_f]).strip(), title, text, act_name))
    return out


def load_corpus(data_dir: str | Path) -> list[Section]:
    """Load every known Act file found in data_dir."""
    data_dir = Path(data_dir)
    sections: list[Section] = []
    for act, (fname, act_name, fields) in ACTS.items():
        path = data_dir / fname
        if path.exists():
            sections.extend(load_act(path, act, act_name, fields))
    if not sections:
        raise FileNotFoundError(
            f"No Act files found in {data_dir}. Run `python scripts/fetch_corpus.py` first."
        )
    return sections
