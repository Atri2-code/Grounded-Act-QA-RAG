import json

from actqa.chunking import chunk_corpus, chunk_section, split_sentences
from actqa.data import Section, clean, load_act, normalize


def test_clean_removes_footnote_markers():
    assert clean("1[(1) A person shall pay] 2*** a fee.") == "(1) A person shall pay a fee."


def test_normalize_straightens_quotes_and_case():
    assert normalize("“Cheque”  IS   here") == '"cheque" is here'


def test_load_act_skips_omitted_and_empty(tmp_path):
    rows = [
        {"section": 1, "title": "Real", "description": "A real section with enough text to keep."},
        {"section": 2, "title": "Gone", "description": "Omitted."},
        {"section": 3, "title": "Blank", "description": ""},
    ]
    p = tmp_path / "x.json"
    p.write_text(json.dumps(rows))
    secs = load_act(p, "X", "Test Act", ("section", "title", "description"))
    assert [s.number for s in secs] == ["1"]
    assert secs[0].header == "Test Act, section 1: Real"


def test_chunks_never_cross_sections(mini_sections):
    chunks = chunk_corpus(mini_sections, max_chars=80)
    for c in chunks:
        sec = next(s for s in mini_sections if s.number == c.section)
        assert c.text in sec.text
        assert c.key == (sec.act, sec.number)


def test_long_section_is_split_and_overlaps():
    text = " ".join(f"Sentence number {i} says something about the topic." for i in range(12))
    sec = Section("T", "1", "Long", text, "Test Act")
    chunks = chunk_section(sec, max_chars=150, overlap_sentences=1)
    assert len(chunks) > 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    first_last = split_sentences(chunks[0].text)[-1]
    assert first_last in chunks[1].text            # one sentence of overlap


def test_short_section_is_one_chunk(mini_sections):
    assert len(chunk_section(mini_sections[0])) == 1


def test_index_text_carries_the_header(mini_sections):
    c = chunk_section(mini_sections[3])[0]
    assert "Sample Library Act, section 4: Late fee" in c.index_text
