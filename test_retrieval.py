import pytest

from actqa.lexical import tokenize
from actqa.retrieval import rrf


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The magistrates are trying cheques") == ["magistr", "tri", "chequ"]


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert [x for x, _ in fused[:2]] == ["a", "b"] or [x for x, _ in fused[:2]] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert {x for x, _ in fused} == {"a", "b", "c", "d"}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_the_obvious_section(mini_pipeline, mode):
    r = mini_pipeline.retrieve("What is the late fee for returning a book after the due date?", k=3, mode=mode)
    assert ("LIB", "4") in [h.key for h in r.hits]


def test_hits_are_unique_sections_with_ranks(mini_pipeline):
    r = mini_pipeline.retrieve("borrow books limit", k=5)
    keys = [h.key for h in r.hits]
    assert len(keys) == len(set(keys))
    assert [h.rank for h in r.hits] == list(range(1, len(r.hits) + 1))


def test_signals_are_in_range(mini_pipeline):
    s = mini_pipeline.retrieve("late fee per day", k=3).signals
    assert 0.0 <= s["coverage"] <= 1.0
    assert -1.0 <= s["dense_top"] <= 1.0


def test_unknown_mode_raises(mini_pipeline):
    with pytest.raises(ValueError):
        mini_pipeline.retrieve("anything", mode="magic")


def test_vector_store_roundtrip(mini_pipeline):
    store = mini_pipeline.retriever.store
    assert store.count() == len(mini_pipeline.chunks)
    vec = mini_pipeline.retriever.embedder.embed(["late fee"])[0]
    ids = [i for i, _ in store.query(vec, k=3)]
    assert all(i in mini_pipeline.retriever.by_id for i in ids)
