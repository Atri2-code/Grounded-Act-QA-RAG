import json
from pathlib import Path

import pytest

from actqa.answer import Answer
from actqa.data import load_corpus, normalize
from actqa.evaluate import (Question, bootstrap_ci, judge, load_questions, paired_bootstrap_diff, rank_of_gold)

QFILE = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
DATA = Path(__file__).resolve().parents[1] / "data"


def test_rank_of_gold():
    keys = [("A", "1"), ("A", "2"), ("B", "3")]
    assert rank_of_gold(keys, [("B", "3")]) == 3
    assert rank_of_gold(keys, [("Z", "9")]) is None
    assert rank_of_gold(keys, [("A", "2"), ("B", "3")]) == 2


def test_bootstrap_interval_brackets_the_mean():
    vals = [1.0] * 30 + [0.0] * 10
    lo, hi = bootstrap_ci(vals, n_boot=500)
    assert lo <= 0.75 <= hi and 0.0 <= lo <= hi <= 1.0


def test_paired_difference_sign():
    a = [1.0] * 20
    b = [0.0] * 10 + [1.0] * 10
    d, lo, hi = paired_bootstrap_diff(a, b, n_boot=500)
    assert d == pytest.approx(0.5) and lo > 0


def _q(answerable=True):
    return Question("x", "q", answerable, [("A", "1")] if answerable else [], ["two years"] if answerable else [], "test")


def test_judge_correct_answer():
    a = Answer("Imprisonment up to two years.", [("A", "1")], False)
    j = judge(a, _q())
    assert j["correct"] and not j["wrong_answer"]


def test_judge_wrong_section_and_missing_fact_are_wrong_answers():
    assert judge(Answer("two years", [("B", "2")], False), _q())["wrong_answer"]
    assert judge(Answer("some other text", [("A", "1")], False), _q())["wrong_answer"]


def test_judge_abstention_and_unanswerable():
    assert not judge(Answer(None, [], True), _q())["wrong_answer"]            # abstaining is not wrong
    assert judge(Answer("anything", [("A", "1")], False), _q(False))["wrong_answer"]
    assert not judge(Answer(None, [], True), _q(False))["wrong_answer"]


def test_question_file_is_well_formed():
    qs = load_questions(QFILE)
    assert len({q.id for q in qs}) == len(qs)
    assert {q.split for q in qs} == {"dev", "test"}
    assert all(q.gold for q in qs if q.answerable) and all(not q.gold for q in qs if not q.answerable)
    assert sum(q.answerable for q in qs) >= 50


@pytest.mark.skipif(not (DATA / "nia.json").exists(), reason="run scripts/fetch_corpus.py first")
def test_every_expected_fact_appears_in_its_gold_section():
    secs = {s.key: s for s in load_corpus(DATA)}
    for q in load_questions(QFILE):
        if not q.answerable:
            continue
        texts = [normalize(secs[g].text) for g in q.gold]
        assert any(all(normalize(p) in t for p in q.must_contain) for t in texts), q.id
