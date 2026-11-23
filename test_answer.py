from actqa.answer import NOT_FOUND, Answer, ExtractiveAnswerer, grounded_fraction
from actqa.llm import LLMAnswerer, build_prompt


def test_extractive_answer_quotes_the_act_and_cites(mini_pipeline):
    a = mini_pipeline.ask("What late fee applies per day when a book is returned late?")
    assert not a.abstained
    assert a.citations[0] == ("LIB", "4")
    sec = mini_pipeline.sections[("LIB", "4")].text
    for sentence in a.text.split(". "):
        assert sentence.strip(". ") in sec          # extractive: nothing is invented


def test_abstains_when_evidence_is_weak(mini_pipeline):
    ans = ExtractiveAnswerer(mini_pipeline.retriever, mini_pipeline.sections, tau_dense=0.0, tau_coverage=0.9)
    a = mini_pipeline.ask("What is the corporate tax rate on shipping profits?", answerer=ans)
    assert a.abstained
    assert a.as_dict()["answer"] == NOT_FOUND


def test_grounded_fraction_flags_invented_sentences():
    ev = ["A member may borrow at most five books at one time."]
    assert grounded_fraction("A member may borrow at most five books.", ev) == 1.0
    assert grounded_fraction("A member may borrow ten dragons from the moon.", ev) == 0.0
    assert grounded_fraction("A member may borrow five books. Dragons live on the moon.", ev) == 0.5


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.messages = self

    def create(self, **kw):
        class R:
            pass
        r = R()
        r.content = [type("B", (), {"text": self.text})()]
        return r


def _ask(pipe, text, q="What is the late fee per day?"):
    llm = LLMAnswerer(pipe.sections, client=FakeClient(text), model="fake")
    return pipe.ask(q, answerer=llm)


def test_llm_answer_with_valid_grounded_citation_is_accepted(mini_pipeline):
    a = _ask(mini_pipeline, "The late fee shall be two rupees per day. [LIB §4]")
    assert not a.abstained and a.citations == [("LIB", "4")]


def test_llm_not_found_becomes_abstention(mini_pipeline):
    assert _ask(mini_pipeline, "NOT_FOUND").abstained


def test_llm_missing_or_invented_citation_is_rejected(mini_pipeline):
    assert _ask(mini_pipeline, "The late fee is two rupees per day.").abstained
    assert _ask(mini_pipeline, "The late fee is two rupees per day. [LIB §99]").abstained


def test_llm_ungrounded_claim_is_rejected(mini_pipeline):
    a = _ask(mini_pipeline, "The late fee is fifty dollars per hour on weekends. [LIB §4]")
    assert a.abstained and a.reason == "ungrounded sentence"


def test_prompt_contains_question_and_sections(mini_sections):
    p = build_prompt("Q?", mini_sections[:2])
    assert "[LIB §1]" in p and "Question: Q?" in p
