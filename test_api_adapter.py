from fastapi.testclient import TestClient

from actqa.api import create_app
from actqa.langchain_adapter import ActRetriever


def test_health_and_ask(mini_pipeline):
    c = TestClient(create_app(mini_pipeline))
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/ask", json={"question": "What is the late fee per day?"}).json()
    assert r["citations"] == ["LIB s.4"] and r["abstained"] is False


def test_ask_validates_input(mini_pipeline):
    c = TestClient(create_app(mini_pipeline))
    assert c.post("/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_langchain_retriever_returns_documents_with_metadata(mini_pipeline):
    docs = ActRetriever(pipeline=mini_pipeline, k=3).invoke("late fee for returning a book")
    assert docs and docs[0].metadata["act"] == "LIB"
    assert any(d.metadata["section"] == "4" for d in docs)
