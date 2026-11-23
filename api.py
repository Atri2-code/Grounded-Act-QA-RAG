"""Small FastAPI service: POST /ask returns a cited answer or an explicit abstention."""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import Pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(title="Grounded Act QA", version="0.1.0")
    state: dict = {"pipeline": pipeline}

    def get_pipeline() -> Pipeline:
        if state["pipeline"] is None:
            state["pipeline"] = Pipeline.from_dir(os.environ.get("ACTQA_DATA_DIR", "data"))
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict:
        p = get_pipeline()
        return {"status": "ok", "sections": len(p.sections), "chunks": len(p.chunks)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return get_pipeline().ask(req.question, k=req.k).as_dict()

    return app


app = create_app()
