"""HTTP API for contract-clause QA (inbound adapter over answer_question).

The pipeline (LLM, retrieval, parsing, guardrail) comes from app.config at startup,
or is injected for tests. A per-document in-memory index is built on first use and
cached by content hash, so repeat questions on the same contract skip re-embedding.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from fastapi import FastAPI # type: ignore
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    document_text: str = Field(min_length=1)
    top_k: int = 8


class CitationOut(BaseModel):
    chunk_id: str
    document_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    score: float


class AnswerResponse(BaseModel):
    answer: str
    abstained: bool
    malformed: bool
    blocked: bool
    no_context: bool
    findings: list[str] = []
    citations: list[CitationOut] = []
    model: str = ""


def create_app(pipeline: Any = None) -> FastAPI:
    app = FastAPI(title="Contract-Intel API", version="1.0")
    state: dict = {"pipeline": pipeline}
    index_cache: dict = {}

    def get_pipeline():
        if state["pipeline"] is None:
            from app.config import build_pipeline
            state["pipeline"] = build_pipeline()
        return state["pipeline"]

    def index_for(text: str):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in index_cache:
            from app.adapters.search.in_memory import InMemorySearch
            p = get_pipeline()
            doc = p.parser.parse(content=text.encode("utf-8"), filename=f"doc-{key[:8]}")
            chunks = p.chunker.chunk(document=doc)
            vectors = [p.embedder.embed_passage(c.text, model=p.embed_model) for c in chunks]
            search = InMemorySearch()
            search.index(chunks, vectors)
            index_cache[key] = search
        return index_cache[key]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def root() -> dict:
        return {"service": "contract-intel", "endpoints": ["/answer", "/health"]}

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        from app.application.answer import answer_question

        p = get_pipeline()
        result = answer_question(
            req.question, llm=p.llm, embedder=p.embedder, search=index_for(req.document_text),
            guardrail=p.guardrail, registry=p.registry, embed_model=p.embed_model,
            model=p.model, top_k=req.top_k,
        )
        return AnswerResponse(
            answer=result.answer, abstained=result.abstained, malformed=result.malformed,
            blocked=result.blocked, no_context=result.no_context, findings=result.findings,
            citations=[
                CitationOut(chunk_id=c.chunk_id, document_id=c.document_id,
                            page=c.page, section=c.section, score=c.score)
                for c in result.citations
            ],
            model=result.model,
        )

    return app


app = create_app()
