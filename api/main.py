"""HTTP API for contract QA, summarization, and a persistent contract repository.

The pipeline (LLM, retrieval, parsing, guardrail) comes from app.config, or is
injected for tests. Two retrieval paths:
  * ephemeral   — /answer with document_text builds a per-request in-memory index.
  * repository  — /contracts indexes into a persistent SearchClient (Azure AI Search
    in prod via SEARCH_PROVIDER=azure, in-memory otherwise); /answer with a
    document_id (or neither) then searches one contract or across the whole library.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile # type: ignore
from pydantic import BaseModel, Field

# Each column of the abstraction grid is one QA question run over a contract; an
# abstention ("not specified") means the clause is absent — a missing-clause flag.
CLAUSE_CHECKLIST: Dict[str, str] = {
    "Governing law": "Which state's or country's law governs this agreement?",
    "Term": "What is the term or duration of this agreement?",
    "Termination notice": "How many days' notice are required to terminate for breach?",
    "Liability cap": "Is there a limitation or cap on liability?",
    "Confidentiality": "Does the agreement impose a confidentiality obligation?",
    "Exclusivity": "Is the arrangement exclusive?",
    "Indemnification": "Does the agreement include an indemnification obligation?",
    "Assignment": "Are there restrictions on assignment or change of control?",
}


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    document_text: Optional[str] = None   # ephemeral path
    document_id: Optional[str] = None      # repository path (None = across all contracts)
    top_k: int = 8


class SummarizeRequest(BaseModel):
    document_text: Optional[str] = None   # paste text, or...
    document_id: Optional[str] = None      # ...summarize a contract already in the library


class IngestResponse(BaseModel):
    text: str
    pages: int
    chars: int


class ContractIn(BaseModel):
    title: str = Field(min_length=1)
    document_text: str = Field(min_length=1)


class ContractMeta(BaseModel):
    document_id: str
    chunks: int
    chars: int


class AbstractRequest(BaseModel):
    document_id: str = Field(min_length=1)


class ClauseCell(BaseModel):
    present: bool
    abstained: bool
    answer: str


class AbstractResponse(BaseModel):
    document_id: str
    clauses: Dict[str, ClauseCell]


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


class ContractRepository:
    """Contract library: chunks go into the injected SearchClient, metadata per doc.

    index() is called with the full chunk set on each add so it works the same for a
    replace-style in-memory index and an upsert-style Azure index.
    """

    def __init__(self, pipeline: Any, search: Any) -> None:
        self._p = pipeline
        self._search = search
        self._chunks: list = []
        self._vectors: list = []
        self._meta: Dict[str, dict] = {}
        self._texts: Dict[str, str] = {}

    def add(self, *, title: str, text: str) -> dict:
        p = self._p
        doc = p.parser.parse(content=text.encode("utf-8"), filename=title)
        chunks = list(p.chunker.chunk(document=doc))
        vectors = [p.embedder.embed_passage(c.text, model=p.embed_model) for c in chunks]
        # Index the full set first; commit local state only if indexing succeeds, so a
        # failed upload can't leave orphan chunks for the next add to flush.
        all_chunks = self._chunks + chunks
        all_vectors = self._vectors + vectors
        self._search.index(all_chunks, all_vectors)
        self._chunks, self._vectors = all_chunks, all_vectors
        meta = {"document_id": title, "chunks": len(chunks), "chars": len(text)}
        self._meta[title] = meta
        self._texts[title] = text
        return meta

    def list(self) -> list:
        return list(self._meta.values())

    def text_for(self, document_id: str) -> Optional[str]:
        return self._texts.get(document_id)

    def answer(self, question: str, *, document_id: Optional[str], top_k: int):
        from app.application.answer import answer_question

        p = self._p
        return answer_question(
            question, llm=p.llm, embedder=p.embedder, search=self._search,
            guardrail=p.guardrail, registry=p.registry, embed_model=p.embed_model,
            model=p.model, top_k=top_k,
            filters={"document_id": document_id} if document_id else None,
        )


def _to_response(result: Any) -> "AnswerResponse":
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


def create_app(pipeline: Any = None) -> FastAPI:
    app = FastAPI(title="Contract-Intel API", version="1.0")
    state: dict = {"pipeline": pipeline, "repo": None}
    index_cache: dict = {}

    def get_pipeline():
        if state["pipeline"] is None:
            from app.config import build_pipeline
            state["pipeline"] = build_pipeline()
        return state["pipeline"]

    def get_repo() -> ContractRepository:
        if state["repo"] is None:
            from app.config import build_search
            state["repo"] = ContractRepository(get_pipeline(), build_search())
        return state["repo"]

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
        return {"service": "contract-intel",
                "endpoints": ["/answer", "/summarize", "/ingest", "/contracts", "/abstract", "/health"]}

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        if req.document_text:
            from app.application.answer import answer_question
            p = get_pipeline()
            result = answer_question(
                req.question, llm=p.llm, embedder=p.embedder, search=index_for(req.document_text),
                guardrail=p.guardrail, registry=p.registry, embed_model=p.embed_model,
                model=p.model, top_k=req.top_k,
            )
        else:
            result = get_repo().answer(req.question, document_id=req.document_id, top_k=req.top_k)
        return _to_response(result)

    @app.post("/summarize")
    def summarize(req: SummarizeRequest) -> dict:
        from app.application.summarize import summarize_contract

        if req.document_id:
            text = get_repo().text_for(req.document_id)
            if text is None:
                raise HTTPException(status_code=404, detail=f"contract {req.document_id!r} not in the library")
        elif req.document_text:
            text = req.document_text
        else:
            raise HTTPException(status_code=422, detail="provide document_text or document_id")
        p = get_pipeline()
        return summarize_contract(text, llm=p.llm, model=p.model)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(file: UploadFile = File(...)) -> IngestResponse:
        from app.adapters.documents.pdf_parser import PyMuPDFDocParser

        content = file.file.read()
        parsed = PyMuPDFDocParser().parse(content=content, filename=file.filename or "upload.pdf")
        return IngestResponse(text=parsed.text, pages=len(parsed.pages), chars=len(parsed.text))

    @app.post("/contracts", response_model=ContractMeta)
    def add_contract(req: ContractIn) -> ContractMeta:
        return ContractMeta(**get_repo().add(title=req.title, text=req.document_text))

    @app.get("/contracts", response_model=list[ContractMeta])
    def list_contracts() -> list:
        return [ContractMeta(**m) for m in get_repo().list()]

    @app.post("/abstract", response_model=AbstractResponse)
    def abstract(req: AbstractRequest) -> AbstractResponse:
        repo = get_repo()
        clauses: Dict[str, ClauseCell] = {}
        for label, question in CLAUSE_CHECKLIST.items():
            res = repo.answer(question, document_id=req.document_id, top_k=6)
            missing = res.abstained or res.no_context
            clauses[label] = ClauseCell(present=not missing, abstained=missing, answer=res.answer)
        return AbstractResponse(document_id=req.document_id, clauses=clauses)

    return app


app = create_app()
