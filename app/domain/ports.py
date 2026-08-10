"""Domain ports: the contracts every adapter implements.

Provider choice (local vs Azure) is wired in app/config.py; business logic in
app/application/* depends on these Protocols, never on a concrete SDK.

Notes:
  * LLMClient.complete takes temperature + seed, so run-to-run control lives in the
    contract, not just the eval judge.
  * Embedder splits embed_query / embed_passage (a query must not be embedded with
    the passage model).
  * Structured output is validated by the caller; a parse failure is a counted
    failure, never silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


# ---- value objects ---------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str            # source contract text, never a summary
    page: int | None = None
    section: str | None = None
    source_revision: str = ""
    text_sha256: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    source_revision: str = ""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    text: str
    pages: list[str] = field(default_factory=list)
    parser_version: str = ""


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    redacted_text: str | None = None
    findings: list[str] = field(default_factory=list)   # e.g. ["pii:PERSON", "ungrounded_claim"]


# ---- ports -----------------------------------------------------------------
class LLMClient(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None,
        model: str,
        prompt_version: str,
        temperature: float = 0.0,
        seed: int | None = 0,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    def embed_query(self, text: str, *, model: str) -> list[float]: ...
    def embed_passage(self, text: str, *, model: str) -> list[float]: ...


class SearchClient(Protocol):
    def index(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None: ...
    def search(
        self, *, query: str, query_vector: list[float], filters: dict | None, top_k: int
    ) -> list[RetrievedChunk]: ...   # hybrid (BM25 + vector) lives inside the adapter


class DocParser(Protocol):
    def parse(self, *, content: bytes, filename: str) -> ParsedDocument: ...


class Chunker(Protocol):
    # section/page-aware, no LLM call
    def chunk(self, *, document: ParsedDocument, max_tokens: int = 800, overlap: int = 80) -> list[Chunk]: ...


class Guardrail(Protocol):
    def inspect_input(self, text: str, *, metadata: dict) -> GuardrailResult: ...
    def inspect_output(
        self, *, answer: str, context: Sequence[RetrievedChunk], metadata: dict
    ) -> GuardrailResult: ...


class PromptRegistry(Protocol):
    def get(self, name: str, version: str) -> str: ...
    def resolve_production(self, name: str) -> str: ...   # -> version string


class EvalRunner(Protocol):
    def run(self, *, dataset: str, system_version: str) -> dict: ...   # -> EvalReport dict
