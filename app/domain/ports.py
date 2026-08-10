"""Domain ports (interfaces) — the stable contracts every adapter implements.

Provider choice (local vs Azure) is wired in `app/config.py`; business logic in
`app/application/*` depends ONLY on these Protocols, never on a concrete SDK.

Design corrections baked in (from the architecture refuter gate):
  * LLMClient.complete takes temperature + seed  → run-to-run control lives in the
    contract, not just the eval judge. (refuter C1/C3)
  * Embedder splits embed_query / embed_passage  → fixes the PoC bug of using a
    passage-model for queries. (refuter C1)
  * Structured output is validated by the caller; a parse failure is a COUNTED
    failure mode (see application layer), never silently dropped. (refuter C1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


# ---- value objects ---------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str            # SOURCE contract text — never a summary (root-cause fix)
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
    # DETERMINISTIC: section/page-aware, no LLM call.
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
