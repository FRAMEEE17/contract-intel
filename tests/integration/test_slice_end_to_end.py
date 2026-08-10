"""End-to-end wiring proof: the REAL adapters compose through answer_question.

Unlike the unit tests (which inject hand-written fakes), this constructs the
concrete adapters — TextDocParser, FixedChunker, SentenceTransformerEmbedder,
InMemorySearch, FilePromptRegistry, PassthroughGuardrail — runs a real
parse→chunk→embed→index ingestion, and drives the use-case through them. Only the
LLM is scripted (offline, deterministic) per the unit-only decision; every other
component is the production adapter. Marked `integration` and skipped when
sentence-transformers is unavailable, so the offline unit suite stays fast.
"""
from __future__ import annotations

import json
import re

import pytest

from app.application.answer import answer_question
from app.domain.ports import LLMResponse

pytestmark = pytest.mark.integration

try:
    import sentence_transformers  # noqa: F401
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _ST_AVAILABLE, reason="sentence-transformers not installed")]

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONTRACT = (
    "1. Term. This Agreement commences on the Effective Date and continues for three (3) years.\n"
    "2. Termination. Either party may terminate this Agreement upon sixty (60) days written notice.\n"
    "3. Governing Law. This Agreement shall be governed by and construed in accordance with the "
    "laws of the State of New York, without regard to its conflict-of-laws principles.\n"
    "4. Confidentiality. Each party shall keep the other's Confidential Information secret for five years.\n"
    "5. Assignment. Neither party may assign this Agreement without the prior written consent of the other.\n"
)


class _CiteFirstLLM:
    """Offline LLMClient: answers and cites the first chunk_id present in the prompt."""

    def __init__(self, answer="New York", not_specified=False):
        self._answer = answer
        self._not_specified = not_specified

    def complete(self, *, system_prompt, user_prompt, response_schema, model,
                 prompt_version, temperature=0.0, seed=0) -> LLMResponse:
        m = re.search(r"chunk_id=(.+?), section=", user_prompt)  # ids embed the filename (spaces/commas)
        cids = [m.group(1)] if m else []
        payload = {"answer": self._answer, "not_specified": self._not_specified, "citations": cids}
        return LLMResponse(text=json.dumps(payload), model="cite-first-llm",
                           prompt_version=prompt_version, input_tokens=len(user_prompt.split()),
                           output_tokens=8, latency_ms=1)


def _build_pipeline():
    from app.adapters.documents.text_parser import TextDocParser
    from app.adapters.chunking.fixed_chunker import FixedChunker
    from app.adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.adapters.search.in_memory import InMemorySearch
    from app.adapters.registry.file_registry import FilePromptRegistry
    from app.adapters.guardrails.passthrough import PassthroughGuardrail

    parser, chunker = TextDocParser(), FixedChunker()
    doc = parser.parse(content=CONTRACT.encode("utf-8"), filename="sample_contract.txt")
    chunks = chunker.chunk(document=doc, max_tokens=40, overlap=8)
    assert len(chunks) >= 2, "sample should split into multiple chunks"

    embedder = SentenceTransformerEmbedder()
    vectors = [embedder.embed_passage(c.text, model=EMB_MODEL) for c in chunks]
    search = InMemorySearch()
    search.index(chunks, vectors)
    return dict(embedder=embedder, search=search, registry=FilePromptRegistry(),
                guardrail=PassthroughGuardrail())


def test_real_adapters_answer_end_to_end():
    pipe = _build_pipeline()
    r = answer_question("Which state's law governs this agreement?", llm=_CiteFirstLLM(),
                        embed_model=EMB_MODEL, model="cite-first-llm", top_k=3, **pipe)

    assert r.llm_called and not r.malformed and not r.blocked and not r.no_context
    assert r.answer == "New York" and r.retrieved_k >= 1
    assert len(r.citations) == 1
    assert "citation_not_in_context" not in r.findings
    assert len(r.citations[0].text_sha256) == 64  # computed in the use-case, not carried by RetrievedChunk


def test_real_adapters_abstain_end_to_end():
    pipe = _build_pipeline()
    r = answer_question("What is the governing law?", llm=_CiteFirstLLM(answer="NOT SPECIFIED", not_specified=True),
                        embed_model=EMB_MODEL, model="cite-first-llm", top_k=3, **pipe)
    assert r.abstained and not r.malformed and r.answer == "NOT SPECIFIED"
