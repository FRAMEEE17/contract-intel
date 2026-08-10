"""Shared test contract: repo on path + deterministic fake ports + gold loader.

The fakes implement the Protocols in app.domain.ports exactly, so unit tests run
offline, fast, and deterministically — no model download, no network, no server.
Real adapters (sentence-transformers embedder, live MLX) are exercised elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.domain.ports import LLMResponse, RetrievedChunk  # noqa: E402

_EMBED_DIM = 64


def _hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-hashed-tokens vector — lexical overlap => cosine similarity.

    No learning, but stable and meaningful enough to test ranking: documents that
    share words with the query score higher. Same input -> byte-identical vector.
    """
    vec = [0.0] * _EMBED_DIM
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[h % _EMBED_DIM] += 1.0
    return vec


class FakeEmbedder:
    """Implements app.domain.ports.Embedder, deterministically."""

    def embed_query(self, text: str, *, model: str) -> list[float]:
        return _hash_embed(text)

    def embed_passage(self, text: str, *, model: str) -> list[float]:
        return _hash_embed(text)


class FakeLLM:
    """Implements app.domain.ports.LLMClient; returns a scripted answer.

    Construct with the text to return, or a callable(user_prompt)->text to react
    to the prompt (e.g. emit a malformed reply to test the fail-closed path).
    """

    def __init__(self, reply="{}", *, model="fake-llm"):
        self._reply = reply
        self._model = model
        self.calls: list[dict] = []

    def complete(self, *, system_prompt, user_prompt, response_schema, model,
                 prompt_version, temperature=0.0, seed=0) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt,
                           "model": model, "prompt_version": prompt_version})
        text = self._reply(user_prompt) if callable(self._reply) else self._reply
        return LLMResponse(text=text, model=self._model, prompt_version=prompt_version,
                           input_tokens=len(user_prompt.split()), output_tokens=len(text.split()),
                           latency_ms=1)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_llm():
    """Factory: make_llm(reply) -> FakeLLM. reply is str or callable(prompt)->str."""
    return lambda reply="{}", **kw: FakeLLM(reply, **kw)


@pytest.fixture
def retrieved_chunk():
    """Factory for a RetrievedChunk with sensible defaults (for answer-path tests)."""
    def _make(text="This Agreement is governed by the laws of the State of New York.",
              *, chunk_id="c1", document_id="d1", score=0.9, page=1, section="Governing Law"):
        return RetrievedChunk(chunk_id=chunk_id, document_id=document_id, text=text,
                              score=score, page=page, section=section, source_revision="rev1")
    return _make


@pytest.fixture(scope="session")
def gold_items() -> list[dict]:
    path = REPO_ROOT / "evals" / "gold" / "gold.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
