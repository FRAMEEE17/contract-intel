"""Unit tests for app.adapters.embeddings.sentence_transformer.SentenceTransformerEmbedder.

sentence-transformers is a heavy optional dependency; these tests skip (rather than
fail) when it is not installed, per repo policy that the search path never depends
on it.
"""
from __future__ import annotations

import pytest

from app.adapters.embeddings.sentence_transformer import DEFAULT_MODEL, SentenceTransformerEmbedder

try:
    import sentence_transformers  # noqa: F401

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _ST_AVAILABLE, reason="sentence-transformers is not installed")


def test_embed_query_and_passage_return_equal_length_nonempty_vectors():
    embedder = SentenceTransformerEmbedder()
    text = "This Agreement shall terminate upon 30 days written notice."

    query_vec = embedder.embed_query(text, model=DEFAULT_MODEL)
    passage_vec = embedder.embed_passage(text, model=DEFAULT_MODEL)

    assert len(query_vec) > 0
    assert len(passage_vec) > 0
    assert len(query_vec) == len(passage_vec)


def test_embed_is_deterministic():
    embedder = SentenceTransformerEmbedder()
    text = "Confidential Information means any non-public information disclosed hereunder."

    first = embedder.embed_query(text, model=DEFAULT_MODEL)
    second = embedder.embed_query(text, model=DEFAULT_MODEL)

    assert first == second


def test_query_and_passage_embeddings_differ_for_same_text():
    embedder = SentenceTransformerEmbedder()
    text = "Governing law of this Agreement is the State of Delaware."

    query_vec = embedder.embed_query(text, model=DEFAULT_MODEL)
    passage_vec = embedder.embed_passage(text, model=DEFAULT_MODEL)

    assert query_vec != passage_vec
