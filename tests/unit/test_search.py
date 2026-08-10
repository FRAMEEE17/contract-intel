"""Unit tests for app.adapters.search.in_memory.InMemorySearch (hybrid BM25 + cosine).

Uses the deterministic FakeEmbedder from tests/conftest.py (hashed bag-of-words) so
ranking assertions stay offline and reproducible, per repo convention.
"""
from __future__ import annotations

import pytest

from app.adapters.search.in_memory import InMemorySearch
from app.domain.ports import Chunk, RetrievedChunk
from tests.conftest import FakeEmbedder


def _make_chunk(chunk_id, document_id, text, *, section=None, page=1, source_revision="rev1"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        page=page,
        section=section,
        source_revision=source_revision,
    )


def _embed_passages(chunks):
    embedder = FakeEmbedder()
    return [embedder.embed_passage(chunk.text, model="fake") for chunk in chunks]


def test_query_sharing_exact_terms_ranks_matching_chunk_first():
    chunks = [
        _make_chunk(
            "c1",
            "d1",
            "This Agreement is governed by the laws of the State of New York.",
            section="Governing Law",
        ),
        _make_chunk(
            "c2", "d1", "The parties shall keep Confidential Information secret.", section="Confidentiality"
        ),
        _make_chunk("c3", "d2", "Payment shall be made within thirty days of invoice.", section="Payment"),
    ]
    vectors = _embed_passages(chunks)
    search = InMemorySearch()
    search.index(chunks, vectors)

    embedder = FakeEmbedder()
    query = "governed by the laws of the State of New York"
    query_vector = embedder.embed_query(query, model="fake")

    results = search.search(query=query, query_vector=query_vector, filters=None, top_k=3)

    assert results
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].chunk_id == "c1"


def test_filters_restrict_results_to_matching_document():
    chunks = [
        _make_chunk(
            "c1",
            "d1",
            "This Agreement is governed by the laws of the State of New York.",
            section="Governing Law",
        ),
        _make_chunk(
            "c2",
            "d2",
            "This Agreement is governed by the laws of the State of New York.",
            section="Governing Law",
        ),
    ]
    vectors = _embed_passages(chunks)
    search = InMemorySearch()
    search.index(chunks, vectors)

    embedder = FakeEmbedder()
    query = "governed by the laws"
    query_vector = embedder.embed_query(query, model="fake")

    results = search.search(query=query, query_vector=query_vector, filters={"document_id": "d1"}, top_k=10)

    assert results
    assert all(r.document_id == "d1" for r in results)


def test_top_k_limits_number_of_results():
    chunks = [_make_chunk(f"c{i}", "d1", f"clause number {i} about termination rights") for i in range(5)]
    vectors = _embed_passages(chunks)
    search = InMemorySearch()
    search.index(chunks, vectors)

    embedder = FakeEmbedder()
    query_vector = embedder.embed_query("termination rights", model="fake")

    results = search.search(query="termination rights", query_vector=query_vector, filters=None, top_k=2)

    assert len(results) == 2


def test_top_k_non_positive_returns_empty_list():
    chunks = [_make_chunk("c1", "d1", "some clause text")]
    vectors = _embed_passages(chunks)
    search = InMemorySearch()
    search.index(chunks, vectors)

    embedder = FakeEmbedder()
    query_vector = embedder.embed_query("some clause text", model="fake")

    results = search.search(query="some clause text", query_vector=query_vector, filters=None, top_k=0)

    assert results == []


def test_empty_index_returns_empty_list():
    search = InMemorySearch()
    search.index([], [])

    embedder = FakeEmbedder()
    query_vector = embedder.embed_query("anything", model="fake")

    results = search.search(query="anything", query_vector=query_vector, filters=None, top_k=5)

    assert results == []


def test_index_raises_value_error_on_mismatched_lengths():
    chunks = [_make_chunk("c1", "d1", "some text")]
    vectors: list[list[float]] = []
    search = InMemorySearch()

    with pytest.raises(ValueError):
        search.index(chunks, vectors)
