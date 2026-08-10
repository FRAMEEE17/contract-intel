"""Unit tests for app.adapters.chunking.fixed_chunker.FixedChunker."""
from __future__ import annotations

import hashlib

import pytest

from app.adapters.chunking.fixed_chunker import FixedChunker
from app.domain.ports import ParsedDocument


def _doc(text: str, *, document_id: str = "doc1") -> ParsedDocument:
    return ParsedDocument(document_id=document_id, text=text, pages=[text], parser_version="text/v1")


def _tokens_text(n: int) -> str:
    """n distinct whitespace tokens: t0 t1 t2 ..."""
    return " ".join(f"t{i}" for i in range(n))


def test_chunking_is_deterministic():
    chunker = FixedChunker()
    document = _doc(_tokens_text(50))

    first = chunker.chunk(document=document, max_tokens=10, overlap=3)
    second = chunker.chunk(document=document, max_tokens=10, overlap=3)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_consecutive_chunks_share_exactly_the_overlap_tokens():
    chunker = FixedChunker()
    # 8 tokens, max_tokens=5, overlap=2 -> step=3 -> guaranteed >= 2 chunks.
    document = _doc(_tokens_text(8))

    chunks = chunker.chunk(document=document, max_tokens=5, overlap=2)

    assert len(chunks) >= 2
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        prev_tokens = prev_chunk.text.split()
        next_tokens = next_chunk.text.split()
        assert prev_tokens[-2:] == next_tokens[:2]


def test_max_tokens_is_never_exceeded():
    chunker = FixedChunker()
    document = _doc(_tokens_text(97))

    chunks = chunker.chunk(document=document, max_tokens=10, overlap=4)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text.split()) <= 10


def test_chunk_ids_embed_text_sha256_prefix_and_are_idempotent():
    chunker = FixedChunker()
    document = _doc(_tokens_text(30))

    first = chunker.chunk(document=document, max_tokens=8, overlap=2)
    second = chunker.chunk(document=document, max_tokens=8, overlap=2)

    for i, c in enumerate(first):
        expected_hash = hashlib.sha256(c.text.encode()).hexdigest()
        assert c.text_sha256 == expected_hash
        assert c.chunk_id == f"{document.document_id}:{i}:{expected_hash[:8]}"

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_empty_document_text_returns_no_chunks():
    chunker = FixedChunker()

    assert chunker.chunk(document=_doc(""), max_tokens=10, overlap=2) == []
    assert chunker.chunk(document=_doc("   \n\t  "), max_tokens=10, overlap=2) == []


@pytest.mark.parametrize(
    "max_tokens, overlap",
    [
        (5, 5),   # overlap == max_tokens
        (5, 6),   # overlap > max_tokens
        (0, 0),   # max_tokens <= 0
        (-1, 0),  # max_tokens negative
    ],
)
def test_invalid_overlap_or_max_tokens_raises_value_error(max_tokens, overlap):
    chunker = FixedChunker()
    document = _doc(_tokens_text(20))

    with pytest.raises(ValueError):
        chunker.chunk(document=document, max_tokens=max_tokens, overlap=overlap)


def test_chunk_field_defaults_are_set_correctly():
    chunker = FixedChunker()
    document = _doc(_tokens_text(3), document_id="my-doc.txt")

    chunks = chunker.chunk(document=document, max_tokens=10, overlap=2)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == "my-doc.txt"
    assert c.page is None
    assert c.section is None
    assert c.source_revision == ""
    assert c.text == "t0 t1 t2"
