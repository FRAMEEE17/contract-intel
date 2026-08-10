"""In-memory hybrid search (BM25 + cosine), no external index.

Combines BM25 over tokenized chunk text with cosine similarity on the stored passage
vectors. Each signal is min-max normalized to [0, 1] and fused 0.5/0.5. Filters are
applied before ranking, so scores are only compared within the eligible subset.
"""
from __future__ import annotations

import math
from typing import Sequence

from rank_bm25 import BM25Okapi # type: ignore

from app.domain.ports import Chunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:  # degenerate: all equal (incl. a single score) -> avoid /0
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class InMemorySearch:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        chunks = list(chunks)
        vectors = list(vectors)
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )
        self._chunks = chunks
        self._vectors = vectors
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(
        self, *, query: str, query_vector: list[float], filters: dict | None, top_k: int
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not self._chunks:
            return []

        filters = filters or {}
        indices = [
            i
            for i, chunk in enumerate(self._chunks)
            if all(getattr(chunk, key, None) == value for key, value in filters.items())
        ]
        if not indices:
            return []

        bm25_all = self._bm25.get_scores(_tokenize(query)) if self._bm25 is not None else [0.0] * len(self._chunks)
        bm25_scores = [float(bm25_all[i]) for i in indices]
        cosine_scores = [_cosine(query_vector, self._vectors[i]) for i in indices]

        bm25_norm = _min_max_normalize(bm25_scores)
        cosine_norm = _min_max_normalize(cosine_scores)
        combined = [0.5 * b + 0.5 * c for b, c in zip(bm25_norm, cosine_norm)]

        ranked = sorted(zip(indices, combined), key=lambda pair: pair[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    page=chunk.page,
                    section=chunk.section,
                    source_revision=chunk.source_revision,
                )
            )
        return results
