"""Deterministic fixed-window Chunker adapter — whitespace-token windowing, no LLM."""
from __future__ import annotations

import hashlib

from app.domain.ports import Chunk, ParsedDocument


class FixedChunker:
    """Implements app.domain.ports.Chunker via deterministic token windowing."""

    def chunk(
        self, *, document: ParsedDocument, max_tokens: int = 800, overlap: int = 80
    ) -> list[Chunk]:
        if max_tokens <= 0 or overlap >= max_tokens:
            raise ValueError(
                f"invalid chunking window: max_tokens={max_tokens}, overlap={overlap} "
                "(require max_tokens > 0 and overlap < max_tokens)"
            )

        tokens = document.text.split()
        if not tokens:
            return []

        step = max_tokens - overlap
        chunks: list[Chunk] = []
        start = 0
        i = 0
        n = len(tokens)
        while start < n:
            window = tokens[start : start + max_tokens]
            text = " ".join(window)
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = f"{document.document_id}:{i}:{text_sha256[:8]}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=text,
                    page=None,
                    section=None,
                    source_revision="",
                    text_sha256=text_sha256,
                )
            )
            i += 1
            start += step

        return chunks
