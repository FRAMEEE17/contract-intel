"""Plain-text parser: decode bytes, split pages on form-feed."""
from __future__ import annotations

from app.domain.ports import ParsedDocument


class TextDocParser:
    def parse(self, *, content: bytes, filename: str) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        pages = text.split("\f") if "\f" in text else [text]
        return ParsedDocument(
            document_id=filename,
            text=text,
            pages=pages,
            parser_version="text/v1",
        )
