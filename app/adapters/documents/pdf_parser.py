"""PyMuPDF DocParser — extract the embedded text layer, OCR only when there isn't one.

When a PDF has no usable text layer (a scan), the work is delegated to an
injected `ocr_fallback` DocParser (e.g. the Azure Document Intelligence adapter).
Implements app.domain.ports.DocParser.
"""
from __future__ import annotations

from app.domain.ports import DocParser, ParsedDocument

_PAGE_SEP = "\f"  # form-feed page separator — TextDocParser splits on the same char


class PyMuPDFDocParser:
    def __init__(self, *, ocr_fallback: DocParser | None = None, min_chars_per_page: int = 10) -> None:
        """`ocr_fallback`: a DocParser used when the PDF looks scanned (no text layer).
        `min_chars_per_page`: below this average, the PDF is treated as scanned."""
        self._ocr_fallback = ocr_fallback
        self._min_chars_per_page = min_chars_per_page

    def parse(self, *, content: bytes, filename: str) -> ParsedDocument:
        import fitz  # PyMuPDF — imported here so the module loads even if it isn't installed

        pages: list[str] = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                pages.append(page.get_text("text"))

        n_pages = max(len(pages), 1)
        total_chars = sum(len(p.strip()) for p in pages)
        looks_scanned = total_chars < self._min_chars_per_page * n_pages

        # No usable text layer -> hand the ORIGINAL bytes to the OCR fallback.
        if looks_scanned and self._ocr_fallback is not None:
            return self._ocr_fallback.parse(content=content, filename=filename)

        return ParsedDocument(
            document_id=filename,
            text=_PAGE_SEP.join(pages),
            pages=pages,
            parser_version="pymupdf/v1",
        )
