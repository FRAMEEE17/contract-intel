"""PyMuPDF parser: extract the embedded text layer.

When a PDF has no usable text layer (a scan), delegates to the injected
ocr_fallback parser (e.g. Azure Document Intelligence).
"""
from __future__ import annotations

from app.domain.ports import DocParser, ParsedDocument

_PAGE_SEP = "\f"  # form-feed; TextDocParser splits on the same char


class PyMuPDFDocParser:
    def __init__(self, *, ocr_fallback: DocParser | None = None, min_chars_per_page: int = 10) -> None:
        """`ocr_fallback`: a DocParser used when the PDF looks scanned (no text layer).
        `min_chars_per_page`: below this average, the PDF is treated as scanned."""
        self._ocr_fallback = ocr_fallback
        self._min_chars_per_page = min_chars_per_page

    def parse(self, *, content: bytes, filename: str) -> ParsedDocument:
        import fitz  # lazy import so the module loads without PyMuPDF installed

        pages: list[str] = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                pages.append(page.get_text("text"))

        n_pages = max(len(pages), 1)
        total_chars = sum(len(p.strip()) for p in pages)
        looks_scanned = total_chars < self._min_chars_per_page * n_pages

        # scanned: hand the original bytes to the OCR fallback
        if looks_scanned and self._ocr_fallback is not None:
            return self._ocr_fallback.parse(content=content, filename=filename)

        return ParsedDocument(
            document_id=filename,
            text=_PAGE_SEP.join(pages),
            pages=pages,
            parser_version="pymupdf/v1",
        )
