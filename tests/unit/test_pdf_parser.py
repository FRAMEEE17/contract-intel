"""PyMuPDF PDF parser + Azure Document Intelligence fallback — routing + wiring.

The mental model under test: extract the embedded text layer (cheap, exact) for
digital-born PDFs; only when a PDF has no usable text layer (scanned) route to the
injected OCR fallback DocParser. Azure Document Intelligence is one such fallback,
tested here with an injected fake client (no network).
"""
from __future__ import annotations

import fitz  # PyMuPDF
import pytest

from app.adapters.documents.azure_doc_intelligence import AzureDocIntelligenceParser
from app.adapters.documents.pdf_parser import PyMuPDFDocParser
from app.domain.ports import ParsedDocument


# ---- helpers: build real PDFs in-memory ------------------------------------
def _text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _blank_pdf(n_pages: int = 1) -> bytes:
    """A PDF with no text layer — simulates a scanned page."""
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


# ---- fake OCR fallback (a DocParser) ---------------------------------------
class FakeOCR:
    def __init__(self):
        self.calls = []

    def parse(self, *, content: bytes, filename: str) -> ParsedDocument:
        self.calls.append(filename)
        return ParsedDocument(document_id=filename, text="OCR TEXT",
                              pages=["OCR TEXT"], parser_version="fake-ocr/v1")


# ---- fake Azure Document Intelligence client -------------------------------
class _Line:
    def __init__(self, content):
        self.content = content


class _Page:
    def __init__(self, lines):
        self.lines = [_Line(x) for x in lines]


class _Result:
    content = "AZURE FULL TEXT"
    pages = [_Page(["l1", "l2"]), _Page(["l3"])]


class _Poller:
    def result(self):
        return _Result()


class FakeDIClient:
    def __init__(self):
        self.calls = []

    def begin_analyze_document(self, model_id, *, body, content_type):
        self.calls.append((model_id, content_type, len(body)))
        return _Poller()


# ---- PyMuPDF routing -------------------------------------------------------
def test_digital_born_uses_text_layer():
    parser = PyMuPDFDocParser(ocr_fallback=FakeOCR())
    doc = parser.parse(content=_text_pdf("Governing Law: laws of the State of New York."),
                       filename="contract.pdf")
    assert "Governing Law" in doc.text
    assert doc.document_id == "contract.pdf" and doc.parser_version == "pymupdf/v1"
    assert len(doc.pages) == 1


def test_scanned_routes_to_ocr_fallback():
    ocr = FakeOCR()
    parser = PyMuPDFDocParser(ocr_fallback=ocr)
    doc = parser.parse(content=_blank_pdf(2), filename="scan.pdf")
    assert doc.text == "OCR TEXT" and doc.parser_version == "fake-ocr/v1"
    assert ocr.calls == ["scan.pdf"]  # fallback actually invoked


def test_scanned_without_fallback_degrades_gracefully():
    parser = PyMuPDFDocParser(ocr_fallback=None)
    doc = parser.parse(content=_blank_pdf(1), filename="scan.pdf")
    assert doc.parser_version == "pymupdf/v1" and doc.text.strip() == ""  # honest empty, no crash


# ---- Azure Document Intelligence adapter (injected fake client) ------------
def test_azure_di_parses_via_injected_client():
    client = FakeDIClient()
    parser = AzureDocIntelligenceParser(client)
    doc = parser.parse(content=b"%PDF-fake-bytes", filename="x.pdf")
    assert doc.text == "AZURE FULL TEXT"
    assert doc.pages == ["l1\nl2", "l3"] and len(doc.pages) == 2
    assert doc.parser_version == "azure-di/prebuilt-read"
    assert client.calls and client.calls[0][0] == "prebuilt-read"  # model id passed through


# ---- the two compose: PyMuPDF fast path -> Azure fallback ------------------
def test_pymupdf_falls_back_to_azure_di_on_scanned():
    azure = AzureDocIntelligenceParser(FakeDIClient())
    parser = PyMuPDFDocParser(ocr_fallback=azure)
    doc = parser.parse(content=_blank_pdf(1), filename="scan.pdf")
    assert doc.text == "AZURE FULL TEXT" and doc.parser_version == "azure-di/prebuilt-read"
