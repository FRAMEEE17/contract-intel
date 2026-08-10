"""Live-verify AzureDocIntelligenceParser against a real Azure resource.

Prereq: provision a Document Intelligence (FormRecognizer) resource, then put its
endpoint + key in .env:
    AZURE_DOC_INTEL_ENDPOINT=https://<name>.cognitiveservices.azure.com/
    AZURE_DOC_INTEL_KEY=<key1>

Then, from the project root:  python smoke_docintel.py

Builds a text PDF from a real gold contract, sends it to Document Intelligence,
and checks the parsed text comes back. This confirms auth + the live SDK call +
result parsing (it does not exercise OCR on a scanned page).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fitz  # type: ignore # PyMuPDF
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.adapters.documents.azure_doc_intelligence import AzureDocIntelligenceParser
from evals.generate import load_contract_text


def _text_to_pdf(text: str, *, fontsize: int = 9) -> bytes:
    # insert_textbox writes nothing (and returns a negative overflow) when the
    # string is too tall for the box, so size each page to what actually fits.
    rect = fitz.Rect(40, 40, 550, 780)

    def fits(s: str) -> bool:
        scratch = fitz.open()
        overflow = scratch.new_page().insert_textbox(rect, s, fontsize=fontsize)
        scratch.close()
        return overflow >= 0

    doc = fitz.open()
    i = 0
    while i < len(text):
        lo, hi, take = 1, len(text) - i, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(text[i:i + mid]):
                take, lo = mid, mid + 1
            else:
                hi = mid - 1
        doc.new_page().insert_textbox(rect, text[i:i + take], fontsize=fontsize)
        i += take
    data = doc.tobytes()
    doc.close()
    return data


def main() -> int:
    endpoint = os.environ.get("AZURE_DOC_INTEL_ENDPOINT")
    key = os.environ.get("AZURE_DOC_INTEL_KEY")
    if not endpoint or not key:
        print("x set AZURE_DOC_INTEL_ENDPOINT and AZURE_DOC_INTEL_KEY in .env first "
              "(provision a Document Intelligence resource).")
        return 1

    contract_id = "FuseMedicalInc_20190321_10-K_EX-10.43_11575454_EX-10.43_Distributor Agreement.pdf"
    pdf = _text_to_pdf(load_contract_text(contract_id)[:6000])
    print(f"-> Document Intelligence  {endpoint}\n   test PDF: {len(pdf)} bytes from {contract_id[:40]}...")

    parser = AzureDocIntelligenceParser.for_azure(endpoint=endpoint, api_key=key)
    try:
        doc = parser.parse(content=pdf, filename="smoke.pdf")
    except Exception as exc:
        print(f"x live call failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"\nOK  parser_version={doc.parser_version}  pages={len(doc.pages)}  chars={len(doc.text)}")
    print(f"    first 120 chars: {doc.text[:120]!r}")
    if not doc.text.strip():
        print("x  empty text returned"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
