"""API smoke: /health and /answer wire the pipeline and return structured output."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from app.adapters.chunking.fixed_chunker import FixedChunker
from app.adapters.documents.text_parser import TextDocParser
from app.adapters.guardrails.passthrough import PassthroughGuardrail
from app.adapters.registry.file_registry import FilePromptRegistry
from app.config import EMBED_MODEL, Pipeline

CONTRACT = "This Agreement is governed by the laws of the State of New York."


def _client(make_llm, fake_embedder, reply):
    pipeline = Pipeline(
        llm=make_llm(reply), embedder=fake_embedder, chunker=FixedChunker(),
        parser=TextDocParser(), guardrail=PassthroughGuardrail(),
        registry=FilePromptRegistry(), embed_model=EMBED_MODEL, model="fake-llm",
    )
    return TestClient(create_app(pipeline))


def test_health():
    assert TestClient(create_app(object())).get("/health").json() == {"status": "ok"}


def test_answer_returns_structured_answer(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder,
                     '{"answer": "New York", "not_specified": false, "citations": []}')
    resp = client.post("/answer", json={"question": "Which law governs?", "document_text": CONTRACT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "New York"
    assert body["abstained"] is False and body["malformed"] is False and body["blocked"] is False


def test_answer_abstains_when_model_says_not_specified(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder,
                     '{"answer": "NOT SPECIFIED", "not_specified": true, "citations": []}')
    body = client.post("/answer", json={"question": "What is the penalty?", "document_text": CONTRACT}).json()
    assert body["abstained"] is True


def test_answer_validation_rejects_empty_question(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder, "{}")
    assert client.post("/answer", json={"question": "", "document_text": CONTRACT}).status_code == 422


def test_summarize_returns_summary(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder, "Parties: A and B. Governing law: New York.")
    body = client.post("/summarize", json={"document_text": CONTRACT}).json()
    assert "New York" in body["summary"]


def test_summarize_a_library_contract_by_id(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder, "Summary: governed by New York law.")
    client.post("/contracts", json={"title": "A.pdf", "document_text": CONTRACT})
    body = client.post("/summarize", json={"document_id": "A.pdf"}).json()
    assert "New York" in body["summary"]


def test_summarize_unknown_contract_is_404(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder, "x")
    assert client.post("/summarize", json={"document_id": "missing.pdf"}).status_code == 404


def test_add_and_list_contracts(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder, "{}")
    client.post("/contracts", json={"title": "A.pdf", "document_text": CONTRACT})
    client.post("/contracts", json={"title": "B.pdf", "document_text": "Term: two years."})
    listed = client.get("/contracts").json()
    assert {c["document_id"] for c in listed} == {"A.pdf", "B.pdf"}
    assert all(c["chunks"] >= 1 for c in listed)


def test_answer_over_repository_by_document_id(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder,
                     '{"answer": "New York", "not_specified": false, "citations": []}')
    client.post("/contracts", json={"title": "A.pdf", "document_text": CONTRACT})
    body = client.post("/answer", json={"question": "Which law governs?", "document_id": "A.pdf"}).json()
    assert body["answer"] == "New York" and body["abstained"] is False


def test_abstract_returns_a_clause_grid(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder,
                     '{"answer": "New York", "not_specified": false, "citations": []}')
    client.post("/contracts", json={"title": "A.pdf", "document_text": CONTRACT})
    grid = client.post("/abstract", json={"document_id": "A.pdf"}).json()
    assert grid["document_id"] == "A.pdf"
    assert "Governing law" in grid["clauses"]
    assert grid["clauses"]["Governing law"]["present"] is True


def test_abstract_flags_missing_clause_on_abstention(make_llm, fake_embedder):
    client = _client(make_llm, fake_embedder,
                     '{"answer": "NOT SPECIFIED", "not_specified": true, "citations": []}')
    client.post("/contracts", json={"title": "A.pdf", "document_text": CONTRACT})
    grid = client.post("/abstract", json={"document_id": "A.pdf"}).json()
    assert grid["clauses"]["Liability cap"]["present"] is False  # abstain -> missing clause


def test_ingest_extracts_pdf_text(make_llm, fake_embedder):
    import fitz  # build a tiny text PDF in-memory

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Governing law: State of New York.")
    pdf_bytes = doc.tobytes()
    doc.close()

    client = _client(make_llm, fake_embedder, "{}")
    resp = client.post("/ingest", files={"file": ("contract.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    assert "New York" in body["text"] and body["pages"] >= 1
