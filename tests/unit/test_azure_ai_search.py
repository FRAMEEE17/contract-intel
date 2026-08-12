"""AzureAISearch adapter — schema, upload, hybrid query, and result mapping (offline via fakes)."""
from __future__ import annotations

import pytest

from app.adapters.search.azure_ai_search import AzureAISearch, _encode_key, _filter_expression
from app.domain.ports import Chunk


class _FakeIndexClient:
    def __init__(self) -> None:
        self.created = None

    def create_or_update_index(self, index):
        self.created = index


class _FakeSearchClient:
    def __init__(self, results=None) -> None:
        self.uploaded = None
        self.search_kwargs = None
        self._results = results or []

    def upload_documents(self, documents):
        self.uploaded = documents

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return list(self._results)


def _adapter(results=None):
    idx, cli = _FakeIndexClient(), _FakeSearchClient(results)
    return AzureAISearch(search_client=cli, index_client=idx, index_name="t", vector_dim=384), idx, cli


def _chunk(cid, doc, text):
    return Chunk(chunk_id=cid, document_id=doc, text=text)


def test_encode_key_is_azure_safe():
    key = _encode_key("Foo Bar.pdf:3:abc123")          # colons, spaces, dot
    assert all(c.isalnum() or c in "-_=" for c in key)  # Azure key charset


def test_filter_expression_escapes_and_handles_empty():
    assert _filter_expression(None) is None
    assert _filter_expression({}) is None
    assert _filter_expression({"document_id": "a"}) == "document_id eq 'a'"
    assert _filter_expression({"document_id": "O'Brien"}) == "document_id eq 'O''Brien'"


def test_index_builds_vector_schema_and_uploads_encoded_docs():
    adapter, idx, cli = _adapter()
    adapter.index([_chunk("C:0:aa", "doc.pdf", "governing law is New York")], [[0.1] * 384])

    fields = {f.name: f for f in idx.created.fields}
    assert fields["id"].key is True
    assert fields["vector"].vector_search_dimensions == 384
    assert idx.created.vector_search is not None

    doc = cli.uploaded[0]
    assert doc["id"] == _encode_key("C:0:aa")   # key is the encoded chunk id
    assert doc["chunk_id"] == "C:0:aa"          # original rides along
    assert doc["document_id"] == "doc.pdf"
    assert len(doc["vector"]) == 384


def test_index_rejects_length_mismatch_and_noops_on_empty():
    adapter, idx, cli = _adapter()
    with pytest.raises(ValueError):
        adapter.index([_chunk("a", "d", "t")], [])       # 1 chunk, 0 vectors
    adapter.index([], [])                                 # empty -> no index, no upload
    assert idx.created is None and cli.uploaded is None


def test_search_issues_hybrid_query_and_maps_results():
    hit = {
        "chunk_id": "C:0:aa", "document_id": "doc.pdf", "text": "New York law governs",
        "page": 2, "section": None, "source_revision": "", "@search.score": 0.87,
    }
    adapter, _, cli = _adapter(results=[hit])
    out = adapter.search(query="which law?", query_vector=[0.2] * 384,
                         filters={"document_id": "doc.pdf"}, top_k=5)

    kw = cli.search_kwargs
    assert kw["search_text"] == "which law?"          # BM25 arm
    assert len(kw["vector_queries"]) == 1             # vector arm -> Azure fuses (RRF)
    assert kw["filter"] == "document_id eq 'doc.pdf'"
    assert kw["top"] == 5

    assert len(out) == 1
    assert out[0].chunk_id == "C:0:aa"
    assert out[0].score == pytest.approx(0.87)
    assert out[0].page == 2


def test_search_short_circuits_on_nonpositive_top_k():
    adapter, _, cli = _adapter()
    assert adapter.search(query="q", query_vector=[0.1] * 384, filters=None, top_k=0) == []
    assert cli.search_kwargs is None


def test_credential_is_key_when_provided():
    from azure.core.credentials import AzureKeyCredential
    from app.adapters.search.azure_ai_search import _build_credential

    assert isinstance(_build_credential("k"), AzureKeyCredential)


def test_credential_is_managed_identity_without_key():
    from azure.identity import DefaultAzureCredential
    from app.adapters.search.azure_ai_search import _build_credential

    assert isinstance(_build_credential(None), DefaultAzureCredential)
    assert isinstance(_build_credential(""), DefaultAzureCredential)
