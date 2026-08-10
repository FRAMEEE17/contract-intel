"""Hybrid search backed by an Azure AI Search index.

index() creates the schema on first use (idempotent) and uploads chunks + vectors.
search() issues one hybrid request (BM25 + vector); Azure fuses them with RRF
server-side. Filters become an OData equality clause applied before ranking.

Azure keys allow only [A-Za-z0-9_-=], but chunk ids carry ':' and spaces, so the key
is a base64url of the chunk id and the original rides along in a `chunk_id` field.
"""
from __future__ import annotations

import base64
from typing import Any, Sequence

from app.domain.ports import Chunk, RetrievedChunk

_KEY_FIELD = "id"
_VECTOR_FIELD = "vector"
_ALGO = "hnsw-config"
_PROFILE = "hnsw-profile"
_SELECT = ["chunk_id", "document_id", "text", "page", "section", "source_revision"]


def _encode_key(chunk_id: str) -> str:
    return base64.urlsafe_b64encode(chunk_id.encode("utf-8")).decode("ascii")


def _filter_expression(filters: dict | None) -> str | None:
    if not filters:
        return None
    clauses = [f"{key} eq '{str(value).replace(chr(39), chr(39) * 2)}'" for key, value in filters.items()]
    return " and ".join(clauses)


class AzureAISearch:
    """Implements app.domain.ports.SearchClient against an Azure AI Search index."""

    def __init__(self, *, search_client: Any, index_client: Any, index_name: str, vector_dim: int) -> None:
        self._client = search_client
        self._index_client = index_client
        self._index_name = index_name
        self._vector_dim = vector_dim

    @classmethod
    def for_azure(
        cls, *, endpoint: str, api_key: str, index_name: str, vector_dim: int,
        retry_total: int = 5, retry_backoff_factor: float = 0.8,
    ) -> "AzureAISearch":
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        # azure-core already retries 429/5xx with exponential backoff and honors
        # Retry-After; these just make the throttling posture explicit and tunable.
        credential = AzureKeyCredential(api_key)
        retry = {"retry_total": retry_total, "retry_backoff_factor": retry_backoff_factor}
        return cls(
            search_client=SearchClient(endpoint=endpoint, index_name=index_name, credential=credential, **retry),
            index_client=SearchIndexClient(endpoint=endpoint, credential=credential, **retry),
            index_name=index_name,
            vector_dim=vector_dim,
        )

    def _ensure_index(self) -> None:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration, SearchableField, SearchField,
            SearchFieldDataType, SearchIndex, SimpleField, VectorSearch,
            VectorSearchProfile,
        )
        fields = [
            SimpleField(name=_KEY_FIELD, type=SearchFieldDataType.String, key=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SearchField(
                name=_VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._vector_dim,
                vector_search_profile_name=_PROFILE,
            ),
            SimpleField(name="page", type=SearchFieldDataType.Int32),
            SimpleField(name="section", type=SearchFieldDataType.String),
            SimpleField(name="source_revision", type=SearchFieldDataType.String),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=_ALGO)],
            profiles=[VectorSearchProfile(name=_PROFILE, algorithm_configuration_name=_ALGO)],
        )
        self._index_client.create_or_update_index(
            SearchIndex(name=self._index_name, fields=fields, vector_search=vector_search)
        )

    def index(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        chunks, vectors = list(chunks), list(vectors)
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks and vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        if not chunks:
            return
        self._ensure_index()
        self._client.upload_documents(documents=[
            {
                _KEY_FIELD: _encode_key(c.chunk_id),
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                _VECTOR_FIELD: list(v),
                "page": c.page,
                "section": c.section,
                "source_revision": c.source_revision,
            }
            for c, v in zip(chunks, vectors)
        ])

    def search(
        self, *, query: str, query_vector: list[float], filters: dict | None, top_k: int
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        from azure.search.documents.models import VectorizedQuery

        results = self._client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields=_VECTOR_FIELD)],
            filter=_filter_expression(filters),
            top=top_k,
            select=_SELECT,
        )
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=float(r["@search.score"]),
                page=r.get("page"),
                section=r.get("section"),
                source_revision=r.get("source_revision") or "",
            )
            for r in results
        ]
