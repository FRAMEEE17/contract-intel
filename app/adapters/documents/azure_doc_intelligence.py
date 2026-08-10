"""Azure Document Intelligence parser (production OCR fallback).

Sends the PDF to Azure (prebuilt-read; swap model_id for a layout/table model) and
returns text + per-page lines. The client is injected, so parse() is unit-tested
with a fake and for_azure() builds the real one. Auth: Managed Identity by default,
API key only for local dev.
"""
from __future__ import annotations

from typing import Any

from app.domain.ports import ParsedDocument


class AzureDocIntelligenceParser:
    def __init__(self, client: Any, *, model_id: str = "prebuilt-read") -> None:
        self._client = client
        self._model_id = model_id

    @classmethod
    def for_azure(
        cls, *, endpoint: str, api_key: str | None = None, model_id: str = "prebuilt-read",
    ) -> "AzureDocIntelligenceParser":
        """Build the real client. `api_key=None` -> Managed Identity / Entra ID."""
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Azure Document Intelligence needs 'azure-ai-documentintelligence' "
                "(pip install azure-ai-documentintelligence)."
            ) from exc
        if api_key:
            from azure.core.credentials import AzureKeyCredential # type: ignore
            credential: Any = AzureKeyCredential(api_key)
        else:
            try:
                from azure.identity import DefaultAzureCredential # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Managed-Identity auth needs 'azure-identity' (pip install azure-identity), "
                    "or pass api_key for local dev."
                ) from exc
            credential = DefaultAzureCredential()
        return cls(DocumentIntelligenceClient(endpoint, credential), model_id=model_id)

    def parse(self, *, content: bytes, filename: str) -> ParsedDocument:
        poller = self._client.begin_analyze_document(
            self._model_id, body=content, content_type="application/pdf",
        )
        result = poller.result()

        pages: list[str] = []
        for page in getattr(result, "pages", None) or []:
            lines = getattr(page, "lines", None) or []
            pages.append("\n".join(getattr(ln, "content", "") for ln in lines))

        full_text = getattr(result, "content", "") or ""
        if not pages:
            pages = [full_text]
        return ParsedDocument(
            document_id=filename,
            text=full_text or "\f".join(pages),
            pages=pages,
            parser_version=f"azure-di/{self._model_id}",
        )
