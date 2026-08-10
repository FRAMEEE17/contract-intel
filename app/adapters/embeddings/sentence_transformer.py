"""Local sentence-transformers embedder.

Queries and passages get different instruction prefixes ("query: " / "passage: ")
so they embed differently. sentence-transformers is a heavy optional dependency, so
the import is guarded and models are cached per model-id.
"""
from __future__ import annotations

from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only when the optional dep is absent
    SentenceTransformer = None  # type: ignore[assignment,misc]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class SentenceTransformerEmbedder:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def embed_query(self, text: str, *, model: str = DEFAULT_MODEL) -> list[float]:
        return self._encode(_QUERY_PREFIX + text, model=model)

    def embed_passage(self, text: str, *, model: str = DEFAULT_MODEL) -> list[float]:
        return self._encode(_PASSAGE_PREFIX + text, model=model)

    def _encode(self, text: str, *, model: str) -> list[float]:
        vector = self._get_model(model).encode(text, convert_to_numpy=True)
        return vector.tolist()

    def _get_model(self, model: str) -> Any:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` to use SentenceTransformerEmbedder."
            )
        if model not in self._models:
            self._models[model] = SentenceTransformer(model)
        return self._models[model]
