"""Eval "generation" helpers: load gold contract text and build a search index.

Composes the concrete adapters (DocParser, Chunker, Embedder, SearchClient) via
app.domain.ports; no scoring/metrics logic lives here (see evals/jury.py / score.py).
"""
from __future__ import annotations

from pathlib import Path

from app.adapters.search.in_memory import InMemorySearch
from app.domain.ports import Chunker, DocParser, Embedder, SearchClient

CONTRACTS_DIR = Path(__file__).resolve().parent / "gold" / "contracts"

_FUZZY_PREFIX_LEN = 40


def resolve_contract_path(contract_id: str, contracts_dir: Path = CONTRACTS_DIR) -> Path:
    """Resolve a gold `contract_id` (e.g. "Foo.pdf") to its full-text .txt file."""
    stem = contract_id[: -len(".pdf")] if contract_id.endswith(".pdf") else contract_id

    direct = contracts_dir / f"{stem}.txt"
    if direct.exists():
        return direct

    prefix = stem[:_FUZZY_PREFIX_LEN]
    for candidate in sorted(contracts_dir.glob("*.txt")):
        if candidate.stem[:_FUZZY_PREFIX_LEN] == prefix:
            return candidate

    raise LookupError(f"no contract file found for contract_id={contract_id!r} in {contracts_dir}")


def load_contract_text(contract_id: str, contracts_dir: Path = CONTRACTS_DIR) -> str:
    """Read the full contract text for `contract_id` as utf-8 (invalid bytes replaced)."""
    path = resolve_contract_path(contract_id, contracts_dir)
    return path.read_text(encoding="utf-8", errors="replace")


def build_contract_index(
    *,
    text: str,
    filename: str,
    parser: DocParser,
    chunker: Chunker,
    embedder: Embedder,
    embed_model: str,
    max_tokens: int = 800,
    overlap: int = 80,
    search: SearchClient | None = None,
) -> SearchClient:
    """Parse -> chunk -> embed `text`, index it, and return the ready-to-query search client.

    `search` defaults to a fresh in-process InMemorySearch; pass an AzureAISearch (or
    any SearchClient) to index into an external backend instead.
    """
    document = parser.parse(content=text.encode("utf-8"), filename=filename)
    chunks = chunker.chunk(document=document, max_tokens=max_tokens, overlap=overlap)
    vectors = [embedder.embed_passage(chunk.text, model=embed_model) for chunk in chunks]

    search = search if search is not None else InMemorySearch()
    search.index(chunks, vectors)
    return search
