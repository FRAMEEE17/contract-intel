"""Unit tests for evals.generate — contract text loading + index-building helpers.

Uses the deterministic FakeEmbedder from tests/conftest.py plus the REAL
TextDocParser and FixedChunker adapters, exercised against a real gold contract
so these tests double as a smoke test of the full parse -> chunk -> embed ->
index pipeline the eval harness composes.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.adapters.chunking.fixed_chunker import FixedChunker
from app.adapters.documents.text_parser import TextDocParser
from app.domain.ports import RetrievedChunk
from evals.generate import (
    build_contract_index,
    load_contract_text,
    resolve_contract_path,
)
from tests.conftest import FakeEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_gold_item() -> dict:
    path = REPO_ROOT / "evals" / "gold" / "gold.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_resolve_contract_path_returns_existing_txt_for_real_contract():
    contract_id = _first_gold_item()["contract_id"]

    path = resolve_contract_path(contract_id)

    assert path.exists()
    assert path.suffix == ".txt"


def test_resolve_contract_path_raises_lookup_error_for_bogus_id():
    try:
        resolve_contract_path("no-such-contract-anywhere_XYZ.pdf")
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError for a bogus contract_id")


def test_load_contract_text_returns_full_text_not_a_snippet():
    contract_id = _first_gold_item()["contract_id"]

    text = load_contract_text(contract_id)

    assert isinstance(text, str)
    assert len(text) > 1000


def test_build_contract_index_returns_searchable_index_over_real_contract():
    item = _first_gold_item()
    contract_id = item["contract_id"]
    text = load_contract_text(contract_id)

    search = build_contract_index(
        text=text,
        filename=contract_id,
        parser=TextDocParser(),
        chunker=FixedChunker(),
        embedder=FakeEmbedder(),
        embed_model="fake-embed-model",
    )

    query = "governing law"
    query_vector = FakeEmbedder().embed_query(query, model="fake-embed-model")
    results = search.search(query=query, query_vector=query_vector, filters=None, top_k=3)

    assert results
    assert all(isinstance(r, RetrievedChunk) for r in results)
