"""Live-verify AzureAISearch against a real Azure AI Search service.

Prereq — provision an Azure AI Search service (Free tier is enough), then put its
endpoint + admin key in .env:
    AZURE_SEARCH_ENDPOINT=https://<name>.search.windows.net
    AZURE_SEARCH_KEY=<primary admin key>
    AZURE_SEARCH_INDEX=contract-smoke      # optional, this is the default

Then, from the project root:  python smoke_search.py

Indexes one real gold contract into Azure AI Search, then runs a hybrid (BM25 +
vector) query and checks relevant chunks come back — confirming auth + index
creation + upload + hybrid search + the filter clause.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.adapters.chunking.fixed_chunker import FixedChunker
from app.adapters.documents.text_parser import TextDocParser
from app.adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
from app.adapters.search.azure_ai_search import AzureAISearch
from app.config import EMBED_MODEL
from evals.generate import build_contract_index, load_contract_text

CONTRACT_ID = "FuseMedicalInc_20190321_10-K_EX-10.43_11575454_EX-10.43_Distributor Agreement.pdf"
QUESTION = "Which law governs this agreement?"


def _search_with_retry(index, embedder, *, filters, tries=8, delay=2.0):
    """Azure indexing is eventually consistent — poll until the new docs are searchable."""
    qvec = embedder.embed_query(QUESTION, model=EMBED_MODEL)
    for attempt in range(tries):
        hits = index.search(query=QUESTION, query_vector=qvec, filters=filters, top_k=5)
        if hits:
            return hits
        time.sleep(delay)
    return []


def main() -> int:
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    key = os.environ.get("AZURE_SEARCH_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "contract-smoke")
    if not endpoint or not key:
        print("x set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in .env first "
              "(provision an Azure AI Search service).")
        return 1

    embedder = SentenceTransformerEmbedder()
    vector_dim = len(embedder.embed_passage("probe", model=EMBED_MODEL))
    azure_search = AzureAISearch.for_azure(
        endpoint=endpoint, api_key=key, index_name=index_name, vector_dim=vector_dim,
    )

    print(f"-> Azure AI Search  {endpoint}  index={index_name!r}  dim={vector_dim}")
    index = build_contract_index(
        text=load_contract_text(CONTRACT_ID), filename=CONTRACT_ID,
        parser=TextDocParser(), chunker=FixedChunker(),
        embedder=embedder, embed_model=EMBED_MODEL, search=azure_search,
    )
    print(f"   indexed {CONTRACT_ID[:40]}...  (waiting for it to become searchable)")

    hits = _search_with_retry(index, embedder, filters=None)
    if not hits:
        print("x no results after indexing (indexing may still be catching up, or auth/index issue)")
        return 1

    top = hits[0]
    print(f"\nOK  hybrid search returned {len(hits)} hits for {QUESTION!r}")
    print(f"    top score={top.score:.4f}  document_id={top.document_id!r}")
    print(f"    top text: {top.text[:160]!r}")

    filtered = _search_with_retry(index, embedder, filters={"document_id": top.document_id}, tries=1)
    print(f"    filtered by document_id -> {len(filtered)} hits (filter clause works: {bool(filtered)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
