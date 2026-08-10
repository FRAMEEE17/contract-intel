"""End-to-end runner wiring: real gold contracts -> real pipeline -> score report.

Loads real full contract text from evals/gold/contracts/, builds the index with
the real adapters, and runs the eval runner. Only the LLM and the jury are
scripted (offline); everything else is production code. Marked integration and
skipped without sentence-transformers.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.domain.ports import LLMResponse
from evals.run_eval import make_answer_fn, score_items

try:
    import sentence_transformers  # noqa: F401
    _ST = True
except ImportError:
    _ST = False

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _ST, reason="sentence-transformers not installed")]

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GOLD = Path(__file__).resolve().parents[1].parent / "contract-intel" / "evals" / "gold" / "gold.jsonl"
if not GOLD.exists():  # running from repo root
    GOLD = Path(__file__).resolve().parents[2] / "evals" / "gold" / "gold.jsonl"


class _AnsweredLLM:
    """Always emits a well-formed 'answered' reply (offline)."""
    def complete(self, *, system_prompt, user_prompt, response_schema, model,
                 prompt_version, temperature=0.0, seed=0) -> LLMResponse:
        payload = json.dumps({"answer": "New York", "not_specified": False, "citations": []})
        return LLMResponse(text=payload, model="answered-llm", prompt_version=prompt_version,
                           input_tokens=len(user_prompt.split()), output_tokens=8, latency_ms=1)


def _scripted_jury(question, reference, answer):
    # unanswerable reference -> credit the abstention; answerable -> correct
    majority = "correct_abstain" if reference.strip().upper() == "NOT SPECIFIED" else "correct"
    return SimpleNamespace(majority=majority, per_judge=(majority,) * 3)


def _two_items_same_contract():
    rows = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    fuse = [r for r in rows if "Fuse" in r["contract_id"]]
    answerable = next(r for r in fuse if r["has_answer"])
    unanswerable = next(r for r in fuse if not r["has_answer"])
    return [answerable, unanswerable]


def test_runner_over_real_contracts_end_to_end():
    from app.adapters.documents.text_parser import TextDocParser
    from app.adapters.chunking.fixed_chunker import FixedChunker
    from app.adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.adapters.registry.file_registry import FilePromptRegistry
    from app.adapters.guardrails.passthrough import PassthroughGuardrail
    from app.config import Pipeline

    pipeline = Pipeline(
        llm=_AnsweredLLM(), embedder=SentenceTransformerEmbedder(), chunker=FixedChunker(),
        parser=TextDocParser(), guardrail=PassthroughGuardrail(), registry=FilePromptRegistry(),
        embed_model=EMB_MODEL, model="answered-llm",
    )
    answer_fn = make_answer_fn(pipeline, top_k=5)
    items = _two_items_same_contract()
    out = score_items(items, answer_fn=answer_fn, judge_fn=_scripted_jury)

    rep = out["report"]
    assert rep["total_items"] == 2 and rep["audit"]["partition_ok"]
    # both produced an 'answered' result -> jury decided each; buckets sum to total
    assert sum(rep["counts"].values()) == 2
    assert rep["answerable"] + rep["unanswerable"] == 2
