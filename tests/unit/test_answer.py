"""Unit tests for the RAG answer use-case — every outcome bucket, offline via fake ports.

Verifies the honesty invariants from evals/METRICS.md: malformed / abstained /
blocked / answered are distinct, malformed fails closed and is never counted as an
abstention, and zero-context abstains without calling the LLM.
"""
from __future__ import annotations

import json

import pytest

from app.application.answer import NOT_SPECIFIED, answer_question
from app.domain.ports import GuardrailResult

EMB, MODEL = "embed-model", "test-model"


# ---- inline fake ports (search / registry / guardrails) --------------------
class FakeSearch:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.calls = []

    def index(self, chunks, vectors):  # pragma: no cover - unused here
        pass

    def search(self, *, query, query_vector, filters, top_k):
        self.calls.append({"query": query, "query_vector": query_vector,
                           "filters": filters, "top_k": top_k})
        return list(self._chunks[:top_k])


class FakeRegistry:
    def get(self, name, version):
        return "SYSTEM" if name.endswith("_system") else "Context:\n{context}\n\nQuestion: {question}"

    def resolve_production(self, name):
        return "v1"


class Pass:
    def inspect_input(self, text, *, metadata):
        return GuardrailResult(allowed=True, findings=[])

    def inspect_output(self, *, answer, context, metadata):
        return GuardrailResult(allowed=True, findings=[])


class BlockIn(Pass):
    def inspect_input(self, text, *, metadata):
        return GuardrailResult(allowed=False, findings=["pii:PERSON"])


class BlockOut(Pass):
    def inspect_output(self, *, answer, context, metadata):
        return GuardrailResult(allowed=False, findings=["ungrounded_claim"])


class RedactOut(Pass):
    def inspect_output(self, *, answer, context, metadata):
        return GuardrailResult(allowed=True, redacted_text="[REDACTED]", findings=["pii:PERSON"])


class ExplodingOutput(Pass):
    def inspect_output(self, *, answer, context, metadata):  # must never be called on malformed/no-context
        raise AssertionError("inspect_output must not run when there is no trustworthy answer")


def _run(llm, embedder, search, guardrail, **kw):
    return answer_question("Which state's law governs?", llm=llm, embedder=embedder,
                           search=search, guardrail=guardrail, registry=FakeRegistry(),
                           embed_model=EMB, model=MODEL, **kw)


# ---- happy path ------------------------------------------------------------
def test_answered_with_valid_citation(make_llm, fake_embedder, retrieved_chunk):
    reply = json.dumps({"answer": "New York", "not_specified": False, "citations": ["c1"]})
    llm = make_llm(reply)
    chunk = retrieved_chunk(chunk_id="c1")
    search = FakeSearch([chunk])
    r = _run(llm, fake_embedder, search, Pass())

    assert r.answer == "New York"
    assert (r.abstained, r.malformed, r.blocked, r.no_context) == (False, False, False, False)
    assert r.llm_called and r.retrieved_k == 1
    assert len(r.citations) == 1 and r.citations[0].chunk_id == "c1"
    assert len(r.citations[0].text_sha256) == 64  # computed here (RetrievedChunk lacks it)


def test_search_receives_both_query_and_vector(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "NY", "not_specified": False}))
    search = FakeSearch([retrieved_chunk(chunk_id="c1")])
    _run(llm, fake_embedder, search, Pass())
    call = search.calls[0]
    assert call["query"] == "Which state's law governs?"          # BM25 needs the text
    assert call["query_vector"] == fake_embedder.embed_query(call["query"], model=EMB)


# ---- abstention (two routes) ----------------------------------------------
def test_abstain_via_not_specified_flag(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "NOT SPECIFIED", "not_specified": True}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), Pass())
    assert r.abstained and not r.malformed and r.answer == NOT_SPECIFIED and r.citations == []


def test_abstain_via_answer_text(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "not  specified", "not_specified": False}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), Pass())
    assert r.abstained and not r.malformed


# ---- malformed = counted failure, fails closed, distinct from abstain ------
def test_malformed_non_json(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm("this is not json")
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), Pass())
    assert r.malformed and not r.abstained
    assert r.answer == NOT_SPECIFIED and r.raw_output == "this is not json"


def test_malformed_missing_required_keys(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm("{}")  # valid JSON but missing 'answer' -> malformed
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), Pass())
    assert r.malformed and not r.abstained


def test_empty_answer_is_malformed_not_answered(make_llm, fake_embedder, retrieved_chunk):
    # an answered reply whose answer is blank asserts nothing -> M4 malformed, not a false answer/abstention
    llm = make_llm(json.dumps({"answer": "   ", "not_specified": False}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), Pass())
    assert r.malformed and not r.abstained and r.answer == NOT_SPECIFIED


def test_malformed_skips_output_guardrail(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm("garbage")
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), ExplodingOutput())
    assert r.malformed  # ExplodingOutput.inspect_output would have raised if called


# ---- no context -> abstain WITHOUT calling the LLM -------------------------
def test_no_context_abstains_without_llm(make_llm, fake_embedder):
    llm = make_llm(json.dumps({"answer": "X", "not_specified": False}))
    r = _run(llm, fake_embedder, FakeSearch([]), ExplodingOutput())
    assert r.no_context and r.abstained and not r.llm_called
    assert r.retrieved_k == 0 and llm.calls == []  # LLM never called


# ---- blocked input -> no embed / no search / no LLM ------------------------
def test_blocked_input_short_circuits(make_llm, fake_embedder):
    llm = make_llm("{}")
    search = FakeSearch([])
    r = _run(llm, fake_embedder, search, BlockIn())
    assert r.blocked and r.answer == "" and not r.llm_called
    assert "pii:PERSON" in r.findings and search.calls == [] and llm.calls == []


# ---- blocked output -> fail closed, answer withheld ------------------------
def test_blocked_output_withholds_answer(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "New York", "not_specified": False, "citations": ["c1"]}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk(chunk_id="c1")]), BlockOut())
    assert r.blocked and r.answer == "" and "ungrounded_claim" in r.findings


# ---- citation pointing outside retrieved set is flagged + dropped ----------
def test_citation_not_in_context_flagged(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "NY", "not_specified": False, "citations": ["c1", "ghost"]}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk(chunk_id="c1")]), Pass())
    assert "citation_not_in_context" in r.findings
    assert [c.chunk_id for c in r.citations] == ["c1"]  # ghost dropped


# ---- output redaction is applied ------------------------------------------
def test_prompt_version_v2_uses_tightened_system_prompt(make_llm, fake_embedder, retrieved_chunk):
    # X->Y: pinning prompt_version="v2" must feed the abstention-tuned system prompt from the registry
    from app.adapters.registry.file_registry import FilePromptRegistry

    llm = make_llm(json.dumps({"answer": "NOT SPECIFIED", "not_specified": True}))
    answer_question("Which law governs?", llm=llm, embedder=fake_embedder,
                    search=FakeSearch([retrieved_chunk()]), guardrail=Pass(),
                    registry=FilePromptRegistry(), embed_model=EMB, model=MODEL, prompt_version="v2")
    assert "EXPLICITLY" in llm.calls[0]["system_prompt"]


def test_default_prompt_version_uses_v1(make_llm, fake_embedder, retrieved_chunk):
    from app.adapters.registry.file_registry import FilePromptRegistry

    llm = make_llm(json.dumps({"answer": "NOT SPECIFIED", "not_specified": True}))
    answer_question("Which law governs?", llm=llm, embedder=fake_embedder,
                    search=FakeSearch([retrieved_chunk()]), guardrail=Pass(),
                    registry=FilePromptRegistry(), embed_model=EMB, model=MODEL)
    assert "EXPLICITLY" not in llm.calls[0]["system_prompt"]


def test_output_redaction_applied(make_llm, fake_embedder, retrieved_chunk):
    llm = make_llm(json.dumps({"answer": "Signed by John Doe", "not_specified": False}))
    r = _run(llm, fake_embedder, FakeSearch([retrieved_chunk()]), RedactOut())
    assert r.answer == "[REDACTED]" and "pii:PERSON" in r.findings and not r.blocked
