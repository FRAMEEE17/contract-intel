"""RAG answer use-case — compose the ports into one honest, source-grounded answer.

Depends ONLY on the Protocols in app.domain.ports (dependency injection): the
concrete provider (Azure OpenAI, local MLX, in-memory search, …) is chosen by the
caller / composition root, never here.

Honesty rules baked in (evals/METRICS.md §0, §2):
  * A malformed LLM reply is a COUNTED failure (M4) and fails CLOSED — it is NOT
    silently dropped and NOT counted as an abstention.
  * "Not specified" (abstention), "blocked" (guardrail), "malformed", and
    "answered" are DISTINCT outcomes; at most one is the primary result.
  * No `grounded: bool` is emitted — we have no verifier model yet, so claiming
    semantic grounding would be an unverifiable overclaim. We surface deterministic
    signals (citation validity, guardrail findings); the eval jury owns verdicts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from app.domain.ports import (
    Embedder,
    Guardrail,
    LLMClient,
    PromptRegistry,
    RetrievedChunk,
    SearchClient,
)

NOT_SPECIFIED = "NOT SPECIFIED"

# Minimal structured-output schema — works on Azure OpenAI JSON mode and local MLX.
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "not_specified": {"type": "boolean"},
        "citations": {"type": "array", "items": {"type": "string"}},  # chunk_ids
    },
    "required": ["answer", "not_specified"],
}


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    page: int | None
    section: str | None
    source_revision: str
    score: float
    text_sha256: str  # computed here from chunk.text — RetrievedChunk does not carry it


@dataclass(frozen=True)
class AnswerResult:
    answer: str                 # user-facing; NOT_SPECIFIED on abstain/malformed/no-context; "" when blocked
    abstained: bool
    malformed: bool             # M4 counted failure
    blocked: bool               # input OR output guardrail refusal
    no_context: bool            # retrieval returned zero chunks
    citations: list[Citation] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    raw_output: str | None = None      # keep raw LLM text so a malformed reply stays auditable
    model: str = ""
    prompt_name: str = ""
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    retrieved_k: int = 0
    llm_called: bool = False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_abstained(answer: str) -> bool:
    return " ".join(answer.strip().upper().split()) == NOT_SPECIFIED


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (chunk_id={c.chunk_id}, section={c.section}, page={c.page})\n{c.text}")
    return "\n\n".join(blocks)


def _parse(raw: str):
    """Return a validated dict, or None if the reply is malformed (M4)."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if "answer" not in obj or not isinstance(obj["answer"], str):
        return None
    if "not_specified" in obj and not isinstance(obj["not_specified"], bool):
        return None
    citations = obj.get("citations", [])
    if not isinstance(citations, list) or not all(isinstance(x, str) for x in citations):
        return None
    return obj


def answer_question(
    question: str,
    *,
    llm: LLMClient,
    embedder: Embedder,
    search: SearchClient,
    guardrail: Guardrail,
    registry: PromptRegistry,
    embed_model: str,
    model: str,
    prompt_name: str = "answer",
    prompt_version: str | None = None,   # None -> registry.resolve_production(); pin for X→Y (§6)
    top_k: int = 8,
    filters: dict | None = None,
    temperature: float = 0.0,
    seed: int | None = 0,
    metadata: dict | None = None,
) -> AnswerResult:
    meta = metadata or {}

    # input guardrail: short-circuit BLOCKED before any embed/LLM call
    gr_in = guardrail.inspect_input(question, metadata=meta)
    if not gr_in.allowed:
        return AnswerResult(answer="", abstained=False, malformed=False, blocked=True,
                            no_context=False, findings=list(gr_in.findings),
                            prompt_name=prompt_name)

    # use redacted text downstream so PII never reaches the index/model
    query_text = gr_in.redacted_text or question

    # embed the QUERY (never the passage model) and hybrid-search (both query + vector)
    qv = embedder.embed_query(query_text, model=embed_model)
    chunks = search.search(query=query_text, query_vector=qv, filters=filters, top_k=top_k)
    chunks = sorted(chunks, key=lambda c: c.score, reverse=True)  # deterministic order for cassettes

    # no context -> honest source-grounded abstention, WITHOUT calling the LLM
    if not chunks:
        return AnswerResult(answer=NOT_SPECIFIED, abstained=True, malformed=False, blocked=False,
                            no_context=True, retrieved_k=0, prompt_name=prompt_name,
                            findings=list(gr_in.findings))

    version = prompt_version or registry.resolve_production(prompt_name)
    system_prompt = registry.get(f"{prompt_name}_system", version)
    user_template = registry.get(prompt_name, version)
    user_prompt = user_template.format(context=_format_context(chunks), question=query_text)

    resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt,
                        response_schema=RESPONSE_SCHEMA, model=model, prompt_version=version,
                        temperature=temperature, seed=seed)

    base = dict(model=resp.model, prompt_name=prompt_name, prompt_version=version,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                latency_ms=resp.latency_ms, retrieved_k=len(chunks), llm_called=True)

    # parse/validate -> MALFORMED fails closed; skip output guardrail (no trustworthy answer)
    parsed = _parse(resp.text)
    if parsed is None:
        return AnswerResult(answer=NOT_SPECIFIED, abstained=False, malformed=True, blocked=False,
                            no_context=False, raw_output=resp.text,
                            findings=list(gr_in.findings), **base) # type: ignore

    # genuine abstention (distinct from malformed)
    if parsed.get("not_specified") is True or _looks_abstained(parsed["answer"]):
        return AnswerResult(answer=NOT_SPECIFIED, abstained=True, malformed=False, blocked=False,
                            no_context=False, raw_output=resp.text,
                            findings=list(gr_in.findings), **base) # type: ignore

    # an empty/whitespace answer asserts nothing: treat as MALFORMED (M4),
    # never as an answer or abstention (would otherwise mislead the eval jury).
    answer_text = parsed["answer"].strip()
    if not answer_text:
        return AnswerResult(answer=NOT_SPECIFIED, abstained=False, malformed=True, blocked=False,
                            no_context=False, raw_output=resp.text,
                            findings=list(gr_in.findings), **base) # type: ignore

    by_id = {c.chunk_id: c for c in chunks}
    declared = parsed.get("citations", [])
    findings = list(gr_in.findings)
    valid_ids = [cid for cid in declared if cid in by_id]
    if any(cid not in by_id for cid in declared):
        findings.append("citation_not_in_context")
    citations = [
        Citation(chunk_id=c.chunk_id, document_id=c.document_id, page=c.page, section=c.section,source_revision=c.source_revision, score=c.score, text_sha256=_sha256(c.text))
        for c in (by_id[cid] for cid in valid_ids)
    ]

    # output guardrail: fail closed on refusal; apply redaction; merge findings
    gr_out = guardrail.inspect_output(answer=answer_text, context=chunks, metadata=meta)
    findings.extend(gr_out.findings)
    if not gr_out.allowed:
        return AnswerResult(answer="", abstained=False, malformed=False, blocked=True,
                            no_context=False, citations=citations, raw_output=resp.text,
                            findings=findings, **base) # type: ignore
    if gr_out.redacted_text is not None:
        answer_text = gr_out.redacted_text

    return AnswerResult(answer=answer_text, abstained=False, malformed=False, blocked=False,
                        no_context=False, citations=citations, raw_output=resp.text,
                        findings=findings, **base) # type: ignore
