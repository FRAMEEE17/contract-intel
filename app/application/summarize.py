"""Contract summarization use-case: one LLM call over the contract text.

Separate from answer_question (which does grounded clause QA with abstention). This
is freeform summary, not an eval-gated metric, so it does not use the jury or the
prompt registry.
"""
from __future__ import annotations

from app.domain.ports import LLMClient

SUMMARY_SYSTEM = (
    "You are a legal analyst. Summarize the contract below in plain English. Cover, "
    "when present: the parties, the term and effective date, key obligations, "
    "termination, exclusivity, confidentiality, liability, and governing law. Be "
    "concise and factual, and do not state terms that are not in the text."
)


def summarize_contract(
    text: str, *, llm: LLMClient, model: str, max_chars: int = 12000
) -> str:
    """Return a plain-English summary of `text` (truncated to max_chars)."""
    response = llm.complete(
        system_prompt=SUMMARY_SYSTEM,
        user_prompt=text[:max_chars],
        response_schema=None,
        model=model,
        prompt_version="summarize/v1",
    )
    return response.text.strip()
