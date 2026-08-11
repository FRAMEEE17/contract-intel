"""Contract summarization use-case: one LLM call returning a structured summary.

Separate from answer_question (grounded clause QA with abstention). This produces a
decision-support summary (parties, term, key conditions with priority, important
dates) as JSON; it is not eval-gated and does not use the jury.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.domain.ports import LLMClient

SUMMARY_SYSTEM = (
    "You are a legal analyst summarizing a contract for a decision-maker. Return the "
    "structured summary as JSON. Use \"not specified\" for anything the contract does "
    "not state, and do not invent terms. Sort key_conditions and important_dates by "
    "priority (high first)."
)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "parties": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "role": {"type": "string"}}}},
        "term": {"type": "object", "properties": {
            "start_date": {"type": "string"}, "end_date": {"type": "string"},
            "initial_term": {"type": "string"}}},
        "overview": {"type": "string"},
        "key_conditions": {"type": "array", "items": {"type": "object", "properties": {
            "priority": {"type": "string"}, "description": {"type": "string"},
            "impact": {"type": "string"}}}},
        "important_dates": {"type": "array", "items": {"type": "object", "properties": {
            "priority": {"type": "string"}, "date": {"type": "string"},
            "description": {"type": "string"}}}},
    },
    "required": ["title", "overview", "parties"],
}


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"overview": text.strip()}


def summarize_contract(text: str, *, llm: LLMClient, model: str, max_chars: int = 12000) -> dict:
    response = llm.complete(
        system_prompt=SUMMARY_SYSTEM,
        user_prompt=text[:max_chars],
        response_schema=SUMMARY_SCHEMA,
        model=model,
        prompt_version="summarize/v2",
    )
    return _parse(response.text)
