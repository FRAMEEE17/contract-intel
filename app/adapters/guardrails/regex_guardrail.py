"""Deterministic guardrail: block prompt-injection, redact PII, no external service.

inspect_input blocks obvious jailbreak/injection attempts and redacts PII from the
question before it reaches the index, model, or logs. inspect_output redacts PII the
model may have echoed into the answer. Regex-based, so it is reproducible and adds no
network call; Azure AI Content Safety is the managed upgrade behind the same Protocol.
"""
from __future__ import annotations

import re
from typing import Sequence

from app.domain.ports import GuardrailResult, RetrievedChunk

_INJECTION = re.compile(
    r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+instructions"
    r"|disregard\s+(the\s+)?(above|previous|prior)"
    r"|system\s+prompt"
    r"|you\s+are\s+now\b"
    r"|jailbreak|developer\s+mode"
    r"|reveal\s+(your\s+)?(instructions|system\s+prompt|prompt)",
    re.IGNORECASE,
)

_PII = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
]


def _redact(text: str) -> tuple[str, list[str]]:
    findings: list[str] = []
    for label, pattern in _PII:
        if pattern.search(text):
            findings.append(f"pii:{label}")
            text = pattern.sub(f"[REDACTED_{label}]", text)
    return text, findings


class RegexGuardrail:
    def inspect_input(self, text: str, *, metadata: dict) -> GuardrailResult:
        if _INJECTION.search(text):
            return GuardrailResult(allowed=False, redacted_text=None, findings=["prompt_injection"])
        redacted, findings = _redact(text)
        return GuardrailResult(
            allowed=True,
            redacted_text=redacted if findings else None,
            findings=findings,
        )

    def inspect_output(
        self, *, answer: str, context: Sequence[RetrievedChunk], metadata: dict
    ) -> GuardrailResult:
        redacted, findings = _redact(answer)
        return GuardrailResult(
            allowed=True,
            redacted_text=redacted if findings else None,
            findings=findings,
        )
