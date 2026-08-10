"""No-op guardrail: always allows, never redacts.

A placeholder so the pipeline has a Guardrail to call. Real PII / content-safety
inspection is a later adapter behind the same Protocol.
"""
from __future__ import annotations

from typing import Sequence

from app.domain.ports import GuardrailResult, RetrievedChunk


class PassthroughGuardrail:
    def inspect_input(self, text: str, *, metadata: dict) -> GuardrailResult:
        findings: list[str] = []
        return GuardrailResult(allowed=True, redacted_text=None, findings=findings)

    def inspect_output(
        self, *, answer: str, context: Sequence[RetrievedChunk], metadata: dict
    ) -> GuardrailResult:
        findings: list[str] = []
        return GuardrailResult(allowed=True, redacted_text=None, findings=findings)
