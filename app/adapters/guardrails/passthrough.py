"""No-op guardrail — implements app.domain.ports.Guardrail.

MVP passthrough: always allows, never redacts, never finds anything. Real
PII / content-safety inspection is a later adapter behind the same Protocol;
this one exists so the pipeline has a Guardrail to call from day one.
"""
from __future__ import annotations

from typing import Sequence

from app.domain.ports import GuardrailResult, RetrievedChunk


class PassthroughGuardrail:
    """Implements app.domain.ports.Guardrail as an always-allow no-op."""

    def inspect_input(self, text: str, *, metadata: dict) -> GuardrailResult:
        findings: list[str] = []
        return GuardrailResult(allowed=True, redacted_text=None, findings=findings)

    def inspect_output(
        self, *, answer: str, context: Sequence[RetrievedChunk], metadata: dict
    ) -> GuardrailResult:
        findings: list[str] = []
        return GuardrailResult(allowed=True, redacted_text=None, findings=findings)
