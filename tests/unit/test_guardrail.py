"""Unit tests for app.adapters.guardrails.passthrough.PassthroughGuardrail."""
from __future__ import annotations

from app.adapters.guardrails.passthrough import PassthroughGuardrail


def test_inspect_input_allows_and_has_no_findings():
    guardrail = PassthroughGuardrail()
    result = guardrail.inspect_input("What is the governing law?", metadata={})
    assert result.allowed is True
    assert result.findings == []


def test_inspect_output_allows_and_has_no_findings():
    guardrail = PassthroughGuardrail()
    result = guardrail.inspect_output(
        answer="The governing law is New York.",
        context=[],
        metadata={},
    )
    assert result.allowed is True
    assert result.findings == []
