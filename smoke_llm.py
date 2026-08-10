"""Smoke test: prove the wired LLMClient can talk to a live backend.

Uses the real composition root (app.config.build_llm), so it exercises provider
selection too. Defaults to the local MLX provider:

    mlx_lm.server --model mlx-community/Qwen3.5-9B-OptiQ-4bit --port 8081
    python smoke_llm.py

Point it elsewhere with the same env vars app/config.py reads, e.g.
    LLM_BASE_URL=http://localhost:8081/v1 python smoke_llm.py
    LLM_PROVIDER=azure AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=... python smoke_llm.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import build_llm, load_llm_settings

SYSTEM = (
    "You answer questions about a legal contract using ONLY the provided context. "
    "If the context does not contain the answer, say it is not specified."
)
CONTEXT = (
    "This Agreement shall be governed by and construed in accordance with the laws "
    "of the State of New York, without regard to its conflict-of-laws principles."
)
QUESTION = "Which state's law governs this agreement?"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "grounded": {"type": "boolean"}},
    "required": ["answer", "grounded"],
}


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def main() -> int:
    settings = load_llm_settings()
    target = settings.azure_endpoint or settings.base_url
    print(f"-> provider={settings.provider}  target={target}  model={settings.model}\n")

    try:
        llm = build_llm(settings)
        resp = llm.complete(
            system_prompt=SYSTEM,
            user_prompt=f"Context:\n{CONTEXT}\n\nQuestion: {QUESTION}",
            response_schema=SCHEMA,
            model=settings.model,
            prompt_version="smoke/v1",
        )
    except Exception as exc:
        print(f"x request failed: {type(exc).__name__}: {exc}")
        print("  is the server running / are the provider env vars set?")
        return 1

    print("--- raw text -------------------------------")
    print(resp.text)
    print("--------------------------------------------")
    print(f"model={resp.model}  in={resp.input_tokens} tok  out={resp.output_tokens} tok  {resp.latency_ms} ms")

    parsed = _extract_json(resp.text.strip())
    if parsed is not None:
        print(f"OK  structured output parsed: {parsed}")
    else:
        print("x  no valid JSON found (in the pipeline this is a COUNTED failure mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
