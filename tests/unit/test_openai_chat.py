"""OpenAIChatLLM request dialect — classic vs reasoning (o-series / gpt-5) backends."""
from __future__ import annotations

from app.adapters.llm.openai_chat import OpenAIChatLLM
from app.config import _is_reasoning_model


class _Recorder:
    """Minimal client exposing .chat.completions.create, capturing the request."""

    def __init__(self) -> None:
        self.seen: dict = {}
        outer = self

        class _Msg:
            content = '{"answer": "x", "grounded": true}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "m"
            usage = None

        class _Completions:
            def create(self, **kwargs):
                outer.seen = kwargs
                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _run(reasoning: bool) -> dict:
    client = _Recorder()
    llm = OpenAIChatLLM(client, max_tokens=16384, json_mode=True, reasoning=reasoning)
    llm.complete(
        system_prompt="s", user_prompt="u", response_schema={"type": "object"},
        model="gpt-5-mini", prompt_version="v", temperature=0.0, seed=0,
    )
    return client.seen


def test_classic_backend_uses_max_tokens_and_sampling_controls():
    req = _run(reasoning=False)
    assert req["max_tokens"] == 16384
    assert req["temperature"] == 0.0
    assert req["seed"] == 0
    assert "max_completion_tokens" not in req


def test_reasoning_backend_switches_param_and_drops_temperature_seed():
    req = _run(reasoning=True)
    assert req["max_completion_tokens"] == 16384
    assert "max_tokens" not in req
    assert "temperature" not in req   # reasoning models reject non-default temperature
    assert "seed" not in req


def test_reasoning_model_detected_by_name():
    assert _is_reasoning_model("gpt-5-mini")
    assert _is_reasoning_model("o3-mini")
    assert not _is_reasoning_model("gpt-4o-mini")
