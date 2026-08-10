"""jury caching: a failed judge call (None) must NOT be frozen — it retries next run.

Regression test for the bug where rate-limited/None verdicts were cached and then
replayed as permanent 'no verdict', poisoning the eval with indeterminate items.
"""
from __future__ import annotations

import json

from evals import jury


def test_extract_verdict_plain_json():
    assert jury._extract_verdict('{"verdict": "correct", "reason": "ok"}') == "correct"


def test_extract_verdict_ignores_reasoning_block():
    # reasoning models emit <think>...{stray braces}...</think> before the real JSON
    raw = ('\n<think>\nMaybe the answer looks like {something} but let me decide.\n</think>\n'
           '{"verdict": "fabricated", "reason": "invented a term"}')
    assert jury._extract_verdict(raw) == "fabricated"


def test_extract_verdict_rejects_unknown_and_missing():
    assert jury._extract_verdict('{"verdict": "banana"}') is None   # not in taxonomy
    assert jury._extract_verdict("no json here") is None


def test_judge_one_does_not_cache_none(tmp_path, monkeypatch):
    monkeypatch.setattr(jury, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(jury, "_ask_judge", lambda model, prompt: None)  # simulate API failure
    assert jury.judge_one("m", "q", "ref", "ans") is None
    assert list(tmp_path.glob("*.json")) == []  # nothing frozen -> retried on the next run


def test_judge_one_caches_valid_verdict_and_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(jury, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(jury, "_ask_judge", lambda model, prompt: '{"verdict": "correct", "reason": "ok"}')
    assert jury.judge_one("m", "q", "ref", "ans") == "correct"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["verdict"] == "correct"

    # second call must hit the cache, never the API
    def _boom(model, prompt):
        raise AssertionError("cached verdict should not re-call the judge")
    monkeypatch.setattr(jury, "_ask_judge", _boom)
    assert jury.judge_one("m", "q", "ref", "ans") == "correct"
