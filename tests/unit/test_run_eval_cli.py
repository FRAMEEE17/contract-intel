"""Unit tests for the runner's cassette layer + replay mode — fully offline.

The live glue (build_llm + live jury calls) is exercised only by `make eval-live`;
here we test the reproducible, deterministic pieces: system-answer cassettes save
and reload, replay errors loudly on a missing cassette, and replay -> score works.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals import run_eval


def _res(answer, *, abstained=False, malformed=False, blocked=False, no_context=False):
    return SimpleNamespace(answer=answer, abstained=abstained, malformed=malformed,
                           blocked=blocked, no_context=no_context)


def test_system_cassette_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SYSTEM_CASSETTE_DIR", tmp_path)
    item = {"id": "itemX"}
    run_eval.save_system_answer("v1", item, _res("New York"))
    got = run_eval.load_system_answer("v1", item)
    assert got.answer == "New York" and got.malformed is False and got.abstained is False


def test_cassette_key_depends_on_system_version(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SYSTEM_CASSETTE_DIR", tmp_path)
    item = {"id": "itemX"}
    run_eval.save_system_answer("v1", item, _res("A"))
    assert run_eval.load_system_answer("v2", item) is None  # different version -> different cassette


def test_replay_missing_cassette_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SYSTEM_CASSETTE_DIR", tmp_path)
    fn = run_eval.replay_answer_fn("v1")
    with pytest.raises(SystemExit):
        fn({"id": "does-not-exist"})


def test_replay_then_score(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SYSTEM_CASSETTE_DIR", tmp_path)
    items = [
        {"id": "a", "has_answer": True, "question": "q1", "ground_truth": "New York"},
        {"id": "u", "has_answer": False, "question": "q2", "ground_truth": "NOT SPECIFIED"},
    ]
    run_eval.save_system_answer("v1", items[0], _res("New York"))
    run_eval.save_system_answer("v1", items[1], _res("NOT SPECIFIED", abstained=True))

    answer_fn = run_eval.replay_answer_fn("v1")
    judge_fn = lambda q, ref, a: SimpleNamespace(majority="correct", per_judge=("correct",) * 3)
    out = run_eval.score_items(items, answer_fn=answer_fn, judge_fn=judge_fn)

    counts = out["report"]["counts"]
    assert out["report"]["total_items"] == 2
    assert counts["A_CORRECT"] == 1          # answered + jury correct
    assert counts["U_CORRECT_ABSTAIN"] == 1  # abstained flag on unanswerable
