"""Unit tests for the eval runner loop — offline, scripted answer_fn + jury.

Verifies the runner wires classify/aggregate correctly and, critically, that the
jury is invoked ONLY for answered items (never for malformed/abstained/etc.).
"""
from __future__ import annotations

from types import SimpleNamespace

from evals.run_eval import score_items


def _res(answer="", *, blocked=False, malformed=False, no_context=False, abstained=False):
    return SimpleNamespace(answer=answer, blocked=blocked, malformed=malformed,
                           no_context=no_context, abstained=abstained)


def test_runner_scores_and_calls_jury_only_on_answered():
    items = [
        {"id": "a1", "has_answer": True,  "question": "q1", "ground_truth": "New York"},
        {"id": "a2", "has_answer": True,  "question": "q2", "ground_truth": "60 days"},   # malformed
        {"id": "u1", "has_answer": False, "question": "q3", "ground_truth": "NOT SPECIFIED"},
        {"id": "u2", "has_answer": False, "question": "q4", "ground_truth": "NOT SPECIFIED"},  # abstained
    ]
    results = {
        "a1": _res("New York"),                 # answered
        "a2": _res(malformed=True),             # malformed -> no jury
        "u1": _res("Exclusive rights granted"), # answered on unanswerable
        "u2": _res("NOT SPECIFIED", abstained=True),  # abstained -> no jury
    }
    verdicts = {"q1": "correct", "q3": "fabricated"}

    jury_calls = []

    def judge_fn(question, reference, answer):
        jury_calls.append(question)
        return SimpleNamespace(majority=verdicts[question], per_judge=(verdicts[question],) * 3)

    out = score_items(items, answer_fn=lambda it: results[it["id"]], judge_fn=judge_fn)

    # jury saw ONLY the two answered items, never the malformed/abstained ones
    assert sorted(jury_calls) == ["q1", "q3"]

    counts = out["report"]["counts"]
    assert counts["A_CORRECT"] == 1 and counts["A_MALFORMED"] == 1
    assert counts["U_FABRICATED"] == 1 and counts["U_CORRECT_ABSTAIN"] == 1

    m1 = out["report"]["M1_hallucination_rate"]
    assert m1["numerator"] == 1 and m1["denominator"] == 4          # only the fabricated item
    assert out["report"]["M4_malformed_rate"]["numerator"] == 1
    assert out["report"]["audit"]["partition_ok"]


def test_runner_records_have_stable_shape():
    items = [{"id": "x", "has_answer": True, "question": "q", "ground_truth": "gt"}]
    out = score_items(items, answer_fn=lambda it: _res("something"),
                      judge_fn=lambda q, r, a: SimpleNamespace(majority="correct", per_judge=("correct",) * 3))
    rec = out["records"][0]
    assert rec["item_id"] == "x" and rec["bucket"] == "A_CORRECT" and rec["disagreement"] is False
