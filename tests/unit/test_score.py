"""Unit tests for eval scoring — totality of classify + exact metric formulas.

Verifies the honesty invariants: the jury is called ONLY for `answered`; failure
buckets (malformed/blocked/no_context/indeterminate) never enter the M1 numerator
or an abstention numerator; M1 is an interval [M1, M1_upper]; denominators match
METRICS.md §3 (M1/M4 over total, M2 over answerable, M3 split over U and A).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.score import ALL_BUCKETS, aggregate, classify, primary_outcome


def _result(*, blocked=False, malformed=False, no_context=False, abstained=False):
    return SimpleNamespace(blocked=blocked, malformed=malformed,
                           no_context=no_context, abstained=abstained)


def _jury(majority, per_judge=(), disagreement=False):
    return SimpleNamespace(majority=majority, per_judge=per_judge, disagreement=disagreement)


def _boom():  # a judge that must never be called for non-answered outcomes
    raise AssertionError("jury called on a non-answered outcome")


# ---- precedence ------------------------------------------------------------
def test_precedence_blocked_beats_all():
    assert primary_outcome(_result(blocked=True, malformed=True, no_context=True, abstained=True)) == "blocked"


def test_precedence_no_context_beats_abstained():
    # no_context sets abstained too; precedence must pick no_context
    assert primary_outcome(_result(no_context=True, abstained=True)) == "no_context"


def test_precedence_malformed_beats_no_context():
    assert primary_outcome(_result(malformed=True, no_context=True)) == "malformed"


# ---- flag-decided cells never call the jury --------------------------------
@pytest.mark.parametrize("has_answer,kwargs,expected", [
    (True,  dict(blocked=True),    "A_BLOCKED"),
    (False, dict(blocked=True),    "U_BLOCKED"),
    (True,  dict(malformed=True),  "A_MALFORMED"),
    (False, dict(malformed=True),  "U_MALFORMED"),
    (True,  dict(no_context=True, abstained=True), "A_NO_CONTEXT"),
    (False, dict(no_context=True, abstained=True), "U_NO_CONTEXT"),
    (True,  dict(abstained=True),  "A_FALSE_ABSTENTION"),
    (False, dict(abstained=True),  "U_CORRECT_ABSTAIN"),
])
def test_flag_cells_no_jury(has_answer, kwargs, expected):
    bucket, majority, per_judge = classify(has_answer=has_answer, result=_result(**kwargs), judge=_boom)
    assert bucket == expected and majority is None and per_judge is None


# ---- answered cells: jury decides ------------------------------------------
@pytest.mark.parametrize("has_answer,majority,expected", [
    (True,  "correct",         "A_CORRECT"),
    (True,  "incorrect",       "A_INCORRECT"),
    (True,  "abstained",       "A_INDETERMINATE"),   # off-axis on answerable
    (True,  None,              "A_INDETERMINATE"),   # no 2-of-3
    (False, "fabricated",      "U_FABRICATED"),
    (False, "correct_abstain", "U_CORRECT_ABSTAIN"),
    (False, "correct",         "U_INDETERMINATE"),   # off-axis: incoherent, NOT credited correct
    (False, None,              "U_INDETERMINATE"),
])
def test_answered_cells_use_jury(has_answer, majority, expected):
    bucket, got_majority, _ = classify(has_answer=has_answer, result=_result(),
                                       judge=lambda: _jury(majority))
    assert bucket == expected and got_majority == majority


def test_every_classify_bucket_is_known():
    # exhaustively drive all 13 buckets and confirm each is a declared bucket
    seen = set()
    for ha in (True, False):
        for kw in (dict(blocked=True), dict(malformed=True), dict(no_context=True, abstained=True),
                   dict(abstained=True)):
            seen.add(classify(has_answer=ha, result=_result(**kw), judge=_boom)[0])
        for maj in ("correct", "incorrect", "abstained", None, "fabricated", "correct_abstain"):
            seen.add(classify(has_answer=ha, result=_result(), judge=lambda m=maj: _jury(m))[0])
    assert seen == set(ALL_BUCKETS)


# ---- aggregate: exact formulas + denominators ------------------------------
def _recs(counts: dict) -> list:
    return [{"bucket": b} for b, n in counts.items() for _ in range(n)]


def test_aggregate_formulas_and_denominators():
    counts = {
        "A_CORRECT": 5, "A_INCORRECT": 3, "A_FALSE_ABSTENTION": 2, "A_NO_CONTEXT": 1,
        "A_MALFORMED": 1, "A_BLOCKED": 1, "A_INDETERMINATE": 1,               # answerable = 14
        "U_CORRECT_ABSTAIN": 4, "U_FABRICATED": 2, "U_NO_CONTEXT": 1,
        "U_MALFORMED": 1, "U_BLOCKED": 1, "U_INDETERMINATE": 1,               # unanswerable = 10
    }
    rep = aggregate(_recs(counts))
    assert rep["total_items"] == 24 and rep["answerable"] == 14 and rep["unanswerable"] == 10

    m1 = rep["M1_hallucination_rate"]
    assert m1["numerator"] == 5 and m1["denominator"] == 24           # (incorrect 3 + fabricated 2)
    assert m1["value"] == pytest.approx(5 / 24)
    assert m1["incorrect"]["count"] == 3 and m1["fabricated"]["count"] == 2
    assert m1["conservative_upper"]["numerator"] == 7                 # + indeterminate (1+1)
    assert m1["conservative_upper"]["value"] == pytest.approx(7 / 24)

    assert rep["M2_answer_correctness"] == {"value": pytest.approx(5 / 14), "numerator": 5, "denominator": 14}
    m3 = rep["M3_abstention"]
    assert m3["correct_abstain_rate"]["value"] == pytest.approx(4 / 10)   # denom = unanswerable
    assert m3["false_abstention_rate"]["value"] == pytest.approx(2 / 14)  # denom = answerable
    assert rep["M4_malformed_rate"]["value"] == pytest.approx(2 / 24) and rep["M4_malformed_rate"]["denominator"] == 24

    # no_context is a diagnostic, NOT folded into the abstention numerators
    assert rep["diagnostics"]["no_context_rate"] == pytest.approx(2 / 24)
    assert rep["audit"]["partition_ok"] and rep["audit"]["m1_le_upper"]
    assert rep["audit"]["answerable_plus_unanswerable_eq_total"]


def test_m1_upper_ge_m1_and_sensitivity_counts_any_judge_flag():
    records = [
        {"bucket": "A_CORRECT", "per_judge": ("correct", "incorrect", "correct")},   # any-flag: yes
        {"bucket": "A_CORRECT", "per_judge": ("correct", "correct", "correct")},     # no
        {"bucket": "U_CORRECT_ABSTAIN", "per_judge": ("fabricated", "abstained", "correct_abstain")},  # yes
    ]
    rep = aggregate(records)
    assert rep["M1_hallucination_rate"]["sensitivity_any_judge"]["numerator"] == 2
    assert rep["M1_hallucination_rate"]["sensitivity_any_judge"]["value"] == pytest.approx(2 / 3)


def test_aggregate_zero_denominator_returns_none():
    rep = aggregate(_recs({"U_CORRECT_ABSTAIN": 3}))  # no answerable items
    assert rep["answerable"] == 0
    assert rep["M2_answer_correctness"]["value"] is None            # 0 denominator -> None, no crash
    assert rep["M3_abstention"]["false_abstention_rate"]["value"] is None
    assert rep["M3_abstention"]["correct_abstain_rate"]["value"] == pytest.approx(3 / 3)
