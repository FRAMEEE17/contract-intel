"""Eval scoring: map each gold item's outcome to a bucket, aggregate into M1-M4.

Grounded in evals/METRICS.md §0–§3. Core rules:

  * The 3-model jury is called for EXACTLY ONE outcome: `answered` (all failure
    flags False). Every other outcome is decided by flags + has_answer alone, so a
    fail-closed "NOT SPECIFIED"/"" can never be laundered by the jury into a
    success/abstention bucket.
  * Flag precedence resolves the no_context⊂abstained overlap:
        blocked > malformed > no_context > abstained > answered
  * malformed / blocked / no_context / indeterminate are failure buckets; they
    sit only in denominators, never in the M1 numerator and never credited as a
    correct-abstain. (§0 "malformed = counted failure"; §1 hallucination = a false
    *assertion*, which these are not.)
  * no_context is a RETRIEVAL failure (M5), kept out of the M3 abstention numerators
    so a retrieval miss is never credited as a correct abstention.
  * Headline M1 is reported as an INTERVAL [M1, M1_upper]; M1_upper folds the
    jury-unresolved items in, so disagreement cannot game the number downward. The
    CI gate keys on M1_upper.

Everything here is a pure function of its inputs (the jury is injected), so it is
unit-testable offline.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

# 13 buckets, mutually exclusive & exhaustive
A_BUCKETS = (
    "A_CORRECT", "A_INCORRECT", "A_FALSE_ABSTENTION",
    "A_NO_CONTEXT", "A_MALFORMED", "A_BLOCKED", "A_INDETERMINATE",
)
U_BUCKETS = (
    "U_CORRECT_ABSTAIN", "U_FABRICATED",
    "U_NO_CONTEXT", "U_MALFORMED", "U_BLOCKED", "U_INDETERMINATE",
)
ALL_BUCKETS = A_BUCKETS + U_BUCKETS

PRECEDENCE = ("blocked", "malformed", "no_context", "abstained", "answered")


def primary_outcome(result) -> str:
    """The single primary outcome of an AnswerResult, by fixed precedence."""
    if result.blocked:
        return "blocked"
    if result.malformed:
        return "malformed"
    if result.no_context:   # must precede 'abstained': no_context also sets abstained
        return "no_context"
    if result.abstained:
        return "abstained"
    return "answered"


def classify(*, has_answer: bool, result, judge: Callable[[], object]):
    """Return (bucket, jury_majority, per_judge).

    `judge` is a zero-arg callable returning an object with `.majority` and
    `.per_judge` (a JuryResult). It is invoked ONLY for an `answered` outcome; for
    every other outcome the return is decided by flags and judge is never called.
    """
    outcome = primary_outcome(result)
    tag = "A" if has_answer else "U"

    if outcome == "blocked":
        return f"{tag}_BLOCKED", None, None
    if outcome == "malformed":
        return f"{tag}_MALFORMED", None, None
    if outcome == "no_context":
        return f"{tag}_NO_CONTEXT", None, None
    if outcome == "abstained":
        return ("A_FALSE_ABSTENTION" if has_answer else "U_CORRECT_ABSTAIN"), None, None

    # answered -> the one semantic judgment that needs the jury
    jr = judge()
    majority: Optional[str] = getattr(jr, "majority", None)
    per_judge = tuple(getattr(jr, "per_judge", ()) or ())
    if has_answer:
        if majority == "correct":
            bucket = "A_CORRECT"
        elif majority == "incorrect":
            bucket = "A_INCORRECT"
        else:  # None, or a verdict incoherent with an answerable item -> not silently merged
            bucket = "A_INDETERMINATE"
    else:
        if majority == "fabricated":
            bucket = "U_FABRICATED"
        elif majority == "correct_abstain":
            bucket = "U_CORRECT_ABSTAIN"
        else:  # off-axis (correct/incorrect/None) on an unanswerable ref => jury incoherent
            bucket = "U_INDETERMINATE"
    return bucket, majority, per_judge


def _rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def aggregate(records: Sequence[dict]) -> dict:
    """Aggregate per-item records into the metric report.

    Each record: {"bucket": str, "per_judge": tuple|None, "disagreement": bool|None}.
    (`bucket` must be one of ALL_BUCKETS.)
    """
    counts = {b: 0 for b in ALL_BUCKETS}
    for r in records:
        counts[r["bucket"]] += 1

    total = len(records)
    answerable = sum(counts[b] for b in A_BUCKETS)
    unanswerable = sum(counts[b] for b in U_BUCKETS)

    incorrect = counts["A_INCORRECT"]
    fabricated = counts["U_FABRICATED"]
    indeterminate = counts["A_INDETERMINATE"] + counts["U_INDETERMINATE"]
    m1_num = incorrect + fabricated

    # §4 sensitivity: answered items where ANY single judge flagged incorrect/fabricated
    any_flag = sum(1 for r in records
                   if r.get("per_judge") and any(v in ("incorrect", "fabricated") for v in r["per_judge"]))
    judged = [r for r in records if r.get("per_judge")]
    disagreements = sum(1 for r in judged if r.get("disagreement"))

    report = {
        "total_items": total,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "counts": counts,
        "M1_hallucination_rate": {
            "value": _rate(m1_num, total),
            "numerator": m1_num, "denominator": total,
            "incorrect": {"count": incorrect, "rate": _rate(incorrect, total)},
            "fabricated": {"count": fabricated, "rate": _rate(fabricated, total)},
            "conservative_upper": {  # folds jury-unresolved in; the CI gate keys on this
                "value": _rate(m1_num + indeterminate, total),
                "numerator": m1_num + indeterminate, "denominator": total,
            },
            "sensitivity_any_judge": {"value": _rate(any_flag, total), "numerator": any_flag,
                                      "denominator": total},
        },
        "M2_answer_correctness": {
            "value": _rate(counts["A_CORRECT"], answerable),
            "numerator": counts["A_CORRECT"], "denominator": answerable,
        },
        "M3_abstention": {
            "correct_abstain_rate": {"value": _rate(counts["U_CORRECT_ABSTAIN"], unanswerable),
                                     "numerator": counts["U_CORRECT_ABSTAIN"], "denominator": unanswerable},
            "false_abstention_rate": {"value": _rate(counts["A_FALSE_ABSTENTION"], answerable),
                                      "numerator": counts["A_FALSE_ABSTENTION"], "denominator": answerable},
        },
        "M4_malformed_rate": {
            "value": _rate(counts["A_MALFORMED"] + counts["U_MALFORMED"], total),
            "numerator": counts["A_MALFORMED"] + counts["U_MALFORMED"], "denominator": total,
        },
        "diagnostics": {
            "no_context_rate": _rate(counts["A_NO_CONTEXT"] + counts["U_NO_CONTEXT"], total),
            "blocked_rate": _rate(counts["A_BLOCKED"] + counts["U_BLOCKED"], total),
            "indeterminate_rate": _rate(indeterminate, total),
            "jury_disagreement_rate": _rate(disagreements, len(judged)),
            "jury_graded_items": len(judged),
        },
        "audit": {
            "partition_ok": sum(counts.values()) == total,
            "answerable_plus_unanswerable_eq_total": answerable + unanswerable == total,
            "flag_precedence": list(PRECEDENCE),
        },
    }
    up = report["M1_hallucination_rate"]["conservative_upper"]["value"]
    lo = report["M1_hallucination_rate"]["value"]
    report["audit"]["m1_le_upper"] = (lo is None) or (up is None) or (lo <= up <= 1.0)
    return report
