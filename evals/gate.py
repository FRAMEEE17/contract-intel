"""Eval-gate --- the CI merge gate for contract-QA quality (METRICS.md).

Two tiers, matching the honest split between what offline replay can and cannot prove:

  * PR tier (default, deterministic, free).  Replays the pinned baseline arm's
    committed cassettes through the CURRENT scoring code and ratchets the result to
    the frozen numbers in gate_baseline.json. Because replay is bit-identical and
    needs no network (jury verdicts are cached), the margin is zero: any real move
    of the measurement --- a scoring/gold/judge-cassette change --- must fail and
    force a conscious re-pin. It does NOT exercise the RAG pipeline, so it cannot
    catch a live-quality regression; only the live tier re-measures the system.

  * Live tier (--live, scheduled/manual).  After `make eval-live` regenerates a
    candidate arm against the real model + jury (genuinely stochastic), a paired
    bootstrap decides significance: FAIL only if the candidate's M1 is resolved
    worse (whole 95% CI of the delta above zero). Here the statistical margin does
    its real job of absorbing model/jury noise.

Exit codes: 0 = pass; 1 = a quality check regressed; 2 = operational/config error
(missing baseline or cassette, denominator mismatch) --- so CI can tell "the gate
broke" from "quality regressed".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

BASELINE_PATH = Path(__file__).resolve().parent / "gate_baseline.json"
EPS = 1e-9


class GateConfigError(Exception):
    """Operational problem (missing file/cassette, wrong denominator) -> exit 2."""


def _load_baseline(path: Optional[Path] = None) -> dict:
    path = path or BASELINE_PATH  # resolved at call time so the path stays overridable
    if not path.exists():
        raise GateConfigError(f"baseline file not found: {path} (pin one from a live run first)")
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_report(system_version: str) -> dict:
    """Score the committed cassettes for `system_version` through the current code."""
    from evals.jury import judge as jury_judge
    from evals.run_eval import load_gold, replay_answer_fn, score_items

    try:
        out = score_items(
            load_gold(),
            answer_fn=replay_answer_fn(system_version),
            judge_fn=lambda q, ref, a: jury_judge(q, ref, a),
        )
    except SystemExit as exc:  # replay_answer_fn raises SystemExit on a missing cassette
        raise GateConfigError(str(exc)) from exc
    return out["report"]


def _checks(report: dict, baseline: dict) -> list[tuple[bool, str, str]]:
    """Return (passed, name, detail) for each ratchet check. Denominator/config
    problems raise GateConfigError rather than reporting a soft failure."""
    n = report["total_items"]
    if n != baseline["n_items"]:
        raise GateConfigError(
            f"denominator mismatch: replay n={n} but baseline n={baseline['n_items']} "
            "(gate requires the full gold set --- do not run with LIMIT>0)"
        )

    audit = report.get("audit", {})
    if not (audit.get("partition_ok") and audit.get("m1_le_upper")):
        raise GateConfigError(f"report audit invariants not satisfied: {audit}")

    m1_upper = report["M1_hallucination_rate"]["conservative_upper"]["value"]
    m2 = report["M2_answer_correctness"]["value"]
    false_abst = report["M3_abstention"]["false_abstention_rate"]["value"]
    indet = report["diagnostics"]["indeterminate_rate"]

    def line(passed, name, got, op, ref):
        return (passed, name, f"{got:.4f} {op} {ref:.4f}")

    return [
        line(m1_upper <= baseline["M1_upper"] + EPS, "M1_upper (hallucination ceiling)",
             m1_upper, "<=", baseline["M1_upper"]),
        line(m2 >= baseline["M2_answer_correctness"] - EPS, "M2 (correctness floor)",
             m2, ">=", baseline["M2_answer_correctness"]),
        line(false_abst <= baseline["false_abstention_rate"] + EPS, "false_abstention (ceiling)",
             false_abst, "<=", baseline["false_abstention_rate"]),
        line(indet <= EPS, "indeterminate (must be 0)", indet, "<=", 0.0),
    ]


def _run_pr_gate(system_version: str, baseline: dict, *, label: str = "PR gate") -> int:
    """Replay `system_version`, ratchet to `baseline`. Returns an exit code."""
    print(f"-> {label}: replaying arm '{system_version}' vs pinned baseline "
          f"'{baseline['system_version']}' (n={baseline['n_items']})")
    report = _replay_report(system_version)
    results = _checks(report, baseline)
    for passed, name, detail in results:
        print(f"   {'PASS' if passed else 'FAIL'}  {name:34s} {detail}")
    ok = all(passed for passed, _, _ in results)
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    print("   NOTE: replay does not exercise the RAG pipeline; run the live tier "
          "(--live) to re-measure the system itself.")
    return 0 if ok else 1


def _run_live_gate(candidate: str, baseline: dict) -> int:
    """Paired bootstrap: FAIL only if candidate M1 is resolved worse than baseline."""
    from evals.bootstrap import bootstrap_delta, hallucination_flags

    base_arm = baseline["system_version"]
    print(f"-> live gate: paired bootstrap candidate '{candidate}' vs baseline '{base_arm}'")
    try:
        res = bootstrap_delta(hallucination_flags(base_arm), hallucination_flags(candidate))
    except SystemExit as exc:
        raise GateConfigError(str(exc)) from exc

    lo, hi = res["delta"]["ci95"]  # delta = candidate - baseline; >0 means worse
    resolved_worse = lo > 0.0
    print(f"   M1 baseline={res['M1_x']['mean']:.3f}  candidate={res['M1_y']['mean']:.3f}")
    print(f"   delta={res['delta']['mean']:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]")
    if resolved_worse:
        print("   FAIL  candidate is resolved WORSE (entire 95% CI above 0)")
        return 1
    print("   PASS  candidate not resolved worse; if it is resolved BETTER, re-pin the baseline")
    return 0


def _self_test(baseline: dict) -> int:
    """Prove the gate has teeth with REAL committed data:
       (1) the baseline arm passes; (2) the strictly-worse v1 arm fails on M1."""
    print("== self-test: gate must PASS the baseline arm and FAIL a worse arm ==")
    good = _run_pr_gate(baseline["system_version"], baseline, label="self-test/pass-case")
    print()
    bad = _run_pr_gate("v1", baseline, label="self-test/fail-case (v1 is strictly worse)")
    ok = (good == 0) and (bad == 1)
    print(f"\n== self-test {'PASSED' if ok else 'FAILED'}: "
          f"pass-case exit={good} (want 0), fail-case exit={bad} (want 1) ==")
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Contract-QA eval-gate (PR ratchet + live bootstrap).")
    ap.add_argument("--live", action="store_true", help="live tier: paired bootstrap vs baseline")
    ap.add_argument("--candidate", default=None, help="arm to bootstrap against baseline (with --live)")
    ap.add_argument("--self-test", action="store_true", help="prove the gate fails on a worse arm")
    args = ap.parse_args(argv)

    try:
        baseline = _load_baseline()
        if args.self_test:
            return _self_test(baseline)
        if args.live:
            if not args.candidate:
                raise GateConfigError("--live requires --candidate <arm>")
            return _run_live_gate(args.candidate, baseline)
        return _run_pr_gate(baseline["system_version"], baseline)
    except GateConfigError as exc:
        print(f"x gate config error: {exc}")
        return 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
