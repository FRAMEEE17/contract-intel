"""Bootstrap confidence intervals for M1 and the X->Y delta.

Answers "is the improvement real at n=150?" via a paired resample over items —
free (recomputed from committed cassettes, no model calls).
"""
from __future__ import annotations

import numpy as np

from evals.run_eval import load_gold, replay_answer_fn, score_items

HALLUCINATION_BUCKETS = {"A_INCORRECT", "U_FABRICATED"}


def hallucination_flags(system_version: str) -> np.ndarray:
    """Per-item 0/1 hallucination indicator for a committed run (replay, offline)."""
    from evals.jury import judge as jury_judge

    records = score_items(
        load_gold(),
        answer_fn=replay_answer_fn(system_version),
        judge_fn=lambda q, ref, a: jury_judge(q, ref, a),
    )["records"]
    return np.array([r["bucket"] in HALLUCINATION_BUCKETS for r in records], dtype=float)


def _summary(samples: np.ndarray) -> dict:
    return {
        "mean": float(samples.mean()),
        "sd": float(samples.std()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def bootstrap_delta(x_flags: np.ndarray, y_flags: np.ndarray, *,
                    n_resamples: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of M1_x, M1_y, and their delta. x and y must align by item."""
    if len(x_flags) != len(y_flags):
        raise ValueError("x_flags and y_flags must be the same length (paired by item)")
    n = len(x_flags)
    resample = np.random.default_rng(seed).integers(0, n, (n_resamples, n))
    m1_x = x_flags[resample].mean(axis=1)
    m1_y = y_flags[resample].mean(axis=1)
    delta = m1_y - m1_x
    return {
        "n": n,
        "n_resamples": n_resamples,
        "M1_x": _summary(m1_x),
        "M1_y": _summary(m1_y),
        "delta": _summary(delta),
        "prob_improved": float((delta < 0).mean()),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Bootstrap CIs for M1 and the X->Y delta.")
    ap.add_argument("--x", default="v1", help="system-version of the baseline arm")
    ap.add_argument("--y", default="v2", help="system-version of the improved arm")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args(argv)

    result = bootstrap_delta(hallucination_flags(args.x), hallucination_flags(args.y),
                             n_resamples=args.resamples)

    def line(name, s):
        return f"  {name:10s} {s['mean']:.3f}  SD={s['sd']:.3f}  95%CI=[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"

    print(f"n={result['n']}  paired bootstrap B={result['n_resamples']}  (X={args.x}, Y={args.y})\n")
    print(line("M1 X", result["M1_x"]))
    print(line("M1 Y", result["M1_y"]))
    print(line("delta", result["delta"]))
    print(f"  prob improved (delta<0) = {result['prob_improved']:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
