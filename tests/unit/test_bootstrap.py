"""Unit tests for the paired bootstrap — offline, synthetic indicators."""
from __future__ import annotations

import numpy as np
import pytest

from evals.bootstrap import bootstrap_delta


def test_bootstrap_recovers_means_and_improvement_direction():
    x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # M1 = 0.3
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # M1 = 0.1
    r = bootstrap_delta(x, y, n_resamples=3000, seed=0)

    assert r["n"] == 10
    assert abs(r["M1_x"]["mean"] - 0.3) < 0.05
    assert abs(r["M1_y"]["mean"] - 0.1) < 0.05
    assert r["delta"]["mean"] < 0 and r["prob_improved"] > 0.5
    lo, hi = r["M1_x"]["ci95"]
    assert lo <= r["M1_x"]["mean"] <= hi


def test_bootstrap_requires_paired_lengths():
    with pytest.raises(ValueError):
        bootstrap_delta(np.array([1.0, 0.0]), np.array([1.0]))
