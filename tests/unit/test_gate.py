"""Eval-gate --- the gate is code too, so it is under test: it must pass the pinned
baseline, keep its teeth (fail a worse arm), and exit 2 (not silently pass) on config error."""
from __future__ import annotations

import copy

import pytest

from evals import gate


@pytest.fixture
def baseline():
    return gate._load_baseline()


def test_pr_gate_passes_on_pinned_baseline():
    assert gate.main([]) == 0


def test_self_test_has_teeth(capsys):
    # self-test asserts the baseline arm passes AND the strictly-worse v1 arm fails.
    assert gate.main(["--self-test"]) == 0
    out = capsys.readouterr().out
    assert "self-test PASSED" in out


def test_tightened_baseline_makes_current_arm_fail(monkeypatch, baseline):
    evil = copy.deepcopy(baseline)
    evil["M1_upper"] = 0.10           # pretend prod beat azure -> azure now regresses
    monkeypatch.setattr(gate, "_load_baseline", lambda *a, **k: evil)
    assert gate.main([]) == 1         # quality regression -> exit 1


def test_partial_gold_is_config_error_not_pass(monkeypatch):
    import evals.run_eval as run_eval
    full = run_eval.load_gold()
    monkeypatch.setattr(run_eval, "load_gold", lambda *a, **k: full[:5])
    assert gate.main([]) == 2         # denominator mismatch -> exit 2, never a silent pass


def test_missing_baseline_is_config_error(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "BASELINE_PATH", tmp_path / "nope.json")
    assert gate.main([]) == 2
